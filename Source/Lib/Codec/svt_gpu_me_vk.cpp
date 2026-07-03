// Vulkan compute backend for the GPU open-loop ME offload (same public API
// as the CUDA backend in svt_gpu_me.cu). Targets integrated GPUs with
// unified memory (developed on Mali-G720 / Radxa Orion O6).
//
// Resource model: three mega-buffer plane pools (full / quarter / sixteenth)
// carved into LRU slots — frames are CPU-pad-copied straight into mapped
// pool memory (no staging, no upload copies), pyramids built by a
// synchronous per-frame downsample submit. Searches run per acquire as one
// command buffer per lane (hme0 -> hme1 -> fullpel per missing pair, with
// pipeline barriers), fenced, results read from a HOST_CACHED buffer.
// Vulkan queue submissions may overlap execution, so uploads are fenced
// before publishing and all intra-submission ordering uses barriers.
#include "svt_gpu_me.h"

#include <vulkan/vulkan.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "downsample_enc.spv.h"
#include "hme0_enc.spv.h"
#include "hme1_enc.spv.h"
#include "fullpel_enc.spv.h"

namespace {

thread_local bool t_isPrefetchThread = false;
inline long long nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

constexpr int B64 = 64;
constexpr int NUM_WORKERS = 2;
constexpr int NUM_LANES = NUM_WORKERS + 1;
constexpr int SLOT_CAP = 64;   // frame slots per plane pool
constexpr int PAIR_CACHE_CAP = 256;
constexpr int RESULT_WORDS = 170; // per b64: u32 sad[85] + u32 mv[85]

struct PushConsts {
    uint32_t a, b, c, d, e, f;
};

struct VkBuf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
};

struct Pipe {
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
};

struct Lane {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer searchCb = VK_NULL_HANDLE, uploadCb = VK_NULL_HANDLE;
    VkFence searchFence = VK_NULL_HANDLE, uploadFence = VK_NULL_HANDLE;
    VkBuf mvL0, mvL1, results, desc;
    VkDescriptorSet hme0Set = VK_NULL_HANDLE, hme1Set = VK_NULL_HANDLE, fpSet = VK_NULL_HANDLE;
};

struct PairEntry {
    std::vector<GpuMeB64Result> results;
    uint64_t lastUse = 0;
    bool ready = false;
};

struct FrameSlot {
    uint64_t gen = 0, pic = 0;
    uint64_t lastUse = 0;
    bool used = false;
};

struct Ctx {
    std::mutex mtx;      // pair cache
    std::mutex frameMtx; // frame pool
    std::mutex meLaneMtx;
    std::mutex queueMtx; // vkQueueSubmit is externally synchronized
    std::condition_variable cv;
    bool initTried = false, ok = false;

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfam = 0;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    Pipe dsPipe, hme0Pipe, hme1Pipe, fpPipe;
    VkDescriptorSet dsSetA = VK_NULL_HANDLE, dsSetB = VK_NULL_HANDLE;
    VkBuf fullPool, quarterPool, sixteenthPool;
    Lane lanes[NUM_LANES];

    int w = 0, h = 0, padW = 0, padH = 0, cols = 0, blocks = 0;
    size_t fullSlotWords = 0, qSlotWords = 0, sSlotWords = 0;

    FrameSlot slots[SLOT_CAP];
    std::map<std::pair<uint64_t, uint64_t>, int> frameMap; // (gen,pic) -> slot
    std::map<std::array<uint64_t, 3>, std::shared_ptr<PairEntry>> pairs;
    uint64_t useCounter = 0;
    uint64_t pairsComputed = 0, cacheHits = 0;
    long long gpuNsWorker = 0, gpuNsMe = 0, waitNsMe = 0;
    uint64_t meWaits = 0, meComputes = 0;
    std::atomic<uint64_t> acquireFails{0};
};

Ctx g_ctx;
thread_local int t_laneIdx = NUM_LANES - 1;

#define VKC(call)                                                                     \
    do {                                                                              \
        VkResult r_ = (call);                                                         \
        if (r_ != VK_SUCCESS) {                                                       \
            fprintf(stderr, "[GPU_ME/VK] error %d at %s:%d\n", r_, __FILE__, __LINE__); \
            return false;                                                             \
        }                                                                             \
    } while (0)

// Detached prefetch workers can race a queue submission against process
// teardown, which wedges the Mali DDK's cleanup. Take the queue mutex for
// good and drain the device before the process exits.
void quiesceAtExit() {
    g_ctx.queueMtx.lock(); // never released: blocks any further submits
    if (g_ctx.dev) vkDeviceWaitIdle(g_ctx.dev);
}

void reportStats() {
    fprintf(stderr,
            "[GPU_ME] pairs: %llu  hits: %llu | gpu ms (worker/ME): %.0f/%.0f | ME cv-wait ms: %.0f "
            "(%llu waits, %llu ME computes, %llu acquire fails)\n",
            (unsigned long long)g_ctx.pairsComputed, (unsigned long long)g_ctx.cacheHits, g_ctx.gpuNsWorker / 1e6,
            g_ctx.gpuNsMe / 1e6, g_ctx.waitNsMe / 1e6, (unsigned long long)g_ctx.meWaits,
            (unsigned long long)g_ctx.meComputes, (unsigned long long)g_ctx.acquireFails.load());
}

bool createBuf(VkDeviceSize size, VkBuf& b, bool wantCached) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKC(vkCreateBuffer(g_ctx.dev, &bci, nullptr, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_ctx.dev, b.buf, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_ctx.phys, &mp);
    const VkMemoryPropertyFlags base = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t idx = UINT32_MAX;
    if (wantCached) {
        const VkMemoryPropertyFlags cached = base | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & cached) == cached) {
                idx = i;
                break;
            }
    }
    if (idx == UINT32_MAX)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & base) == base) {
                idx = i;
                break;
            }
    if (idx == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = idx;
    VKC(vkAllocateMemory(g_ctx.dev, &mai, nullptr, &b.mem));
    VKC(vkBindBufferMemory(g_ctx.dev, b.buf, b.mem, 0));
    VKC(vkMapMemory(g_ctx.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return true;
}

bool createPipe(const uint32_t* code, size_t codeBytes, int numBindings, Pipe& p) {
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = codeBytes;
    smi.pCode = code;
    VkShaderModule mod;
    VKC(vkCreateShaderModule(g_ctx.dev, &smi, nullptr, &mod));
    std::vector<VkDescriptorSetLayoutBinding> binds(numBindings);
    for (int i = 0; i < numBindings; i++)
        binds[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = (uint32_t)binds.size();
    dli.pBindings = binds.data();
    VKC(vkCreateDescriptorSetLayout(g_ctx.dev, &dli, nullptr, &p.dsl));
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConsts)};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VKC(vkCreatePipelineLayout(g_ctx.dev, &pli, nullptr, &p.layout));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = mod;
    cpi.stage.pName = "main";
    cpi.layout = p.layout;
    VKC(vkCreateComputePipelines(g_ctx.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &p.pipe));
    vkDestroyShaderModule(g_ctx.dev, mod, nullptr);
    return true;
}

VkDescriptorSet makeSet(const Pipe& p, std::vector<VkBuffer> bufs) {
    VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsa.descriptorPool = g_ctx.descPool;
    dsa.descriptorSetCount = 1;
    dsa.pSetLayouts = &p.dsl;
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(g_ctx.dev, &dsa, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
    std::vector<VkWriteDescriptorSet> wds(bufs.size());
    for (size_t i = 0; i < bufs.size(); i++) {
        dbi[i] = {bufs[i], 0, VK_WHOLE_SIZE};
        wds[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wds[i].dstSet = set;
        wds[i].dstBinding = (uint32_t)i;
        wds[i].descriptorCount = 1;
        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wds[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(g_ctx.dev, (uint32_t)wds.size(), wds.data(), 0, nullptr);
    return set;
}

void barrierCompute(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                         0, nullptr, 0, nullptr);
}

bool submitFenced(VkCommandBuffer cb, VkFence fence) {
    VKC(vkResetFences(g_ctx.dev, 1, &fence));
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    {
        std::lock_guard<std::mutex> qlk(g_ctx.queueMtx);
        VKC(vkQueueSubmit(g_ctx.queue, 1, &si, fence));
    }
    return true;
}

bool initContext(int w, int h) {
    g_ctx.initTried = true;
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "svt_gpu_me_vk";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VKC(vkCreateInstance(&ici, nullptr, &g_ctx.instance));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_ctx.instance, &n, nullptr);
    if (!n) return false;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g_ctx.instance, &n, devs.data());
    for (auto d : devs) {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &qn, qf.data());
        for (uint32_t i = 0; i < qn; i++)
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                g_ctx.phys = d;
                g_ctx.qfam = i;
                break;
            }
        if (g_ctx.phys) break;
    }
    if (!g_ctx.phys) return false;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(g_ctx.phys, &props);
    fprintf(stderr, "[GPU_ME] Vulkan device: %s\n", props.deviceName);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_ctx.qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VKC(vkCreateDevice(g_ctx.phys, &dci, nullptr, &g_ctx.dev));
    vkGetDeviceQueue(g_ctx.dev, g_ctx.qfam, 0, &g_ctx.queue);

    g_ctx.w = w;
    g_ctx.h = h;
    g_ctx.padW = (w + B64 - 1) / B64 * B64;
    g_ctx.padH = (h + B64 - 1) / B64 * B64;
    g_ctx.cols = g_ctx.padW / B64;
    g_ctx.blocks = g_ctx.cols * (g_ctx.padH / B64);
    g_ctx.fullSlotWords = (size_t)g_ctx.padW * g_ctx.padH / 4;
    g_ctx.qSlotWords = g_ctx.fullSlotWords / 4;
    g_ctx.sSlotWords = g_ctx.fullSlotWords / 16;

    if (!createBuf(4ull * g_ctx.fullSlotWords * SLOT_CAP + 16, g_ctx.fullPool, false)) return false;
    if (!createBuf(4ull * g_ctx.qSlotWords * SLOT_CAP + 16, g_ctx.quarterPool, false)) return false;
    if (!createBuf(4ull * g_ctx.sSlotWords * SLOT_CAP + 16, g_ctx.sixteenthPool, false)) return false;

    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 32;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VKC(vkCreateDescriptorPool(g_ctx.dev, &dpi, nullptr, &g_ctx.descPool));

    if (!createPipe(downsample_enc_spv, sizeof(downsample_enc_spv), 2, g_ctx.dsPipe)) return false;
    if (!createPipe(hme0_enc_spv, sizeof(hme0_enc_spv), 3, g_ctx.hme0Pipe)) return false;
    if (!createPipe(hme1_enc_spv, sizeof(hme1_enc_spv), 4, g_ctx.hme1Pipe)) return false;
    if (!createPipe(fullpel_enc_spv, sizeof(fullpel_enc_spv), 4, g_ctx.fpPipe)) return false;

    g_ctx.dsSetA = makeSet(g_ctx.dsPipe, {g_ctx.fullPool.buf, g_ctx.quarterPool.buf});
    g_ctx.dsSetB = makeSet(g_ctx.dsPipe, {g_ctx.quarterPool.buf, g_ctx.sixteenthPool.buf});
    if (!g_ctx.dsSetA || !g_ctx.dsSetB) return false;

    for (int i = 0; i < NUM_LANES; i++) {
        Lane& L = g_ctx.lanes[i];
        VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi.queueFamilyIndex = g_ctx.qfam;
        VKC(vkCreateCommandPool(g_ctx.dev, &cpi, nullptr, &L.pool));
        VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = L.pool;
        cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = 2;
        VkCommandBuffer cbs[2];
        VKC(vkAllocateCommandBuffers(g_ctx.dev, &cai, cbs));
        L.searchCb = cbs[0];
        L.uploadCb = cbs[1];
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VKC(vkCreateFence(g_ctx.dev, &fci, nullptr, &L.searchFence));
        VKC(vkCreateFence(g_ctx.dev, &fci, nullptr, &L.uploadFence));
        if (!createBuf(4ull * g_ctx.blocks * GPU_ME_MAX_REFS, L.mvL0, false)) return false;
        if (!createBuf(4ull * g_ctx.blocks * GPU_ME_MAX_REFS, L.mvL1, false)) return false;
        if (!createBuf(4ull * RESULT_WORDS * g_ctx.blocks * GPU_ME_MAX_REFS, L.results, true)) return false;
        if (!createBuf(4ull * 8 * GPU_ME_MAX_REFS, L.desc, false)) return false;
        L.hme0Set = makeSet(g_ctx.hme0Pipe, {g_ctx.sixteenthPool.buf, L.mvL0.buf, L.desc.buf});
        L.hme1Set = makeSet(g_ctx.hme1Pipe, {g_ctx.quarterPool.buf, L.mvL0.buf, L.mvL1.buf, L.desc.buf});
        L.fpSet = makeSet(g_ctx.fpPipe, {g_ctx.fullPool.buf, L.mvL1.buf, L.results.buf, L.desc.buf});
        if (!L.hme0Set || !L.hme1Set || !L.fpSet) return false;
    }
    atexit(reportStats);
    atexit(quiesceAtExit); // LIFO: runs before reportStats
    return true;
}

// Under frameMtx: CPU pad-copy the strided source into a pool slot, then run
// the pyramid downsamples synchronously (submission overlap on Vulkan queues
// means the frame must be complete before any later search references it).
int ensureFrame(Lane& L, uint64_t gen, uint64_t pic, const uint8_t* y, int stride) {
    auto it = g_ctx.frameMap.find({gen, pic});
    if (it != g_ctx.frameMap.end()) {
        g_ctx.slots[it->second].lastUse = ++g_ctx.useCounter;
        return it->second;
    }
    // LRU slot
    int victim = 0;
    for (int i = 0; i < SLOT_CAP; i++) {
        if (!g_ctx.slots[i].used) {
            victim = i;
            break;
        }
        if (g_ctx.slots[i].lastUse < g_ctx.slots[victim].lastUse) victim = i;
    }
    if (g_ctx.slots[victim].used) g_ctx.frameMap.erase({g_ctx.slots[victim].gen, g_ctx.slots[victim].pic});

    // CPU pad-copy into mapped pool memory (unified memory).
    uint8_t* dst = (uint8_t*)g_ctx.fullPool.map + 4ull * g_ctx.fullSlotWords * victim;
    for (int r = 0; r < g_ctx.h; r++) {
        uint8_t* d = dst + (size_t)r * g_ctx.padW;
        memcpy(d, y + (size_t)r * stride, g_ctx.w);
        memset(d + g_ctx.w, d[g_ctx.w - 1], g_ctx.padW - g_ctx.w);
    }
    for (int r = g_ctx.h; r < g_ctx.padH; r++)
        memcpy(dst + (size_t)r * g_ctx.padW, dst + (size_t)(g_ctx.h - 1) * g_ctx.padW, g_ctx.padW);

    // Pyramid: synchronous downsample submit on this lane's upload cmdbuf.
    if (vkWaitForFences(g_ctx.dev, 1, &L.uploadFence, VK_TRUE, ~0ull) != VK_SUCCESS) return -1;
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(L.uploadCb, &bi) != VK_SUCCESS) return -1;
    uint32_t qw = g_ctx.padW / 2, qh = g_ctx.padH / 2, sw = g_ctx.padW / 4, sh = g_ctx.padH / 4;
    PushConsts pcA = {(uint32_t)(g_ctx.fullSlotWords * victim), (uint32_t)(g_ctx.qSlotWords * victim),
                      (uint32_t)g_ctx.padW, qw, qh, 0};
    vkCmdBindPipeline(L.uploadCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.dsPipe.pipe);
    vkCmdBindDescriptorSets(L.uploadCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.dsPipe.layout, 0, 1, &g_ctx.dsSetA, 0,
                            nullptr);
    vkCmdPushConstants(L.uploadCb, g_ctx.dsPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcA), &pcA);
    vkCmdDispatch(L.uploadCb, (qw / 4 * qh + 63) / 64, 1, 1);
    barrierCompute(L.uploadCb);
    PushConsts pcB = {(uint32_t)(g_ctx.qSlotWords * victim), (uint32_t)(g_ctx.sSlotWords * victim), qw, sw, sh, 0};
    vkCmdBindDescriptorSets(L.uploadCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.dsPipe.layout, 0, 1, &g_ctx.dsSetB, 0,
                            nullptr);
    vkCmdPushConstants(L.uploadCb, g_ctx.dsPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcB), &pcB);
    vkCmdDispatch(L.uploadCb, (sw / 4 * sh + 63) / 64, 1, 1);
    if (vkEndCommandBuffer(L.uploadCb) != VK_SUCCESS) return -1;
    if (!submitFenced(L.uploadCb, L.uploadFence)) return -1;
    if (vkWaitForFences(g_ctx.dev, 1, &L.uploadFence, VK_TRUE, ~0ull) != VK_SUCCESS) return -1;

    g_ctx.slots[victim] = {gen, pic, ++g_ctx.useCounter, true};
    g_ctx.frameMap[{gen, pic}] = victim;
    return victim;
}

// Under mtx.
void evictPairsLocked() {
    while (g_ctx.pairs.size() >= PAIR_CACHE_CAP) {
        auto lru = g_ctx.pairs.end();
        for (auto it = g_ctx.pairs.begin(); it != g_ctx.pairs.end(); ++it)
            if (it->second->ready && (lru == g_ctx.pairs.end() || it->second->lastUse < lru->second->lastUse))
                lru = it;
        if (lru == g_ctx.pairs.end()) return;
        g_ctx.pairs.erase(lru);
    }
}

struct PicHandle {
    std::shared_ptr<PairEntry> entries[GPU_ME_MAX_REFS];
    int n = 0;
};

} // namespace

extern "C" int svt_gpu_me_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SVT_GPU_ME");
        enabled = (env && env[0] == '1') ? 1 : 0;
        if (enabled) fprintf(stderr, "[GPU_ME] GPU open-loop ME offload ENABLED (Vulkan)\n");
    }
    return enabled;
}

extern "C" void* svt_gpu_me_acquire(const uint8_t* src_y, int src_stride, uint64_t src_pic_num,
                                    uint64_t generation, const GpuMeRef* refs, int num_refs, int width, int height,
                                    const GpuMeB64Result** bases) {
    if (num_refs <= 0 || num_refs > GPU_ME_MAX_REFS) return nullptr;

    std::shared_ptr<PairEntry> entries[GPU_ME_MAX_REFS];
    bool mustCompute[GPU_ME_MAX_REFS] = {};
    int numMisses = 0;
    {
        std::unique_lock<std::mutex> lk(g_ctx.mtx);
        if (!g_ctx.initTried) g_ctx.ok = initContext(width, height);
        if (!g_ctx.ok || width != g_ctx.w || height != g_ctx.h) {
            g_ctx.acquireFails++;
            return nullptr;
        }
        for (int i = 0; i < num_refs; i++) {
            const std::array<uint64_t, 3> key{generation, src_pic_num, refs[i].pic_num};
            auto it = g_ctx.pairs.find(key);
            if (it == g_ctx.pairs.end()) {
                evictPairsLocked();
                entries[i] = std::make_shared<PairEntry>();
                g_ctx.pairs[key] = entries[i];
                mustCompute[i] = true;
                numMisses++;
            } else {
                entries[i] = it->second;
                g_ctx.cacheHits++;
            }
            entries[i]->lastUse = ++g_ctx.useCounter;
        }
    }

    if (numMisses) {
        const bool meThread = !t_isPrefetchThread;
        if (meThread) g_ctx.meLaneMtx.lock();
        Lane& L = g_ctx.lanes[t_laneIdx];

        bool ok = true;
        int slotToRef[GPU_ME_MAX_REFS];
        std::vector<GpuMeB64Result> staged[GPU_ME_MAX_REFS];
        int launched = 0;
        long long gpuT0 = nowNs();
        {
            int srcSlot, refSlot[GPU_ME_MAX_REFS];
            {
                std::lock_guard<std::mutex> flk(g_ctx.frameMtx);
                srcSlot = ensureFrame(L, generation, src_pic_num, src_y, src_stride);
                ok = srcSlot >= 0;
                if (ok)
                    for (int i = 0; i < num_refs; i++) {
                        if (!mustCompute[i]) continue;
                        refSlot[launched] = ensureFrame(L, generation, refs[i].pic_num, refs[i].y, refs[i].stride);
                        if (refSlot[launched] < 0) {
                            ok = false;
                            break;
                        }
                        slotToRef[launched++] = i;
                    }
            }
            if (ok && launched) {
                // Record one command buffer: hme0 -> hme1 -> fullpel per pair.
                ok = vkWaitForFences(g_ctx.dev, 1, &L.searchFence, VK_TRUE, ~0ull) == VK_SUCCESS;
                VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                ok = ok && vkBeginCommandBuffer(L.searchCb, &bi) == VK_SUCCESS;
                if (ok) {
                    // Per-pair slot offsets via the lane's desc SSBO; all
                    // pairs run in one grid.y-batched dispatch per stage.
                    uint32_t* dw = (uint32_t*)L.desc.map;
                    for (int sl = 0; sl < launched; sl++) {
                        int rs = refSlot[sl];
                        dw[sl * 8 + 0] = (uint32_t)(g_ctx.sSlotWords * srcSlot);
                        dw[sl * 8 + 1] = (uint32_t)(g_ctx.sSlotWords * rs);
                        dw[sl * 8 + 2] = (uint32_t)(g_ctx.qSlotWords * srcSlot);
                        dw[sl * 8 + 3] = (uint32_t)(g_ctx.qSlotWords * rs);
                        dw[sl * 8 + 4] = (uint32_t)(g_ctx.fullSlotWords * srcSlot);
                        dw[sl * 8 + 5] = (uint32_t)(g_ctx.fullSlotWords * rs);
                        dw[sl * 8 + 6] = (uint32_t)(RESULT_WORDS * g_ctx.blocks * sl);
                        dw[sl * 8 + 7] = 0;
                    }
                    uint32_t qw = (uint32_t)g_ctx.padW / 2, qh = (uint32_t)g_ctx.padH / 2;
                    uint32_t sw = (uint32_t)g_ctx.padW / 4, shh = (uint32_t)g_ctx.padH / 4;
                    PushConsts p0 = {sw, shh, (uint32_t)g_ctx.cols, (uint32_t)g_ctx.blocks, 0, 0};
                    vkCmdBindPipeline(L.searchCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.hme0Pipe.pipe);
                    vkCmdBindDescriptorSets(L.searchCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.hme0Pipe.layout, 0, 1,
                                            &L.hme0Set, 0, nullptr);
                    vkCmdPushConstants(L.searchCb, g_ctx.hme0Pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(p0), &p0);
                    vkCmdDispatch(L.searchCb, g_ctx.blocks, launched, 1);
                    barrierCompute(L.searchCb);
                    PushConsts p1 = {qw, qh, (uint32_t)g_ctx.cols, (uint32_t)g_ctx.blocks, 0, 0};
                    vkCmdBindPipeline(L.searchCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.hme1Pipe.pipe);
                    vkCmdBindDescriptorSets(L.searchCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.hme1Pipe.layout, 0, 1,
                                            &L.hme1Set, 0, nullptr);
                    vkCmdPushConstants(L.searchCb, g_ctx.hme1Pipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(p1), &p1);
                    vkCmdDispatch(L.searchCb, g_ctx.blocks, launched, 1);
                    barrierCompute(L.searchCb);
                    PushConsts p2 = {(uint32_t)g_ctx.padW, (uint32_t)g_ctx.padH, (uint32_t)g_ctx.cols,
                                     (uint32_t)g_ctx.blocks, 0, 0};
                    vkCmdBindPipeline(L.searchCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.fpPipe.pipe);
                    vkCmdBindDescriptorSets(L.searchCb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ctx.fpPipe.layout, 0, 1,
                                            &L.fpSet, 0, nullptr);
                    vkCmdPushConstants(L.searchCb, g_ctx.fpPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p2),
                                       &p2);
                    vkCmdDispatch(L.searchCb, g_ctx.blocks, launched, 1);
                    VkMemoryBarrier hb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkCmdPipelineBarrier(L.searchCb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0, nullptr);
                    ok = vkEndCommandBuffer(L.searchCb) == VK_SUCCESS && submitFenced(L.searchCb, L.searchFence) &&
                        vkWaitForFences(g_ctx.dev, 1, &L.searchFence, VK_TRUE, ~0ull) == VK_SUCCESS;
                }
                if (ok)
                    for (int sl = 0; sl < launched; sl++) {
                        const GpuMeB64Result* src =
                            (const GpuMeB64Result*)((const uint8_t*)L.results.map +
                                                    4ull * RESULT_WORDS * g_ctx.blocks * sl);
                        staged[sl].assign(src, src + g_ctx.blocks);
                    }
            }
        }
        if (meThread) g_ctx.meLaneMtx.unlock();

        std::lock_guard<std::mutex> lk(g_ctx.mtx);
        if (t_isPrefetchThread)
            g_ctx.gpuNsWorker += nowNs() - gpuT0;
        else {
            g_ctx.gpuNsMe += nowNs() - gpuT0;
            g_ctx.meComputes++;
        }
        if (ok) {
            for (int sl = 0; sl < launched; sl++) {
                int i = slotToRef[sl];
                entries[i]->results = std::move(staged[sl]);
                entries[i]->ready = true;
                g_ctx.pairsComputed++;
            }
        } else {
            for (int i = 0; i < num_refs; i++) {
                if (!mustCompute[i]) continue;
                g_ctx.pairs.erase({generation, src_pic_num, refs[i].pic_num});
                entries[i]->ready = true;
            }
        }
        g_ctx.cv.notify_all();
        if (!ok) {
            g_ctx.acquireFails++;
            return nullptr;
        }
    }

    {
        std::unique_lock<std::mutex> lk(g_ctx.mtx);
        for (int i = 0; i < num_refs; i++) {
            if (!mustCompute[i] && !entries[i]->ready) {
                long long w0 = nowNs();
                g_ctx.cv.wait(lk, [&] { return entries[i]->ready; });
                if (!t_isPrefetchThread) {
                    g_ctx.waitNsMe += nowNs() - w0;
                    g_ctx.meWaits++;
                }
            }
        }
    }
    for (int i = 0; i < num_refs; i++)
        if (entries[i]->results.empty()) {
            g_ctx.acquireFails++;
            return nullptr;
        }

    PicHandle* h = new PicHandle;
    h->n = num_refs;
    for (int i = 0; i < num_refs; i++) {
        h->entries[i] = entries[i];
        bases[i] = entries[i]->results.data();
    }
    return h;
}

extern "C" void svt_gpu_me_release(void* handle) { delete (PicHandle*)handle; }

// ------------------------------------------------------------- prefetch ----

namespace {

struct PrefetchReq {
    const uint8_t* srcY;
    int srcStride;
    uint64_t srcPicNum;
    uint64_t generation;
    GpuMeRef refs[GPU_ME_MAX_REFS];
    int numRefs;
    int w, h;
};

struct PrefetchPool {
    std::mutex qMtx;
    std::condition_variable qCv;
    std::deque<PrefetchReq> queue;
    bool started = false;

    static constexpr size_t QUEUE_CAP = 64;

    void run(int laneIdx) {
        t_isPrefetchThread = true;
        t_laneIdx = laneIdx;
        for (;;) {
            PrefetchReq req;
            {
                std::unique_lock<std::mutex> lk(qMtx);
                qCv.wait(lk, [&] { return !queue.empty(); });
                req = queue.front();
                queue.pop_front();
            }
            const GpuMeB64Result* bases[GPU_ME_MAX_REFS];
            void* h = svt_gpu_me_acquire(req.srcY, req.srcStride, req.srcPicNum, req.generation, req.refs,
                                         req.numRefs, req.w, req.h, bases);
            if (h) svt_gpu_me_release(h);
        }
    }

    void enqueue(const PrefetchReq& req) {
        {
            std::lock_guard<std::mutex> lk(qMtx);
            if (!started) {
                for (int i = 0; i < NUM_WORKERS; i++) std::thread([this, i] { run(i); }).detach();
                started = true;
            }
            if (queue.size() >= QUEUE_CAP) queue.pop_front();
            queue.push_back(req);
        }
        qCv.notify_one();
    }
};

PrefetchPool g_prefetch;

} // namespace

extern "C" void svt_gpu_me_prefetch(const uint8_t* src_y, int src_stride, uint64_t src_pic_num,
                                    uint64_t generation, const GpuMeRef* refs, int num_refs, int width,
                                    int height) {
    if (num_refs <= 0 || num_refs > GPU_ME_MAX_REFS) return;
    PrefetchReq req;
    req.srcY = src_y;
    req.srcStride = src_stride;
    req.srcPicNum = src_pic_num;
    req.generation = generation;
    memcpy(req.refs, refs, sizeof(GpuMeRef) * num_refs);
    req.numRefs = num_refs;
    req.w = width;
    req.h = height;
    g_prefetch.enqueue(req);
}
