// CUDA implementation of the open-loop hierarchical ME offload.
// Kernels are the (bit-exact-verified) design from the standalone PoC:
// 3-level pyramid, HME L0 full search / L1 refine / full-pel 85-sub-block
// search with warp-synchronous aggregation and __vsadu4 SIMD SADs.
#include "svt_gpu_me.h"

#include <cuda_runtime.h>
#include <array>
#include <atomic>
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
#include <chrono>

namespace {
thread_local bool t_isPrefetchThread = false;
inline long long nowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
} // namespace

namespace {

constexpr int B64 = 64;
constexpr int HME0_RANGE = 24; // +/- at 1/16 area -> effective +/-96 full-pel
constexpr int HME1_RANGE = 4;
constexpr int FP_RANGE = 12;

#define GPU_ME_CHECK(call)                    \
    do {                                      \
        if ((call) != cudaSuccess) return false; \
    } while (0)

// ---------------------------------------------------------------- kernels ---

// Copy a strided luma plane into a 64-aligned padded plane, edge-replicated.
__global__ void padKernel(const uint8_t* __restrict__ src, int srcStride, int w, int h, uint8_t* __restrict__ dst,
                          int padW, int padH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= padW || y >= padH) return;
    int sx = x < w ? x : w - 1, sy = y < h ? y : h - 1;
    dst[(size_t)y * padW + x] = src[(size_t)sy * srcStride + sx];
}

__global__ void downsample2xKernel(const uint8_t* __restrict__ in, int inW, uint8_t* __restrict__ out, int outW,
                                   int outH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH) return;
    const uint8_t* r0 = in + (size_t)(2 * y) * inW + 2 * x;
    const uint8_t* r1 = r0 + inW;
    out[(size_t)y * outW + x] = (uint8_t)((r0[0] + r0[1] + r1[0] + r1[1] + 2) >> 2);
}

__device__ inline void clampWindow(int center, int range, int lo, int hi, int& c0, int& c1) {
    c0 = max(center - range, lo);
    c1 = min(center + range, hi);
    if (c0 > c1) c0 = c1 = min(max(center, lo), hi);
}

__device__ inline uint32_t sad8px(uint32_t s0, uint32_t s1, const uint32_t* __restrict__ words, int shiftBits) {
    uint32_t w0 = words[0], w1 = words[1], w2 = words[2];
    return __vsadu4(s0, __funnelshift_r(w0, w1, shiftBits)) + __vsadu4(s1, __funnelshift_r(w1, w2, shiftBits));
}

// Block-wide (min SAD, min position) reduction; returns winner to all threads.
template <int THREADS>
__device__ unsigned long long reduceMinKey(unsigned long long key, unsigned long long* shKey) {
    shKey[threadIdx.x] = key;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shKey[threadIdx.x] = min(shKey[threadIdx.x], shKey[threadIdx.x + s]);
        __syncthreads();
    }
    unsigned long long r = shKey[0];
    __syncthreads();
    return r;
}

// HME L0 phase: 16x16 at 1/16 area, full search around (0,0).
template <int THREADS>
__device__ short2 hme0Phase(const uint8_t* __restrict__ src, const uint8_t* __restrict__ ref, int w, int h, int bx,
                            int by, uint8_t* srcSh, uint8_t* refSh, unsigned long long* shKey) {
    constexpr int REF_STRIDE = 68;
    int x0, x1, y0, y1;
    clampWindow(0, HME0_RANGE, -bx, w - 16 - bx, x0, x1);
    clampWindow(0, HME0_RANGE, -by, h - 16 - by, y0, y1);
    int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
    int rw = 15 + nx, rh = 15 + ny;

    for (int i = threadIdx.x; i < 256; i += THREADS) srcSh[i] = src[(size_t)(by + i / 16) * w + bx + i % 16];
    for (int i = threadIdx.x; i < rh * REF_STRIDE; i += THREADS) {
        int r = i / REF_STRIDE, c = i % REF_STRIDE;
        refSh[i] = c < rw ? ref[(size_t)(by + y0 + r) * w + bx + x0 + c] : 0;
    }
    __syncthreads();

    const uint32_t* srcW = (const uint32_t*)srcSh;
    unsigned long long bestKey = ~0ull;
    int numPos = nx * ny;
    for (int pos = threadIdx.x; pos < numPos; pos += THREADS) {
        int px = pos % nx, py = pos / nx;
        uint32_t sad = 0;
        for (int y = 0; y < 16; y++) {
            int o = (py + y) * REF_STRIDE + px;
            const uint32_t* rw32 = (const uint32_t*)(refSh + (o & ~3));
            int shift = (o & 3) * 8;
            sad += __vsadu4(srcW[y * 4 + 0], __funnelshift_r(rw32[0], rw32[1], shift)) +
                __vsadu4(srcW[y * 4 + 1], __funnelshift_r(rw32[1], rw32[2], shift)) +
                __vsadu4(srcW[y * 4 + 2], __funnelshift_r(rw32[2], rw32[3], shift)) +
                __vsadu4(srcW[y * 4 + 3], __funnelshift_r(rw32[3], rw32[4], shift));
        }
        bestKey = min(bestKey, ((unsigned long long)sad << 24) | (unsigned)pos);
    }
    unsigned long long best = reduceMinKey<THREADS>(bestKey, shKey);
    int pos = (int)(best & 0xffffff);
    return make_short2((short)(x0 + pos % nx), (short)(y0 + pos / nx));
}

// HME L1 phase: 32x32 at 1/4 area around 2x the L0 vector.
template <int THREADS>
__device__ short2 hme1Phase(const uint8_t* __restrict__ src, const uint8_t* __restrict__ ref, int w, int h, int bx,
                            int by, short2 l0, uint8_t* srcSh, uint8_t* refSh, unsigned long long* shKey) {
    constexpr int REF_STRIDE = 44;
    int x0, x1, y0, y1;
    clampWindow(l0.x * 2, HME1_RANGE, -bx, w - 32 - bx, x0, x1);
    clampWindow(l0.y * 2, HME1_RANGE, -by, h - 32 - by, y0, y1);
    int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
    int rw = 31 + nx, rh = 31 + ny;

    for (int i = threadIdx.x; i < 1024; i += THREADS) srcSh[i] = src[(size_t)(by + i / 32) * w + bx + i % 32];
    for (int i = threadIdx.x; i < rh * REF_STRIDE; i += THREADS) {
        int r = i / REF_STRIDE, c = i % REF_STRIDE;
        refSh[i] = c < rw ? ref[(size_t)(by + y0 + r) * w + bx + x0 + c] : 0;
    }
    __syncthreads();

    const uint32_t* srcW = (const uint32_t*)srcSh;
    unsigned long long bestKey = ~0ull;
    int numPos = nx * ny;
    if (threadIdx.x < numPos) {
        int pos = threadIdx.x, px = pos % nx, py = pos / nx;
        uint32_t sad = 0;
        for (int y = 0; y < 32; y++) {
            int o = (py + y) * REF_STRIDE + px;
            const uint32_t* rw32 = (const uint32_t*)(refSh + (o & ~3));
            int shift = (o & 3) * 8;
            uint32_t prev = rw32[0];
            for (int k = 0; k < 8; k++) {
                uint32_t next = rw32[k + 1];
                sad += __vsadu4(srcW[y * 8 + k], __funnelshift_r(prev, next, shift));
                prev = next;
            }
        }
        bestKey = ((unsigned long long)sad << 24) | (unsigned)pos;
    }
    unsigned long long best = reduceMinKey<THREADS>(bestKey, shKey);
    int pos = (int)(best & 0xffffff);
    return make_short2((short)(x0 + pos % nx), (short)(y0 + pos / nx));
}

// SVT-AV1 index of the 16x16 at raster position (x16, y16): z-order grouped
// under the parent 32x32.
__device__ inline int svt16Idx(int x16, int y16) {
    return ((y16 >> 1) * 2 + (x16 >> 1)) * 4 + (y16 & 1) * 2 + (x16 & 1);
}

struct PairPtrs {
    const uint8_t *full, *quarter, *sixteenth;
};
struct MeLaunchArgs {
    PairPtrs src;
    PairPtrs refs[GPU_ME_MAX_REFS];
};

// Fused per-pair search: one launch runs HME L0 -> L1 -> full-pel for every
// 64x64 of every launched reference (grid = (blocks, refs)). The levels of
// one 64x64 depend only on each other, so they chain inside the block with
// shared-memory reuse; results land directly in SVT-AV1's ME layout.
__global__ void mePairKernel(MeLaunchArgs args, int w, int h, int cols, int blocks,
                             GpuMeB64Result* __restrict__ results) {
    constexpr int WARPS = 8;
    constexpr int THREADS = WARPS * 32;
    constexpr int REF_STRIDE = 92;
    int blk = blockIdx.x, ri = blockIdx.y;
    int bx = (blk % cols) * B64, by = (blk / cols) * B64;
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const uint8_t* src = args.src.full;
    const uint8_t* ref = args.refs[ri].full;

    __shared__ __align__(4) uint8_t srcShSmall[32 * 32];
    __shared__ __align__(4) uint8_t refShSmall[64 * 68 + 4];
    __shared__ unsigned long long shKey[THREADS];

    short2 l0 = hme0Phase<THREADS>(args.src.sixteenth, args.refs[ri].sixteenth, w / 4, h / 4, (blk % cols) * 16,
                                   (blk / cols) * 16, srcShSmall, refShSmall, shKey);
    short2 l1 = hme1Phase<THREADS>(args.src.quarter, args.refs[ri].quarter, w / 2, h / 2, (blk % cols) * 32,
                                   (blk / cols) * 32, l0, srcShSmall, refShSmall, shKey);
    // Full-pel searches two windows: around the HME predictor and around
    // (0,0) — SVT's CPU search has an equivalent zero-MV check; without it,
    // HME wandering on flat/noisy blocks yields worse-than-zero MVs.
    int winCx[2] = {l1.x * 2, 0}, winCy[2] = {l1.y * 2, 0};
    int numWin = (l1.x | l1.y) ? 2 : 1;

    __shared__ __align__(4) uint8_t refSh[88 * REF_STRIDE + 4];
    int sub8x = (lane & 7) * 8, sub8y = (lane >> 3) * 8;
    uint32_t srcA[16], srcB[16];
    {
        const uint8_t* pA = src + (size_t)(by + sub8y) * w + bx + sub8x;
        const uint8_t* pB = pA + (size_t)32 * w;
        for (int y = 0; y < 8; y++) {
            srcA[y * 2] = *(const uint32_t*)(pA + (size_t)y * w);
            srcA[y * 2 + 1] = *(const uint32_t*)(pA + (size_t)y * w + 4);
            srcB[y * 2] = *(const uint32_t*)(pB + (size_t)y * w);
            srcB[y * 2 + 1] = *(const uint32_t*)(pB + (size_t)y * w + 4);
        }
    }
    __shared__ uint32_t scr[WARPS][84];
    unsigned long long key8A = ~0ull, key8B = ~0ull, key16 = ~0ull, key32 = ~0ull, key64 = ~0ull;

    for (int wi = 0; wi < numWin; wi++) {
    int x0, x1, y0, y1;
    clampWindow(winCx[wi], FP_RANGE, -bx, w - B64 - bx, x0, x1);
    clampWindow(winCy[wi], FP_RANGE, -by, h - B64 - by, y0, y1);
    int nx = x1 - x0 + 1, ny = y1 - y0 + 1;
    int numPos = nx * ny;
    __syncthreads(); // previous window's refSh readers done
    {
        int rw = B64 - 1 + nx, rh = B64 - 1 + ny;
        for (int i = tid; i < rh * REF_STRIDE; i += THREADS) {
            int r = i / REF_STRIDE, c = i % REF_STRIDE;
            refSh[i] = c < rw ? ref[(size_t)(by + y0 + r) * w + bx + x0 + c] : 0;
        }
    }
    __syncthreads();

    for (int pos = warp; pos < numPos; pos += WARPS) {
        int px = pos % nx, py = pos / nx;
        uint32_t sadA = 0, sadB = 0;
        {
            int oA = (py + sub8y) * REF_STRIDE + px + sub8x;
            int shift = (oA & 3) * 8;
            for (int y = 0; y < 8; y++) {
                const uint32_t* rowA = (const uint32_t*)(refSh + ((oA + y * REF_STRIDE) & ~3));
                const uint32_t* rowB = rowA + 8 * REF_STRIDE;
                sadA += sad8px(srcA[y * 2], srcA[y * 2 + 1], rowA, shift);
                sadB += sad8px(srcB[y * 2], srcB[y * 2 + 1], rowB, shift);
            }
        }
        // MV-cost regularization: bias toward short/zero vectors so the
        // exhaustive search does not noise-fit on flat content.
        {
            int pdx = x0 + px, pdy = y0 + py;
            uint32_t pen = 4u * (uint32_t)(abs(pdx) + abs(pdy));
            sadA += pen;
            sadB += pen;
        }
        unsigned long long posKey = (unsigned)((wi << 20) | pos);
        key8A = min(key8A, ((unsigned long long)sadA << 32) | posKey);
        key8B = min(key8B, ((unsigned long long)sadB << 32) | posKey);
        scr[warp][lane] = sadA;
        scr[warp][lane + 32] = sadB;
        __syncwarp();
        if (lane < 16) {
            int gx = (lane & 3) * 2, gy = (lane >> 2) * 2;
            uint32_t s16 = scr[warp][gy * 8 + gx] + scr[warp][gy * 8 + gx + 1] + scr[warp][(gy + 1) * 8 + gx] +
                scr[warp][(gy + 1) * 8 + gx + 1];
            key16 = min(key16, ((unsigned long long)s16 << 32) | posKey);
            scr[warp][64 + lane] = s16;
        }
        __syncwarp();
        if (lane < 4) {
            int gx = (lane & 1) * 2, gy = (lane >> 1) * 2;
            uint32_t s32 = scr[warp][64 + gy * 4 + gx] + scr[warp][64 + gy * 4 + gx + 1] +
                scr[warp][64 + (gy + 1) * 4 + gx] + scr[warp][64 + (gy + 1) * 4 + gx + 1];
            key32 = min(key32, ((unsigned long long)s32 << 32) | posKey);
            scr[warp][80 + lane] = s32;
        }
        __syncwarp();
        if (lane == 0) {
            uint32_t s64 = scr[warp][80] + scr[warp][81] + scr[warp][82] + scr[warp][83];
            key64 = min(key64, ((unsigned long long)s64 << 32) | posKey);
        }
        __syncwarp();
    }
    } // window loop

    // Cross-warp merge in the PoC's internal (raster) index space...
    __shared__ unsigned long long mergeBuf[WARPS][85];
    mergeBuf[warp][21 + lane] = key8A;
    mergeBuf[warp][21 + lane + 32] = key8B;
    if (lane < 16) mergeBuf[warp][5 + lane] = key16;
    if (lane < 4) mergeBuf[warp][1 + lane] = key32;
    if (lane == 0) mergeBuf[warp][0] = key64;
    __syncthreads();

    // ...then write decoded winners at SVT-AV1's indices with SVT MV packing.
    if (tid < 85) {
        unsigned long long best = ~0ull;
        for (int wp = 0; wp < WARPS; wp++) best = min(best, mergeBuf[wp][tid]);
        int wi = (int)((best >> 20) & 0xfff);
        int pos = (int)(best & 0xfffff);
        int x0, x1, y0, y1;
        clampWindow(winCx[wi], FP_RANGE, -bx, w - B64 - bx, x0, x1);
        clampWindow(winCy[wi], FP_RANGE, -by, h - B64 - by, y0, y1);
        int nx = x1 - x0 + 1;
        int dx = x0 + pos % nx, dy = y0 + pos / nx;
        int svt;
        if (tid == 0) svt = 0;
        else if (tid < 5) svt = tid; // 32x32: raster == SVT order
        else if (tid < 21) {
            int r = tid - 5;
            svt = 5 + svt16Idx(r & 3, r >> 2);
        } else {
            int r = tid - 21, x8 = r & 7, y8 = r >> 3;
            svt = 21 + svt16Idx(x8 >> 1, y8 >> 1) * 4 + (y8 & 1) * 2 + (x8 & 1);
        }
        GpuMeB64Result& out = results[(size_t)ri * blocks + blk];
        // The MV-cost penalty steered selection only; report the raw SAD
        // (recoverable: penalty is a function of the winning MV and level).
        uint32_t nSub8 = tid == 0 ? 64u : tid < 5 ? 16u : tid < 21 ? 4u : 1u;
        uint32_t pen = 4u * nSub8 * (uint32_t)(abs(dx) + abs(dy));
        out.sad[svt] = (uint32_t)(best >> 32) - pen;
        out.mv[svt] = ((uint32_t)(uint16_t)(int16_t)dy << 16) | (uint16_t)(int16_t)dx;
    }
}
// ------------------------------------------------------------------- host ---

struct GpuFrame {
    uint8_t *full = nullptr, *quarter = nullptr, *sixteenth = nullptr;
    uint64_t lastUse = 0;
};

struct PairEntry {
    std::vector<GpuMeB64Result> results;
    uint64_t lastUse = 0;
    bool ready = false; // false while a thread is computing it
};

// A lane is one independent GPU execution context: its own stream, upload
// staging and result buffers. Lanes 0..NUM_WORKERS-1 belong exclusively to
// the prefetch worker threads; the last lane serves ME-thread misses (under
// meLaneMtx). The frame pool is shared across lanes under frameMtx; an
// upload synchronizes its own stream before publishing, so frames in the map
// are always fully resident.
struct GpuLane {
    cudaStream_t stream = nullptr;
    uint8_t* dUpload = nullptr;
    uint8_t* hUpload = nullptr;
    size_t uploadCap = 0;
    GpuMeB64Result* dResults = nullptr;
    GpuMeB64Result* hResults = nullptr;
};

constexpr int NUM_WORKERS = 2;
constexpr int NUM_LANES = NUM_WORKERS + 1;

struct GpuMeContext {
    std::mutex mtx;      // pair cache
    std::mutex frameMtx; // frame pool
    std::mutex meLaneMtx;
    std::condition_variable cv;
    bool initTried = false, ok = false;
    int w = 0, h = 0, padW = 0, padH = 0, cols = 0, rows = 0, blocks = 0;
    size_t fullSize = 0, qSize = 0, sSize = 0;
    std::map<std::pair<uint64_t, uint64_t>, GpuFrame> frames; // (generation, picNum)
    std::deque<GpuFrame> retired; // deferred frees: other lanes may still read
    std::map<std::array<uint64_t, 3>, std::shared_ptr<PairEntry>> pairs; // (gen, src, ref)
    uint64_t useCounter = 0;
    uint64_t pairsComputed = 0, cacheHits = 0;
    long long gpuNsWorker = 0, gpuNsMe = 0, waitNsMe = 0;
    uint64_t meWaits = 0, meComputes = 0;
    std::atomic<uint64_t> acquireFails{0};
    GpuLane lanes[NUM_LANES];

    static constexpr size_t FRAME_POOL_CAP = 128;
    static constexpr size_t RETIRED_CAP = 24;
    static constexpr size_t PAIR_CACHE_CAP = 256;
};

GpuMeContext g_ctx;
thread_local int t_laneIdx = NUM_LANES - 1; // workers override; ME threads use last lane

void reportStats() {
    fprintf(stderr,
            "[GPU_ME] pairs: %llu  hits: %llu | gpu ms (worker/ME): %.0f/%.0f | ME cv-wait ms: %.0f "
            "(%llu waits, %llu ME computes, %llu acquire fails)\n",
            (unsigned long long)g_ctx.pairsComputed, (unsigned long long)g_ctx.cacheHits,
            g_ctx.gpuNsWorker / 1e6, g_ctx.gpuNsMe / 1e6, g_ctx.waitNsMe / 1e6,
            (unsigned long long)g_ctx.meWaits, (unsigned long long)g_ctx.meComputes,
            (unsigned long long)g_ctx.acquireFails.load());
}

bool initContext(int w, int h) {
    g_ctx.initTried = true;
    int dev = 0;
    if (cudaGetDeviceCount(&dev) != cudaSuccess || dev == 0) return false;
    g_ctx.w = w;
    g_ctx.h = h;
    g_ctx.padW = (w + B64 - 1) / B64 * B64;
    g_ctx.padH = (h + B64 - 1) / B64 * B64;
    g_ctx.cols = g_ctx.padW / B64;
    g_ctx.rows = g_ctx.padH / B64;
    g_ctx.blocks = g_ctx.cols * g_ctx.rows;
    g_ctx.fullSize = (size_t)g_ctx.padW * g_ctx.padH;
    g_ctx.qSize = g_ctx.fullSize / 4;
    g_ctx.sSize = g_ctx.fullSize / 16;
    for (int i = 0; i < NUM_LANES; i++) {
        GpuLane& L = g_ctx.lanes[i];
        GPU_ME_CHECK(cudaStreamCreateWithFlags(&L.stream, cudaStreamNonBlocking));
        GPU_ME_CHECK(cudaMalloc(&L.dResults, sizeof(GpuMeB64Result) * g_ctx.blocks * GPU_ME_MAX_REFS));
        GPU_ME_CHECK(cudaMallocHost(&L.hResults, sizeof(GpuMeB64Result) * g_ctx.blocks * GPU_ME_MAX_REFS));
    }
    atexit(reportStats);
    return true;
}

// Under frameMtx.
void evictFramesIfNeeded() {
    while (g_ctx.frames.size() >= GpuMeContext::FRAME_POOL_CAP) {
        auto lru = g_ctx.frames.begin();
        for (auto it = g_ctx.frames.begin(); it != g_ctx.frames.end(); ++it)
            if (it->second.lastUse < lru->second.lastUse) lru = it;
        // Defer the free: another lane may still have kernels reading this
        // frame. By the time RETIRED_CAP more evictions happen, any such
        // kernels (sub-millisecond) are long finished.
        g_ctx.retired.push_back(lru->second);
        g_ctx.frames.erase(lru);
        while (g_ctx.retired.size() > GpuMeContext::RETIRED_CAP) {
            cudaFree(g_ctx.retired.front().full);
            cudaFree(g_ctx.retired.front().quarter);
            cudaFree(g_ctx.retired.front().sixteenth);
            g_ctx.retired.pop_front();
        }
    }
}

// Under mtx. Only ready entries are eviction candidates.
void evictPairsLocked() {
    while (g_ctx.pairs.size() >= GpuMeContext::PAIR_CACHE_CAP) {
        auto lru = g_ctx.pairs.end();
        for (auto it = g_ctx.pairs.begin(); it != g_ctx.pairs.end(); ++it)
            if (it->second->ready && (lru == g_ctx.pairs.end() || it->second->lastUse < lru->second->lastUse))
                lru = it;
        if (lru == g_ctx.pairs.end()) return; // everything in flight
        g_ctx.pairs.erase(lru);
    }
}

// Takes frameMtx internally. Uploads via the caller's lane and synchronizes
// that stream before publishing, so the frame is resident for all lanes.
GpuFrame* ensureFrame(GpuLane& L, uint64_t gen, uint64_t picNum, const uint8_t* y, int stride) {
    std::lock_guard<std::mutex> flk(g_ctx.frameMtx);
    auto it = g_ctx.frames.find({gen, picNum});
    if (it != g_ctx.frames.end()) {
        it->second.lastUse = ++g_ctx.useCounter;
        return &it->second;
    }
    evictFramesIfNeeded();
    GpuFrame f;
    if (cudaMalloc(&f.full, g_ctx.fullSize) != cudaSuccess) return nullptr;
    if (cudaMalloc(&f.quarter, g_ctx.qSize) != cudaSuccess || cudaMalloc(&f.sixteenth, g_ctx.sSize) != cudaSuccess) {
        cudaFree(f.full);
        cudaFree(f.quarter);
        return nullptr;
    }
    size_t need = (size_t)stride * g_ctx.h;
    if (need > L.uploadCap) {
        cudaFree(L.dUpload);
        cudaFreeHost(L.hUpload);
        if (cudaMalloc(&L.dUpload, need) != cudaSuccess || cudaMallocHost(&L.hUpload, need) != cudaSuccess) {
            L.dUpload = nullptr;
            L.hUpload = nullptr;
            L.uploadCap = 0;
            cudaFree(f.full);
            cudaFree(f.quarter);
            cudaFree(f.sixteenth);
            return nullptr;
        }
        L.uploadCap = need;
    }
    memcpy(L.hUpload, y, need);
    bool ok = cudaMemcpyAsync(L.dUpload, L.hUpload, need, cudaMemcpyHostToDevice, L.stream) == cudaSuccess;
    if (ok) {
        dim3 blk(16, 16);
        padKernel<<<dim3((g_ctx.padW + 15) / 16, (g_ctx.padH + 15) / 16), blk, 0, L.stream>>>(
            L.dUpload, stride, g_ctx.w, g_ctx.h, f.full, g_ctx.padW, g_ctx.padH);
        downsample2xKernel<<<dim3((g_ctx.padW / 2 + 15) / 16, (g_ctx.padH / 2 + 15) / 16), blk, 0, L.stream>>>(
            f.full, g_ctx.padW, f.quarter, g_ctx.padW / 2, g_ctx.padH / 2);
        downsample2xKernel<<<dim3((g_ctx.padW / 4 + 15) / 16, (g_ctx.padH / 4 + 15) / 16), blk, 0, L.stream>>>(
            f.quarter, g_ctx.padW / 2, f.sixteenth, g_ctx.padW / 4, g_ctx.padH / 4);
        ok = cudaStreamSynchronize(L.stream) == cudaSuccess;
    }
    if (!ok) {
        cudaFree(f.full);
        cudaFree(f.quarter);
        cudaFree(f.sixteenth);
        return nullptr;
    }
    f.lastUse = ++g_ctx.useCounter;
    auto res = g_ctx.frames.emplace(std::make_pair(gen, picNum), f);
    return &res.first->second;
}

} // namespace

extern "C" int svt_gpu_me_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char* env = getenv("SVT_GPU_ME");
        enabled = (env && env[0] == '1') ? 1 : 0;
        if (enabled) fprintf(stderr, "[GPU_ME] GPU open-loop ME offload ENABLED\n");
    }
    return enabled;
}

namespace {
struct PicHandle {
    std::shared_ptr<PairEntry> entries[GPU_ME_MAX_REFS];
    int n = 0;
};
} // namespace

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
        // One GPU trip on this thread's lane for all missing pairs. Results
        // are staged into local vectors on the lane so the publish below is
        // an O(1) move under the cache mutex.
        const bool meThread = !t_isPrefetchThread;
        if (meThread) g_ctx.meLaneMtx.lock();
        GpuLane& L = g_ctx.lanes[t_laneIdx];

        bool ok;
        int slotToRef[GPU_ME_MAX_REFS];
        std::vector<GpuMeB64Result> staged[GPU_ME_MAX_REFS];
        int launched = 0;
        long long gpuT0 = nowNs();
        {
            GpuFrame* fs = ensureFrame(L, generation, src_pic_num, src_y, src_stride);
            ok = fs != nullptr;
            if (ok) {
                MeLaunchArgs largs = {};
                largs.src = {fs->full, fs->quarter, fs->sixteenth};
                for (int i = 0; i < num_refs; i++) {
                    if (!mustCompute[i]) continue;
                    GpuFrame* fr = ensureFrame(L, generation, refs[i].pic_num, refs[i].y, refs[i].stride);
                    if (!fr) {
                        ok = false;
                        break;
                    }
                    largs.refs[launched] = {fr->full, fr->quarter, fr->sixteenth};
                    slotToRef[launched++] = i;
                }
                if (ok) {
                    mePairKernel<<<dim3(g_ctx.blocks, launched), 256, 0, L.stream>>>(
                        largs, g_ctx.padW, g_ctx.padH, g_ctx.cols, g_ctx.blocks, L.dResults);
                    ok = cudaMemcpyAsync(L.hResults, L.dResults, sizeof(GpuMeB64Result) * g_ctx.blocks * launched,
                                         cudaMemcpyDeviceToHost, L.stream) == cudaSuccess &&
                        cudaStreamSynchronize(L.stream) == cudaSuccess && cudaGetLastError() == cudaSuccess;
                }
            }
            if (ok)
                for (int sl = 0; sl < launched; sl++)
                    staged[sl].assign(L.hResults + (size_t)sl * g_ctx.blocks,
                                      L.hResults + (size_t)(sl + 1) * g_ctx.blocks);
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
                entries[i]->ready = true; // empty results signal failure
            }
        }
        g_ctx.cv.notify_all();
        if (!ok) {
            g_ctx.acquireFails++;
            return nullptr;
        }
    }

    // Wait for any pairs being computed by other threads, then pin and hand out.
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
        h->entries[i] = entries[i]; // pins the vectors beyond any cache eviction
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

    static constexpr size_t QUEUE_CAP = 64; // must cover a mini-GOP burst

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
            if (h) svt_gpu_me_release(h); // entries stay in the shared cache
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
