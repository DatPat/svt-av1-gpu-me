/*
 * GPU (CUDA) open-loop motion estimation offload — experimental.
 *
 * Computes, per (source picture, reference picture) pair, the best full-pel
 * MV + SAD for all 85 sub-blocks of every 64x64, in SVT-AV1's ME results
 * layout (64x64 @0, 32x32 @1 raster, 16x16 @5 z-order-by-32x32, 8x8 @21
 * quadrants-under-16x16; MVs packed ((uint16)yMv << 16) | (uint16)xMv,
 * full-pel). Results are injected into MeContext::p_sb_best_sad / p_sb_best_mv
 * in place of the CPU HME + integer search.
 *
 * Enabled at runtime with environment variable SVT_GPU_ME=1.
 * Any failure (no CUDA device, allocation, dimension change) returns NULL and
 * the caller falls back to the CPU search path.
 */
#ifndef SVT_GPU_ME_H
#define SVT_GPU_ME_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuMeB64Result {
    uint32_t sad[85];
    uint32_t mv[85];
} GpuMeB64Result;

/* 1 if SVT_GPU_ME=1 is set and a CUDA device is usable (evaluated once). */
int svt_gpu_me_enabled(void);

typedef struct GpuMeRef {
    const uint8_t* y;
    int stride;
    uint64_t pic_num;
} GpuMeRef;

#define GPU_ME_MAX_REFS 8

/* Acquire the picture's pinned result set against all its references. All
 * cache misses among the pairs are computed in a single GPU trip; bases[i]
 * then points at refs[i]'s per-b64 result array (index with b64_index,
 * lock-free) and stays valid until svt_gpu_me_release(handle).
 * `generation` disambiguates cache entries when picture buffers are
 * rewritten in place: 0 for open-loop ME (post-TF content); for temporal
 * filtering pass center_pic_num+1 (pre-filter content, unique per TF op).
 * Returns NULL on any failure (caller falls back to CPU). Thread-safe. */
void* svt_gpu_me_acquire(const uint8_t* src_y, int src_stride, uint64_t src_pic_num, uint64_t generation,
                         const GpuMeRef* refs, int num_refs, int width, int height,
                         const GpuMeB64Result** bases);
void svt_gpu_me_release(void* handle);

/* Asynchronously compute the picture's ME on a worker thread so results are
 * cached before the ME stage requests them. Best effort; never blocks. */
void svt_gpu_me_prefetch(const uint8_t* src_y, int src_stride, uint64_t src_pic_num, uint64_t generation,
                         const GpuMeRef* refs, int num_refs, int width, int height);

#ifdef __cplusplus
}
#endif

#endif /* SVT_GPU_ME_H */
