# SVT-AV1 with CUDA GPU motion-estimation offload

An experimental fork of [SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1)
that offloads open-loop motion estimation (and temporal filtering's motion
search) to an NVIDIA GPU via CUDA, and uses the GPU's search quality to
safely prune mode decision. Built for people whose NVIDIA cards have no AV1
hardware encoder (GTX 10/16, RTX 20/30): the idle GPU buys **better
compression at the same speed** rather than raw speed.

Measured on real-world 720p25 content (preset 6, CRF 32, RTX 4080 +
24-thread CPU, full 76-minute video, same binary and settings):

| | stock CPU | GPU-assisted |
|---|---|---|
| Speed | 297 fps | 305 fps |
| Video size | 1.156 GB | **916 MB (-20.7%)** |
| VMAF | 81.30 | **84.96 (+3.7)** |

Why quality instead of speed: the GPU searches exhaustively (effective
+/-96 hierarchical + dual-window +/-12 full-pel including a zero-MV window)
where the CPU's ME must prune aggressively; better MVs both shrink residuals
and help mode decision converge. Open-loop ME is only ~4-13% of encoder CPU
cycles, so removing it cannot speed the encoder up much — improving its
output can improve everything downstream.

## What's in the patch (one commit on top of the upstream snapshot)

- `Source/Lib/Codec/svt_gpu_me.cu/.h` — CUDA module: fused
  HME-L0/L1/full-pel kernel producing best SAD+MV for all 85 sub-blocks of
  each 64x64 in SVT's ME layout; GPU frame pool with generation-tagged keys
  (temporal filtering reads pre-filter content that is later rewritten in
  place); per-picture-pair result cache; two prefetch worker threads with
  independent CUDA streams; per-picture pinned result handles so the
  per-block hot path is lock-free.
- `motion_estimation.c` — injection hook: replaces `hme_b64` +
  `integer_search_b64` with GPU results in `p_sb_best_sad/mv` (plus the
  per-PU pointer fields temporal filtering reads); CPU fallback on any GPU
  failure.
- `pd_process.c` — prefetch triggers when picture decision posts ME and TF
  tasks, so results are cached before the ME stage asks.
- `enc_mode_config.c` — GPU-SAD-gated depth-removal boost
  (`SVT_GPU_MD_PRUNE`): pruning is strengthened only on SBs where a single
  64x64 MV already matches the sum of the best 8x8 MVs within 5%.
  +7.7% fps at unchanged VMAF; the ungated equivalent is strictly harmful.
- Instrumentation used during development (per-stage CPU-cycle timers,
  `[GPU_ME]` statistics at exit) is left in for reproducibility.

## Build & run

Requires CUDA Toolkit 12/13, CMake 3.24+, nasm.

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSVT_GPU_ME=ON
cmake --build build
SVT_GPU_ME=1 SVT_GPU_MD_PRUNE=4 ./Bin/Release/SvtAv1EncApp -i in.y4m --preset 6 --crf 32 -b out.ivf
```

`SVT_GPU_ME=1` enables the offload at runtime (off = stock behavior;
any GPU failure falls back to the CPU search per picture).
`SVT_GPU_MD_PRUNE=4` additionally enables the gated MD pruning.

## Caveats

- Experimental / proof-of-concept quality. 8-bit only; no super-res/resize
  (falls back to CPU); tested on Windows + RTX 4080 with one content type.
  Expect rough edges elsewhere.
- Warning to patch authors: fields added to `MeContext` must go at the END
  of the struct — the ASM_* build targets compile without
  `SVT_ENABLE_GPU_ME` and will silently read later fields at wrong offsets
  otherwise.
- The research trail (standalone bit-exact PoC, per-stage profiling data,
  full benchmark history) lives in the companion repo: `cuda-av1-me-poc`.

Base: upstream commit `7a88adebaf4189d59487ba55ed3d9bbbbc645f35`.
License: same as SVT-AV1 (BSD-2-Clause + AOM Patent License).
