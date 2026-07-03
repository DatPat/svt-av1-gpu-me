# GPU-Assisted SVT-AV1 on Radxa Orion O6 — Vulkan Port Report

Date: 2026-07-04. Hardware: Radxa Orion O6 (CIX P1/CD8180: 8x Cortex-A720 +
4x A520, Mali-G720-Immortalis MC10, 16 GB LPDDR5, Debian 12, Arm DDK r53p0,
Vulkan 1.3.296) driven remotely; desktop reference: RTX 4080 + 24-thread
x86 (Windows, CUDA 13.3).

## 1. What was delivered

- **Armv9.2-optimized SVT-AV1 build** on the board: clang-19 with
  `-mcpu=cortex-a720` (gcc 12 caps at armv9-a, so clang was installed to
  honor the 9.2 target); all five Arm SIMD tiers compiled and runtime-
  dispatched: NEON, NEON_DOTPROD, NEON_I8MM, SVE, SVE2.
- **Complete Vulkan compute backend** (`svt_gpu_me_vk.cpp` +
  `gpu_me_shaders/*.comp`) implementing the same
  `svt_gpu_me_acquire/release/prefetch` API as the CUDA backend, so every
  encoder hook (ME injection, TF path, prefetch, MD-prune gate) is shared.
  Mali/portability specifics: 32-bit `SAD<<k|pos` min-keys (no int64),
  workgroup barriers only (subgroup-size agnostic), unified-memory
  mega-buffer frame pools with LRU slots and CPU pad-copy (no staging, no
  transfers), grid.y-batched per-pair dispatches driven by a descriptor
  SSBO, HOST_CACHED readback, per-lane fenced command buffers, and
  synchronous pyramid submits (Vulkan queue submissions may overlap,
  unlike CUDA streams). Shaders are embedded at build time
  (`glslangValidator --vn`). Build: `-DSVT_GPU_ME_VULKAN=ON`; runtime:
  `SVT_GPU_ME=1`.

## 2. Validation (all on the board, all passing)

- Encodes decode correctly (dav1d), frame-exact counts.
- Byte-deterministic across repeated runs.
- Zero GPU-path failures across the whole benchmark campaign
  (`0 acquire fails`); CPU fallback verified.
- With `SVT_GPU_ME` unset, the Vulkan-enabled binary produces bitstreams
  **byte-identical** to the pure-CPU build.
- Two teardown bugs found and fixed during bring-up: detached prefetch
  workers racing Mali DDK cleanup at exit (fixed by a queue-mutex +
  vkDeviceWaitIdle atexit quiesce) and glibc static-destruction of mutexes
  still referenced by detached threads (fixed by leaky singletons).

## 3. Performance

### Kernel level (standalone PoC, bit-exact vs CPU reference)

| | RTX 4080 (CUDA) | Mali-G720 (Vulkan) |
|---|---|---|
| 720p ME pairs/s (kernels) | ~5,800–7,200 | 158 |
| 1080p ME pairs/s (kernels) | ~2,900* | 72 |

*4080 number scaled from measured 64-pair batches. The Mali is ~40–50x
slower — consistent with its compute class. Mali-specific optimization
journey (63 -> 158 pairs/s at 720p): HOST_CACHED readback (uncached mapped
reads cost ~10x), funnel-shifted whole-word SADs, 4 position-groups x 64
threads per workgroup (512 threads regressed — register pressure).

### Encoder level (board, 1080p50 240-frame clips, CRF 35, pipeline frozen)

| clip | preset | CPU fps | GPU fps | GPU wide fps |
|---|---|---|---|---|
| crowd_run | 6 | 11.06 | 9.68 (−12%) | — |
| crowd_run | 4 | 7.52 | 5.45 (−28%) | 3.05 |
| crowd_run | 2 | 1.72 | 1.70 (−1%) | 1.58 (−8%) |
| old_town_cross | 6 | 19.00 | 11.61 (−39%) | — |
| old_town_cross | 4 | 14.15 | 5.98 (−58%) | 3.17 |
| old_town_cross | 2 | 2.96 | 2.64 (−11%) | 2.24 (−24%) |

The GPU is the bottleneck at fast presets (its ~7 multi-ref pictures/s at
1080p caps the pipeline); at preset 2 the CPU is slow enough that the GPU
keeps pace. "Wide" = desktop search parameters (±96 HME, ±12 dual-window).

## 4. Quality (board, clean encoder — VMAF v0.6.1 vs source)

| clip | preset | CPU | GPU narrow | GPU wide |
|---|---|---|---|---|
| crowd_run | 6 | 97.03 / 18.82 MB | 96.98 / 18.60 MB | — |
| crowd_run | 4 | 96.07 / 15.62 MB | 96.01 / 15.49 MB | 96.00 / 15.46 MB |
| crowd_run | 2 | 97.73 / 15.01 MB | 97.71 / 15.13 MB | 97.72 / 15.13 MB |
| old_town_cross | 6 | 94.69 / 1.43 MB | 94.72 / 1.46 MB | — |
| old_town_cross | 4 | 94.67 / 1.19 MB | 95.00 / 1.35 MB | 94.99 / 1.33 MB |
| old_town_cross | 2 | 95.86 / 1.39 MB | 95.96 / 1.62 MB | 95.95 / 1.61 MB |

**Verdict: on this board and content, GPU-assisted ME is quality-neutral**
(VMAF deltas within ±0.35) with size −1% (hard content) to +17% (static
content, concentrated in TPL keyframe boosting on a 4.8 s clip — heavily
amortized on real-length content). Widening the search to desktop
parameters does not change quality on these clips. Net: **on the Orion O6
the offload works mechanically but does not currently pay** — the GPU is
below the compute threshold where extra search converts into compression
gains, and it costs fps at all presets except ~parity at preset 2.

## 5. Major incidental discovery: upstream corruption on x86

While reconciling desktop-vs-board numbers we found that **pristine
upstream SVT-AV1 master (7a88adeb) produces visibly corrupt bitstreams on
the x86/MSVC desktop build** — deterministic, present in pure-C mode
(`--asm 0`), engaged when encodes exceed roughly the default lookahead
depth (~120–180 frames), absent entirely on the Arm/clang build of the
same source. Symptom: checkerboard block corruption, VMAF collapsing to
~34–37 where the Arm build scores 96–97 on identical input/settings.

Consequences for previously-reported desktop results: all long desktop
SvtAv1EncApp encodes (CUDA-vs-CPU quality campaign on 720p content,
including the "−21% bitrate / +3.7 VMAF" full-video result) were measured
on this corrupt-mode encoder. Both arms of those comparisons ran the same
corrupt mode, so the deltas may retain directional meaning, but **the
desktop quality claims must be considered unverified until re-measured on
a clean x86 build** (older release base or clang-cl are the candidate
fixes). This also retroactively explains the unexplained gap between our
master builds (VMAF ~81) and ffmpeg's bundled libsvtav1 release (VMAF
~94.5) on the same content — previously misattributed to keyint defaults.

## 6. Engineering findings worth keeping

1. Vulkan queue submissions may overlap execution (no CUDA-stream
   ordering); every cross-submission dependency needs fences/semaphores.
2. Mali: workgroup "shared memory" is emulated — prefer L2-served reads;
   HOST_CACHED for CPU readback; subgroup width is 16.
3. SVT renames threads (`svt-app-main`), which breaks `pkill -x` on the
   binary name.
4. Detached worker threads must never outlive GPU driver teardown or
   static destructors (quiesce at exit; leaky singletons).
5. Fields added to cross-target structs (MeContext) must go at the END —
   ASM_* targets compile without the feature define (desktop lesson,
   revalidated in this port).
6. An MV-cost penalty in exhaustive search must steer selection only;
   leaking it into reported SADs poisons TF blend weights downstream.
7. The whole GPU-ME module API (acquire/release/prefetch, generation
   keys) ported from CUDA to Vulkan with zero changes to encoder hooks —
   the API boundary was drawn in the right place.

## 7. Recommendations

- For the Orion O6 as an encode box: run CPU-only SVT-AV1 with the
  Armv9.2/SVE2 build (`build-cpu`); it is the better configuration today.
  Revisit GPU-assist there if/when a pre-analysis stage with cheaper GPU
  math (MD-prune hints only, no full search) is built.
- For the desktop: re-baseline all quality numbers on a clean x86 build
  (older SVT base or clang-cl), then re-run the CUDA quality campaign.
- Report the x86 lookahead corruption upstream with the reproduction
  recipe (this repo's base commit, 1080p50/720p50 input, >=180 frames,
  any preset, CRF 35, Windows/MSVC).

## Addendum (step-3 follow-up): MD-prune hints evaluated — negative

The recommended "cheaper GPU-assist" (mode-decision pruning hints without
full GPU search) was evaluated on the board's clean testbed. Because the
prune gate's inputs (`me_*_distortion`) are populated by the stock CPU ME
anyway, the cheapest form needs no GPU at all; the gate was decoupled from
the GPU module (`SVT_GPU_MD_PRUNE` now works on CPU-only builds) and swept
at k=0/4/6 on both clips at presets 2/4/6:

- fps: within +/-2% run-to-run noise in every cell
- size: within 0.15% in every cell

**The gated depth-removal boost has no measurable effect on the clean Arm
testbed.** The desktop's previously-reported "+7.7% fps at flat VMAF" for
this mechanism was measured on the corrupt-mode x86 encoder and should be
considered an artifact until reproduced cleanly. Building a GPU hint
kernel to feed this gate would inherit its ineffectiveness, so the
"light GPU hints" variant is not worth implementing as formulated. If MD
pruning on this encoder is to be pursued, it needs a different actuation
point than `depth_removal_level` (e.g., direct depth-list surgery in
`build_starting_cand_block_array` territory) and a BD-rate methodology on
a trusted testbed first.
