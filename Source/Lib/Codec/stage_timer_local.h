// Local experiment: per-pipeline-stage CPU-time accounting, portable.
// Windows: QueryThread/ProcessCycleTime (CPU cycles). POSIX: CLOCK_THREAD/
// PROCESS_CPUTIME_ID (CPU nanoseconds). Shares (stage/process) are
// comparable either way.
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
static inline uint64_t stage_thread_ticks(void) {
    ULONG64 c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return c;
}
static inline uint64_t stage_process_ticks(void) {
    ULONG64 c = 0;
    QueryProcessCycleTime(GetCurrentProcess(), &c);
    return c;
}
#define STAGE_ADD64(p, v) InterlockedExchangeAdd64((volatile LONG64*)(p), (LONG64)(v))
#define STAGE_ONCE(p) (InterlockedCompareExchange((volatile LONG*)(p), 1, 0) == 0)
typedef volatile LONG64 stage_acc64;
typedef volatile LONG stage_flag32;
#else
#include <time.h>
static inline uint64_t stage_thread_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static inline uint64_t stage_process_ticks(void) {
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
#define STAGE_ADD64(p, v) __atomic_fetch_add((p), (int64_t)(v), __ATOMIC_RELAXED)
#define STAGE_ONCE(p) (__sync_val_compare_and_swap((p), 0, 1) == 0)
typedef volatile int64_t stage_acc64;
typedef volatile int32_t stage_flag32;
#endif

#define DEFINE_STAGE_TIMER(label, inner_fn, wrapper_fn)                                                  static stage_acc64  g_##label##_cycles;                                                              static stage_flag32 g_##label##_registered;                                                          static void         label##_stage_report(void) {                                                         uint64_t pc = stage_process_ticks();                                                                 fprintf(stderr,                                                                                              "[STAGE] %-24s cycles: %14lld  share: %5.1f%%
",                                                   #label,                                                                                              (long long)g_##label##_cycles,                                                                       100.0 * (double)g_##label##_cycles / (double)pc);                                        }                                                                                                    EbErrorType wrapper_fn(void* context) {                                                                  if (STAGE_ONCE(&g_##label##_registered))                                                                 atexit(label##_stage_report);                                                                    uint64_t    c0 = stage_thread_ticks();                                                               EbErrorType r  = inner_fn(context);                                                                  STAGE_ADD64(&g_##label##_cycles, stage_thread_ticks() - c0);                                         return r;                                                                                        }
