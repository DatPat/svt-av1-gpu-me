// Local experiment: per-pipeline-stage CPU-cycle accounting.
// Wrap a process's *_kernel_iter so every iteration's thread-cycles are
// accumulated; an atexit hook prints the stage's share of process cycles.
#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFINE_STAGE_TIMER(label, inner_fn, wrapper_fn)                                              \
    static volatile LONG64 g_##label##_cycles;                                                       \
    static volatile LONG   g_##label##_registered;                                                   \
    static void            label##_stage_report(void) {                                              \
        ULONG64 pc = 0;                                                                              \
        QueryProcessCycleTime(GetCurrentProcess(), &pc);                                             \
        fprintf(stderr,                                                                              \
                "[STAGE] %-24s cycles: %14lld  share: %5.1f%%\n",                                    \
                #label,                                                                              \
                (long long)g_##label##_cycles,                                                       \
                100.0 * (double)g_##label##_cycles / (double)pc);                                    \
    }                                                                                                \
    EbErrorType wrapper_fn(void* context) {                                                          \
        if (!InterlockedCompareExchange(&g_##label##_registered, 1, 0))                              \
            atexit(label##_stage_report);                                                            \
        ULONG64 c0 = 0, c1 = 0;                                                                      \
        QueryThreadCycleTime(GetCurrentThread(), &c0);                                               \
        EbErrorType r = inner_fn(context);                                                           \
        QueryThreadCycleTime(GetCurrentThread(), &c1);                                               \
        InterlockedExchangeAdd64(&g_##label##_cycles, (LONG64)(c1 - c0));                            \
        return r;                                                                                    \
    }
