#include "kops_internal.h"

#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#else
#include <pthread.h>
#endif

int kops_cpu_count(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) {
        return 1;
    }
    if (n > 64) {
        return 64;
    }
    return (int)n;
}

uint64_t kops_now_ns(void)
{
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
}

#ifdef __APPLE__

typedef struct kops_gcd_ctx {
    kops_parallel_fn fn;
    void *ctx;
} kops_gcd_ctx;

static void kops_gcd_trampoline(void *arg, size_t idx)
{
    kops_gcd_ctx *g = (kops_gcd_ctx *)arg;
    g->fn(g->ctx, idx);
}

void kops_parallel_for(size_t count, kops_parallel_fn fn, void *ctx)
{
    kops_gcd_ctx g;
    if (count == 0) {
        return;
    }
    if (count == 1) {
        fn(ctx, 0);
        return;
    }
    g.fn = fn;
    g.ctx = ctx;
    dispatch_apply_f(count, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                     &g, kops_gcd_trampoline);
}

#else

typedef struct kops_pt_arg {
    kops_parallel_fn fn;
    void *ctx;
    size_t idx;
} kops_pt_arg;

static void *kops_pt_trampoline(void *p)
{
    kops_pt_arg *a = (kops_pt_arg *)p;
    a->fn(a->ctx, a->idx);
    return NULL;
}

void kops_parallel_for(size_t count, kops_parallel_fn fn, void *ctx)
{
    pthread_t threads[KOPS_PZ_MAX_SLICES];
    kops_pt_arg args[KOPS_PZ_MAX_SLICES];
    size_t i;
    if (count == 0) {
        return;
    }
    if (count == 1) {
        fn(ctx, 0);
        return;
    }
    if (count > KOPS_PZ_MAX_SLICES) {
        for (i = 0; i < count; ++i) {
            fn(ctx, i);
        }
        return;
    }
    for (i = 1; i < count; ++i) {
        args[i].fn = fn;
        args[i].ctx = ctx;
        args[i].idx = i;
        if (pthread_create(&threads[i], NULL, kops_pt_trampoline, &args[i]) != 0) {
            threads[i] = 0;
            fn(ctx, i);
        }
    }
    fn(ctx, 0);
    for (i = 1; i < count; ++i) {
        if (threads[i]) {
            pthread_join(threads[i], NULL);
        }
    }
}

#endif
