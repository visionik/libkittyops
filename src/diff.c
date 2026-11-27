#include "kops_internal.h"

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define KOPS_HAVE_NEON 1
#elif (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define KOPS_HAVE_X86_SIMD 1

static int kops_x86_simd_level(void)
{
    static int cached = -1;
    int lv = cached;
    if (lv < 0) {
        if (__builtin_cpu_supports("avx2")) {
            lv = 2;
        } else if (__builtin_cpu_supports("ssse3")) {
            lv = 1;
        } else {
            lv = 0;
        }
        cached = lv;
    }
    return lv;
}

__attribute__((target("avx2")))
static bool kops_row_chunk_equal_avx2(const uint8_t *a, const uint8_t *b, size_t bytes)
{
    size_t i = 0;
    __m256i acc = _mm256_setzero_si256();
    for (; i + 64u <= bytes; i += 64u) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)),
                                      _mm256_loadu_si256((const __m256i *)(b + i)));
        __m256i x1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i + 32)),
                                      _mm256_loadu_si256((const __m256i *)(b + i + 32)));
        acc = _mm256_or_si256(acc, _mm256_or_si256(x0, x1));
    }
    for (; i + 32u <= bytes; i += 32u) {
        __m256i x0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(a + i)),
                                      _mm256_loadu_si256((const __m256i *)(b + i)));
        acc = _mm256_or_si256(acc, x0);
    }
    if (!_mm256_testz_si256(acc, acc)) {
        return false;
    }
    for (; i < bytes; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

__attribute__((target("ssse3")))
static bool kops_row_chunk_equal_ssse3(const uint8_t *a, const uint8_t *b, size_t bytes)
{
    size_t i = 0;
    __m128i acc = _mm_setzero_si128();
    for (; i + 32u <= bytes; i += 32u) {
        __m128i x0 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(a + i)),
                                   _mm_loadu_si128((const __m128i *)(b + i)));
        __m128i x1 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(a + i + 16)),
                                   _mm_loadu_si128((const __m128i *)(b + i + 16)));
        acc = _mm_or_si128(acc, _mm_or_si128(x0, x1));
    }
    for (; i + 16u <= bytes; i += 16u) {
        __m128i x0 = _mm_xor_si128(_mm_loadu_si128((const __m128i *)(a + i)),
                                   _mm_loadu_si128((const __m128i *)(b + i)));
        acc = _mm_or_si128(acc, x0);
    }
    if (_mm_movemask_epi8(_mm_cmpeq_epi8(acc, _mm_setzero_si128())) != 0xFFFF) {
        return false;
    }
    for (; i < bytes; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

static inline void kops_store12(uint8_t *dst, __m128i v)
{
    uint64_t lo = (uint64_t)_mm_cvtsi128_si64(v);
    uint32_t hi = (uint32_t)_mm_cvtsi128_si32(_mm_srli_si128(v, 8));
    memcpy(dst, &lo, 8);
    memcpy(dst + 8, &hi, 4);
}

static inline __m128i kops_load12(const uint8_t *src)
{
    uint64_t lo = 0;
    uint32_t hi = 0;
    memcpy(&lo, src, 8);
    memcpy(&hi, src + 8, 4);
    return _mm_or_si128(_mm_cvtsi64_si128((long long)lo),
                        _mm_slli_si128(_mm_cvtsi32_si128((int)hi), 8));
}

/* Tighten [sx0, sx1) from a 12-bit byte-mismatch mask covering 4 RGB24 pixels. */
static inline void kops_x86_rgb4_span(uint32_t diff_mask12, int32_t base_x,
                                      int32_t *sx0, int32_t *sx1)
{
    if (diff_mask12 != 0u) {
        const int32_t first = base_x + (__builtin_ctz(diff_mask12) / 3);
        const int32_t last_byte = 31 - __builtin_clz(diff_mask12);
        const int32_t last = base_x + (last_byte / 3) + 1;
        if (first < *sx0) { *sx0 = first; }
        if (last > *sx1) { *sx1 = last; }
    }
}

__attribute__((target("avx2")))
static bool kops_xrgb_row_avx2(const uint8_t *srow, uint8_t *drow, int32_t tw, bool use_rgba,
                               int32_t *out_x, int32_t *sx0, int32_t *sx1)
{
    int32_t x = 0;
    bool row_dirty = false;

    if (!use_rgba) {
        const __m256i shuf = _mm256_setr_epi8(
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1,
            2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
        while (x + 8 <= tw) {
            const __m256i src = _mm256_loadu_si256((const __m256i *)(srow + (size_t)x * 4u));
            const __m256i rgb = _mm256_shuffle_epi8(src, shuf);
            const __m128i lo = _mm256_castsi256_si128(rgb);
            const __m128i hi = _mm256_extracti128_si256(rgb, 1);
            const __m128i dlo = kops_load12(drow + (size_t)x * 3u);
            const __m128i dhi = kops_load12(drow + (size_t)(x + 4) * 3u);
            const uint32_t m0 = (uint32_t)(~_mm_movemask_epi8(_mm_cmpeq_epi8(lo, dlo))) & 0xFFFu;
            const uint32_t m1 = (uint32_t)(~_mm_movemask_epi8(_mm_cmpeq_epi8(hi, dhi))) & 0xFFFu;
            if ((m0 | m1) != 0u) {
                if (m0) {
                    kops_store12(drow + (size_t)x * 3u, lo);
                    kops_x86_rgb4_span(m0, x, sx0, sx1);
                }
                if (m1) {
                    kops_store12(drow + (size_t)(x + 4) * 3u, hi);
                    kops_x86_rgb4_span(m1, x + 4, sx0, sx1);
                }
                row_dirty = true;
            }
            x += 8;
        }
    } else {
        const __m256i shuf = _mm256_setr_epi8(
            2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
            2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        while (x + 8 <= tw) {
            const __m256i src = _mm256_loadu_si256((const __m256i *)(srow + (size_t)x * 4u));
            const __m256i rgba = _mm256_shuffle_epi8(src, shuf);
            const __m256i dst = _mm256_loadu_si256((const __m256i *)(drow + (size_t)x * 4u));
            const uint32_t diff_m = ~(uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(rgba, dst));
            if (diff_m != 0u) {
                const int32_t first = x + (__builtin_ctz(diff_m) >> 2);
                const int32_t last = x + 8 - (__builtin_clz(diff_m) >> 2);
                _mm256_storeu_si256((__m256i *)(drow + (size_t)x * 4u), rgba);
                if (first < *sx0) { *sx0 = first; }
                if (last > *sx1) { *sx1 = last; }
                row_dirty = true;
            }
            x += 8;
        }
    }
    *out_x = x;
    return row_dirty;
}

__attribute__((target("ssse3")))
static bool kops_xrgb_row_ssse3(const uint8_t *srow, uint8_t *drow, int32_t tw, bool use_rgba,
                                int32_t *out_x, int32_t *sx0, int32_t *sx1)
{
    int32_t x = 0;
    bool row_dirty = false;

    if (!use_rgba) {
        const __m128i shuf = _mm_setr_epi8(2, 1, 0, 6, 5, 4, 10, 9, 8, 14, 13, 12, -1, -1, -1, -1);
        while (x + 4 <= tw) {
            const __m128i src = _mm_loadu_si128((const __m128i *)(srow + (size_t)x * 4u));
            const __m128i rgb = _mm_shuffle_epi8(src, shuf);
            const __m128i dlo = kops_load12(drow + (size_t)x * 3u);
            const uint32_t m0 = (uint32_t)(~_mm_movemask_epi8(_mm_cmpeq_epi8(rgb, dlo))) & 0xFFFu;
            if (m0 != 0u) {
                kops_store12(drow + (size_t)x * 3u, rgb);
                kops_x86_rgb4_span(m0, x, sx0, sx1);
                row_dirty = true;
            }
            x += 4;
        }
    } else {
        const __m128i shuf = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        while (x + 4 <= tw) {
            const __m128i src = _mm_loadu_si128((const __m128i *)(srow + (size_t)x * 4u));
            const __m128i rgba = _mm_shuffle_epi8(src, shuf);
            const __m128i dst = _mm_loadu_si128((const __m128i *)(drow + (size_t)x * 4u));
            const uint32_t diff_m = (uint32_t)(~_mm_movemask_epi8(_mm_cmpeq_epi8(rgba, dst))) & 0xFFFFu;
            if (diff_m != 0u) {
                const int32_t first = x + (__builtin_ctz(diff_m) >> 2);
                const int32_t last = x + ((31 - __builtin_clz(diff_m)) >> 2) + 1;
                _mm_storeu_si128((__m128i *)(drow + (size_t)x * 4u), rgba);
                if (first < *sx0) { *sx0 = first; }
                if (last > *sx1) { *sx1 = last; }
                row_dirty = true;
            }
            x += 4;
        }
    }
    *out_x = x;
    return row_dirty;
}
#endif

static inline int32_t kops_min32(int32_t a, int32_t b) { return a < b ? a : b; }
static inline int32_t kops_max32(int32_t a, int32_t b) { return a > b ? a : b; }

static inline bool kops_clip_rect(int32_t fb_w, int32_t fb_h, const kops_rect *in, kops_rect *out)
{
    int32_t x0 = in->x < 0 ? 0 : in->x;
    int32_t y0 = in->y < 0 ? 0 : in->y;
    int32_t x1 = in->x + in->w;
    int32_t y1 = in->y + in->h;
    if (x1 > fb_w) { x1 = fb_w; }
    if (y1 > fb_h) { y1 = fb_h; }
    if (x1 <= x0 || y1 <= y0) {
        return false;
    }
    out->x = x0;
    out->y = y0;
    out->w = x1 - x0;
    out->h = y1 - y0;
    return true;
}

bool kops_union_rects(int32_t fb_w, int32_t fb_h,
                      const kops_rect *rects, int32_t n_rects,
                      kops_rect *out)
{
    int32_t i;
    int32_t min_x = fb_w, min_y = fb_h, max_x = 0, max_y = 0;
    bool any = false;

    if (!out || fb_w <= 0 || fb_h <= 0) {
        return false;
    }
    if (!rects || n_rects <= 0) {
        out->x = 0;
        out->y = 0;
        out->w = fb_w;
        out->h = fb_h;
        return true;
    }
    for (i = 0; i < n_rects; ++i) {
        kops_rect cl;
        if (!kops_clip_rect(fb_w, fb_h, &rects[i], &cl)) {
            continue;
        }
        if (cl.x < min_x) { min_x = cl.x; }
        if (cl.y < min_y) { min_y = cl.y; }
        if (cl.x + cl.w > max_x) { max_x = cl.x + cl.w; }
        if (cl.y + cl.h > max_y) { max_y = cl.y + cl.h; }
        any = true;
    }
    if (!any) {
        return false;
    }
    out->x = min_x;
    out->y = min_y;
    out->w = max_x - min_x;
    out->h = max_y - min_y;
    return true;
}

float kops_coverage_ratio(int32_t fb_w, int32_t fb_h, const kops_rect *r)
{
    double fb_area, r_area, cov;
    if (!r || fb_w <= 0 || fb_h <= 0 || r->w <= 0 || r->h <= 0) {
        return 0.0f;
    }
    fb_area = (double)fb_w * (double)fb_h;
    r_area = (double)r->w * (double)r->h;
    cov = r_area / fb_area;
    if (cov < 0.0) { cov = 0.0; }
    if (cov > 1.0) { cov = 1.0; }
    return (float)cov;
}

int32_t kops_coalesce_rects(int32_t fb_w, int32_t fb_h,
                            const kops_rect *in, int32_t n_in,
                            int32_t gap_px, size_t max_bloat_px,
                            kops_rect *out, int32_t max_out)
{
    kops_rect tmp[64];
    kops_rect clipped_in[64];
    int32_t n = 0;
    int32_t n_raw = 0;
    int32_t i;
    bool merged;
    size_t raw_px = 0;

    if (!in || !out || max_out <= 0 || fb_w <= 0 || fb_h <= 0 || n_in <= 0) {
        return 0;
    }
    if (gap_px < 0) {
        gap_px = 0;
    }

    for (i = 0; i < n_in && n < 64; ++i) {
        kops_rect cl;
        if (!kops_clip_rect(fb_w, fb_h, &in[i], &cl)) {
            continue;
        }
        tmp[n] = cl;
        clipped_in[n] = cl;
        raw_px += (size_t)cl.w * (size_t)cl.h;
        n++;
    }
    n_raw = n;
    if (n == 0) {
        return 0;
    }

    do {
        merged = false;
        for (i = 0; i < n; ++i) {
            int32_t j = i + 1;
            int32_t ax0 = tmp[i].x, ay0 = tmp[i].y;
            int32_t ax1 = ax0 + tmp[i].w, ay1 = ay0 + tmp[i].h;
            while (j < n) {
                const int32_t bx0 = tmp[j].x, by0 = tmp[j].y;
                const int32_t bx1 = bx0 + tmp[j].w, by1 = by0 + tmp[j].h;
                if (ax0 - gap_px < bx1 && ax1 + gap_px > bx0 &&
                    ay0 - gap_px < by1 && ay1 + gap_px > by0) {
                    ax0 = kops_min32(ax0, bx0);
                    ay0 = kops_min32(ay0, by0);
                    ax1 = kops_max32(ax1, bx1);
                    ay1 = kops_max32(ay1, by1);
                    tmp[i].x = ax0;
                    tmp[i].y = ay0;
                    tmp[i].w = ax1 - ax0;
                    tmp[i].h = ay1 - ay0;
                    tmp[j] = tmp[n - 1];
                    n--;
                    merged = true;
                } else {
                    j++;
                }
            }
        }
    } while (merged && n > 1);

    if (n > max_out) {
        kops_rect u = tmp[0];
        for (i = 1; i < n; ++i) {
            const int32_t x0 = kops_min32(u.x, tmp[i].x);
            const int32_t y0 = kops_min32(u.y, tmp[i].y);
            const int32_t x1 = kops_max32(u.x + u.w, tmp[i].x + tmp[i].w);
            const int32_t y1 = kops_max32(u.y + u.h, tmp[i].y + tmp[i].h);
            u.x = x0;
            u.y = y0;
            u.w = x1 - x0;
            u.h = y1 - y0;
        }
        tmp[0] = u;
        n = 1;
    }

    if (max_bloat_px > 0 && n < n_raw && n_raw <= max_out) {
        size_t coal_px = 0;
        for (i = 0; i < n; ++i) {
            coal_px += (size_t)tmp[i].w * (size_t)tmp[i].h;
        }
        if (coal_px > raw_px + max_bloat_px) {
            for (i = 0; i < n_raw; ++i) {
                out[i] = clipped_in[i];
            }
            return n_raw;
        }
    }

    for (i = 0; i < n; ++i) {
        out[i] = tmp[i];
    }
    return n;
}

uint64_t kops_fast_hash64(const uint8_t *data, size_t len)
{
    uint64_t h0 = 0xcbf29ce484222325ULL;
    uint64_t h1 = 0x9e3779b97f4a7c15ULL;
    uint64_t h2 = 0x517cc1b727220a95ULL;
    uint64_t h3 = 0x94d049bb133111ebULL;
    size_t i = 0;

    if (!data || len == 0) {
        return h0;
    }
    for (; i + 32u <= len; i += 32u) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, data + i, 8);
        memcpy(&w1, data + i + 8, 8);
        memcpy(&w2, data + i + 16, 8);
        memcpy(&w3, data + i + 24, 8);
        h0 = (h0 ^ w0) * 0x100000001b3ULL;
        h1 = (h1 ^ w1) * 0x100000001b3ULL;
        h2 = (h2 ^ w2) * 0x100000001b3ULL;
        h3 = (h3 ^ w3) * 0x100000001b3ULL;
    }
    h0 ^= h1 * 0x9e3779b97f4a7c15ULL;
    h0 ^= h2 * 0xbf58476d1ce4e5b9ULL;
    h0 ^= h3 * 0x94d049bb133111ebULL;
    for (; i < len; ++i) {
        h0 ^= (uint64_t)data[i];
        h0 *= 0x100000001b3ULL;
    }
    h0 ^= h0 >> 33;
    h0 *= 0xff51afd7ed558ccdULL;
    h0 ^= h0 >> 33;
    return h0;
}

bool kops_pack_rect(const uint8_t *fb, int32_t fb_w, int32_t fb_h, int32_t bpp,
                    const kops_rect *rect, uint8_t *out, size_t out_cap, size_t *out_len)
{
    kops_rect cl;
    size_t row_bytes, fb_stride, need;
    int32_t y;

    if (!fb || !rect || !out || fb_w <= 0 || fb_h <= 0 || (bpp != 3 && bpp != 4)) {
        return false;
    }
    if (!kops_clip_rect(fb_w, fb_h, rect, &cl)) {
        return false;
    }
    row_bytes = (size_t)cl.w * (size_t)bpp;
    need = row_bytes * (size_t)cl.h;
    if (out_cap < need) {
        return false;
    }
    fb_stride = (size_t)fb_w * (size_t)bpp;
    if (cl.x == 0 && cl.w == fb_w) {
        memcpy(out, fb + (size_t)cl.y * fb_stride, need);
    } else {
        const uint8_t *src = fb + (size_t)cl.y * fb_stride + (size_t)cl.x * (size_t)bpp;
        uint8_t *dst = out;
        for (y = 0; y < cl.h; ++y) {
            memcpy(dst, src, row_bytes);
            src += fb_stride;
            dst += row_bytes;
        }
    }
    if (out_len) {
        *out_len = need;
    }
    return true;
}

/* --------------------------------------------------------------------------
 * Parallel SIMD multi-rect tile diff (RGB24 / RGBA32) with 2D pixel trimming
 * -------------------------------------------------------------------------- */

static inline bool kops_row_chunk_equal(const uint8_t *a, const uint8_t *b, size_t bytes)
{
#ifdef KOPS_HAVE_NEON
    size_t i = 0;
    uint8x16_t acc = vdupq_n_u8(0);
    for (; i + 32u <= bytes; i += 32u) {
        uint8x16_t x0 = veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i));
        uint8x16_t x1 = veorq_u8(vld1q_u8(a + i + 16), vld1q_u8(b + i + 16));
        acc = vorrq_u8(acc, vorrq_u8(x0, x1));
    }
    for (; i + 16u <= bytes; i += 16u) {
        acc = vorrq_u8(acc, veorq_u8(vld1q_u8(a + i), vld1q_u8(b + i)));
    }
    if (vmaxvq_u32(vreinterpretq_u32_u8(acc)) != 0u) {
        return false;
    }
    for (; i < bytes; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
#elif defined(KOPS_HAVE_X86_SIMD)
    const int lv = kops_x86_simd_level();
    if (lv >= 2 && bytes >= 32u) {
        return kops_row_chunk_equal_avx2(a, b, bytes);
    }
    if (lv >= 1 && bytes >= 16u) {
        return kops_row_chunk_equal_ssse3(a, b, bytes);
    }
    return memcmp(a, b, bytes) == 0;
#else
    return memcmp(a, b, bytes) == 0;
#endif
}

static int32_t kops_diff_band_rgb(const uint8_t *prev, const uint8_t *curr,
                                  int32_t fb_w, int32_t bpp,
                                  int32_t tile, int32_t tiles_x,
                                  int32_t ty, int32_t th,
                                  uint32_t *tile_spans)
{
    const size_t stride = (size_t)fb_w * (size_t)bpp;
    uint8_t sy0[512], sy1[512], sx0[512], sx1[512];
    int32_t tx_idx, ry;
    int32_t dirty_count = 0;

    if (tiles_x <= 512) {
        for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
            const int32_t tx = tx_idx * tile;
            const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
            sy0[tx_idx] = (uint8_t)th;
            sy1[tx_idx] = 0u;
            sx0[tx_idx] = (uint8_t)tw;
            sx1[tx_idx] = 0u;
        }
        /* Row-major streaming pass: linear DRAM reads maximize hardware prefetcher BW. */
        for (ry = 0; ry < th; ++ry) {
            const size_t row_off = (size_t)(ty + ry) * stride;
            const uint8_t *prow = prev + row_off;
            const uint8_t *crow = curr + row_off;
            for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
                const int32_t tx = tx_idx * tile;
                const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
                const size_t x_off = (size_t)tx * (size_t)bpp;
                const uint8_t *pa = prow + x_off;
                const uint8_t *pb = crow + x_off;
                if (kops_row_chunk_equal(pa, pb, (size_t)tw * (size_t)bpp)) {
                    continue;
                }
                if ((uint8_t)ry < sy0[tx_idx]) { sy0[tx_idx] = (uint8_t)ry; }
                sy1[tx_idx] = (uint8_t)(ry + 1);
                {
                    int32_t lx = 0, rx = tw;
                    while (lx < rx) {
                        const size_t o = (size_t)lx * (size_t)bpp;
                        if (pa[o] != pb[o] || pa[o + 1] != pb[o + 1] || pa[o + 2] != pb[o + 2] ||
                            (bpp == 4 && pa[o + 3] != pb[o + 3])) {
                            break;
                        }
                        lx++;
                    }
                    while (rx > lx) {
                        const size_t o = (size_t)(rx - 1) * (size_t)bpp;
                        if (pa[o] != pb[o] || pa[o + 1] != pb[o + 1] || pa[o + 2] != pb[o + 2] ||
                            (bpp == 4 && pa[o + 3] != pb[o + 3])) {
                            break;
                        }
                        rx--;
                    }
                    if ((uint8_t)lx < sx0[tx_idx]) { sx0[tx_idx] = (uint8_t)lx; }
                    if ((uint8_t)rx > sx1[tx_idx]) { sx1[tx_idx] = (uint8_t)rx; }
                }
            }
        }
        for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
            if (sy0[tx_idx] < sy1[tx_idx] && sx0[tx_idx] < sx1[tx_idx]) {
                tile_spans[tx_idx] = ((uint32_t)sy0[tx_idx] << 24) |
                                     ((uint32_t)sy1[tx_idx] << 16) |
                                     ((uint32_t)sx0[tx_idx] << 8) |
                                     (uint32_t)sx1[tx_idx];
                dirty_count++;
            } else {
                tile_spans[tx_idx] = 0u;
            }
        }
        return dirty_count;
    }

    for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
        const int32_t tx = tx_idx * tile;
        const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
        const size_t row_bytes = (size_t)tw * (size_t)bpp;
        int32_t t_sy0 = th, t_sy1 = 0, t_sx0 = tw, t_sx1 = 0;
        for (ry = 0; ry < th; ++ry) {
            const size_t off = (size_t)(ty + ry) * stride + (size_t)tx * (size_t)bpp;
            const uint8_t *pa = prev + off;
            const uint8_t *pb = curr + off;
            if (kops_row_chunk_equal(pa, pb, row_bytes)) { continue; }
            if (ry < t_sy0) { t_sy0 = ry; }
            t_sy1 = ry + 1;
            t_sx0 = 0;
            t_sx1 = tw;
        }
        if (t_sy0 < t_sy1 && t_sx0 < t_sx1) {
            tile_spans[tx_idx] = ((uint32_t)(uint8_t)t_sy0 << 24) |
                                 ((uint32_t)(uint8_t)t_sy1 << 16) |
                                 ((uint32_t)(uint8_t)t_sx0 << 8) |
                                 (uint32_t)(uint8_t)t_sx1;
            dirty_count++;
        } else {
            tile_spans[tx_idx] = 0u;
        }
    }
    return dirty_count;
}

typedef struct kops_par_rgb_diff_ctx {
    const uint8_t *prev;
    const uint8_t *curr;
    int32_t fb_w, fb_h, bpp, tile, tiles_x, num_bands, n_workers;
    uint32_t *all_spans;
    int32_t *band_counts;
} kops_par_rgb_diff_ctx;

static void kops_par_rgb_diff_worker(void *arg, size_t widx)
{
    kops_par_rgb_diff_ctx *ctx = (kops_par_rgb_diff_ctx *)arg;
    const int32_t b0 = ((int32_t)widx * ctx->num_bands) / ctx->n_workers;
    const int32_t b1 = (((int32_t)widx + 1) * ctx->num_bands) / ctx->n_workers;
    int32_t b;
    for (b = b0; b < b1; ++b) {
        const int32_t ty = b * ctx->tile;
        const int32_t th = (ty + ctx->tile <= ctx->fb_h) ? ctx->tile : (ctx->fb_h - ty);
        uint32_t *ts = ctx->all_spans + (size_t)b * (size_t)ctx->tiles_x;
        ctx->band_counts[b] = kops_diff_band_rgb(ctx->prev, ctx->curr, ctx->fb_w, ctx->bpp,
                                                 ctx->tile, ctx->tiles_x, ty, th, ts);
    }
}

static kops_diff_outcome kops_collect_spans_to_rects(const uint32_t *all_spans,
                                                     const int32_t *band_counts,
                                                     int32_t fb_w, int32_t fb_h,
                                                     int32_t tile, int32_t tiles_x, int32_t num_bands,
                                                     float cover_limit,
                                                     kops_rect *rects_out, int32_t max_rects,
                                                     int32_t *out_count, kops_rect *bbox_out)
{
    kops_rect acc[64];
    int32_t n_acc = 0;
    bool overflow = false;
    int32_t min_x = fb_w, min_y = fb_h, max_x = 0, max_y = 0;
    size_t dirty_px = 0;
    int32_t b, tx_idx;
    kops_rect bbox;

    for (b = 0; b < num_bands; ++b) {
        const int32_t ty = b * tile;
        const int32_t th = (ty + tile <= fb_h) ? tile : (fb_h - ty);
        const uint32_t *tile_span = all_spans + (size_t)b * (size_t)tiles_x;
        int32_t run_x0 = -1, run_x1 = 0, run_y0 = th, run_y1 = 0;
        if (band_counts[b] == 0) {
            continue;
        }
        for (tx_idx = 0; tx_idx <= tiles_x; ++tx_idx) {
            const uint32_t sp = (tx_idx < tiles_x) ? tile_span[tx_idx] : 0u;
            if (sp != 0u) {
                const int32_t tx = tx_idx * tile;
                const int32_t sy0 = (int32_t)((sp >> 24) & 0xFFu);
                const int32_t sy1 = (int32_t)((sp >> 16) & 0xFFu);
                const int32_t sx0 = tx + (int32_t)((sp >> 8) & 0xFFu);
                const int32_t sx1 = tx + (int32_t)(sp & 0xFFu);
                if (sx0 < min_x) { min_x = sx0; }
                if (sx1 > max_x) { max_x = sx1; }
                if (ty + sy0 < min_y) { min_y = ty + sy0; }
                if (ty + sy1 > max_y) { max_y = ty + sy1; }
                if (run_x0 < 0) {
                    run_x0 = sx0;
                    run_x1 = sx1;
                    run_y0 = sy0;
                    run_y1 = sy1;
                } else {
                    run_x1 = sx1;
                    if (sy0 < run_y0) { run_y0 = sy0; }
                    if (sy1 > run_y1) { run_y1 = sy1; }
                }
            } else if (run_x0 >= 0) {
                const int32_t run_w = run_x1 - run_x0;
                const int32_t ry0 = ty + run_y0;
                const int32_t rh = (ty + run_y1) - ry0;
                dirty_px += (size_t)run_w * (size_t)rh;
                if (!overflow) {
                    bool extended = false;
                    int32_t k;
                    for (k = n_acc - 1; k >= 0; --k) {
                        if (acc[k].x == run_x0 && acc[k].w == run_w && acc[k].y + acc[k].h == ry0) {
                            acc[k].h += rh;
                            extended = true;
                            break;
                        }
                    }
                    if (!extended) {
                        if (n_acc < 64) {
                            acc[n_acc].x = run_x0;
                            acc[n_acc].y = ry0;
                            acc[n_acc].w = run_w;
                            acc[n_acc].h = rh;
                            n_acc++;
                        } else {
                            overflow = true;
                        }
                    }
                }
                run_x0 = -1;
                run_x1 = 0;
                run_y0 = th;
                run_y1 = 0;
            }
        }
    }

    if (dirty_px == 0) {
        if (bbox_out) {
            bbox_out->x = bbox_out->y = bbox_out->w = bbox_out->h = 0;
        }
        return KOPS_DIFF_NONE;
    }

    bbox.x = min_x;
    bbox.y = min_y;
    bbox.w = max_x - min_x;
    bbox.h = max_y - min_y;
    if (bbox_out) {
        *bbox_out = bbox;
    }

    {
        const double fb_area = (double)fb_w * (double)fb_h;
        const float cov = fb_area > 0.0 ? (float)((double)dirty_px / fb_area) : 1.0f;
        if (cov >= cover_limit) {
            if (rects_out && max_rects > 0) {
                rects_out[0] = bbox;
            }
            if (out_count) {
                *out_count = (rects_out && max_rects > 0) ? 1 : 0;
            }
            return KOPS_DIFF_FULL;
        }
    }

    if (rects_out && max_rects > 0) {
        if (overflow) {
            rects_out[0] = bbox;
            if (out_count) { *out_count = 1; }
        } else if (n_acc > max_rects) {
            const int32_t cn = kops_coalesce_rects(fb_w, fb_h, acc, n_acc, tile, 0u, rects_out, max_rects);
            if (cn > 0) {
                if (out_count) { *out_count = cn; }
            } else {
                rects_out[0] = bbox;
                if (out_count) { *out_count = 1; }
            }
        } else {
            int32_t i;
            for (i = 0; i < n_acc; ++i) {
                rects_out[i] = acc[i];
            }
            if (out_count) { *out_count = n_acc; }
        }
    }
    return KOPS_DIFF_RECT;
}

kops_diff_outcome kops_tile_diff_rects(const uint8_t *prev, const uint8_t *curr,
                                       int32_t fb_w, int32_t fb_h, int32_t bpp,
                                       int32_t tile, float cover_limit,
                                       kops_rect *rects_out, int32_t max_rects,
                                       int32_t *out_count, kops_rect *bbox_out)
{
    uint32_t stack_spans[4096];
    int32_t stack_counts[256];
    uint32_t *all_spans = stack_spans;
    int32_t *band_counts = stack_counts;
    int32_t tiles_x, num_bands;
    size_t total_tiles;
    kops_diff_outcome res;

    if (out_count) { *out_count = 0; }
    if (!prev || !curr || fb_w <= 0 || fb_h <= 0 || (bpp != 3 && bpp != 4) || tile <= 0) {
        if (bbox_out) {
            bbox_out->x = bbox_out->y = 0;
            bbox_out->w = fb_w > 0 ? fb_w : 0;
            bbox_out->h = fb_h > 0 ? fb_h : 0;
        }
        return KOPS_DIFF_FULL;
    }
    if (tile > 128) { tile = 128; }

    tiles_x = (fb_w + tile - 1) / tile;
    num_bands = (fb_h + tile - 1) / tile;
    total_tiles = (size_t)num_bands * (size_t)tiles_x;
    if (total_tiles > 4096u) {
        all_spans = (uint32_t *)malloc(total_tiles * sizeof(uint32_t));
        if (!all_spans) { return KOPS_DIFF_FULL; }
    }
    if (num_bands > 256) {
        band_counts = (int32_t *)malloc((size_t)num_bands * sizeof(int32_t));
        if (!band_counts) {
            if (all_spans != stack_spans) { free(all_spans); }
            return KOPS_DIFF_FULL;
        }
    }

    if ((size_t)fb_w * (size_t)fb_h >= (size_t)640 * 480 && num_bands >= 4) {
        kops_par_rgb_diff_ctx ctx;
        ctx.prev = prev;
        ctx.curr = curr;
        ctx.fb_w = fb_w;
        ctx.fb_h = fb_h;
        ctx.bpp = bpp;
        ctx.tile = tile;
        ctx.tiles_x = tiles_x;
        ctx.num_bands = num_bands;
        ctx.n_workers = ((size_t)fb_w * (size_t)fb_h >= (size_t)1280 * 720 && num_bands >= 6) ? 6 : 4;
        ctx.all_spans = all_spans;
        ctx.band_counts = band_counts;
        kops_parallel_for((size_t)ctx.n_workers, kops_par_rgb_diff_worker, &ctx);
    } else {
        int32_t b;
        for (b = 0; b < num_bands; ++b) {
            const int32_t ty = b * tile;
            const int32_t th = (ty + tile <= fb_h) ? tile : (fb_h - ty);
            band_counts[b] = kops_diff_band_rgb(prev, curr, fb_w, bpp, tile, tiles_x, ty, th,
                                                all_spans + (size_t)b * (size_t)tiles_x);
        }
    }

    res = kops_collect_spans_to_rects(all_spans, band_counts, fb_w, fb_h, tile, tiles_x, num_bands,
                                      cover_limit, rects_out, max_rects, out_count, bbox_out);
    if (band_counts != stack_counts) { free(band_counts); }
    if (all_spans != stack_spans) { free(all_spans); }
    return res;
}

/* --------------------------------------------------------------------------
 * Fused XRGB8888/ARGB8888 -> RGB24/RGBA32 convert + 32px NEON clean-tile diff
 * -------------------------------------------------------------------------- */

#ifdef KOPS_HAVE_NEON
/* Return exact first and past-last nonzero byte lane in a 16-byte NEON diff vector. */
static inline void kops_neon_diff_span16(uint8x16_t diff, int32_t base_x,
                                         int32_t *sx0, int32_t *sx1)
{
    /* Pack 16 nonzero-byte flags into a 64-bit nibble mask (4 bits per lane). */
    const uint8x16_t inv = vmvnq_u8(vceqq_u8(diff, vdupq_n_u8(0)));
    const uint8x8_t mask8 = vshrn_n_u16(vreinterpretq_u16_u8(inv), 4);
    const uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(mask8), 0);
    if (mask != 0u) {
        const int32_t first = base_x + (int32_t)(__builtin_ctzll(mask) >> 2);
        const int32_t last = base_x + 16 - (int32_t)(__builtin_clzll(mask) >> 2);
        if (first < *sx0) { *sx0 = first; }
        if (last > *sx1) { *sx1 = last; }
    }
}
#endif

static int32_t kops_convert_and_diff_band_xrgb(const uint8_t *src_pixels, int32_t src_pitch,
                                               uint8_t *scratch_rgb, bool use_rgba,
                                               int32_t fb_w, int32_t tile, int32_t tiles_x,
                                               int32_t ty, int32_t th, uint32_t *tile_spans)
{
    const int32_t bpp = use_rgba ? 4 : 3;
    const size_t dst_stride = (size_t)fb_w * (size_t)bpp;
    uint8_t t_sy0[512], t_sy1[512], t_sx0[512], t_sx1[512];
    int32_t tx_idx, ry;
    int32_t dirty_count = 0;

    if (tiles_x <= 512) {
        for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
            const int32_t tx = tx_idx * tile;
            const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
            t_sy0[tx_idx] = (uint8_t)th;
            t_sy1[tx_idx] = 0u;
            t_sx0[tx_idx] = (uint8_t)tw;
            t_sx1[tx_idx] = 0u;
        }

        /* Row-major streaming pass across the full scanline: saturates DRAM prefetcher
         * (~180 GB/s on Apple Silicon, cutting 1080p diff time in half vs tile-major). */
        for (ry = 0; ry < th; ++ry) {
            const uint8_t *sline = src_pixels + (size_t)(ty + ry) * (size_t)src_pitch;
            uint8_t *dline = scratch_rgb + (size_t)(ty + ry) * dst_stride;

            for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
                const int32_t tx = tx_idx * tile;
                const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
                const uint8_t *srow = sline + (size_t)tx * 4u;
                uint8_t *drow = dline + (size_t)tx * (size_t)bpp;
                int32_t x = 0;
                int32_t sx0 = (int32_t)t_sx0[tx_idx];
                int32_t sx1 = (int32_t)t_sx1[tx_idx];
                bool row_dirty = false;

#ifdef KOPS_HAVE_NEON
                if (!use_rgba) {
                    /* 32-pixel combined NEON clean-tile fast path (XRGB8888 LE: B,G,R,X) */
                    while (x + 32 <= tw) {
                        const uint8x16x4_t a = vld4q_u8(srow + (size_t)x * 4u);
                        const uint8x16x4_t b = vld4q_u8(srow + (size_t)(x + 16) * 4u);
                        const uint8x16x3_t da = vld3q_u8(drow + (size_t)x * 3u);
                        const uint8x16x3_t db = vld3q_u8(drow + (size_t)(x + 16) * 3u);
                        const uint8x16_t diff_a = vorrq_u8(vorrq_u8(veorq_u8(a.val[2], da.val[0]),
                                                                    veorq_u8(a.val[1], da.val[1])),
                                                           veorq_u8(a.val[0], da.val[2]));
                        const uint8x16_t diff_b = vorrq_u8(vorrq_u8(veorq_u8(b.val[2], db.val[0]),
                                                                    veorq_u8(b.val[1], db.val[1])),
                                                           veorq_u8(b.val[0], db.val[2]));
                        if (vmaxvq_u32(vreinterpretq_u32_u8(vorrq_u8(diff_a, diff_b))) != 0u) {
                            uint8x16x3_t out_a, out_b;
                            out_a.val[0] = a.val[2]; out_a.val[1] = a.val[1]; out_a.val[2] = a.val[0];
                            out_b.val[0] = b.val[2]; out_b.val[1] = b.val[1]; out_b.val[2] = b.val[0];
                            vst3q_u8(drow + (size_t)x * 3u, out_a);
                            vst3q_u8(drow + (size_t)(x + 16) * 3u, out_b);
                            row_dirty = true;
                            kops_neon_diff_span16(diff_a, x, &sx0, &sx1);
                            kops_neon_diff_span16(diff_b, x + 16, &sx0, &sx1);
                        }
                        x += 32;
                    }
                    while (x + 16 <= tw) {
                        const uint8x16x4_t a = vld4q_u8(srow + (size_t)x * 4u);
                        const uint8x16x3_t da = vld3q_u8(drow + (size_t)x * 3u);
                        const uint8x16_t diff = vorrq_u8(vorrq_u8(veorq_u8(a.val[2], da.val[0]),
                                                                  veorq_u8(a.val[1], da.val[1])),
                                                         veorq_u8(a.val[0], da.val[2]));
                        if (vmaxvq_u32(vreinterpretq_u32_u8(diff)) != 0u) {
                            uint8x16x3_t out_a;
                            out_a.val[0] = a.val[2]; out_a.val[1] = a.val[1]; out_a.val[2] = a.val[0];
                            vst3q_u8(drow + (size_t)x * 3u, out_a);
                            row_dirty = true;
                            kops_neon_diff_span16(diff, x, &sx0, &sx1);
                        }
                        x += 16;
                    }
                } else {
                    while (x + 16 <= tw) {
                        const uint8x16x4_t a = vld4q_u8(srow + (size_t)x * 4u);
                        const uint8x16x4_t da = vld4q_u8(drow + (size_t)x * 4u);
                        const uint8x16_t diff = vorrq_u8(
                            vorrq_u8(veorq_u8(a.val[2], da.val[0]), veorq_u8(a.val[1], da.val[1])),
                            vorrq_u8(veorq_u8(a.val[0], da.val[2]), veorq_u8(a.val[3], da.val[3])));
                        if (vmaxvq_u32(vreinterpretq_u32_u8(diff)) != 0u) {
                            uint8x16x4_t out_a;
                            out_a.val[0] = a.val[2]; out_a.val[1] = a.val[1];
                            out_a.val[2] = a.val[0]; out_a.val[3] = a.val[3];
                            vst4q_u8(drow + (size_t)x * 4u, out_a);
                            row_dirty = true;
                            kops_neon_diff_span16(diff, x, &sx0, &sx1);
                        }
                        x += 16;
                    }
                }
#elif defined(KOPS_HAVE_X86_SIMD)
                {
                    const int lv = kops_x86_simd_level();
                    if (lv >= 2 && tw >= 8) {
                        if (kops_xrgb_row_avx2(srow, drow, tw, use_rgba, &x, &sx0, &sx1)) {
                            row_dirty = true;
                        }
                    } else if (lv >= 1 && tw >= 4) {
                        if (kops_xrgb_row_ssse3(srow, drow, tw, use_rgba, &x, &sx0, &sx1)) {
                            row_dirty = true;
                        }
                    }
                }
#endif
                for (; x < tw; ++x) {
                    const uint8_t b_ch = srow[(size_t)x * 4u + 0u];
                    const uint8_t g_ch = srow[(size_t)x * 4u + 1u];
                    const uint8_t r_ch = srow[(size_t)x * 4u + 2u];
                    if (!use_rgba) {
                        uint8_t *dp = drow + (size_t)x * 3u;
                        if (dp[0] != r_ch || dp[1] != g_ch || dp[2] != b_ch) {
                            dp[0] = r_ch; dp[1] = g_ch; dp[2] = b_ch;
                            row_dirty = true;
                            if (x < sx0) { sx0 = x; }
                            if (x + 1 > sx1) { sx1 = x + 1; }
                        }
                    } else {
                        const uint8_t a_ch = srow[(size_t)x * 4u + 3u];
                        uint8_t *dp = drow + (size_t)x * 4u;
                        if (dp[0] != r_ch || dp[1] != g_ch || dp[2] != b_ch || dp[3] != a_ch) {
                            dp[0] = r_ch; dp[1] = g_ch; dp[2] = b_ch; dp[3] = a_ch;
                            row_dirty = true;
                            if (x < sx0) { sx0 = x; }
                            if (x + 1 > sx1) { sx1 = x + 1; }
                        }
                    }
                }
                if (row_dirty) {
                    if ((uint8_t)ry < t_sy0[tx_idx]) { t_sy0[tx_idx] = (uint8_t)ry; }
                    t_sy1[tx_idx] = (uint8_t)(ry + 1);
                    t_sx0[tx_idx] = (uint8_t)sx0;
                    t_sx1[tx_idx] = (uint8_t)sx1;
                }
            }
        }

        for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
            if (t_sy0[tx_idx] < t_sy1[tx_idx] && t_sx0[tx_idx] < t_sx1[tx_idx]) {
                tile_spans[tx_idx] = ((uint32_t)t_sy0[tx_idx] << 24) |
                                     ((uint32_t)t_sy1[tx_idx] << 16) |
                                     ((uint32_t)t_sx0[tx_idx] << 8) |
                                     (uint32_t)t_sx1[tx_idx];
                dirty_count++;
            } else {
                tile_spans[tx_idx] = 0u;
            }
        }
        return dirty_count;
    }

    /* Rare wide FB (> 512 tiles across): scalar tile-major fallback */
    for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
        const int32_t tx = tx_idx * tile;
        const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
        int32_t sy0 = th, sy1 = 0, sx0 = tw, sx1 = 0;
        for (ry = 0; ry < th; ++ry) {
            const uint8_t *srow = src_pixels + (size_t)(ty + ry) * (size_t)src_pitch + (size_t)tx * 4u;
            uint8_t *drow = scratch_rgb + (size_t)(ty + ry) * dst_stride + (size_t)tx * (size_t)bpp;
            int32_t x;
            bool row_dirty = false;
            for (x = 0; x < tw; ++x) {
                const uint8_t b_ch = srow[(size_t)x * 4u + 0u];
                const uint8_t g_ch = srow[(size_t)x * 4u + 1u];
                const uint8_t r_ch = srow[(size_t)x * 4u + 2u];
                uint8_t *dp = drow + (size_t)x * (size_t)bpp;
                if (!use_rgba) {
                    if (dp[0] != r_ch || dp[1] != g_ch || dp[2] != b_ch) {
                        dp[0] = r_ch; dp[1] = g_ch; dp[2] = b_ch;
                        row_dirty = true;
                        if (x < sx0) { sx0 = x; }
                        if (x + 1 > sx1) { sx1 = x + 1; }
                    }
                } else {
                    const uint8_t a_ch = srow[(size_t)x * 4u + 3u];
                    if (dp[0] != r_ch || dp[1] != g_ch || dp[2] != b_ch || dp[3] != a_ch) {
                        dp[0] = r_ch; dp[1] = g_ch; dp[2] = b_ch; dp[3] = a_ch;
                        row_dirty = true;
                        if (x < sx0) { sx0 = x; }
                        if (x + 1 > sx1) { sx1 = x + 1; }
                    }
                }
            }
            if (row_dirty) {
                if (ry < sy0) { sy0 = ry; }
                sy1 = ry + 1;
            }
        }
        if (sy0 < sy1 && sx0 < sx1) {
            tile_spans[tx_idx] = ((uint32_t)(uint8_t)sy0 << 24) |
                                 ((uint32_t)(uint8_t)sy1 << 16) |
                                 ((uint32_t)(uint8_t)sx0 << 8) |
                                 (uint32_t)(uint8_t)sx1;
            dirty_count++;
        } else {
            tile_spans[tx_idx] = 0u;
        }
    }
    return dirty_count;
}

typedef struct kops_par_xrgb_ctx {
    const uint8_t *src_pixels;
    int32_t src_pitch;
    uint8_t *scratch_rgb;
    bool use_rgba;
    int32_t fb_w, fb_h, tile, tiles_x, num_bands, n_workers;
    uint32_t *all_spans;
    int32_t *band_counts;
} kops_par_xrgb_ctx;

static void kops_par_xrgb_worker(void *arg, size_t widx)
{
    kops_par_xrgb_ctx *ctx = (kops_par_xrgb_ctx *)arg;
    const int32_t b0 = ((int32_t)widx * ctx->num_bands) / ctx->n_workers;
    const int32_t b1 = (((int32_t)widx + 1) * ctx->num_bands) / ctx->n_workers;
    int32_t b;
    for (b = b0; b < b1; ++b) {
        const int32_t ty = b * ctx->tile;
        const int32_t th = (ty + ctx->tile <= ctx->fb_h) ? ctx->tile : (ctx->fb_h - ty);
        ctx->band_counts[b] = kops_convert_and_diff_band_xrgb(
            ctx->src_pixels, ctx->src_pitch, ctx->scratch_rgb, ctx->use_rgba,
            ctx->fb_w, ctx->tile, ctx->tiles_x, ty, th,
            ctx->all_spans + (size_t)b * (size_t)ctx->tiles_x);
    }
}

kops_diff_outcome kops_convert_xrgb_and_diff(const uint8_t *src_pixels, int32_t src_pitch,
                                             uint8_t *scratch_rgb,
                                             int32_t fb_w, int32_t fb_h, bool use_rgba,
                                             int32_t tile, float cover_limit,
                                             kops_rect *rects_out, int32_t max_rects,
                                             int32_t *out_count, kops_rect *bbox_out)
{
    uint32_t stack_spans[4096];
    int32_t stack_counts[256];
    uint32_t *all_spans = stack_spans;
    int32_t *band_counts = stack_counts;
    int32_t tiles_x, num_bands;
    size_t total_tiles;
    kops_diff_outcome res;

    if (out_count) { *out_count = 0; }
    if (!src_pixels || !scratch_rgb || fb_w <= 0 || fb_h <= 0 || src_pitch < fb_w * 4 || tile <= 0) {
        return KOPS_DIFF_FULL;
    }
    if (tile > 128) { tile = 128; }

    tiles_x = (fb_w + tile - 1) / tile;
    num_bands = (fb_h + tile - 1) / tile;
    total_tiles = (size_t)num_bands * (size_t)tiles_x;
    if (total_tiles > 4096u) {
        all_spans = (uint32_t *)malloc(total_tiles * sizeof(uint32_t));
        if (!all_spans) { return KOPS_DIFF_FULL; }
    }
    if (num_bands > 256) {
        band_counts = (int32_t *)malloc((size_t)num_bands * sizeof(int32_t));
        if (!band_counts) {
            if (all_spans != stack_spans) { free(all_spans); }
            return KOPS_DIFF_FULL;
        }
    }

    if ((size_t)fb_w * (size_t)fb_h >= (size_t)640 * 480 && num_bands >= 4) {
        kops_par_xrgb_ctx ctx;
        ctx.src_pixels = src_pixels;
        ctx.src_pitch = src_pitch;
        ctx.scratch_rgb = scratch_rgb;
        ctx.use_rgba = use_rgba;
        ctx.fb_w = fb_w;
        ctx.fb_h = fb_h;
        ctx.tile = tile;
        ctx.tiles_x = tiles_x;
        ctx.num_bands = num_bands;
        ctx.n_workers = ((size_t)fb_w * (size_t)fb_h >= (size_t)1280 * 720 && num_bands >= 6) ? 6 : 4;
        ctx.all_spans = all_spans;
        ctx.band_counts = band_counts;
        kops_parallel_for((size_t)ctx.n_workers, kops_par_xrgb_worker, &ctx);
    } else {
        int32_t b;
        for (b = 0; b < num_bands; ++b) {
            const int32_t ty = b * tile;
            const int32_t th = (ty + tile <= fb_h) ? tile : (fb_h - ty);
            band_counts[b] = kops_convert_and_diff_band_xrgb(
                src_pixels, src_pitch, scratch_rgb, use_rgba, fb_w, tile, tiles_x, ty, th,
                all_spans + (size_t)b * (size_t)tiles_x);
        }
    }

    res = kops_collect_spans_to_rects(all_spans, band_counts, fb_w, fb_h, tile, tiles_x, num_bands,
                                      cover_limit, rects_out, max_rects, out_count, bbox_out);
    if (band_counts != stack_counts) { free(band_counts); }
    if (all_spans != stack_spans) { free(all_spans); }
    return res;
}
