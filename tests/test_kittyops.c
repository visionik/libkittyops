#include "kittyops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static int failures = 0;

static void check(bool cond, const char *msg)
{
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        failures++;
    }
}

static void check_int(int64_t got, int64_t want, const char *msg)
{
    if (got != want) {
        fprintf(stderr, "FAIL: %s (got %lld want %lld)\n", msg, (long long)got, (long long)want);
        failures++;
    }
}

static void fill_pattern(uint8_t *fb, int32_t w, int32_t h, int32_t bpp, int32_t scroll_x, int32_t scroll_y)
{
    int32_t y, x;
    for (y = 0; y < h; ++y) {
        const uint32_t sy = (uint32_t)(y - scroll_y);
        const uint32_t by = sy / 4u;
        for (x = 0; x < w; ++x) {
            const uint32_t sx = (uint32_t)(x - scroll_x);
            const uint32_t bx = sx / 4u;
            uint8_t *p = fb + ((size_t)y * (size_t)w + (size_t)x) * (size_t)bpp;
            p[0] = (uint8_t)((bx * 7u + sy * 13u + ((by / 5u) * 53u)) & 0xFFu);
            p[1] = (uint8_t)((bx * 11u ^ (by * 29u)) & 0xFFu);
            p[2] = (uint8_t)(((sx / 9u) * 37u + by * 5u) & 0xFFu);
            if (bpp == 4) {
                p[3] = 255u;
            }
        }
    }
}

static void draw_box(uint8_t *fb, int32_t w, int32_t h, int32_t bpp,
                     int32_t bx, int32_t by, int32_t bw, int32_t bh,
                     uint8_t r, uint8_t g, uint8_t b)
{
    int32_t y, x;
    for (y = by; y < by + bh && y < h; ++y) {
        if (y < 0) { continue; }
        for (x = bx; x < bx + bw && x < w; ++x) {
            if (x < 0) { continue; }
            uint8_t *p = fb + ((size_t)y * (size_t)w + (size_t)x) * (size_t)bpp;
            p[0] = r; p[1] = g; p[2] = b;
            if (bpp == 4) { p[3] = 255u; }
        }
    }
}

static void test_rect_and_coalesce(void)
{
    kops_rect in[2] = { { 10, 10, 20, 20 }, { 31, 10, 20, 20 } };
    kops_rect out[4];
    int32_t n = kops_coalesce_rects(200, 200, in, 2, 2, 0, out, 4);
    check_int(n, 1, "near rects merge with gap=2");
    check_int(out[0].w, 41, "merged width");

    /* Bloat guard: 4:3 game box + overhanging HUD card touching within gap=2 */
    {
        kops_rect game_and_hud[2] = {
            { 240, 0, 1440, 1080 },
            { 1480, 12, 434, 148 }
        };
        /* Merging adds ~234*932 = ~218K clean pixels (> 65536), so bloat guard rejects merge */
        n = kops_coalesce_rects(1920, 1080, game_and_hud, 2, 2, 65536u, out, 4);
        check_int(n, 2, "bloat guard keeps game + HUD separate");
    }
}

static void test_tile_diff_and_xrgb(void)
{
    const int32_t w = 256, h = 128;
    uint8_t *a = (uint8_t *)calloc((size_t)w * (size_t)h * 3u, 1);
    uint8_t *b = (uint8_t *)calloc((size_t)w * (size_t)h * 3u, 1);
    kops_rect rects[16], bbox;
    int32_t count = 0;
    kops_diff_outcome res;

    res = kops_tile_diff_rects(a, b, w, h, 3, 32, 0.90f, rects, 16, &count, &bbox);
    check_int((int)res, (int)KOPS_DIFF_NONE, "identical RGB -> NONE");

    draw_box(b, w, h, 3, 10, 12, 8, 6, 200, 50, 10);
    draw_box(b, w, h, 3, 200, 90, 12, 10, 10, 220, 80);
    res = kops_tile_diff_rects(a, b, w, h, 3, 32, 0.90f, rects, 16, &count, &bbox);
    check_int((int)res, (int)KOPS_DIFF_RECT, "two boxes -> RECT");
    check_int(count, 2, "two boxes -> 2 rects");
    check_int(rects[0].x, 10, "box0 trimmed x");
    check_int(rects[0].y, 12, "box0 trimmed y");
    check_int(rects[0].w, 8, "box0 trimmed w");
    check_int(rects[0].h, 6, "box0 trimmed h");

    /* Fused XRGB8888 -> RGB24 convert + diff */
    {
        uint8_t *xrgb = (uint8_t *)calloc((size_t)w * (size_t)h * 4u, 1);
        uint8_t *scratch = (uint8_t *)calloc((size_t)w * (size_t)h * 3u, 1);
        /* Set pixel (40, 20) to R=0xAA, G=0xBB, B=0xCC in little-endian XRGB8888 (B,G,R,X) */
        const size_t off = ((size_t)20 * (size_t)w + 40u) * 4u;
        xrgb[off + 0] = 0xCCu;
        xrgb[off + 1] = 0xBBu;
        xrgb[off + 2] = 0xAAu;
        xrgb[off + 3] = 0xFFu;
        res = kops_convert_xrgb_and_diff(xrgb, w * 4, scratch, w, h, false, 32, 0.90f,
                                         rects, 16, &count, &bbox);
        check_int((int)res, (int)KOPS_DIFF_RECT, "xrgb single pixel -> RECT");
        check_int(count, 1, "xrgb single pixel count");
        check_int(rects[0].x, 40, "xrgb exact trimmed x");
        check_int(rects[0].y, 20, "xrgb exact trimmed y");
        check_int(rects[0].w, 1, "xrgb exact trimmed w");
        check_int(rects[0].h, 1, "xrgb exact trimmed h");
        check_int(scratch[((size_t)20 * (size_t)w + 40u) * 3u + 0], 0xAA, "xrgb updated R");
        check_int(scratch[((size_t)20 * (size_t)w + 40u) * 3u + 1], 0xBB, "xrgb updated G");
        check_int(scratch[((size_t)20 * (size_t)w + 40u) * 3u + 2], 0xCC, "xrgb updated B");

        /* Second call must report NONE because scratch was updated in-place */
        res = kops_convert_xrgb_and_diff(xrgb, w * 4, scratch, w, h, false, 32, 0.90f,
                                         rects, 16, &count, &bbox);
        check_int((int)res, (int)KOPS_DIFF_NONE, "xrgb repeat -> NONE");

        /* ARGB8888 -> RGBA32 fused path */
        {
            uint8_t *rgba_scratch = (uint8_t *)calloc((size_t)w * (size_t)h * 4u, 1);
            res = kops_convert_xrgb_and_diff(xrgb, w * 4, rgba_scratch, w, h, true, 32, 0.90f,
                                             rects, 16, &count, &bbox);
            check_int((int)res, (int)KOPS_DIFF_RECT, "argb single pixel -> RECT");
            check_int(rects[0].x, 40, "argb exact trimmed x");
            check_int(rects[0].w, 1, "argb exact trimmed w");
            check_int(rgba_scratch[off + 0], 0xAA, "argb updated R");
            check_int(rgba_scratch[off + 1], 0xBB, "argb updated G");
            check_int(rgba_scratch[off + 2], 0xCC, "argb updated B");
            check_int(rgba_scratch[off + 3], 0xFF, "argb updated A");
            res = kops_convert_xrgb_and_diff(xrgb, w * 4, rgba_scratch, w, h, true, 32, 0.90f,
                                             rects, 16, &count, &bbox);
            check_int((int)res, (int)KOPS_DIFF_NONE, "argb repeat -> NONE");
            free(rgba_scratch);
        }
        free(xrgb);
        free(scratch);
    }

    free(a);
    free(b);
}

static void test_compress_roundtrip(void)
{
    kops_encoder *enc = kops_encoder_create(3);
    const int32_t w = 640, h = 480;
    const size_t full_len = (size_t)w * (size_t)h * 3u;
    uint8_t *fb = (uint8_t *)malloc(full_len);
    uint8_t *dec = (uint8_t *)malloc(full_len);
    const uint8_t *zptr = NULL;
    size_t zlen = 0;
    uLongf dlen;

    check(enc != NULL, "encoder create");
    fill_pattern(fb, w, h, 3, 0, 0);

    /* Large buffer (>= 224 KB) -> parallel multi-slice zlib */
    check(kops_compress_zlib(enc, fb, full_len, &zptr, &zlen), "parallel zlib compress");
    check(zlen > 0 && zlen < full_len, "parallel zlib compressed smaller");
    dlen = (uLongf)full_len;
    check_int(uncompress(dec, &dlen, zptr, (uLong)zlen), Z_OK, "parallel zlib inflate OK");
    check_int((int64_t)dlen, (int64_t)full_len, "parallel zlib inflate len");
    check(memcmp(fb, dec, full_len) == 0, "parallel zlib exact roundtrip");

    /* Strided sub-rect >= 224 KB -> zero-copy strided parallel zlib */
    {
        kops_rect r = { 80, 40, 400, 320 }; /* 384 KB */
        const size_t rlen = (size_t)r.w * (size_t)r.h * 3u;
        uint8_t *packed = (uint8_t *)malloc(rlen);
        size_t plen = 0;
        check(kops_pack_rect(fb, w, h, 3, &r, packed, rlen, &plen), "pack ref rect");
        check(kops_compress_rect_strided(enc, fb, w, h, 3, &r, &zptr, &zlen), "strided parallel zlib");
        dlen = (uLongf)rlen;
        check_int(uncompress(dec, &dlen, zptr, (uLong)zlen), Z_OK, "strided inflate OK");
        check(memcmp(packed, dec, rlen) == 0, "strided exact roundtrip");
        free(packed);
    }

    /* Small rect (< 224 KB) -> single-thread libdeflate */
    {
        kops_rect r = { 10, 10, 64, 64 }; /* 12 KB */
        const size_t rlen = (size_t)r.w * (size_t)r.h * 3u;
        uint8_t *packed = (uint8_t *)malloc(rlen);
        size_t plen = 0;
        check(kops_pack_rect(fb, w, h, 3, &r, packed, rlen, &plen), "pack small rect");
        check(kops_compress_rect_strided(enc, fb, w, h, 3, &r, &zptr, &zlen), "small rect compress");
        dlen = (uLongf)rlen;
        check_int(uncompress(dec, &dlen, zptr, (uLong)zlen), Z_OK, "small inflate OK");
        check(memcmp(packed, dec, rlen) == 0, "small exact roundtrip");
        free(packed);
    }

    free(fb);
    free(dec);
    kops_encoder_destroy(enc);
}

static void test_scroll_and_compose_pipeline(void)
{
    const int32_t w = 640, h = 480;
    const size_t len = (size_t)w * (size_t)h * 3u;
    uint8_t *f0 = (uint8_t *)malloc(len);
    uint8_t *f1 = (uint8_t *)malloc(len);
    uint8_t *f2 = (uint8_t *)malloc(len);
    int32_t dx = 0, dy = 0;
    float conf = 0.0f;
    kops_encoder *enc = kops_encoder_create(3);
    kops_caps caps;
    const char *apc = NULL;
    size_t apc_len = 0;
    kops_stats st;

    kops_caps_default(&caps);

    /* Vertical scroll down by +16 px + a fixed foreground sprite */
    fill_pattern(f0, w, h, 3, 0, 0);
    fill_pattern(f1, w, h, 3, 0, 16);
    draw_box(f0, w, h, 3, 300, 220, 32, 32, 255, 0, 0);
    draw_box(f1, w, h, 3, 300, 220, 32, 32, 255, 0, 0);

    check(kops_detect_scroll(f0, f1, w, h, 3, NULL, &dx, &dy, &conf), "detect vertical scroll");
    check_int(dx, 0, "vertical scroll dx == 0");
    check_int(dy, 16, "vertical scroll dy == 16");
    check(conf > 0.90f, "vertical scroll high confidence");

    /* Horizontal scroll left by -12 px (side-scroller) */
    fill_pattern(f2, w, h, 3, -12, 0);
    check(kops_detect_scroll(f0, f2, w, h, 3, NULL, &dx, &dy, &conf), "detect horizontal scroll");
    check_int(dx, -12, "horizontal scroll dx == -12");
    check_int(dy, 0, "horizontal scroll dy == 0");

    /* Full stateful pipeline: f0 full -> f0 skip -> f1 scroll compose (1->2) -> f1->f2 scroll compose (2->1) */
    check(kops_encode_frame(enc, 42, NULL, f0, w, h, 3, &caps, &apc, &apc_len, &st), "encode full");
    check_int((int)st.mode, (int)KOPS_MODE_FULL_DIRECT, "mode FULL_DIRECT");
    check(strstr(apc, "\x1b_Ga=T,") != NULL, "contains a=T");

    check(kops_encode_frame(enc, 42, f0, f0, w, h, 3, &caps, &apc, &apc_len, &st), "encode skip");
    check_int((int)st.mode, (int)KOPS_MODE_SKIP, "mode SKIP");
    check_int((int64_t)apc_len, 0, "skip outputs 0 bytes");

    check(kops_encode_frame(enc, 42, f0, f1, w, h, 3, &caps, &apc, &apc_len, &st), "encode scroll 1->2");
    check_int((int)st.mode, (int)KOPS_MODE_SCROLL_COMPOSE, "mode SCROLL_COMPOSE 1->2");
    check_int(st.scroll_dy, 16, "stats scroll_dy == 16");
    check(strstr(apc, "\x1b_Ga=c,i=42,r=1,c=2,w=640,h=464,x=0,y=16,X=0,Y=0,C=1,q=2\x1b\\") != NULL,
          "a=c r=1->c=2 exact keys");
    check(strstr(apc, "\x1b_Ga=a,i=42,c=2,s=1,q=2\x1b\\") != NULL, "a=a selects frame 2");
    check(st.raw_bytes < len / 4u, "scroll compose raw_bytes << full frame");

    /* Next scroll step: f1 -> f1_next (+16 more) must ping-pong r=2 -> c=1 */
    fill_pattern(f2, w, h, 3, 0, 32);
    check(kops_encode_frame(enc, 42, f1, f2, w, h, 3, &caps, &apc, &apc_len, &st), "encode scroll 2->1");
    check_int((int)st.mode, (int)KOPS_MODE_SCROLL_COMPOSE, "mode SCROLL_COMPOSE 2->1");
    check(strstr(apc, "\x1b_Ga=c,i=42,r=2,c=1,w=640,h=464,x=0,y=16,X=0,Y=0,C=1,q=2\x1b\\") != NULL,
          "a=c r=2->c=1 ping-pong exact keys");
    check(strstr(apc, "\x1b_Ga=a,i=42,c=1,s=1,q=2\x1b\\") != NULL, "a=a selects frame 1");

    /* Storage-cap guard: Rex 10 MB cap disables 2-frame a=c when 2*w*h*4 > storage_bytes */
    caps.storage_bytes = 1000000u; /* 1 MB < 2 * 640*480*4 = 2.45 MB */
    kops_encoder_reset(enc);
    check(kops_encode_frame(enc, 43, f0, f1, w, h, 3, &caps, &apc, &apc_len, &st), "storage-capped scroll");
    check(st.mode != KOPS_MODE_SCROLL_COMPOSE, "storage cap prevents OOM a=c");

    /* Adaptive backoff: 4 non-scrolling large frames skip probing on frame 5 (scroll_ns == 0),
     * then re-engage automatically on the 30th frame when scrolling resumes. */
    {
        uint8_t *solid_a = (uint8_t *)malloc(len);
        uint8_t *solid_b = (uint8_t *)malloc(len);
        int k;
        memset(solid_a, 0x20, len);
        memset(solid_b, 0x90, len);
        kops_caps_default(&caps);
        kops_encoder_reset(enc);
        for (k = 0; k < 4; ++k) {
            check(kops_encode_frame(enc, 44, solid_a, solid_b, w, h, 3, &caps, &apc, &apc_len, &st),
                  "backoff warmup miss");
            check(st.scroll_ns > 0u, "first 4 large frames probe scroll");
        }
        /* Frame 5 (miss_streak == 4): skipped completely */
        check(kops_encode_frame(enc, 44, solid_a, solid_b, w, h, 3, &caps, &apc, &apc_len, &st),
              "backoff skip frame 5");
        check_int((int64_t)st.scroll_ns, 0, "backoff skips scroll probe after 4 misses");
        /* Advance to miss_streak == 30 -> periodic re-probe catches a real scroll */
        for (k = 5; k < 30; ++k) {
            check(kops_encode_frame(enc, 44, solid_a, solid_b, w, h, 3, &caps, &apc, &apc_len, &st),
                  "backoff idle frame");
        }
        check(kops_encode_frame(enc, 44, f0, f1, w, h, 3, &caps, &apc, &apc_len, &st),
              "backoff re-probe catches real scroll");
        check_int((int)st.mode, (int)KOPS_MODE_SCROLL_COMPOSE, "re-probe engages a=c scroll compose");
        free(solid_a);
        free(solid_b);
    }

    free(f0);
    free(f1);
    free(f2);
    kops_encoder_destroy(enc);
}

int main(void)
{
    printf("libkittyops tests (backend: %s)\n", kops_compression_backend());
    test_rect_and_coalesce();
    test_tile_diff_and_xrgb();
    test_compress_roundtrip();
    test_scroll_and_compose_pipeline();
    if (failures > 0) {
        fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    printf("test_kittyops: ALL PASSED\n");
    return 0;
}
