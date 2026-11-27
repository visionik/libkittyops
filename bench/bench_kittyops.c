#include "kittyops.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_scene(uint8_t *fb, int32_t w, int32_t h, int32_t sx, int32_t sy, bool pillarbox)
{
    const int32_t x0 = pillarbox ? (w - 810) / 2 : 0;
    const int32_t x1 = pillarbox ? x0 + 810 : w;
    int32_t y, x;
    memset(fb, 0, (size_t)w * (size_t)h * 3u);
    for (y = 0; y < h; ++y) {
        const uint32_t py = (uint32_t)(y - sy);
        const uint32_t row_band = py / 24u;
        for (x = x0; x < x1; ++x) {
            const uint32_t px = (uint32_t)(x - sx);
            uint8_t *p = fb + ((size_t)y * (size_t)w + (size_t)x) * 3u;
            /* Pixel-art / web-page-like structured tiles (compressible like real games/browsers) */
            const uint32_t tile = ((px / 16u) * 31u + row_band * 17u) & 255u;
            const uint32_t edge = ((px % 16u) == 0 || (py % 24u) == 0) ? 40u : 0u;
            p[0] = (uint8_t)((tile + edge) & 255u);
            p[1] = (uint8_t)(((tile * 3u) ^ (row_band * 19u)) & 255u);
            p[2] = (uint8_t)((64u + ((px / 32u) * 23u) + edge) & 255u);
        }
    }
}

static void draw_sprite(uint8_t *fb, int32_t w, int32_t h,
                        int32_t bx, int32_t by, int32_t bw, int32_t bh,
                        uint8_t r, uint8_t g, uint8_t b)
{
    int32_t y, x;
    for (y = by; y < by + bh && y < h; ++y) {
        for (x = bx; x < bx + bw && x < w; ++x) {
            uint8_t *p = fb + ((size_t)y * (size_t)w + (size_t)x) * 3u;
            p[0] = r; p[1] = g; p[2] = b;
        }
    }
}

static void run_case(const char *label, const uint8_t *prev, const uint8_t *curr,
                     int32_t w, int32_t h, const kops_caps *caps, int iters)
{
    kops_encoder *enc = kops_encoder_create(3);
    const char *apc = NULL;
    size_t apc_len = 0;
    kops_stats st;
    double tot_ms = 0.0, diff_ms = 0.0, scroll_ms = 0.0, comp_ms = 0.0, enc_ms = 0.0;
    size_t raw_kb = 0, wire_kb = 0;
    int32_t rects = 0, sdx = 0, sdy = 0;
    kops_present_mode mode = KOPS_MODE_SKIP;
    int i;

    /* Warmup */
    kops_encode_frame(enc, 1, prev, curr, w, h, 3, caps, &apc, &apc_len, &st);

    for (i = 0; i < iters; ++i) {
        kops_encode_frame(enc, 1, prev, curr, w, h, 3, caps, &apc, &apc_len, &st);
        diff_ms += (double)st.diff_ns / 1e6;
        scroll_ms += (double)st.scroll_ns / 1e6;
        comp_ms += (double)st.compress_ns / 1e6;
        enc_ms += (double)st.encode_ns / 1e6;
        tot_ms += (double)(st.diff_ns + st.scroll_ns + st.compress_ns + st.encode_ns) / 1e6;
        raw_kb = st.raw_bytes / 1024u;
        wire_kb = st.wire_bytes / 1024u;
        rects = st.rect_count;
        sdx = st.scroll_dx;
        sdy = st.scroll_dy;
        mode = st.mode;
    }
    printf("%-38s | mode=%d | total=%5.2f ms (diff=%.2f scr=%.2f comp=%.2f b64=%.2f) | raw=%4zu KB | wire=%4zu KB | rects=%2d | d=(%d,%d)\n",
           label, (int)mode, tot_ms / iters, diff_ms / iters, scroll_ms / iters,
           comp_ms / iters, enc_ms / iters, raw_kb, wire_kb, rects, (int)sdx, (int)sdy);
    kops_encoder_destroy(enc);
}

int main(void)
{
    const int32_t w = 1920, h = 1080;
    const size_t sz = (size_t)w * (size_t)h * 3u;
    uint8_t *a = (uint8_t *)malloc(sz);
    uint8_t *b = (uint8_t *)malloc(sz);
    kops_caps caps_no_scroll, caps_scroll;

    kops_caps_default(&caps_no_scroll);
    caps_no_scroll.allow_scroll_compose = false;
    caps_scroll = caps_no_scroll;
    caps_scroll.allow_scroll_compose = true;

    printf("libkittyops 1920x1080 benchmark (backend: %s)\n", kops_compression_backend());
    printf("---------------------------------------------------------------------------------------------------------------------\n");

    /* 1. Static frame */
    fill_scene(a, w, h, 0, 0, true);
    memcpy(b, a, sz);
    run_case("1080p static identical", a, b, w, h, &caps_no_scroll, 50);

    /* 2. Vertical 3:4 arcade shmup (1942/Raiden-like: dy=+4 px + player ship + 3 bullets) */
    fill_scene(a, w, h, 0, 0, true);
    fill_scene(b, w, h, 0, 4, true);
    draw_sprite(a, w, h, 930, 860, 64, 64, 240, 220, 40);
    draw_sprite(b, w, h, 934, 860, 64, 64, 240, 220, 40);
    draw_sprite(b, w, h, 956, 700, 12, 28, 255, 80, 40);
    run_case("3:4 shmup scroll dy=+4 (standard a=f)", a, b, w, h, &caps_no_scroll, 30);
    run_case("3:4 shmup scroll dy=+4 (a=c compose)", a, b, w, h, &caps_scroll, 30);

    /* 3. Full 1080p browser page scroll (dy=-36 px + sticky 64px top navbar) */
    fill_scene(a, w, h, 0, 0, false);
    fill_scene(b, w, h, 0, -36, false);
    draw_sprite(a, w, h, 0, 0, w, 64, 28, 32, 42);
    draw_sprite(b, w, h, 0, 0, w, 64, 28, 32, 42);
    run_case("1080p browser dy=-36 (standard a=f)", a, b, w, h, &caps_no_scroll, 30);
    run_case("1080p browser dy=-36 (a=c compose)", a, b, w, h, &caps_scroll, 30);

    free(a);
    free(b);
    return 0;
}
