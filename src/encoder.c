#include "kops_internal.h"

#include <stdio.h>

void kops_caps_default(kops_caps *caps)
{
    if (!caps) {
        return;
    }
    memset(caps, 0, sizeof(*caps));
    caps->max_escape_bytes = 262144u;
    caps->storage_bytes = 320000000u;
    caps->frame_edit = true;
    caps->frame_edit_zlib = true;
    caps->compose = true;
    caps->shm = false;
    caps->placeholders = false;
    caps->allow_scroll_compose = true;
    caps->af_max_coverage = 0.90f;
    caps->af_max_pixels = (size_t)1920 * 1080u;
    caps->tile_size = 32;
    caps->coalesce_gap = 2;
    caps->coalesce_max_bloat = 65536u;
    caps->zlib_level = 3;
}

kops_encoder *kops_encoder_create(int32_t zlib_level)
{
    kops_encoder *enc = (kops_encoder *)calloc(1, sizeof(kops_encoder));
    int ncpu;
    if (!enc) {
        return NULL;
    }
    enc->zlib_level = (zlib_level >= 1 && zlib_level <= 9) ? zlib_level : 3;
    ncpu = kops_cpu_count();
    enc->max_slices = ncpu >= KOPS_PZ_MAX_SLICES ? KOPS_PZ_MAX_SLICES : (ncpu >= 2 ? ncpu : 1);
    enc->active_frame = 1u;
    return enc;
}

void kops_encoder_reset(kops_encoder *enc)
{
    if (!enc) {
        return;
    }
    enc->tracked_image_id = 0u;
    enc->tracked_w = 0;
    enc->tracked_h = 0;
    enc->active_frame = 1u;
    enc->frame2_created = false;
    enc->scroll_miss_streak = 0;
}

void kops_encoder_destroy(kops_encoder *enc)
{
    int i;
    if (!enc) {
        return;
    }
    for (i = 0; i < KOPS_PZ_MAX_SLICES; ++i) {
        if (enc->slices[i].stream_init) {
            kops_deflateEnd(&enc->slices[i].stream);
        }
        free(enc->slices[i].pack_buf);
        free(enc->slices[i].out_buf);
    }
    if (enc->zstream_init) {
        kops_deflateEnd(&enc->zstream);
    }
#ifdef KOPS_HAVE_LIBDEFLATE
    if (enc->ld) {
        libdeflate_free_compressor(enc->ld);
    }
#endif
    for (i = 0; i < KOPS_MAX_RECTS; ++i) {
        free(enc->mr[i].pack_buf);
        free(enc->mr[i].z_buf);
        free(enc->mr[i].apc_buf);
#ifdef KOPS_HAVE_LIBDEFLATE
        if (enc->mr[i].ld) {
            libdeflate_free_compressor(enc->mr[i].ld);
        }
#endif
        if (enc->mr[i].zs_init) {
            kops_deflateEnd(&enc->mr[i].zs);
        }
    }
    free(enc->z_buf);
    free(enc->pack_buf);
    free(enc->out_buf);
    free(enc);
}

uint8_t *kops_ensure_z_buf(kops_encoder *enc, size_t need)
{
    uint8_t *p;
    if (enc->z_buf && enc->z_cap >= need) {
        return enc->z_buf;
    }
    p = (uint8_t *)realloc(enc->z_buf, need);
    if (!p) { return NULL; }
    enc->z_buf = p;
    enc->z_cap = need;
    return p;
}

uint8_t *kops_ensure_pack_buf(kops_encoder *enc, size_t need)
{
    uint8_t *p;
    if (enc->pack_buf && enc->pack_cap >= need) {
        return enc->pack_buf;
    }
    p = (uint8_t *)realloc(enc->pack_buf, need);
    if (!p) { return NULL; }
    enc->pack_buf = p;
    enc->pack_cap = need;
    return p;
}

char *kops_ensure_out_buf(kops_encoder *enc, size_t need)
{
    char *p;
    size_t cap;
    if (enc->out_buf && enc->out_cap >= need) {
        return enc->out_buf;
    }
    cap = enc->out_cap ? enc->out_cap : 65536u;
    while (cap < need) {
        cap *= 2u;
    }
    p = (char *)realloc(enc->out_buf, cap);
    if (!p) { return NULL; }
    enc->out_buf = p;
    enc->out_cap = cap;
    return p;
}

bool kops_out_append(kops_encoder *enc, const void *data, size_t len)
{
    char *dst;
    if (len == 0) {
        return true;
    }
    dst = kops_ensure_out_buf(enc, enc->out_len + len + 1u);
    if (!dst) {
        return false;
    }
    memcpy(dst + enc->out_len, data, len);
    enc->out_len += len;
    dst[enc->out_len] = '\0';
    return true;
}

static bool kops_encode_full_frame(kops_encoder *enc,
                                   uint32_t image_id,
                                   const uint8_t *curr_fb,
                                   int32_t fb_w, int32_t fb_h, int32_t bpp,
                                   const kops_caps *caps,
                                   kops_stats *st)
{
    const size_t raw_len = (size_t)fb_w * (size_t)fb_h * (size_t)bpp;
    const int kitty_f = (bpp == 4) ? 32 : 24;
    const size_t chunk = caps->max_escape_bytes >= 64u ? caps->max_escape_bytes : 262144u;

    /* Reset ping-pong frame state since full transmit recreates image from frame 1. */
    enc->tracked_image_id = image_id;
    enc->tracked_w = fb_w;
    enc->tracked_h = fb_h;
    enc->active_frame = 1u;
    enc->frame2_created = false;

    st->raw_bytes = raw_len;
    st->rect_count = 1;

    if (caps->shm) {
        char shm_apc[256];
        const uint64_t e0 = kops_now_ns();
        const size_t n = kops_emit_shm_transmit(shm_apc, sizeof(shm_apc), image_id,
                                                curr_fb, fb_w, fb_h, bpp,
                                                !caps->placeholders, 0, 0);
        if (n > 0) {
            if (!kops_out_append(enc, shm_apc, n)) { return false; }
            st->encode_ns += (kops_now_ns() - e0);
            st->compressed_bytes = raw_len;
            st->wire_bytes = enc->out_len;
            st->mode = KOPS_MODE_FULL_SHM;
            return true;
        }
    }

    {
        const uint8_t *zptr = NULL;
        size_t zlen = 0;
        char first_hdr[160];
        const uint64_t c0 = kops_now_ns();
        uint64_t e0;
        if (!kops_compress_zlib(enc, curr_fb, raw_len, &zptr, &zlen)) {
            return false;
        }
        st->compress_ns += (kops_now_ns() - c0);
        st->compressed_bytes = zlen;

        if (caps->placeholders) {
            snprintf(first_hdr, sizeof(first_hdr),
                     "\x1b_Ga=t,f=%d,t=d,o=z,i=%u,s=%d,v=%d,q=2",
                     kitty_f, (unsigned)image_id, (int)fb_w, (int)fb_h);
        } else {
            snprintf(first_hdr, sizeof(first_hdr),
                     "\x1b_Ga=T,f=%d,t=d,o=z,i=%u,p=1,s=%d,v=%d,C=1,q=2",
                     kitty_f, (unsigned)image_id, (int)fb_w, (int)fb_h);
        }
        e0 = kops_now_ns();
        if (!kops_append_apc_chunked(enc, first_hdr, "\x1b_Gq=2", zptr, zlen, chunk)) {
            return false;
        }
        st->encode_ns += (kops_now_ns() - e0);
        st->wire_bytes = enc->out_len;
        st->mode = KOPS_MODE_FULL_DIRECT;
        return true;
    }
}

bool kops_encode_frame(kops_encoder *enc,
                       uint32_t image_id,
                       const uint8_t *prev_fb,
                       const uint8_t *curr_fb,
                       int32_t fb_w, int32_t fb_h, int32_t bpp,
                       const kops_caps *caps_in,
                       const char **out_apc,
                       size_t *out_len,
                       kops_stats *out_stats)
{
    kops_caps caps;
    kops_stats st;
    kops_rect rects[KOPS_MAX_RECTS];
    kops_rect coal[KOPS_MAX_RECTS];
    kops_rect bbox;
    int32_t n_rects = 0;
    int32_t tile;
    size_t chunk;
    kops_diff_outcome diff;
    uint64_t t0;

    if (!enc || !curr_fb || image_id == 0 || fb_w <= 0 || fb_h <= 0 || (bpp != 3 && bpp != 4)) {
        return false;
    }
    if (caps_in) {
        caps = *caps_in;
    } else {
        kops_caps_default(&caps);
    }
    memset(&st, 0, sizeof(st));
    enc->out_len = 0;
    tile = (caps.tile_size >= 8 && caps.tile_size <= 128) ? caps.tile_size : 32;
    chunk = caps.max_escape_bytes >= 64u ? caps.max_escape_bytes : 262144u;

    /* Reset tracked image state if image_id or geometry changed */
    if (enc->tracked_image_id != image_id || enc->tracked_w != fb_w || enc->tracked_h != fb_h) {
        enc->tracked_image_id = image_id;
        enc->tracked_w = fb_w;
        enc->tracked_h = fb_h;
        enc->active_frame = 1u;
        enc->frame2_created = false;
        enc->scroll_miss_streak = 0;
    }

    /* Full frame required if no previous frame or terminal doesn't support a=f */
    if (!prev_fb || !caps.frame_edit) {
        if (!kops_encode_full_frame(enc, image_id, curr_fb, fb_w, fb_h, bpp, &caps, &st)) {
            return false;
        }
        goto done;
    }

    /* 1. Tile diff */
    t0 = kops_now_ns();
    diff = kops_tile_diff_rects(prev_fb, curr_fb, fb_w, fb_h, bpp, tile,
                                caps.af_max_coverage > 0.0f ? caps.af_max_coverage : 0.90f,
                                rects, KOPS_MAX_RECTS, &n_rects, &bbox);
    st.diff_ns = kops_now_ns() - t0;

    if (diff == KOPS_DIFF_NONE) {
        st.mode = KOPS_MODE_SKIP;
        goto done;
    }

    /* 2. Scroll detection + a=c ping-pong (only when dirty area is large enough
     *    that scroll-shift saves major bytes AND terminal RAM fits 2 RGBA frames). */
    {
        const size_t two_frame_rgba = (size_t)fb_w * (size_t)fb_h * 4u * 2u;
        const float raw_cov = kops_coverage_ratio(fb_w, fb_h, &bbox);
        const bool can_compose =
            caps.compose && caps.allow_scroll_compose &&
            (caps.storage_bytes == 0 || caps.storage_bytes >= two_frame_rgba) &&
            raw_cov >= 0.25f;

        if (can_compose) {
            /* Adaptive backoff: after 4 consecutive misses (e.g. webcam/video/3D),
             * probe only every 30th large frame so non-scrolling streams pay ~0 overhead. */
            const bool probe_now = (enc->scroll_miss_streak < 4) ||
                                   ((enc->scroll_miss_streak % 30) == 0);
            int32_t sdx = 0, sdy = 0;
            float sconf = 0.0f;
            bool scrolled = false;
            if (probe_now) {
                const uint64_t sc0 = kops_now_ns();
                scrolled = kops_detect_scroll(prev_fb, curr_fb, fb_w, fb_h, bpp,
                                              &bbox, &sdx, &sdy, &sconf);
                st.scroll_ns = kops_now_ns() - sc0;
            }

            if (scrolled) {
                kops_rect srects[KOPS_MAX_RECTS];
                kops_rect sbbox;
                int32_t sn = 0;
                const uint64_t sd0 = kops_now_ns();
                const kops_diff_outcome sdiff = kops_diff_shifted_rects(
                    prev_fb, curr_fb, fb_w, fb_h, bpp, sdx, sdy, tile, 0.90f,
                    srects, KOPS_MAX_RECTS, &sn, &sbbox);
                st.scroll_ns += (kops_now_ns() - sd0);

                if (sdiff == KOPS_DIFF_RECT && sn > 0) {
                    size_t shifted_px = 0, unshifted_px = 0;
                    int32_t i;
                    for (i = 0; i < sn; ++i) {
                        shifted_px += (size_t)srects[i].w * (size_t)srects[i].h;
                    }
                    for (i = 0; i < n_rects; ++i) {
                        unshifted_px += (size_t)rects[i].w * (size_t)rects[i].h;
                    }
                    /* Use a=c scroll ping-pong only when it cuts dirty pixels by >= 40%. */
                    if (shifted_px * 5u < unshifted_px * 3u) {
                        const uint32_t src_f = enc->active_frame;
                        const uint32_t dst_f = (src_f == 1u) ? 2u : 1u;
                        const int32_t copy_w = fb_w - (sdx >= 0 ? sdx : -sdx);
                        const int32_t copy_h = fb_h - (sdy >= 0 ? sdy : -sdy);
                        const int32_t src_x = sdx < 0 ? -sdx : 0;
                        const int32_t src_y = sdy < 0 ? -sdy : 0;
                        const int32_t dst_x = sdx > 0 ? sdx : 0;
                        const int32_t dst_y = sdy > 0 ? sdy : 0;
                        char cmd[192];
                        size_t clen;
                        const uint64_t e0 = kops_now_ns();

                        if (!enc->frame2_created) {
                            /* Lazily create frame 2 (1x1 seed; a=c + a=f fill it completely). */
                            int fn = snprintf(cmd, sizeof(cmd),
                                              "\x1b_Ga=f,i=%u,f=24,s=1,v=1,t=d,q=2;AAAA\x1b\\",
                                              (unsigned)image_id);
                            if (fn <= 0 || !kops_out_append(enc, cmd, (size_t)fn)) {
                                return false;
                            }
                            enc->frame2_created = true;
                        }

                        clen = kops_format_compose_apc(cmd, sizeof(cmd), image_id,
                                                       src_f, dst_f,
                                                       src_x, src_y, dst_x, dst_y,
                                                       copy_w, copy_h);
                        if (clen == 0 || !kops_out_append(enc, cmd, clen)) {
                            return false;
                        }
                        st.encode_ns += (kops_now_ns() - e0);

                        /* Coalesce shifted dirty rects with bloat guard */
                        {
                            int32_t cn = sn;
                            const kops_rect *use_r = srects;
                            if (sn > 1 && caps.coalesce_gap > 0) {
                                cn = kops_coalesce_rects(fb_w, fb_h, srects, sn,
                                                         caps.coalesce_gap, caps.coalesce_max_bloat,
                                                         coal, KOPS_MAX_RECTS);
                                if (cn > 0) { use_r = coal; }
                                else { cn = sn; }
                            }
                            if (!kops_append_multi_rect_edit(enc, image_id, dst_f,
                                                             curr_fb, fb_w, fb_h, bpp,
                                                             use_r, cn, caps.frame_edit_zlib, chunk,
                                                             &st.raw_bytes, &st.compressed_bytes,
                                                             &st.compress_ns, &st.encode_ns)) {
                                return false;
                            }
                            st.rect_count = cn;
                        }

                        /* Flip displayed frame to dst_f */
                        {
                            const uint64_t f0 = kops_now_ns();
                            clen = kops_format_animate_select_apc(cmd, sizeof(cmd), image_id, dst_f);
                            if (clen == 0 || !kops_out_append(enc, cmd, clen)) {
                                return false;
                            }
                            st.encode_ns += (kops_now_ns() - f0);
                        }
                        enc->active_frame = dst_f;
                        enc->scroll_miss_streak = 0;
                        st.scroll_dx = sdx;
                        st.scroll_dy = sdy;
                        st.wire_bytes = enc->out_len;
                        st.mode = KOPS_MODE_SCROLL_COMPOSE;
                        goto done;
                    }
                }
            }
            enc->scroll_miss_streak++;
        }
    }

    /* 3. Standard a=f (single or multi-rect) vs full-frame fallback */
    {
        const kops_rect *use_r = rects;
        int32_t use_n = n_rects;
        size_t total_px = 0;
        int32_t i;
        float cov;

        if (diff == KOPS_DIFF_RECT && n_rects > 1 && caps.coalesce_gap > 0) {
            const int32_t cn = kops_coalesce_rects(fb_w, fb_h, rects, n_rects,
                                                   caps.coalesce_gap, caps.coalesce_max_bloat,
                                                   coal, KOPS_MAX_RECTS);
            if (cn > 0) {
                use_r = coal;
                use_n = cn;
            }
        } else if (diff == KOPS_DIFF_FULL) {
            use_r = &bbox;
            use_n = 1;
        }

        for (i = 0; i < use_n; ++i) {
            total_px += (size_t)use_r[i].w * (size_t)use_r[i].h;
        }
        cov = (float)((double)total_px / ((double)fb_w * (double)fb_h));

        if (cov >= caps.af_max_coverage || (caps.af_max_pixels > 0 && total_px > caps.af_max_pixels)) {
            if (!kops_encode_full_frame(enc, image_id, curr_fb, fb_w, fb_h, bpp, &caps, &st)) {
                return false;
            }
            goto done;
        }

        if (!kops_append_multi_rect_edit(enc, image_id, enc->active_frame,
                                         curr_fb, fb_w, fb_h, bpp,
                                         use_r, use_n, caps.frame_edit_zlib, chunk,
                                         &st.raw_bytes, &st.compressed_bytes,
                                         &st.compress_ns, &st.encode_ns)) {
            return false;
        }
        st.rect_count = use_n;
        st.wire_bytes = enc->out_len;
        st.mode = (use_n > 1) ? KOPS_MODE_AF_MULTI : KOPS_MODE_AF_SINGLE;
    }

done:
    if (out_apc) { *out_apc = enc->out_buf ? enc->out_buf : ""; }
    if (out_len) { *out_len = enc->out_len; }
    if (out_stats) { *out_stats = st; }
    return true;
}
