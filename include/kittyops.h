/*
 * libkittyops — High-performance Kitty graphics pixel & protocol kernel.
 *
 * Pure C99, zero SDL / Python dependencies. Designed to be shared by
 * sdl-kitty (C/SDL3) and pixcell (Python/Textual via optional ctypes binding).
 *
 * MIT License — Copyright (c) 2026 Visionik <visionik@pobox.com>
 */
#ifndef KITTYOPS_H
#define KITTYOPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KOPS_VERSION_MAJOR 0
#define KOPS_VERSION_MINOR 1
#define KOPS_VERSION_PATCH 0

/* Maximum rects handled in a single parallel multi-rect batch. */
#define KOPS_MAX_RECTS 16

typedef struct kops_rect {
    int32_t x;
    int32_t y;
    int32_t w;
    int32_t h;
} kops_rect;

typedef enum kops_diff_outcome {
    KOPS_DIFF_NONE = 0, /* Buffers identical */
    KOPS_DIFF_RECT = 1, /* Partial dirty rects */
    KOPS_DIFF_FULL = 2  /* Coverage >= cover_limit; use bbox / full upload */
} kops_diff_outcome;

typedef enum kops_present_mode {
    KOPS_MODE_SKIP = 0,           /* Identical frame; 0 bytes written */
    KOPS_MODE_AF_SINGLE = 1,      /* Single a=f rect edit (parallel-zlib / libdeflate) */
    KOPS_MODE_AF_MULTI = 2,       /* Multiple a=f rect edits */
    KOPS_MODE_SCROLL_COMPOSE = 3, /* a=c ping-pong shift + edge/sprite a=f + a=a flip */
    KOPS_MODE_FULL_DIRECT = 4,    /* Full-frame a=T / a=t direct (o=z) */
    KOPS_MODE_FULL_SHM = 5        /* Full-frame a=T / a=t POSIX SHM (t=s) */
} kops_present_mode;

/*
 * Terminal capabilities & tuning policy (typically hydrated from terminal-facts).
 */
typedef struct kops_caps {
    size_t max_escape_bytes;   /* Max payload bytes per APC chunk (e.g. 262144 Kitty, 2 MiB Rex) */
    size_t storage_bytes;      /* Terminal image RAM budget (e.g. 10 MB Rex, 320 MB Kitty/Ghostty) */
    bool frame_edit;           /* Supports a=f in-place edits */
    bool frame_edit_zlib;      /* Supports o=z on a=f */
    bool compose;              /* Supports a=c frame compose + a=a frame flip */
    bool shm;                  /* Supports t=s POSIX shared memory */
    bool placeholders;         /* Uses U=1 virtual placements (a=t instead of a=T on full) */
    bool allow_scroll_compose; /* Enable scroll detection + a=c ping-pong when storage fits */
    float af_max_coverage;     /* Max dirty area ratio for a=f before falling back to full (0..1) */
    size_t af_max_pixels;      /* Max dirty pixel count for a=f */
    int32_t tile_size;         /* Tile diff granularity (default 32, clamped [8, 128]) */
    int32_t coalesce_gap;      /* Merge gap in pixels for multi-rect coalesce (default 2) */
    size_t coalesce_max_bloat; /* Reject coalesce if added clean pixels exceed this (default 65536) */
    int32_t zlib_level;        /* 1..9 (0 = default 3) */
} kops_caps;

/*
 * Per-frame stage telemetry returned by kops_encode_frame().
 */
typedef struct kops_stats {
    uint64_t diff_ns;
    uint64_t scroll_ns;
    uint64_t compress_ns;
    uint64_t encode_ns;
    size_t raw_bytes;
    size_t compressed_bytes;
    size_t wire_bytes;
    int32_t rect_count;
    int32_t scroll_dx;
    int32_t scroll_dy;
    kops_present_mode mode;
} kops_stats;

/* Opaque reusable encoder holding persistent compression streams & scratch buffers. */
typedef struct kops_encoder kops_encoder;

/* --------------------------------------------------------------------------
 * Initialization & defaults
 * -------------------------------------------------------------------------- */

void kops_caps_default(kops_caps *caps);
const char *kops_compression_backend(void);

kops_encoder *kops_encoder_create(int32_t zlib_level);
void kops_encoder_destroy(kops_encoder *enc);
/* Reset frame-tracking state (e.g. when switching image_id or resizing). */
void kops_encoder_reset(kops_encoder *enc);

/* --------------------------------------------------------------------------
 * Rect math, hashing & SIMD tile-diff
 * -------------------------------------------------------------------------- */

bool kops_union_rects(int32_t fb_w, int32_t fb_h,
                      const kops_rect *rects, int32_t n_rects,
                      kops_rect *out);

float kops_coverage_ratio(int32_t fb_w, int32_t fb_h, const kops_rect *r);

/*
 * Merge overlapping/near rects within gap_px. If max_bloat_px > 0 and the
 * coalesced total area exceeds raw area + max_bloat_px, copies the clipped
 * input rects unchanged (up to max_out) instead.
 */
int32_t kops_coalesce_rects(int32_t fb_w, int32_t fb_h,
                            const kops_rect *in, int32_t n_in,
                            int32_t gap_px, size_t max_bloat_px,
                            kops_rect *out, int32_t max_out);

/* Fast 4-lane 64-bit hash (for upload-skip fingerprinting & row hashing). */
uint64_t kops_fast_hash64(const uint8_t *data, size_t len);

/* Tightly pack rect from full-frame RGB24/RGBA32 buffer (bpp = 3 or 4). */
bool kops_pack_rect(const uint8_t *fb, int32_t fb_w, int32_t fb_h, int32_t bpp,
                    const kops_rect *rect, uint8_t *out, size_t out_cap, size_t *out_len);

/*
 * Parallel SIMD multi-rect tile diff between two tightly packed RGB/RGBA
 * frames (bpp = 3 or 4) with exact 2D pixel trimming.
 */
kops_diff_outcome kops_tile_diff_rects(const uint8_t *prev, const uint8_t *curr,
                                       int32_t fb_w, int32_t fb_h, int32_t bpp,
                                       int32_t tile, float cover_limit,
                                       kops_rect *rects_out, int32_t max_rects,
                                       int32_t *out_count, kops_rect *bbox_out);

/*
 * Fused single-pass XRGB8888/ARGB8888 -> RGB24/RGBA32 convert + in-place
 * multi-rect tile diff (32-pixel NEON clean-tile fast path + 6-worker 1080p).
 * Updates changed pixels in scratch_rgb in place; clean tiles are read-only.
 */
kops_diff_outcome kops_convert_xrgb_and_diff(const uint8_t *src_pixels, int32_t src_pitch,
                                             uint8_t *scratch_rgb,
                                             int32_t fb_w, int32_t fb_h, bool use_rgba,
                                             int32_t tile, float cover_limit,
                                             kops_rect *rects_out, int32_t max_rects,
                                             int32_t *out_count, kops_rect *bbox_out);

/* --------------------------------------------------------------------------
 * Scroll detection & shifted diff (for a=c ping-pong scrolling)
 * -------------------------------------------------------------------------- */

/*
 * Detect dominant integer scroll translation (dx, dy) between prev and curr
 * within roi (pass NULL for full frame). Positive dy means content moved DOWN
 * (curr[y] matches prev[y - dy]); positive dx means content moved RIGHT.
 * Returns true if a confident non-zero scroll was detected (confidence in [0,1]).
 */
bool kops_detect_scroll(const uint8_t *prev, const uint8_t *curr,
                        int32_t fb_w, int32_t fb_h, int32_t bpp,
                        const kops_rect *roi,
                        int32_t *out_dx, int32_t *out_dy,
                        float *out_confidence);

/*
 * Compute dirty rects in curr that differ from prev shifted by (dx, dy),
 * including the newly exposed border strip(s) and any foreground sprites that
 * did not move with the scroll.
 */
kops_diff_outcome kops_diff_shifted_rects(const uint8_t *prev, const uint8_t *curr,
                                          int32_t fb_w, int32_t fb_h, int32_t bpp,
                                          int32_t dx, int32_t dy,
                                          int32_t tile, float cover_limit,
                                          kops_rect *rects_out, int32_t max_rects,
                                          int32_t *out_count, kops_rect *bbox_out);

/* --------------------------------------------------------------------------
 * Compression (libdeflate + zero-copy strided parallel-zlib / zlib-ng / ISA-L)
 * -------------------------------------------------------------------------- */

/*
 * Compress a contiguous buffer into RFC 1950 zlib format.
 * Automatically picks libdeflate (< 224 KB) or multi-core parallel zlib (>= 224 KB).
 * *out_ptr points into enc's internal reusable buffer (valid until next call on enc).
 */
bool kops_compress_zlib(kops_encoder *enc, const uint8_t *src, size_t len,
                        const uint8_t **out_ptr, size_t *out_len);

/*
 * Zero-copy strided compress of a single rect directly from a full framebuffer
 * (each parallel worker packs its row slice L2-hot directly into deflate).
 */
bool kops_compress_rect_strided(kops_encoder *enc,
                                const uint8_t *fb, int32_t fb_w, int32_t fb_h, int32_t bpp,
                                const kops_rect *rect,
                                const uint8_t **out_ptr, size_t *out_len);

/* --------------------------------------------------------------------------
 * Base64 & Kitty APC escape-code framing
 * -------------------------------------------------------------------------- */

size_t kops_base64_encoded_len(size_t raw_len);
size_t kops_base64_encode(const uint8_t *src, size_t len, char *dst, size_t dst_cap);

/* Emit a=c (copy rect from src_frame to dst_frame, overwrite mode C=1). */
size_t kops_format_compose_apc(char *dst, size_t dst_cap,
                               uint32_t image_id, uint32_t src_frame, uint32_t dst_frame,
                               int32_t src_x, int32_t src_y,
                               int32_t dst_x, int32_t dst_y,
                               int32_t w, int32_t h);

/* Emit a=a (switch displayed frame to frame_num, paused s=1). */
size_t kops_format_animate_select_apc(char *dst, size_t dst_cap,
                                      uint32_t image_id, uint32_t frame_num);

/* Emit a=p virtual placement (U=1) for Unicode placeholders. */
size_t kops_format_place_virtual_apc(char *dst, size_t dst_cap,
                                     uint32_t image_id, uint32_t placement_id,
                                     int32_t cols, int32_t rows);

/* Emit a=d delete image + placements (d=I). */
size_t kops_format_delete_apc(char *dst, size_t dst_cap, uint32_t image_id);

/*
 * Write raw pixels into a new POSIX shared-memory object and emit a single
 * t=s Kitty APC (a=t or a=T). Terminal unlinks shm after reading.
 */
size_t kops_emit_shm_transmit(char *dst, size_t dst_cap,
                              uint32_t image_id,
                              const uint8_t *pixels, int32_t w, int32_t h, int32_t bpp,
                              bool place_now, int32_t place_cols, int32_t place_rows);

/* --------------------------------------------------------------------------
 * High-level single-call frame encoder (ideal for pixcell.LiveImage & TUIs)
 * -------------------------------------------------------------------------- */

/*
 * Encode the delta from prev_fb -> curr_fb (or full frame if prev_fb == NULL)
 * into a single ready-to-write Kitty APC byte buffer.
 *
 * Supports:
 *   - Identical frame skip (0 bytes)
 *   - Scroll detection + a=c ping-pong + minimal a=f edge/sprite patches + a=a
 *   - Multi-rect & single-rect a=f (strided parallel-zlib / parallel libdeflate)
 *   - Full-frame a=t / a=T (SHM t=s or parallel-zlib direct t=d)
 *
 * *out_apc points into enc's reusable output buffer (valid until next encode call).
 */
bool kops_encode_frame(kops_encoder *enc,
                       uint32_t image_id,
                       const uint8_t *prev_fb,
                       const uint8_t *curr_fb,
                       int32_t fb_w, int32_t fb_h, int32_t bpp,
                       const kops_caps *caps,
                       const char **out_apc,
                       size_t *out_len,
                       kops_stats *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* KITTYOPS_H */
