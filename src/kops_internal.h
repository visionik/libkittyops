#ifndef KOPS_INTERNAL_H
#define KOPS_INTERNAL_H

#include "kittyops.h"
#include <stdlib.h>
#include <string.h>

#ifdef KOPS_HAVE_ZLIB_NG
#include <zlib-ng.h>
#define kops_z_stream zng_stream
#define kops_deflateInit2 zng_deflateInit2
#define kops_deflateReset zng_deflateReset
#define kops_deflate zng_deflate
#define kops_deflateEnd zng_deflateEnd
#define kops_deflateBound zng_deflateBound
#define kops_adler32 zng_adler32
#define kops_adler32_combine zng_adler32_combine
#else
#include <zlib.h>
#define kops_z_stream z_stream
#define kops_deflateInit2 deflateInit2
#define kops_deflateReset deflateReset
#define kops_deflate deflate
#define kops_deflateEnd deflateEnd
#define kops_deflateBound deflateBound
#define kops_adler32 adler32
#define kops_adler32_combine adler32_combine
#endif

#ifdef KOPS_HAVE_LIBDEFLATE
#include <libdeflate.h>
#endif

#ifdef KOPS_HAVE_ISAL
#include <isa-l/igzip_lib.h>
#endif

#define KOPS_PZ_MAX_SLICES 24
#define KOPS_PZ_MIN_BYTES ((size_t)224 * 1024u)
#define KOPS_MR_LARGE_RECT_BYTES ((size_t)256 * 1024u)

typedef void (*kops_parallel_fn)(void *ctx, size_t idx);
void kops_parallel_for(size_t count, kops_parallel_fn fn, void *ctx);
int kops_cpu_count(void);
uint64_t kops_now_ns(void);

typedef struct kops_pz_slice {
    kops_z_stream stream;
    bool stream_init;
    uint8_t *pack_buf;
    size_t pack_cap;
    uint8_t *out_buf;
    size_t out_cap;
    size_t out_len;
    uint32_t adler;
    size_t in_len;
    bool ok;
} kops_pz_slice;

typedef struct kops_mr_slot {
    kops_rect rect;
    uint8_t *pack_buf;
    size_t pack_cap;
    uint8_t *z_buf;
    size_t z_cap;
    char *apc_buf;
    size_t apc_cap;
    size_t raw_len;
    size_t comp_len;
    size_t wire_len;
    bool ok;
#ifdef KOPS_HAVE_LIBDEFLATE
    struct libdeflate_compressor *ld;
    int ld_level;
#endif
    kops_z_stream zs;
    bool zs_init;
} kops_mr_slot;

struct kops_encoder {
    int32_t zlib_level;
    int max_slices;

    /* Parallel zlib slices */
    kops_pz_slice slices[KOPS_PZ_MAX_SLICES];
    bool pz_ready;

    /* Single-thread zlib fallback */
    kops_z_stream zstream;
    bool zstream_init;

#ifdef KOPS_HAVE_LIBDEFLATE
    struct libdeflate_compressor *ld;
    int ld_level;
#endif

    /* Reusable scratch buffers */
    uint8_t *z_buf;
    size_t z_cap;
    uint8_t *pack_buf;
    size_t pack_cap;
    char *out_buf;
    size_t out_cap;
    size_t out_len;

    /* Parallel multi-rect worker slots */
    kops_mr_slot mr[KOPS_MAX_RECTS];

    /* Stateful a=c ping-pong tracking per image_id + dimensions */
    uint32_t tracked_image_id;
    int32_t tracked_w;
    int32_t tracked_h;
    uint32_t active_frame; /* 1 or 2 */
    bool frame2_created;
};

uint8_t *kops_ensure_z_buf(kops_encoder *enc, size_t need);
uint8_t *kops_ensure_pack_buf(kops_encoder *enc, size_t need);
char *kops_ensure_out_buf(kops_encoder *enc, size_t need);
bool kops_out_append(kops_encoder *enc, const void *data, size_t len);

/* Internal APC builders writing into enc->out_buf */
bool kops_append_apc_chunked(kops_encoder *enc,
                             const char *first_hdr,
                             const char *cont_hdr,
                             const uint8_t *body, size_t body_len,
                             size_t chunk_bytes);

bool kops_append_rect_edit(kops_encoder *enc,
                           uint32_t image_id, uint32_t frame_num,
                           const uint8_t *fb, int32_t fb_w, int32_t fb_h, int32_t bpp,
                           const kops_rect *rect, bool use_zlib, size_t chunk_bytes,
                           size_t *out_raw_bytes, size_t *out_comp_bytes,
                           uint64_t *out_comp_ns, uint64_t *out_enc_ns);

bool kops_append_multi_rect_edit(kops_encoder *enc,
                                 uint32_t image_id, uint32_t frame_num,
                                 const uint8_t *fb, int32_t fb_w, int32_t fb_h, int32_t bpp,
                                 const kops_rect *rects, int32_t n_rects,
                                 bool use_zlib, size_t chunk_bytes,
                                 size_t *out_raw_bytes, size_t *out_comp_bytes,
                                 uint64_t *out_comp_ns, uint64_t *out_enc_ns);

#endif /* KOPS_INTERNAL_H */
