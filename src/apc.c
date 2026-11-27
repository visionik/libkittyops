#include "kops_internal.h"

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

static const char kops_b64_table[65] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

size_t kops_base64_encoded_len(size_t raw_len)
{
    return ((raw_len + 2u) / 3u) * 4u;
}

size_t kops_base64_encode(const uint8_t *src, size_t len, char *dst, size_t dst_cap)
{
    const size_t need = kops_base64_encoded_len(len);
    size_t i = 0, o = 0;
    if (!dst || dst_cap < need) {
        return 0;
    }
    if (!src || len == 0) {
        if (dst_cap > 0) { dst[0] = '\0'; }
        return 0;
    }
    while (i + 3u <= len) {
        const uint32_t v = ((uint32_t)src[i] << 16) | ((uint32_t)src[i + 1] << 8) | (uint32_t)src[i + 2];
        dst[o + 0] = kops_b64_table[(v >> 18) & 63u];
        dst[o + 1] = kops_b64_table[(v >> 12) & 63u];
        dst[o + 2] = kops_b64_table[(v >> 6) & 63u];
        dst[o + 3] = kops_b64_table[v & 63u];
        i += 3u;
        o += 4u;
    }
    if (i < len) {
        const uint32_t b0 = src[i];
        const uint32_t b1 = (i + 1u < len) ? src[i + 1] : 0u;
        const uint32_t v = (b0 << 16) | (b1 << 8);
        dst[o + 0] = kops_b64_table[(v >> 18) & 63u];
        dst[o + 1] = kops_b64_table[(v >> 12) & 63u];
        dst[o + 2] = (i + 1u < len) ? kops_b64_table[(v >> 6) & 63u] : '=';
        dst[o + 3] = '=';
        o += 4u;
    }
    if (o < dst_cap) {
        dst[o] = '\0';
    }
    return o;
}

bool kops_append_apc_chunked(kops_encoder *enc,
                             const char *first_hdr,
                             const char *cont_hdr,
                             const uint8_t *body, size_t body_len,
                             size_t chunk_bytes)
{
    const size_t total_b64 = kops_base64_encoded_len(body_len);
    size_t max_chunk = chunk_bytes >= 64u ? (chunk_bytes & ~(size_t)3u) : 4096u;
    size_t raw_off = 0;
    size_t chunk_idx = 0;

    if (total_b64 <= max_chunk) {
        /* Single chunk: omit m=0 (Rex compatibility) */
        const size_t hlen = strlen(first_hdr);
        const size_t need = hlen + total_b64 + 3u;
        char *dst = kops_ensure_out_buf(enc, enc->out_len + need);
        size_t n;
        if (!dst) { return false; }
        dst += enc->out_len;
        memcpy(dst, first_hdr, hlen);
        dst[hlen] = ';';
        n = kops_base64_encode(body, body_len, dst + hlen + 1u, total_b64 + 1u);
        if (body_len > 0 && n == 0) { return false; }
        dst[hlen + 1u + n] = '\x1b';
        dst[hlen + 2u + n] = '\\';
        enc->out_len += hlen + 3u + n;
        return true;
    }

    /* Multi-chunk: encode directly from raw slices into out_buf (zero extra b64 temp buffer) */
    {
        const size_t raw_per_chunk = (max_chunk / 4u) * 3u;
        while (raw_off < body_len) {
            const size_t take = (body_len - raw_off > raw_per_chunk) ? raw_per_chunk : (body_len - raw_off);
            const bool more = (raw_off + take < body_len);
            const char *hdr = (chunk_idx == 0) ? first_hdr : cont_hdr;
            const size_t hlen = strlen(hdr);
            const size_t b64_n = kops_base64_encoded_len(take);
            const size_t need = hlen + 5u + b64_n + 2u; /* ",m=X;" + b64 + ST */
            char *dst = kops_ensure_out_buf(enc, enc->out_len + need);
            size_t pos = 0;
            size_t enc_n;
            if (!dst) { return false; }
            dst += enc->out_len;
            memcpy(dst + pos, hdr, hlen);
            pos += hlen;
            dst[pos++] = ',';
            dst[pos++] = 'm';
            dst[pos++] = '=';
            dst[pos++] = more ? '1' : '0';
            dst[pos++] = ';';
            enc_n = kops_base64_encode(body + raw_off, take, dst + pos, b64_n + 1u);
            if (enc_n == 0) { return false; }
            pos += enc_n;
            dst[pos++] = '\x1b';
            dst[pos++] = '\\';
            enc->out_len += pos;
            raw_off += take;
            chunk_idx++;
        }
    }
    return true;
}

bool kops_append_rect_edit(kops_encoder *enc,
                           uint32_t image_id, uint32_t frame_num,
                           const uint8_t *fb, int32_t fb_w, int32_t fb_h, int32_t bpp,
                           const kops_rect *rect, bool use_zlib, size_t chunk_bytes,
                           size_t *out_raw_bytes, size_t *out_comp_bytes,
                           uint64_t *out_comp_ns, uint64_t *out_enc_ns)
{
    const int kitty_f = (bpp == 4) ? 32 : 24;
    const size_t raw_len = (size_t)rect->w * (size_t)rect->h * (size_t)bpp;
    const uint8_t *body = NULL;
    size_t body_len = 0;
    char first_hdr[160];
    char cont_hdr[48];
    uint64_t t0, t1;

    if (raw_len == 0) {
        return true;
    }
    if (out_raw_bytes) { *out_raw_bytes += raw_len; }

    t0 = kops_now_ns();
    if (use_zlib) {
        if (!kops_compress_rect_strided(enc, fb, fb_w, fb_h, bpp, rect, &body, &body_len)) {
            return false;
        }
    } else {
        uint8_t *pbuf = kops_ensure_pack_buf(enc, raw_len);
        if (!pbuf || !kops_pack_rect(fb, fb_w, fb_h, bpp, rect, pbuf, raw_len, &body_len)) {
            return false;
        }
        body = pbuf;
    }
    t1 = kops_now_ns();
    if (out_comp_ns) { *out_comp_ns += (t1 - t0); }
    if (out_comp_bytes) { *out_comp_bytes += body_len; }

    snprintf(first_hdr, sizeof(first_hdr),
             "\x1b_Ga=f,i=%u,r=%u,x=%d,y=%d,f=%d,s=%d,v=%d,t=d,%sX=1,q=2",
             (unsigned)image_id, (unsigned)frame_num,
             (int)rect->x, (int)rect->y, kitty_f, (int)rect->w, (int)rect->h,
             use_zlib ? "o=z," : "");
    snprintf(cont_hdr, sizeof(cont_hdr), "\x1b_Ga=f,q=2");

    t0 = kops_now_ns();
    if (!kops_append_apc_chunked(enc, first_hdr, cont_hdr, body, body_len, chunk_bytes)) {
        return false;
    }
    if (out_enc_ns) { *out_enc_ns += (kops_now_ns() - t0); }
    return true;
}

size_t kops_format_compose_apc(char *dst, size_t dst_cap,
                               uint32_t image_id, uint32_t src_frame, uint32_t dst_frame,
                               int32_t src_x, int32_t src_y,
                               int32_t dst_x, int32_t dst_y,
                               int32_t w, int32_t h)
{
    int n;
    if (!dst || dst_cap == 0 || image_id == 0 || src_frame == 0 || dst_frame == 0 || w <= 0 || h <= 0) {
        return 0;
    }
    /* Verified against both Kitty graphics.c & Ghostty graphics_command.zig:
     * r=src_frame, c=dst_frame, x/y=dst coords, X/Y=src coords, C=1 overwrite. */
    n = snprintf(dst, dst_cap,
                 "\x1b_Ga=c,i=%u,r=%u,c=%u,w=%d,h=%d,x=%d,y=%d,X=%d,Y=%d,C=1,q=2\x1b\\",
                 (unsigned)image_id, (unsigned)src_frame, (unsigned)dst_frame,
                 (int)w, (int)h, (int)dst_x, (int)dst_y, (int)src_x, (int)src_y);
    return (n > 0 && (size_t)n < dst_cap) ? (size_t)n : 0u;
}

size_t kops_format_animate_select_apc(char *dst, size_t dst_cap,
                                      uint32_t image_id, uint32_t frame_num)
{
    int n;
    if (!dst || dst_cap == 0 || image_id == 0 || frame_num == 0) {
        return 0;
    }
    n = snprintf(dst, dst_cap, "\x1b_Ga=a,i=%u,c=%u,s=1,q=2\x1b\\",
                 (unsigned)image_id, (unsigned)frame_num);
    return (n > 0 && (size_t)n < dst_cap) ? (size_t)n : 0u;
}

size_t kops_format_place_virtual_apc(char *dst, size_t dst_cap,
                                     uint32_t image_id, uint32_t placement_id,
                                     int32_t cols, int32_t rows)
{
    int n;
    if (!dst || dst_cap == 0 || image_id == 0 || cols <= 0 || rows <= 0) {
        return 0;
    }
    n = snprintf(dst, dst_cap, "\x1b_Ga=p,U=1,i=%u,p=%u,c=%d,r=%d,q=2\x1b\\",
                 (unsigned)image_id, (unsigned)(placement_id ? placement_id : 1u),
                 (int)cols, (int)rows);
    return (n > 0 && (size_t)n < dst_cap) ? (size_t)n : 0u;
}

size_t kops_format_delete_apc(char *dst, size_t dst_cap, uint32_t image_id)
{
    int n;
    if (!dst || dst_cap == 0 || image_id == 0) {
        return 0;
    }
    n = snprintf(dst, dst_cap, "\x1b_Ga=d,d=I,i=%u,q=2\x1b\\", (unsigned)image_id);
    return (n > 0 && (size_t)n < dst_cap) ? (size_t)n : 0u;
}

size_t kops_emit_shm_transmit(char *dst, size_t dst_cap,
                              uint32_t image_id,
                              const uint8_t *pixels, int32_t w, int32_t h, int32_t bpp,
                              bool place_now, int32_t place_cols, int32_t place_rows)
{
    static uint32_t seq = 1u;
    char name[64];
    char b64_name[128];
    const size_t raw_len = (size_t)w * (size_t)h * (size_t)bpp;
    const int kitty_f = (bpp == 4) ? 32 : 24;
    int fd;
    void *map;
    size_t b64_n;
    int n;

    if (!dst || dst_cap < 128u || !pixels || image_id == 0 ||
        w <= 0 || h <= 0 || (bpp != 3 && bpp != 4)) {
        return 0;
    }

    snprintf(name, sizeof(name), "/kops-%ld-%u", (long)getpid(), (unsigned)(seq++));
    fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        return 0;
    }
    if (ftruncate(fd, (off_t)raw_len) != 0) {
        close(fd);
        shm_unlink(name);
        return 0;
    }
    map = mmap(NULL, raw_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        shm_unlink(name);
        return 0;
    }
    memcpy(map, pixels, raw_len);
    munmap(map, raw_len);
    close(fd);

    b64_n = kops_base64_encode((const uint8_t *)name, strlen(name), b64_name, sizeof(b64_name));
    if (b64_n == 0) {
        shm_unlink(name);
        return 0;
    }
    if (place_now && place_cols > 0 && place_rows > 0) {
        n = snprintf(dst, dst_cap,
                     "\x1b_Ga=T,f=%d,t=s,i=%u,p=1,s=%d,v=%d,c=%d,r=%d,C=1,q=2;%s\x1b\\",
                     kitty_f, (unsigned)image_id, (int)w, (int)h,
                     (int)place_cols, (int)place_rows, b64_name);
    } else if (place_now) {
        n = snprintf(dst, dst_cap,
                     "\x1b_Ga=T,f=%d,t=s,i=%u,p=1,s=%d,v=%d,C=1,q=2;%s\x1b\\",
                     kitty_f, (unsigned)image_id, (int)w, (int)h, b64_name);
    } else {
        n = snprintf(dst, dst_cap,
                     "\x1b_Ga=t,f=%d,t=s,i=%u,s=%d,v=%d,q=2;%s\x1b\\",
                     kitty_f, (unsigned)image_id, (int)w, (int)h, b64_name);
    }
    if (n <= 0 || (size_t)n >= dst_cap) {
        shm_unlink(name);
        return 0;
    }
    return (size_t)n;
}
