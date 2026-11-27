#include "kops_internal.h"

/*
 * Fast 1D/2D scroll detector (~15-30 us at 1080p) + shifted multi-rect diff.
 *
 * Strategy:
 *   1. Sample 8-pixel-strided row fingerprints across the middle 60% of the ROI
 *      in `prev` and `curr` (avoiding HUD/status bars at the outer 20% edges).
 *   2. Ignore solid/uniform rows (e.g. black letterbox or flat white margin) so
 *      blank background bands never vote for bogus shifts.
 *   3. Match sampled `curr` row hashes against `prev` rows within dy in [-h/2, h/2]
 *      to find the dominant vertical shift `dy`, and similarly sample columns to
 *      detect horizontal scroll `dx` (side-scrollers like Metal Slug).
 *   4. Verify the top candidate `(dx, dy)` by direct pixel comparison across 16
 *      evenly spaced verification rows inside the overlapped interior; require
 *      >= 60% pixel match (and significantly better than unshifted (0,0)).
 */

static inline bool kops_row_fingerprint(const uint8_t *fb, int32_t fb_w, int32_t bpp,
                                        int32_t x0, int32_t x1, int32_t y,
                                        uint64_t *out_hash)
{
    const uint8_t *row = fb + ((size_t)y * (size_t)fb_w + (size_t)x0) * (size_t)bpp;
    const int32_t w = x1 - x0;
    const int32_t step = w >= 256 ? 8 : (w >= 64 ? 4 : 2);
    uint64_t h = 0xcbf29ce484222325ULL;
    uint32_t first_px = 0;
    int32_t transitions = 0;
    int32_t x;

    for (x = 0; x < w; x += step) {
        const uint8_t *p = row + (size_t)x * (size_t)bpp;
        const uint32_t px = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
        if (x == 0) {
            first_px = px;
        } else if (px != first_px) {
            transitions++;
        }
        h = (h ^ (uint64_t)px) * 0x100000001b3ULL;
    }
    *out_hash = h ^ (h >> 33);
    /* Reject uniform solid rows (e.g. blank white webpage margin / black sky) */
    return transitions >= 3;
}

static inline bool kops_col_fingerprint(const uint8_t *fb, int32_t fb_w, int32_t bpp,
                                        int32_t y0, int32_t y1, int32_t x,
                                        uint64_t *out_hash)
{
    const size_t stride = (size_t)fb_w * (size_t)bpp;
    const int32_t h_span = y1 - y0;
    const int32_t step = h_span >= 256 ? 8 : (h_span >= 64 ? 4 : 2);
    uint64_t h = 0x9e3779b97f4a7c15ULL;
    uint32_t first_px = 0;
    int32_t transitions = 0;
    int32_t y;

    for (y = y0; y < y1; y += step) {
        const uint8_t *p = fb + (size_t)y * stride + (size_t)x * (size_t)bpp;
        const uint32_t px = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16);
        if (y == y0) {
            first_px = px;
        } else if (px != first_px) {
            transitions++;
        }
        h = (h ^ (uint64_t)px) * 0x100000001b3ULL;
    }
    *out_hash = h ^ (h >> 33);
    return transitions >= 3;
}

static float kops_verify_shift(const uint8_t *prev, const uint8_t *curr,
                               int32_t fb_w, int32_t fb_h, int32_t bpp,
                               int32_t x0, int32_t y0, int32_t x1, int32_t y1,
                               int32_t dx, int32_t dy)
{
    const size_t stride = (size_t)fb_w * (size_t)bpp;
    const int32_t ox0 = (x0 > x0 + dx) ? x0 : (x0 + dx);
    const int32_t ox1 = (x1 < x1 + dx) ? x1 : (x1 + dx);
    const int32_t oy0 = (y0 > y0 + dy) ? y0 : (y0 + dy);
    const int32_t oy1 = (y1 < y1 + dy) ? y1 : (y1 + dy);
    const int32_t ow = ox1 - ox0;
    const int32_t oh = oy1 - oy0;
    int32_t s, x;
    size_t matched = 0, total = 0;
    const int32_t xstep = ow >= 256 ? 8 : 4;

    if (ow < 32 || oh < 32) {
        return 0.0f;
    }
    for (s = 0; s < 16; ++s) {
        const int32_t cy = oy0 + ((s * oh) / 16);
        const int32_t py = cy - dy;
        const uint8_t *crow = curr + (size_t)cy * stride;
        const uint8_t *prow = prev + (size_t)py * stride;
        if (py < 0 || py >= fb_h || cy < 0 || cy >= fb_h) {
            continue;
        }
        for (x = ox0; x < ox1; x += xstep) {
            const int32_t px = x - dx;
            const uint8_t *cp = crow + (size_t)x * (size_t)bpp;
            const uint8_t *pp = prow + (size_t)px * (size_t)bpp;
            total++;
            if (cp[0] == pp[0] && cp[1] == pp[1] && cp[2] == pp[2]) {
                matched++;
            }
        }
    }
    return total > 0 ? (float)((double)matched / (double)total) : 0.0f;
}

bool kops_detect_scroll(const uint8_t *prev, const uint8_t *curr,
                        int32_t fb_w, int32_t fb_h, int32_t bpp,
                        const kops_rect *roi,
                        int32_t *out_dx, int32_t *out_dy,
                        float *out_confidence)
{
    int32_t rx0 = 0, ry0 = 0, rx1 = fb_w, ry1 = fb_h;
    int32_t rw, rh;
    int32_t ix0, ix1, iy0, iy1;
    int32_t cand_dx = 0, cand_dy = 0;
    float best_conf = 0.0f;
    float zero_conf = 0.0f;

    if (out_dx) { *out_dx = 0; }
    if (out_dy) { *out_dy = 0; }
    if (out_confidence) { *out_confidence = 0.0f; }
    if (!prev || !curr || fb_w < 64 || fb_h < 64 || (bpp != 3 && bpp != 4)) {
        return false;
    }

    if (roi) {
        if (roi->x > rx0) { rx0 = roi->x; }
        if (roi->y > ry0) { ry0 = roi->y; }
        if (roi->x + roi->w < rx1) { rx1 = roi->x + roi->w; }
        if (roi->y + roi->h < ry1) { ry1 = roi->y + roi->h; }
    }
    rw = rx1 - rx0;
    rh = ry1 - ry0;
    if (rw < 64 || rh < 64) {
        return false;
    }

    /* Inner band trims static chrome/HUD margins */
    ix0 = rx0 + rw / 8;
    ix1 = rx1 - rw / 8;
    iy0 = ry0 + rh / 8;
    iy1 = ry1 - rh / 8;

    /* Quick reject: if > 85% of pixels are already identical at (0,0), not a scroll. */
    zero_conf = kops_verify_shift(prev, curr, fb_w, fb_h, bpp, rx0, ry0, rx1, ry1, 0, 0);
    if (zero_conf >= 0.85f) {
        return false;
    }

    /* --- 1. Vertical scroll search (dy) --- */
    {
        uint64_t prev_h[2048];
        uint8_t prev_ok[2048];
        const int32_t max_y = (fb_h <= 2048) ? fb_h : 2048;
        int32_t y;
        int32_t votes_dy[32];
        int32_t votes_cnt[32];
        int32_t n_votes = 0;
        int32_t s;
        const int32_t max_dy = rh / 2;

        for (y = ry0; y < ry1 && y < max_y; ++y) {
            prev_ok[y] = kops_row_fingerprint(prev, fb_w, bpp, ix0, ix1, y, &prev_h[y]) ? 1u : 0u;
        }
        for (s = 0; s < 24; ++s) {
            const int32_t cy = iy0 + ((s * (iy1 - iy0)) / 24);
            uint64_t ch = 0;
            int32_t lo, hi, py;
            if (cy >= max_y || !kops_row_fingerprint(curr, fb_w, bpp, ix0, ix1, cy, &ch)) {
                continue;
            }
            lo = cy - max_dy;
            if (lo < ry0) { lo = ry0; }
            hi = cy + max_dy;
            if (hi > ry1) { hi = ry1; }
            if (hi > max_y) { hi = max_y; }
            for (py = lo; py < hi; ++py) {
                if (py != cy && prev_ok[py] && prev_h[py] == ch) {
                    const int32_t dy = cy - py;
                    int32_t vi;
                    for (vi = 0; vi < n_votes; ++vi) {
                        if (votes_dy[vi] == dy) {
                            votes_cnt[vi]++;
                            break;
                        }
                    }
                    if (vi == n_votes && n_votes < 32) {
                        votes_dy[n_votes] = dy;
                        votes_cnt[n_votes] = 1;
                        n_votes++;
                    }
                }
            }
        }
        {
            int32_t best_vi = -1, best_vc = 2;
            int32_t vi;
            for (vi = 0; vi < n_votes; ++vi) {
                if (votes_cnt[vi] > best_vc) {
                    best_vc = votes_cnt[vi];
                    best_vi = vi;
                }
            }
            if (best_vi >= 0) {
                const int32_t dy = votes_dy[best_vi];
                const float c = kops_verify_shift(prev, curr, fb_w, fb_h, bpp, rx0, ry0, rx1, ry1, 0, dy);
                if (c > best_conf) {
                    best_conf = c;
                    cand_dx = 0;
                    cand_dy = dy;
                }
            }
        }
    }

    /* --- 2. Horizontal scroll search (dx) if vertical wasn't a clear hit --- */
    if (best_conf < 0.65f) {
        uint64_t prev_h[4096];
        uint8_t prev_ok[4096];
        const int32_t max_x = (fb_w <= 4096) ? fb_w : 4096;
        int32_t x, s;
        int32_t votes_dx[32];
        int32_t votes_cnt[32];
        int32_t n_votes = 0;
        const int32_t max_dx = rw / 3;

        for (x = rx0; x < rx1 && x < max_x; ++x) {
            prev_ok[x] = kops_col_fingerprint(prev, fb_w, bpp, iy0, iy1, x, &prev_h[x]) ? 1u : 0u;
        }
        for (s = 0; s < 24; ++s) {
            const int32_t cx = ix0 + ((s * (ix1 - ix0)) / 24);
            uint64_t ch = 0;
            int32_t lo, hi, px;
            if (cx >= max_x || !kops_col_fingerprint(curr, fb_w, bpp, iy0, iy1, cx, &ch)) {
                continue;
            }
            lo = cx - max_dx;
            if (lo < rx0) { lo = rx0; }
            hi = cx + max_dx;
            if (hi > rx1) { hi = rx1; }
            if (hi > max_x) { hi = max_x; }
            for (px = lo; px < hi; ++px) {
                if (px != cx && prev_ok[px] && prev_h[px] == ch) {
                    const int32_t dx = cx - px;
                    int32_t vi;
                    for (vi = 0; vi < n_votes; ++vi) {
                        if (votes_dx[vi] == dx) {
                            votes_cnt[vi]++;
                            break;
                        }
                    }
                    if (vi == n_votes && n_votes < 32) {
                        votes_dx[n_votes] = dx;
                        votes_cnt[n_votes] = 1;
                        n_votes++;
                    }
                }
            }
        }
        {
            int32_t best_vi = -1, best_vc = 2;
            int32_t vi;
            for (vi = 0; vi < n_votes; ++vi) {
                if (votes_cnt[vi] > best_vc) {
                    best_vc = votes_cnt[vi];
                    best_vi = vi;
                }
            }
            if (best_vi >= 0) {
                const int32_t dx = votes_dx[best_vi];
                const float c = kops_verify_shift(prev, curr, fb_w, fb_h, bpp, rx0, ry0, rx1, ry1, dx, 0);
                if (c > best_conf) {
                    best_conf = c;
                    cand_dx = dx;
                    cand_dy = 0;
                }
            }
        }
    }

    if ((cand_dx != 0 || cand_dy != 0) && best_conf >= 0.60f && best_conf > zero_conf + 0.15f) {
        if (out_dx) { *out_dx = cand_dx; }
        if (out_dy) { *out_dy = cand_dy; }
        if (out_confidence) { *out_confidence = best_conf; }
        return true;
    }
    return false;
}

/*
 * Shifted tile diff: compares curr(x, y) against prev(x - dx, y - dy) inside
 * the shifted overlap region (parallelized across 6 bands at >= 720p), and
 * automatically marks the newly exposed edge strip(s) dirty as well.
 */
static int32_t kops_diff_shifted_band(const uint8_t *prev, const uint8_t *curr,
                                      int32_t fb_w, int32_t fb_h, int32_t bpp,
                                      int32_t dx, int32_t dy,
                                      int32_t tile, int32_t tiles_x,
                                      int32_t ty, int32_t th,
                                      uint32_t *tile_spans)
{
    const size_t stride = (size_t)fb_w * (size_t)bpp;
    int32_t tx_idx;
    int32_t dirty_count = 0;

    for (tx_idx = 0; tx_idx < tiles_x; ++tx_idx) {
        const int32_t tx = tx_idx * tile;
        const int32_t tw = (tx + tile <= fb_w) ? tile : (fb_w - tx);
        int32_t ry;
        int32_t sy0 = th, sy1 = 0, sx0 = tw, sx1 = 0;
        bool tile_dirty = false;

        for (ry = 0; ry < th; ++ry) {
            const int32_t cy = ty + ry;
            const int32_t py = cy - dy;
            if (py < 0 || py >= fb_h) {
                tile_dirty = true;
                if (ry < sy0) { sy0 = ry; }
                sy1 = ry + 1;
                sx0 = 0;
                sx1 = tw;
                continue;
            }
            {
                const uint8_t *crow = curr + (size_t)cy * stride;
                const uint8_t *prow = prev + (size_t)py * stride;
                const int32_t px0 = tx - dx;
                const int32_t px1 = px0 + tw;
                if (px0 >= 0 && px1 <= fb_w) {
                    const uint8_t *cp = crow + (size_t)tx * (size_t)bpp;
                    const uint8_t *pp = prow + (size_t)px0 * (size_t)bpp;
                    if (memcmp(cp, pp, (size_t)tw * (size_t)bpp) == 0) {
                        continue;
                    }
                }
                {
                    int32_t lx;
                    for (lx = 0; lx < tw; ++lx) {
                        const int32_t cx = tx + lx;
                        const int32_t px = cx - dx;
                        bool diff = (px < 0 || px >= fb_w);
                        if (!diff) {
                            const uint8_t *cp = crow + (size_t)cx * (size_t)bpp;
                            const uint8_t *pp = prow + (size_t)px * (size_t)bpp;
                            diff = (cp[0] != pp[0] || cp[1] != pp[1] || cp[2] != pp[2] ||
                                    (bpp == 4 && cp[3] != pp[3]));
                        }
                        if (diff) {
                            tile_dirty = true;
                            if (ry < sy0) { sy0 = ry; }
                            sy1 = ry + 1;
                            if (lx < sx0) { sx0 = lx; }
                            if (lx + 1 > sx1) { sx1 = lx + 1; }
                        }
                    }
                }
            }
        }
        if (tile_dirty && sy0 < sy1 && sx0 < sx1) {
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

typedef struct kops_par_shift_ctx {
    const uint8_t *prev;
    const uint8_t *curr;
    int32_t fb_w, fb_h, bpp, dx, dy, tile, tiles_x, num_bands, n_workers;
    uint32_t *all_spans;
    int32_t *band_counts;
} kops_par_shift_ctx;

static void kops_par_shift_worker(void *arg, size_t widx)
{
    kops_par_shift_ctx *ctx = (kops_par_shift_ctx *)arg;
    const int32_t b0 = ((int32_t)widx * ctx->num_bands) / ctx->n_workers;
    const int32_t b1 = (((int32_t)widx + 1) * ctx->num_bands) / ctx->n_workers;
    int32_t b;
    for (b = b0; b < b1; ++b) {
        const int32_t ty = b * ctx->tile;
        const int32_t th = (ty + ctx->tile <= ctx->fb_h) ? ctx->tile : (ctx->fb_h - ty);
        ctx->band_counts[b] = kops_diff_shifted_band(
            ctx->prev, ctx->curr, ctx->fb_w, ctx->fb_h, ctx->bpp,
            ctx->dx, ctx->dy, ctx->tile, ctx->tiles_x, ty, th,
            ctx->all_spans + (size_t)b * (size_t)ctx->tiles_x);
    }
}

kops_diff_outcome kops_diff_shifted_rects(const uint8_t *prev, const uint8_t *curr,
                                          int32_t fb_w, int32_t fb_h, int32_t bpp,
                                          int32_t dx, int32_t dy,
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
    kops_rect acc[64];
    int32_t n_acc = 0;
    bool overflow = false;
    int32_t min_x = fb_w, min_y = fb_h, max_x = 0, max_y = 0;
    size_t dirty_px = 0;
    int32_t b, tx_idx;
    kops_rect bbox;

    if (out_count) { *out_count = 0; }
    if (!prev || !curr || fb_w <= 0 || fb_h <= 0 || (bpp != 3 && bpp != 4) || tile <= 0) {
        return KOPS_DIFF_FULL;
    }
    if (dx == 0 && dy == 0) {
        return kops_tile_diff_rects(prev, curr, fb_w, fb_h, bpp, tile, cover_limit,
                                    rects_out, max_rects, out_count, bbox_out);
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
        kops_par_shift_ctx ctx;
        ctx.prev = prev;
        ctx.curr = curr;
        ctx.fb_w = fb_w;
        ctx.fb_h = fb_h;
        ctx.bpp = bpp;
        ctx.dx = dx;
        ctx.dy = dy;
        ctx.tile = tile;
        ctx.tiles_x = tiles_x;
        ctx.num_bands = num_bands;
        ctx.n_workers = ((size_t)fb_w * (size_t)fb_h >= (size_t)1280 * 720 && num_bands >= 6) ? 6 : 4;
        ctx.all_spans = all_spans;
        ctx.band_counts = band_counts;
        kops_parallel_for((size_t)ctx.n_workers, kops_par_shift_worker, &ctx);
    } else {
        for (b = 0; b < num_bands; ++b) {
            const int32_t ty = b * tile;
            const int32_t th = (ty + tile <= fb_h) ? tile : (fb_h - ty);
            band_counts[b] = kops_diff_shifted_band(
                prev, curr, fb_w, fb_h, bpp, dx, dy, tile, tiles_x, ty, th,
                all_spans + (size_t)b * (size_t)tiles_x);
        }
    }

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
                    run_x0 = sx0; run_x1 = sx1; run_y0 = sy0; run_y1 = sy1;
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
                run_x0 = -1; run_x1 = 0; run_y0 = th; run_y1 = 0;
            }
        }
    }

    if (band_counts != stack_counts) { free(band_counts); }
    if (all_spans != stack_spans) { free(all_spans); }

    if (dirty_px == 0) {
        if (bbox_out) { bbox_out->x = bbox_out->y = bbox_out->w = bbox_out->h = 0; }
        return KOPS_DIFF_NONE;
    }

    bbox.x = min_x;
    bbox.y = min_y;
    bbox.w = max_x - min_x;
    bbox.h = max_y - min_y;
    if (bbox_out) { *bbox_out = bbox; }

    {
        const double fb_area = (double)fb_w * (double)fb_h;
        const float cov = fb_area > 0.0 ? (float)((double)dirty_px / fb_area) : 1.0f;
        if (cov >= cover_limit) {
            if (rects_out && max_rects > 0) { rects_out[0] = bbox; }
            if (out_count) { *out_count = (rects_out && max_rects > 0) ? 1 : 0; }
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
            for (i = 0; i < n_acc; ++i) { rects_out[i] = acc[i]; }
            if (out_count) { *out_count = n_acc; }
        }
    }
    return KOPS_DIFF_RECT;
}
