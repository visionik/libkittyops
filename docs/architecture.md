# libkittyops Architecture

`libkittyops` isolates the CPU-hot, framework-agnostic pixel diffing, scroll compensation, parallel zlib compression, and Kitty APC framing kernel so both C/SDL (`sdl-kitty`) and Python/Textual (`pixcell`) share a single tuned implementation without pulling SDL or Python into each other.

## Pipeline (`kops_encode_frame`)

1. **SIMD Tile Diff (`src/diff.c`):** Parallel band scan (4–6 workers) produces up to 16 tightly trimmed dirty rects (`KOPS_DIFF_NONE` skips the frame in `~0.08 ms`).
2. **Scroll Detection & `a=c` Ping-Pong (`src/scroll.c`, `src/encoder.c`):**
   - Triggered when `caps.compose && caps.allow_scroll_compose`, dirty coverage `>= 25%`, and terminal `storage_bytes >= 2 * w * h * 4` (protects Rex's 10 MB storage limit on 1080p full-screen while still fitting `<= 720p` and vertical `810×1080` arcade playfields).
   - Computes 8px-strided row/column fingerprints across the interior ROI, votes on `(dx, dy)`, verifies pixel match `>= 60%`, and emits:
     - `a=c,i=<id>,r=<src>,c=<dst>,w=...,h=...,x=<dst_x>,y=<dst_y>,X=<src_x>,Y=<src_y>,C=1,q=2`
     - minimal `a=f,r=<dst>,...` edits for the newly revealed edge strip + moving sprites/sticky chrome
     - `a=a,i=<id>,c=<dst>,s=1,q=2` to display `<dst>` (compatible with `U=1` Unicode placeholders).
3. **Coalesce with Bloat Guard (`kops_coalesce_rects`):** Merges rects within `coalesce_gap` unless the union adds more than `coalesce_max_bloat` (default `64K`) clean pixels.
4. **Hybrid Compression (`src/compress.c`):**
   - `< 224 KB`: single-thread `libdeflate` (avoids thread wakeup jitter on small HUD/sprite rects).
   - `>= 224 KB`: zero-copy strided parallel-pack + parallel-zlib (up to 24 slices via GCD on macOS / pthreads on POSIX), auto-detecting `zlib-ng` / `ISA-L` / system `zlib`.
   - Multi-rect (`3..16` rects): parallel `libdeflate` workers, guarded so any rect `>= 256 KB` routes to multi-core strided parallel-zlib instead of serializing on 1 thread.
5. **APC Framing (`src/apc.c`):** Chunked by `caps.max_escape_bytes` (omitting `m=0` on single-chunk frames for Rex), plus optional `t=s` POSIX shared memory for full frames on Kitty/Ghostty.

## Consumers
- **`sdl-kitty`:** Links `libkittyops` directly into `libSDL3` (replacing duplicated rect-coalesce, `FastHash64`, `PackRect`, and `ConvertAndTileDiff` kernels while keeping SDL surface/window/HUD/async-TTY glue in SDL).
- **`pixcell`:** Stays pure Python by default; imports `kittyops.Encoder` when installed to accelerate `LiveImage` and `BrowserView` with zero-copy buffer handoff.
