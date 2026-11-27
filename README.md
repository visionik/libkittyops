# libkittyops

High-performance, dependency-free **C99** pixel & protocol kernel for the [Kitty Graphics Protocol](https://sw.kovidgoyal.net/kitty/graphics-protocol/), shared by [`sdl-kitty`](https://github.com/visionik/sdl-kitty) and [`pixcell`](https://github.com/visionik/pixcell).

## Features
- **SIMD multi-rect tile diff** (`ARM64 NEON` 32px + runtime-dispatched `x86_64 AVX2`/`SSSE3` clean-tile fast paths with exact 1px X-trimming + 6-worker 1080p bands + fused `XRGB8888`/`ARGB8888 → RGB24`/`RGBA32`).
- **Coalesce with dead-area bloat guard** (merges nearby rects without inflating pillarboxed games + corner HUDs).
- **Scroll detection + `a=c` ping-pong frame compose** (`~20 µs` row/column-hash scroll matcher → copies unchanged background inside terminal RAM in 45 bytes, sends `a=f` only for the new edge strip + moving sprites, and flips with `a=a`; auto-gated on terminal `storage_bytes >= 2 * w * h * 4`).
- **Hybrid parallel compression:** `libdeflate` for small/medium rects (`< 224 KB`) + zero-copy strided multi-slice parallel `zlib` / `zlib-ng` / `ISA-L` (`>= 224 KB`, up to 24 slices) + parallel multi-rect worker with `>= 256 KB` large-rect guard.
- **Kitty APC framing:** `a=t`, `a=T`, `a=f`, `a=c`, `a=a`, `a=p` (`U=1` virtual placements), `a=d`, `max_escape_bytes` chunking (omitting `m=0` on single-chunk for Rex compatibility), and POSIX `t=s` shared memory.
- **Thin Python binding** (`python/kittyops`) for optional `pixcell[fast]` acceleration.

## Build, Test & Benchmark

```bash
make check   # Build C99 library + run C and Python unit tests
make bench   # Run 1920x1080 diff, scroll-compose, and compression benchmarks
```

See [docs/architecture.md](docs/architecture.md) for API details and `sdl-kitty` / `pixcell` integration notes.
