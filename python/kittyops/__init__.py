"""Thin zero-copy ctypes binding for libkittyops (optional accelerator for pixcell)."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KOPS_MODE_SKIP = 0
KOPS_MODE_AF_SINGLE = 1
KOPS_MODE_AF_MULTI = 2
KOPS_MODE_SCROLL_COMPOSE = 3
KOPS_MODE_FULL_DIRECT = 4
KOPS_MODE_FULL_SHM = 5


class _KopsCaps(ctypes.Structure):
    _fields_ = [
        ("max_escape_bytes", ctypes.c_size_t),
        ("storage_bytes", ctypes.c_size_t),
        ("frame_edit", ctypes.c_bool),
        ("frame_edit_zlib", ctypes.c_bool),
        ("compose", ctypes.c_bool),
        ("shm", ctypes.c_bool),
        ("placeholders", ctypes.c_bool),
        ("allow_scroll_compose", ctypes.c_bool),
        ("af_max_coverage", ctypes.c_float),
        ("af_max_pixels", ctypes.c_size_t),
        ("tile_size", ctypes.c_int32),
        ("coalesce_gap", ctypes.c_int32),
        ("coalesce_max_bloat", ctypes.c_size_t),
        ("zlib_level", ctypes.c_int32),
    ]


class _KopsStats(ctypes.Structure):
    _fields_ = [
        ("diff_ns", ctypes.c_uint64),
        ("scroll_ns", ctypes.c_uint64),
        ("compress_ns", ctypes.c_uint64),
        ("encode_ns", ctypes.c_uint64),
        ("raw_bytes", ctypes.c_size_t),
        ("compressed_bytes", ctypes.c_size_t),
        ("wire_bytes", ctypes.c_size_t),
        ("rect_count", ctypes.c_int32),
        ("scroll_dx", ctypes.c_int32),
        ("scroll_dy", ctypes.c_int32),
        ("mode", ctypes.c_int),
    ]


@dataclass(frozen=True)
class Caps:
    """Terminal capability and tuning profile (hydrated from terminal-facts)."""

    max_escape_bytes: int = 262144
    storage_bytes: int = 320_000_000
    frame_edit: bool = True
    frame_edit_zlib: bool = True
    compose: bool = True
    shm: bool = False
    placeholders: bool = True
    allow_scroll_compose: bool = True
    af_max_coverage: float = 0.90
    af_max_pixels: int = 1920 * 1080
    tile_size: int = 32
    coalesce_gap: int = 2
    coalesce_max_bloat: int = 65536
    zlib_level: int = 3

    def _to_c(self) -> _KopsCaps:
        return _KopsCaps(
            max_escape_bytes=self.max_escape_bytes,
            storage_bytes=self.storage_bytes,
            frame_edit=self.frame_edit,
            frame_edit_zlib=self.frame_edit_zlib,
            compose=self.compose,
            shm=self.shm,
            placeholders=self.placeholders,
            allow_scroll_compose=self.allow_scroll_compose,
            af_max_coverage=self.af_max_coverage,
            af_max_pixels=self.af_max_pixels,
            tile_size=self.tile_size,
            coalesce_gap=self.coalesce_gap,
            coalesce_max_bloat=self.coalesce_max_bloat,
            zlib_level=self.zlib_level,
        )


@dataclass(frozen=True)
class FrameStats:
    diff_ms: float
    scroll_ms: float
    compress_ms: float
    encode_ms: float
    raw_bytes: int
    compressed_bytes: int
    wire_bytes: int
    rect_count: int
    scroll_dx: int
    scroll_dy: int
    mode: int


def _load_library() -> ctypes.CDLL:
    candidates: list[str] = []
    if env := os.environ.get("KITTYOPS_LIB"):
        candidates.append(env)
    here = Path(__file__).resolve().parent
    for name in ("libkittyops.dylib", "libkittyops.so", "kittyops.dll"):
        candidates.append(str(here / name))
        candidates.append(str(here.parent.parent / "build" / name))
    found = ctypes.util.find_library("kittyops")
    if found:
        candidates.append(found)
    last_err: Exception | None = None
    for path in candidates:
        if path and Path(path).exists() or path == found:
            try:
                lib = ctypes.CDLL(path)
                _configure_signatures(lib)
                return lib
            except OSError as exc:
                last_err = exc
    raise OSError(f"libkittyops shared library not found (tried {candidates}): {last_err}")


def _configure_signatures(lib: ctypes.CDLL) -> None:
    lib.kops_compression_backend.argtypes = []
    lib.kops_compression_backend.restype = ctypes.c_char_p

    lib.kops_encoder_create.argtypes = [ctypes.c_int32]
    lib.kops_encoder_create.restype = ctypes.c_void_p

    lib.kops_encoder_destroy.argtypes = [ctypes.c_void_p]
    lib.kops_encoder_destroy.restype = None

    lib.kops_encoder_reset.argtypes = [ctypes.c_void_p]
    lib.kops_encoder_reset.restype = None

    lib.kops_encode_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(_KopsCaps),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(_KopsStats),
    ]
    lib.kops_encode_frame.restype = ctypes.c_bool


_LIB: ctypes.CDLL | None = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load_library()
    return _LIB


def compression_backend() -> str:
    return _lib().kops_compression_backend().decode("ascii")


class Encoder:
    """Reusable Kitty graphics frame encoder wrapping kops_encoder."""

    def __init__(self, zlib_level: int = 3) -> None:
        self._lib = _lib()
        ptr = self._lib.kops_encoder_create(zlib_level)
        if not ptr:
            raise MemoryError("kops_encoder_create failed")
        self._ptr: int | None = ptr

    def close(self) -> None:
        if self._ptr:
            self._lib.kops_encoder_destroy(self._ptr)
            self._ptr = None

    def reset(self) -> None:
        if self._ptr:
            self._lib.kops_encoder_reset(self._ptr)

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def encode_frame(
        self,
        image_id: int,
        prev_bytes: bytes | bytearray | memoryview | None,
        curr_bytes: bytes | bytearray | memoryview,
        width: int,
        height: int,
        bpp: int = 3,
        caps: Caps | None = None,
    ) -> tuple[bytes, FrameStats]:
        """Encode delta (or full frame when prev_bytes is None) into Kitty APC bytes."""
        if not self._ptr:
            raise RuntimeError("Encoder is closed")
        expected = width * height * bpp
        if len(curr_bytes) != expected:
            raise ValueError(f"curr_bytes len {len(curr_bytes)} != {expected}")
        if prev_bytes is not None and len(prev_bytes) != expected:
            raise ValueError(f"prev_bytes len {len(prev_bytes)} != {expected}")

        c_curr = (ctypes.c_char * expected).from_buffer_copy(curr_bytes) if isinstance(curr_bytes, memoryview) else curr_bytes
        c_prev = None
        if prev_bytes is not None:
            c_prev = (ctypes.c_char * expected).from_buffer_copy(prev_bytes) if isinstance(prev_bytes, memoryview) else prev_bytes

        c_caps = (caps or Caps())._to_c()
        out_ptr = ctypes.c_char_p()
        out_len = ctypes.c_size_t(0)
        st = _KopsStats()

        ok = self._lib.kops_encode_frame(
            self._ptr,
            image_id,
            c_prev,
            c_curr,
            width,
            height,
            bpp,
            ctypes.byref(c_caps),
            ctypes.byref(out_ptr),
            ctypes.byref(out_len),
            ctypes.byref(st),
        )
        if not ok:
            raise RuntimeError("kops_encode_frame failed")

        raw_apc = ctypes.string_at(out_ptr, out_len.value) if out_len.value > 0 else b""
        stats = FrameStats(
            diff_ms=st.diff_ns / 1e6,
            scroll_ms=st.scroll_ns / 1e6,
            compress_ms=st.compress_ns / 1e6,
            encode_ms=st.encode_ns / 1e6,
            raw_bytes=st.raw_bytes,
            compressed_bytes=st.compressed_bytes,
            wire_bytes=st.wire_bytes,
            rect_count=st.rect_count,
            scroll_dx=st.scroll_dx,
            scroll_dy=st.scroll_dy,
            mode=st.mode,
        )
        return raw_apc, stats
