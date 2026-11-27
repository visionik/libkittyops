import unittest

import kittyops


def _make_pattern(w: int, h: int, scroll_y: int = 0) -> bytes:
    buf = bytearray(w * h * 3)
    for y in range(h):
        sy = (y - scroll_y) & 0xFFFF
        row_off = y * w * 3
        for x in range(w):
            o = row_off + x * 3
            buf[o] = (x * 7 + sy * 13 + (sy // 17) * 53) & 0xFF
            buf[o + 1] = ((x * 11) ^ (sy * 29)) & 0xFF
            buf[o + 2] = ((x // 9) * 37 + sy * 5) & 0xFF
    return bytes(buf)


class TestKittyOpsPython(unittest.TestCase):
    def test_backend_and_pipeline(self) -> None:
        self.assertTrue(len(kittyops.compression_backend()) > 0)
        w, h = 320, 240
        f0 = _make_pattern(w, h, 0)
        f1 = _make_pattern(w, h, 16)
        caps = kittyops.Caps(placeholders=True, allow_scroll_compose=True)

        with kittyops.Encoder(zlib_level=3) as enc:
            apc, st = enc.encode_frame(7, None, f0, w, h, 3, caps)
            self.assertEqual(st.mode, kittyops.KOPS_MODE_FULL_DIRECT)
            self.assertIn(b"\x1b_Ga=t,", apc)

            apc_skip, st_skip = enc.encode_frame(7, f0, f0, w, h, 3, caps)
            self.assertEqual(st_skip.mode, kittyops.KOPS_MODE_SKIP)
            self.assertEqual(apc_skip, b"")

            apc_scr, st_scr = enc.encode_frame(7, f0, f1, w, h, 3, caps)
            self.assertEqual(st_scr.mode, kittyops.KOPS_MODE_SCROLL_COMPOSE)
            self.assertEqual(st_scr.scroll_dy, 16)
            self.assertIn(b"\x1b_Ga=c,i=7,r=1,c=2,", apc_scr)
            self.assertIn(b"\x1b_Ga=a,i=7,c=2,s=1,q=2\x1b\\", apc_scr)
            self.assertLess(st_scr.raw_bytes, len(f0) // 4)


if __name__ == "__main__":
    unittest.main()
