"""Regression test for the shared RapidOCR mode across different files."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from kotaemon.loaders.utils import image_ocr


class StatefulEngine:
    def __init__(self):
        self.cfg = SimpleNamespace(Global=SimpleNamespace(use_det=True, use_cls=True))
        self.use_det = True
        self.use_cls = True

    def __call__(self, image, use_det=None, use_cls=None):
        if use_det is not None:
            self.use_det = use_det
        if use_cls is not None:
            self.use_cls = use_cls
        text = "full image" if self.use_det and self.use_cls else "single line"
        return SimpleNamespace(txts=[text], boxes=None)


class OcrStateTest(unittest.TestCase):
    def test_full_image_mode_is_restored_after_single_line(self):
        engine = StatefulEngine()
        with patch.object(image_ocr, "_ocr", return_value=engine):
            self.assertEqual(image_ocr.read_image_text(b"first", single_line=True), ("single line",))
            self.assertEqual(image_ocr.read_image_text(b"second"), ("full image",))


if __name__ == "__main__":
    unittest.main()
