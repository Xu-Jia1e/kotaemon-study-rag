from functools import lru_cache
from threading import Lock

_ocr_lock = Lock()


@lru_cache(maxsize=1)
def _ocr():
    from rapidocr import RapidOCR

    return RapidOCR(params={"Global.log_level": "warning"})


def read_image_text(
    image: bytes, *, group_labels: bool = False, single_line: bool = False
) -> tuple[str, ...]:
    """Recognize text in a raster image with the local OCR model."""
    with _ocr_lock:
        engine = _ocr()
        if single_line:
            output = engine(image, use_det=False, use_cls=False)
        else:
            # RapidOCR retains call-time flags on the cached engine. Restore the
            # configured defaults after a single-line call before reading an image.
            output = engine(
                image,
                use_det=engine.cfg.Global.use_det,
                use_cls=engine.cfg.Global.use_cls,
            )
    texts = [text.strip() for text in (getattr(output, "txts", None) or ())]
    if group_labels and len(texts) > 2 and output.boxes is not None:
        _group_wrapped_labels(texts, output.boxes)
    return tuple(filter(None, texts))


def _group_wrapped_labels(texts, boxes):
    """Keep nearby wrapped labels in their own column on UI screenshots."""
    bounds = [
        (min(xs), min(ys), max(xs), max(ys))
        for xs, ys in (zip(*box) for box in boxes)
    ]
    for i, (left, top, right, bottom) in enumerate(bounds):
        if not texts[i]:
            continue
        width, height = right - left, bottom - top
        matches = []
        for j in range(i + 2, len(bounds)):
            x0, y0, x1, y1 = bounds[j]
            if (
                texts[j]
                and 0 <= y0 - bottom <= height * 0.3
                and 0.8 * height <= y1 - y0 <= 1.25 * height
                and abs((x0 + x1) - (left + right)) <= 0.5 * width
                and min(x1, right) - max(x0, left) >= 0.65 * min(width, x1 - x0)
            ):
                matches.append(j)
        if len(matches) != 1:
            continue
        j = matches[0]
        # Only correct OCR's interleaving of different columns on the same row.
        if any(
            abs(y0 - top) <= height * 0.3 and (x1 <= left or x0 >= right)
            for x0, y0, x1, _ in bounds[i + 1 : j]
        ):
            texts[i] += " " + texts[j]
            texts[j] = ""
