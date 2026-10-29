"""EasyOCR on CPU with explicit full-image rotations and source-frame boxes."""

import io

from PIL import Image

from annotation.contracts import Box, OCRResult, OCRSpan, VisualInput


def source_point(x: float, y: float, rotation: int) -> tuple[float, float]:
    # PIL rotates counterclockwise; inverse mapping uses normalized coordinates.
    if rotation == 90:
        return 1 - y, x
    if rotation == 180:
        return 1 - x, 1 - y
    if rotation == 270:
        return y, 1 - x
    return x, y


def source_box(points: list, width: int, height: int, rotation: int) -> Box:
    coordinates = [source_point(float(x) / width, float(y) / height, rotation) for x, y in points]
    xs, ys = zip(*coordinates, strict=True)
    return Box(
        x0=max(0.0, min(xs)), y0=max(0.0, min(ys)), x1=min(1.0, max(xs)), y1=min(1.0, max(ys))
    )


class EasyOCRWorker:
    def __init__(self, languages: list[str], rotations: list[int]):
        self.languages, self.rotations = languages, rotations
        self.reader = None

    def read(self, image: VisualInput) -> OCRResult:
        import easyocr
        import numpy as np

        if self.reader is None:
            self.reader = easyocr.Reader(self.languages, gpu=False)
        source = Image.open(io.BytesIO(image.image_bytes)).convert("RGB")
        spans = []
        for rotation in self.rotations:
            rotated = source.rotate(rotation, expand=True)
            results = self.reader.readtext(np.array(rotated), detail=1, paragraph=False)
            for points, text, confidence in results:
                if text.strip():
                    spans.append(
                        OCRSpan(
                            text=text.strip(),
                            confidence=float(confidence),
                            box=source_box(points, rotated.width, rotated.height, rotation),
                            rotation=rotation,
                            crop_id="full",
                        )
                    )
        return OCRResult(spans=spans)
