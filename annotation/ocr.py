"""Adapter interface. GPU OCR and rotation/crop mapping are implemented in T03."""

from typing import Protocol

from annotation.contracts import OCRResult, VisualInput


class OCRWorker(Protocol):
    def read(self, image: VisualInput) -> OCRResult: ...
