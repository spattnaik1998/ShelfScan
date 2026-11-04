"""Conservative title evidence within one OCR rotation and adjacent text spans."""

import re
import unicodedata
from collections import defaultdict


def title_tokens(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def token_contains(text, quote):
    left, right = title_tokens(text), title_tokens(quote)
    return bool(right) and any(
        left[i : i + len(right)] == right for i in range(len(left) - len(right) + 1)
    )


def rotated_box(span):
    box = span.box
    points = [(x, y) for x in (box.x0, box.x1) for y in (box.y0, box.y1)]

    def transform(x, y):
        return {0: (x, y), 90: (y, 1 - x), 180: (1 - x, 1 - y), 270: (1 - y, x)}[span.rotation]

    points = [transform(x, y) for x, y in points]
    return (
        min(x for x, y in points),
        min(y for x, y in points),
        max(x for x, y in points),
        max(y for x, y in points),
    )


def adjacent(left, right):
    x0, y0, x1, y1 = rotated_box(left)
    a0, b0, a1, b1 = rotated_box(right)
    height = max(y1 - y0, b1 - b0)
    if abs((y0 + y1 - b0 - b1) / 2) <= height * 0.6:
        return -0.02 <= a0 - x1 <= 0.12
    overlap = max(0, min(x1, a1) - max(x0, a0))
    return overlap >= 0.25 * min(x1 - x0, a1 - a0) and -0.02 <= b0 - y1 <= max(0.08, 2 * height)


def title_windows(ocr, min_confidence):
    groups = defaultdict(list)
    for span in ocr.spans:
        if span.confidence >= min_confidence and title_tokens(span.text):
            groups[(span.rotation, span.crop_id)].append(span)
    windows = []
    for (rotation, crop), spans in groups.items():
        for start in range(len(spans)):
            chain = []
            for span in spans[start : start + 6]:
                if chain and not adjacent(chain[-1], span):
                    break
                chain.append(span)
                windows.append(
                    {"rotation": rotation, "crop_id": crop, "text": " ".join(s.text for s in chain)}
                )
    return windows


def title_support(claim, ocr, min_confidence):
    quoted = (
        claim.status == "present"
        and isinstance(claim.value, str)
        and any(token_contains(e.quote, claim.value) for e in claim.evidence)
    )
    windows = title_windows(ocr, min_confidence)
    if quoted:
        for evidence in claim.evidence:
            if not token_contains(evidence.quote, claim.value):
                continue
            for window in windows:
                if token_contains(window["text"], evidence.quote):
                    return {
                        "quoted": True,
                        "supported": True,
                        "matched_window": window,
                        "reason": "Ordered title words match one OCR region; punctuation ignored",
                    }
    return {
        "quoted": bool(quoted),
        "supported": False,
        "matched_window": None,
        "reason": "No supported quote in one rotation/region; word substitutions are rejected",
    }
