"""Typed proposals plus deterministic OCR attachment; evidence is never invented."""

import json
from collections import defaultdict

from annotation.contracts import FIELDS, Box, Candidate, Evidence, FieldClaim
from annotation.title_evidence import adjacent, title_tokens
from annotation.verify import contains


def abstain(reason="No unambiguous metadata proposal"):
    return FieldClaim(status="unknown", value=None, reason=reason)


def proposal(name, raw):
    if isinstance(raw, list) and len(raw) == 1 and name != "authors":
        raw = raw[0]
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        raw = raw[0]
    if name == "other_text" and isinstance(raw, list) and all(isinstance(v, str) for v in raw):
        raw = "\n".join(raw)
    if isinstance(raw, list) and all(
        isinstance(x, str) and x.strip().casefold() in ("unknown", "null", "") for x in raw
    ):
        return abstain()
    if raw is None or (isinstance(raw, str) and raw.strip().casefold() in ("unknown", "null", "")):
        return abstain()
    status = "present"
    if isinstance(raw, dict):
        status = raw.get("status", "unknown")
        value = raw.get("value")
    else:
        value = raw
    if status in ("unknown", "absent"):
        if value is not None:
            return abstain("Contradictory status/value")
        return FieldClaim(status=status, value=None, reason="Explicit model abstention/absence")
    if name == "authors" and isinstance(value, str):
        # A single credit can be represented as one name; never split ambiguous strings.
        if any(separator in value.casefold() for separator in (",", " and ", "&", ";")):
            return abstain("Ambiguous author segmentation")
        value = [value]
    if (
        name == "year"
        and isinstance(value, str)
        and value.isascii()
        and len(value) == 4
        and value.isdigit()
    ):
        value = int(value)
    try:
        claims = {field: abstain() for field in FIELDS}
        claims[name] = FieldClaim(
            status="present", value=value, reason="Model value proposal; evidence required"
        )
        return Candidate(fields=claims).fields[name]
    except (ValueError, TypeError):
        return abstain("Invalid proposed field type")


def tokens(text):
    # Match written initials without fuzzy spelling or digit edits.
    out = []
    initials = []
    for word in title_tokens(text):
        if len(word) == 1 and word.isalpha():
            initials.append(word)
        else:
            if initials:
                out.append("".join(initials))
                initials = []
            out.append(word)
    if initials:
        out.append("".join(initials))
    return out


def value_contains(text, value):
    left, right = tokens(text), tokens(value)
    return bool(right) and any(
        left[i : i + len(right)] == right for i in range(len(left) - len(right) + 1)
    )


def regions(ocr, minimum):
    groups = defaultdict(list)
    for index, span in enumerate(ocr.spans):
        if span.confidence >= minimum and tokens(span.text):
            groups[(span.rotation, span.crop_id)].append((index, span))
    for (rotation, crop), spans in groups.items():
        for start in range(len(spans)):
            chain = []
            for index, span in spans[start : start + 6]:
                if chain and not adjacent(chain[-1][1], span):
                    break
                chain.append((index, span))
                yield {
                    "rotation": rotation,
                    "crop_id": crop,
                    "span_indices": [i for i, s in chain],
                    "text": " ".join(s.text for i, s in chain),
                    "box": Box(
                        x0=min(s.box.x0 for i, s in chain),
                        y0=min(s.box.y0 for i, s in chain),
                        x1=max(s.box.x1 for i, s in chain),
                        y1=max(s.box.y1 for i, s in chain),
                    ),
                }


def ground(name, claim, ocr, minimum=0.5):
    if claim.status != "present":
        return {"supported": False, "claim": claim, "provenance": None}
    values = claim.value if isinstance(claim.value, list) else [str(claim.value)]
    for region in regions(ocr, minimum):
        matches = all(
            (
                contains(region["text"], value)
                if name in ("price", "year")
                else value_contains(region["text"], value)
            )
            for value in values
        )
        if matches:
            evidence = Evidence(quote=region["text"], box=region["box"])
            result = claim.model_copy(
                update={
                    "evidence": [evidence],
                    "reason": "Evidence attached from a matching OCR region",
                }
            )
            return {
                "supported": True,
                "claim": result,
                "provenance": {
                    "method": "ocr_region_v1",
                    "rotation": region["rotation"],
                    "crop_id": region["crop_id"],
                    "span_indices": region["span_indices"],
                    "quote": region["text"],
                    "box": region["box"].model_dump(mode="json"),
                },
            }
    return {"supported": False, "claim": claim, "provenance": None}


def value_key(claim):
    if claim.status != "present":
        return claim.status
    values = claim.value if isinstance(claim.value, list) else [str(claim.value)]
    return json.dumps([tokens(value) for value in values], ensure_ascii=False)
