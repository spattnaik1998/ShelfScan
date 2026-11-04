"""Token-aligned partial supervision; unknown metadata never supervises null."""

import json

from annotation.contracts import FIELDS
from shelfscan.schemas import BookRecord


def render_target(example):
    fields = example["fields"]
    values = {name: None for name in FIELDS}
    values["authors"] = []
    known = {"category"}
    for name, decision in fields.items():
        if decision["status"] in ("accepted", "absent"):
            values[name] = (
                [] if name == "authors" and decision["status"] == "absent" else decision["value"]
            )
            known.add(name)
    if fields.get("title", {}).get("status") != "accepted":
        raise ValueError("Partial supervision requires an accepted title")
    values["category"] = example["category"]
    record = BookRecord.model_validate(values).model_dump()
    parts = ["{"]
    mask = [True]
    for index, (name, value) in enumerate(record.items()):
        if index:
            parts.append(",")
            mask.append(True)
        text = json.dumps(name) + ":" + json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        text = text.replace("<", "\\u003c").replace(">", "\\u003e")
        parts.append(text)
        mask.extend([name in known] * len(text))
    parts.append("}")
    mask.append(True)
    return "".join(parts), mask


def token_mask(text, character_mask, offsets):
    if len(text) != len(character_mask):
        raise ValueError("Character mask mismatch")
    result = []
    for start, end in offsets:
        if not 0 <= start <= end <= len(text):
            raise ValueError("Invalid tokenizer offset")
        # Mask a token if ANY portion belongs to an unknown field.
        result.append(end > start and all(character_mask[start:end]))
    return result


def partial_labels(processor, prompt_ids, full_ids, example):
    text, characters = render_target(example)
    tokenizer = processor.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Partial loss masking requires a fast tokenizer with offsets")
    decoded = tokenizer.decode(
        full_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    encoded = tokenizer(decoded, add_special_tokens=False, return_offsets_mapping=True)
    if encoded["input_ids"] != full_ids:
        raise RuntimeError("Full sequence token roundtrip failed; refusing unsafe supervision")
    if decoded.count(text) != 1:
        raise RuntimeError("Assistant target is not uniquely aligned; refusing unsafe supervision")
    at = decoded.index(text)
    labels = [-100] * len(full_ids)
    supervised = 0
    title_tokens = 0
    unknown_supervised = 0
    title_start = len('{"title":')
    title_end = title_start + len(
        json.dumps(example["fields"]["title"]["value"], ensure_ascii=False)
    )
    special = set(getattr(tokenizer, "all_special_ids", []))
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if not (0 <= start <= end <= len(decoded)):
            raise RuntimeError("Tokenizer offsets are outside the decoded input")
        if not (at <= start < end <= at + len(text)) or full_ids[index] in special:
            continue
        lo, hi = start - at, end - at
        if all(characters[lo:hi]):
            labels[index] = full_ids[index]
            supervised += 1
            if max(lo, title_start) < min(hi, title_end):
                title_tokens += 1
        if labels[index] != -100 and not all(characters[lo:hi]):
            unknown_supervised += 1
    if not title_tokens:
        raise RuntimeError("No accepted title-value supervision survives masking")
    return labels, {
        "target_characters": len(text),
        "supervised_tokens": supervised,
        "title_supervised_tokens": title_tokens,
        "unknown_supervised_tokens": unknown_supervised,
        "prompt_and_image_masked": True,
        "alignment_verified": True,
        "policy": "partial-fields-v1",
    }
