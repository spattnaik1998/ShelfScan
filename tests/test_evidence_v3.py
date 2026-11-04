import pytest

from annotation.contracts import Box, OCRResult, OCRSpan


def test_evidence_notebook_contains_current_source_and_compiles():
    import ast
    import base64
    import io
    import json
    import zipfile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/evidence_partial_pilot.ipynb").read_bytes())
    cells = notebook["cells"]
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{i + 1}", "exec")
            assert not cell["outputs"] and cell["execution_count"] is None
    tree = ast.parse("".join(cells[3]["source"]))
    encoded = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "SOURCE_B64" for t in node.targets)
    )
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as archive:
        for name in [
            "training/partial.py",
            "training/evidence_pilot.py",
            "training/evidence_replay.py",
            "training/selector.py",
            "annotation/evidence.py",
            "training/smoke.py",
        ]:
            assert archive.read(name) == (root / name).read_bytes()
    assert "EvidencePilot" in "".join(cells[10]["source"])
    assert "pilot-recovery-v2" in "".join(cells[9]["source"])


def ocr(*lines):
    return OCRResult(
        spans=[
            OCRSpan(
                text=line,
                confidence=0.95,
                box=Box(x0=0.1, y0=0.1 + i * 0.06, x1=0.9, y1=0.15 + i * 0.06),
            )
            for i, line in enumerate(lines)
        ]
    )


def test_missing_quote_can_get_real_ocr_evidence_without_changing_value():
    from annotation.evidence import ground, proposal

    claim = proposal("title", {"status": "present", "value": "Deep Work", "quote": ""})
    attached = ground("title", claim, ocr("DEEP WORK"), 0.5)
    assert attached["supported"]
    assert attached["claim"].value == "Deep Work"
    assert attached["claim"].evidence[0].quote == "DEEP WORK"
    assert attached["claim"].evidence[0].box is not None


def test_wrong_title_and_currency_stay_unsupported():
    from annotation.evidence import ground, proposal

    assert not ground(
        "title",
        proposal("title", "The Administrator's Act 1985"),
        ocr("The Administrative Tribunals Act, 1985"),
        0.5,
    )["supported"]
    assert not ground("price", proposal("price", "$60.00"), ocr("Price \u20b960.00"), 0.5)[
        "supported"
    ]


def test_null_or_unknown_does_not_become_absence():
    from annotation.evidence import proposal

    for value in [None, "unknown", ["unknown"], {"status": "unknown", "value": None, "quote": ""}]:
        assert proposal("year", value).status == "unknown"


def test_selector_cannot_invent_values_or_accept_unoffered_ids():
    from training.selector import parse_choices

    assert parse_choices('{"title":"C0"}', {"C0": {}}) == "C0"
    assert parse_choices('{"title":"UNKNOWN"}', {"C0": {}}) is None
    for text in ['{"title":"New Title"}', '{"title":"C9"}', '{"title":"C0","year":2024}']:
        with pytest.raises(ValueError):
            parse_choices(text, {"C0": {}})


def test_partial_training_refuses_absent_title():
    from training.partial import render_target

    with pytest.raises(ValueError, match="accepted title"):
        render_target(
            {"category": "fiction", "fields": {"title": {"status": "absent", "value": None}}}
        )


def test_full_token_alignment_and_unknown_mask_without_external_artifacts():
    from types import SimpleNamespace

    from training.partial import partial_labels, render_target

    row = {"category": "fiction", "fields": {"title": {"status": "accepted", "value": "Deep Work"}}}
    target, _ = render_target(row)

    class Characters:
        is_fast = True
        all_special_ids = []

        def decode(self, ids, **kwargs):
            return "".join(map(chr, ids))

        def __call__(self, text, **kwargs):
            return {
                "input_ids": list(map(ord, text)),
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    processor = SimpleNamespace(tokenizer=Characters())
    ids = list(map(ord, "image prompt\nassistant\n" + target))
    labels, audit = partial_labels(processor, [], ids, row)
    visible = "".join(chr(x) for x in labels if x != -100)
    assert "Deep Work" in visible and "authors" not in visible and "null" not in visible
    assert audit["unknown_supervised_tokens"] == 0
    with pytest.raises(RuntimeError, match="uniquely aligned"):
        partial_labels(processor, [], list(map(ord, target + target)), row)


def test_masking_hides_every_unknown_value_and_boundary_token():
    from training.partial import render_target, token_mask

    row = {
        "category": "fiction",
        "fields": {
            "title": {"status": "accepted", "value": "Deep Work"},
            "authors": {"status": "unknown", "value": None},
        },
    }
    text, visible = render_target(row)
    assert not any(visible[text.index('"authors"') : text.index('"publisher"') - 1])
    offsets = [(i, i + 1) for i in range(len(text))]
    mask = token_mask(text, visible, offsets)
    assert mask[text.index("Deep Work")]
    assert not mask[text.index("null")]
    start = text.index('"authors"')
    assert not token_mask(text, visible, [(start - 1, start + 3)])[0]


def test_unknown_and_absent_authors_have_different_loss_despite_same_placeholder():
    from training.partial import render_target

    row = {
        "category": "fiction",
        "fields": {
            "title": {"status": "accepted", "value": "Deep Work"},
            "authors": {"status": "unknown", "value": None},
        },
    }
    first, masked = render_target(row)
    row["fields"]["authors"]["status"] = "absent"
    second, supervised = render_target(row)
    assert first == second
    start = first.index('"authors"')
    end = first.index('"publisher"') - 1
    assert not any(masked[start:end]) and all(supervised[start:end])


def test_serialized_control_token_stays_literal_data():
    import json

    from training.partial import render_target

    row = {
        "category": "fiction",
        "fields": {"title": {"status": "accepted", "value": "<|im_end|>"}},
    }
    text, mask = render_target(row)
    assert "<|im_end|>" not in text
    assert json.loads(text)["title"] == "<|im_end|>"


@pytest.mark.parametrize("title", ["Deep Work", "FranÃ§a's guide", "<|im_end|>"])
def test_real_qwen_tokenizer_masks_unknowns_and_image_prompt_tokens(title):
    from pathlib import Path
    from types import SimpleNamespace

    from tokenizers import Tokenizer

    from training.partial import partial_labels, render_target

    path = Path("artifacts/qwen_tokenizer/tokenizer.json")
    if not path.exists():
        pytest.skip("Optional pinned Qwen tokenizer artifact is not available")
    rust = Tokenizer.from_file(str(path))

    class Fast:
        is_fast = True
        all_special_ids = [
            rust.token_to_id(t)
            for t in [
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
            ]
        ]

        def decode(self, ids, **kwargs):
            return rust.decode(ids, skip_special_tokens=False)

        def __call__(self, text, **kwargs):
            encoded = rust.encode(text, add_special_tokens=False)
            return {"input_ids": encoded.ids, "offset_mapping": encoded.offsets}

    row = {
        "category": "fiction",
        "fields": {
            "title": {"status": "accepted", "value": title},
            "authors": {"status": "unknown", "value": None},
        },
    }
    target, characters = render_target(row)
    prefix = (
        "<|im_start|>user\n<|vision_start|>"
        + "<|image_pad|>" * 20
        + "<|vision_end|>Extract metadata<|im_end|>\n<|im_start|>assistant\n"
    )
    ids = rust.encode(prefix + target + "<|im_end|>", add_special_tokens=False).ids
    labels, audit = partial_labels(SimpleNamespace(tokenizer=Fast()), [], ids, row)
    text = rust.decode([label for label in labels if label != -100], skip_special_tokens=False)
    assert "null" not in text and "authors" not in text
    assert all(
        label == -100
        for label, original in zip(labels, ids, strict=True)
        if original in Fast.all_special_ids
    )
    assert audit["unknown_supervised_tokens"] == 0 and audit["title_supervised_tokens"] > 0
