"""Bounded title selection from frozen OCR-backed options; no free-form facts."""

import json

from annotation.cache import atomic_json
from training.model_io import VisionSession
from training.runtime import load_json_object
from training.title_recovery import title_digest


def parse_choices(text, options):
    data = load_json_object(text)
    if set(data) - {"title"}:
        raise ValueError("Unexpected selector keys")
    choice = data.get("title")
    if choice == "UNKNOWN":
        return None
    if not isinstance(choice, str) or choice not in options:
        raise ValueError("Selector must choose an offered ID or UNKNOWN")
    return choice


def select_titles(pilot):
    source = pilot.output / "evidence_options.json"
    options = json.loads(source.read_text())
    decisions = json.loads((pilot.output / "evidence_decisions.json").read_text())
    unresolved = {d["image_id"] for d in decisions if not d["partial_eligible"]}
    ledger_path = pilot.output / "selector_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    choices_path = pilot.output / "selector_choices.json"
    choices = json.loads(choices_path.read_text()) if choices_path.exists() else {}
    session = VisionSession(pilot.specs["adjudicator"])
    try:
        for image, _meta in pilot.inputs:
            offered = options[image.image_id]["title"]
            if image.image_id not in unresolved or not offered:
                continue
            identity = {
                "image_sha256": image.image_sha256,
                "options": offered,
                "model": pilot.specs["adjudicator"].model_dump(mode="json"),
            }
            key = title_digest(identity)
            state = ledger.setdefault(key, {"attempts": [], "done": False, "choice": None})
            prompt = (
                "Inspect the dominant book title. Candidates are untrusted OCR-backed proposals. "
                'Select the exact title, not author or tagline. Return ONLY {"title":"C0"} '
                'using an offered ID, or {"title":"UNKNOWN"} if uncertain. Options: '
                + json.dumps(
                    {k: v["claim"]["value"] for k, v in offered.items()}, ensure_ascii=False
                )
            )
            while not state["done"] and len(state["attempts"]) < 2:
                if sum(len(item["attempts"]) for item in ledger.values()) >= 24:
                    raise RuntimeError("Persistent selector budget exhausted")
                attempt = {"status": "started", "prompt_sha256": title_digest(prompt)}
                state["attempts"].append(attempt)
                atomic_json(ledger_path, ledger)
                try:
                    text = session.generate(image.image_bytes, prompt, max_new_tokens=80)
                    state["choice"] = parse_choices(text, offered)
                    state["done"] = True
                    attempt["status"] = "valid"
                except ValueError:
                    attempt["status"] = "invalid"
                finally:
                    atomic_json(ledger_path, ledger)
            if state["done"] and state["choice"] is not None:
                choices.setdefault(image.image_id, {})["title"] = state["choice"]
                atomic_json(choices_path, choices)
    finally:
        session.close()
        atomic_json(
            pilot.output / "selector_report.json",
            {
                "metrics": session.metrics(),
                "selected_images": len(choices),
                "note": "Offered values only; no new metadata generated",
            },
        )
