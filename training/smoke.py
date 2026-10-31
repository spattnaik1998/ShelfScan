"""Two-step LoRA feasibility check with assistant-only loss and clean adapter reload."""

import json
import math
from pathlib import Path

from annotation.cache import atomic_json
from shelfscan.prompt import EXTRACT_PROMPT
from training.model_io import VisionSession, make_batch
from training.runtime import native_bf16_supported


def target_start(prompt_ids: list[int], full_ids: list[int]) -> int:
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("Chat template prompt is not a prefix; refusing unsafe loss masking")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("No supervised assistant tokens")
    return len(prompt_ids)


def run_smoke(spec, examples: list[tuple[bytes, dict]], output: Path, settings: dict) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not examples:
        raise RuntimeError("No complete accepted pilot records for safe smoke supervision")
    session = VisionSession(spec).load()
    adapter = output / "smoke_adapter"
    raw, record = examples[0]
    before = session.generate(raw, EXTRACT_PROMPT)
    baseline = {"text": before, "note": "Training-only pilot baseline; not a held-out score"}
    atomic_json(output / "baseline.json", baseline)
    try:
        if spec.parameters.get("quantize", False):
            session.model = prepare_model_for_kbit_training(session.model)
        # Restrict LoRA to language attention, excluding vision layers with similar names.
        targets = [
            name
            for name, _ in session.model.named_modules()
            if name.endswith(("q_proj", "v_proj")) and "visual" not in name and "vision" not in name
        ]
        if not targets:
            raise RuntimeError("No supported language attention targets found")
        session.model = get_peft_model(
            session.model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        session.model.config.use_cache = False
        session.model.gradient_checkpointing_enable()
        session.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in session.model.parameters() if p.requires_grad],
            lr=settings["smoke_learning_rate"],
        )
        scaler = torch.amp.GradScaler("cuda", enabled=not native_bf16_supported(torch))
        losses = []
        dtype = torch.bfloat16 if native_bf16_supported(torch) else torch.float16
        for step in range(settings["smoke_steps"]):
            raw, record = examples[step % len(examples)]
            target = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            prompt = make_batch(session.processor, raw, EXTRACT_PROMPT)
            batch = make_batch(session.processor, raw, EXTRACT_PROMPT, target)
            start = target_start(prompt["input_ids"][0].tolist(), batch["input_ids"][0].tolist())
            if batch["input_ids"].shape[1] > settings["max_training_tokens"]:
                raise RuntimeError("Training example exceeds token budget; no unsafe truncation")
            labels = batch["input_ids"].clone()
            labels[:, :start] = -100
            labels[batch["attention_mask"] == 0] = -100
            if int((labels != -100).sum()) == 0:
                raise RuntimeError("No assistant supervision")
            batch = batch.to(session.model.device)
            labels = labels.to(session.model.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                loss = session.model(**batch, labels=labels).loss
            if not math.isfinite(loss.item()):
                raise RuntimeError("Non-finite smoke loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(session.model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        session.model.eval()
        session.model.config.use_cache = True
        expected = session.generate(raw, EXTRACT_PROMPT)
        session.model.save_pretrained(adapter)
        session.processor.save_pretrained(adapter)
        atomic_json(
            adapter / "shelfscan_manifest.json",
            {
                "base": spec.model_dump(mode="json"),
                "scope": "pilot smoke only; not final model",
            },
        )
        # Release optimizer and graph tensors too; otherwise reload can falsely OOM.
        del optimizer, batch, prompt, labels, loss, scaler
        session.close()
        reloaded = VisionSession(spec).load()
        try:
            reloaded.load_adapter(adapter)
            actual = reloaded.generate(raw, EXTRACT_PROMPT)
        finally:
            reloaded.close()
        report = {
            "ok": expected == actual,
            "steps": len(losses),
            "losses": losses,
            "same_decoded_output_after_reload": expected == actual,
            "adapter_directory": "smoke_adapter",
            "base_revision": spec.revision,
        }
        atomic_json(output / "smoke_reload_report.json", report)
        return report
    finally:
        session.close()
