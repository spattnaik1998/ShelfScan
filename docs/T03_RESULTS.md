# T03 preparation report

Prepared on 2026-10-29. **Ready for user execution; GPU feasibility gate not yet passed.**

Delivered `notebooks/00_pilot.ipynb`, model/runtime configs, locked candidate Linux GPU
requirements, real HF-native VLM adapters, rotation-aware CPU OCR, phased cached pilot
orchestration, two-step LoRA save/reload helpers, authenticated serving/gateway smoke
and a deterministic self-contained-code notebook builder. The revised notebook uses local runtime storage and image-only upload, with downloadable results and adapter.

Locally verified:

- 77 tests passed, including foundation/annotation regressions, phased role replay
  with injected test workers, HF token/revision propagation, safe JSON parsing,
  source-frame OCR boxes, strict prompt-prefix loss masking, gate logic, secret-safe
  export and authenticated endpoint behavior. One upstream TestClient deprecation
  warning remains; it did not fail a test.
- Notebook code cells compile and carry no execution outputs; GPU metadata is present.
- Ruff lint and formatting checks passed.
- The candidate GPU dependency graph resolves for Linux Python 3.11. LiteLLM required
  compatible FastAPI/Uvicorn pins; those conflicts were resolved before the handoff.
- Embedded supporting source has per-member checksums and an exact SHA-256 in the notebook's
  bootstrap cell; the image upload has its own checksum. No HF token, original image dataset, model weights or generated role
  outputs are included in committed source.

Not executed locally: CUDA model loading/generation, real OCR, LoRA updates, live
LiteLLM-to-GPU requests, or the real 8/12 title gate. The local adapter tests use
injected stubs; they do not substitute for the user's Colab run. The notebook must
produce its runtime reports before T04 is authorized as data-ready.

See [T03_COLAB.md](T03_COLAB.md) for direct notebook run instructions. The generated
`artifacts/t03_handoff.json` records embedded-source and image-input checksums. Download
the full results ZIP to save the adapter and progress, and return `pilot_reports.zip`
for review. No Drive or OneDrive mount is used. Nothing has been published to HF or deployed publicly by this preparation.

Installation follow-up: the user's Python 3.13 log reproduced a Pillow 13.0.0 versus
project Pillow<13 conflict. Fixed by resolving GPU requirements and pyproject.toml
together for Python 3.13, selecting Pillow 12.3.0. The original resolver conflict was
reproduced and the same resolution check passed after the fix. Added project/lockfile
compatibility and redacted-install-error regression checks; GPU execution remains pending.


## Returned GPU pilot and correction (2026-10-31)

Reviewed pilot_reports.zip and pilot-v1_results.zip: dependency install and 12 OCR
results succeeded, but Teacher A produced 0/12 valid candidates and Teacher B raised
ImportError. Later stages were blocked; no smoke adapter exists. T03 has not passed.

Corrected missing num2words dependency (required by the pinned SmolVLM processor),
selected native BF16 only (FP16 on T4), and replaced conflicting teacher instructions
with one explicit annotation envelope. Raw invalid responses and stage tracebacks are
redacted before saving and included in the small report bundle. Teacher passes stop
on a failed first-image check rather than spending GPU time on all 12 invalid responses.
Run the regenerated notebooks/00_pilot.ipynb with new run ID pilot-v2. These changes
need a new Colab execution; they do not establish the precise cause of the old invalid
responses, which the original reports discarded.

References: pinned SmolVLM processor source:
https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/smolvlm/processing_smolvlm.py
and PyTorch native/emulated BF16 API:
https://docs.pytorch.org/docs/main/generated/torch.cuda.is_bf16_supported.html


## pilot-v2 diagnostics and pilot-v3 correction

Returned diagnostics establish that some Qwen responses failed only on unsupported
model box formats (pixel arrays and text labels), while SmolVLM generated a truncated
repetitive string. The earlier import error no longer occurs in the supplied trace.
Replaced the all-unknown example with compact mixed-status field examples. Teachers
return status/value/quote; deterministic conversion retains strict value validation
and supplies evidence with null boxes. Unsupported legacy model localization is
explicitly discarded, not converted into guessed coordinates. Truncated JSON remains
invalid. Added configurable repetition penalty and smaller teacher output budgets.
The rebuilt notebook uses pilot-v3 and requires both cached one-image checks before
full passes. This does not establish live model quality or completion of T03.

Local verification after pilot-v3 changes: 86 tests passed; Ruff lint/format passed.
All four supplied Qwen response payloads validate after dropping unsupported model
boxes; the supplied truncated SmolVLM payload remains correctly rejected. Actual
SmolVLM regeneration and the complete pilot gate await the user GPU run.
