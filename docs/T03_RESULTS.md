# T03 preparation report

Prepared on 2026-10-29. **Ready for user execution; GPU feasibility gate not yet passed.**

Delivered `notebooks/00_pilot.ipynb`, model/runtime configs, locked candidate Linux GPU
requirements, real HF-native VLM adapters, rotation-aware CPU OCR, phased cached pilot
orchestration, two-step LoRA save/reload helpers, authenticated serving/gateway smoke
and a deterministic self-contained-code notebook builder. The revised notebook uses local runtime storage and image-only upload, with downloadable results and adapter.

Locally verified:

- 75 tests passed, including foundation/annotation regressions, phased role replay
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
