# T03 preparation report

Prepared on 2026-10-29. **Ready for user execution; GPU feasibility gate not yet passed.**

Delivered `notebooks/00_pilot.ipynb`, model/runtime configs, locked candidate Linux GPU
requirements, real HF-native VLM adapters, rotation-aware CPU OCR, phased cached pilot
orchestration, two-step LoRA save/reload helpers, authenticated serving/gateway smoke
and a deterministic source/input handoff builder.

Locally verified:

- 73 tests passed, including foundation/annotation regressions, phased role replay
  with injected test workers, HF token/revision propagation, safe JSON parsing,
  source-frame OCR boxes, strict prompt-prefix loss masking, gate logic, secret-safe
  export and authenticated endpoint behavior. One upstream TestClient deprecation
  warning remains; it did not fail a test.
- Notebook code cells compile and carry no execution outputs; GPU metadata is present.
- Ruff lint and formatting checks passed.
- The candidate GPU dependency graph resolves for Linux Python 3.11. LiteLLM required
  compatible FastAPI/Uvicorn pins; those conflicts were resolved before the handoff.
- Source/input bundle has per-member checksums and an exact SHA-256 in the notebook's
  bootstrap cell. No HF token, original image dataset, model weights or generated role
  outputs are included in committed source.

Not executed locally: CUDA model loading/generation, real OCR, LoRA updates, live
LiteLLM-to-GPU requests, or the real 8/12 title gate. The local adapter tests use
injected stubs; they do not substitute for the user's Colab run. The notebook must
produce its runtime reports before T04 is authorized as data-ready.

See [T03_COLAB.md](T03_COLAB.md) for the two-file handoff and run instructions. The
generated `artifacts/t03_handoff.json` records the matching archive checksum/size.
GPU weights and the smoke adapter remain on Drive; return `pilot_reports.zip` for
review. Nothing has been published to HF or deployed publicly by this preparation.
