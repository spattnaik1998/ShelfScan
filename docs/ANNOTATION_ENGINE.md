# Running the T02 annotation engine

T02 implements the CPU/offline orchestration and its adapter interfaces. The only CLI
backend is **fake**. It deliberately generates invented fixture text rather than reading
the books. Real OCR, vision models and GPU execution belong to the T03 Colab pilot.

## Run the prepared pilot

Prepare T01 inputs first as described in [README](../README.md). Then, from ShelfScan:

```powershell
uv sync --locked
uv run python -m scripts.pilot --config configs/annotation.yaml --backend fake
```

Repeat the command to demonstrate resume. With the shipped configuration, a fresh run
has 12 simulated accepted records, 48 calls and no pending records. A repeat has zero
new calls and 48 cache hits. Teacher B deliberately disagrees about the fixture title,
so every image exercises selective adjudication. This is a control-flow demonstration,
not an annotation accuracy result or fulfillment of the real-model pilot gate.

## Input and config contract

The default input is T01's `artifacts/pilot_inputs.zip`; there is no manual extraction
step. The loader verifies member hashes, prepared image dimensions, prompt hash, split
hash and training-only pilot membership before invoking workers. IDs sent to workers
are opaque hashes. Category/group/split metadata goes only to curation/reporting.

To simulate the full prepared inventory, copy `configs/annotation.yaml` to another
file directly under `configs/`, and change:

```yaml
input_kind: inventory
input_path: data/inventory.jsonl
image_root: data
output_root: data/annotation_full_fake
```

Run it with `uv run python -m scripts.annotate --config configs/<copy>.yaml --backend fake`.
All paths resolve against the project directory containing `configs/`. Inputs and
outputs must remain inside the project. Data preparation owns access to originals;
the annotation runner uses prepared copies only. This full fake run costs no API calls.

## Worker boundaries and decisions

- `contracts.py`: typed boxes, OCR spans, role inputs, claims and decisions; candidate
  fields exclude category and distinguish present/absent/unknown.
- `ocr.py`, `teachers.py`: replaceable worker protocols and synthetic fixtures. Workers
  receive bytes, dimensions, opaque IDs and task prompts; teacher B never receives A's
  response. No browsing, shell or credential tools are exposed to workers.
- `verify.py`: normalized title/ordered-author comparison, exact other-field comparison,
  quote/value containment and OCR support. Unsupported agreements remain unresolved.
- `adjudicate.py`: logged seeded anonymous ordering, one disputed-field resolution round;
  undisputed fields cannot be overwritten. Visual-only resolutions require quotes and
  valid normalized image boxes and retain a lower evidence tier.
- `curate.py`: unresolved title or any optional unknown excludes the whole example.
  Supported absence alone becomes null (or an empty author list). No partial-label
  loss masking is enabled in this MVP.
- `cache.py`, `orchestrator.py`: durable attempts/results, bounded execution, curation
  and coverage reporting. A role failure leaves unresolved fields and quarantine reasons.

OCR quotes and model-proposed boxes are fallible evidence, not proof of correctness.
Adjudication is selective model inference, not a guarantee that a disagreement is solved.

## Cache, retry and budget semantics

Runs are isolated by input/split hashes, effective policy/model config and source-file
hashes. Role keys also include verified image bytes' digest, model revision/parameters,
request hash and response schema hash. A model/prompt/source change creates new work;
raising an execution budget does not invalidate successful cached work.

The runner allows one writer per run and saves an attempt start **before** invoking its
worker. At most two attempts per role/image are permitted, including interrupted calls.
Only retryable technical failures get the second attempt; malformed responses are not
repeated. Failed/exhausted calls remain exhausted across restarts. A corrupt cache fails
closed instead of silently reinvoking models. Role latency and error codes are recorded;
arbitrary provider exception text is not copied into diagnostics.

`max_total_attempts` counts the durable ledger across resumes. `max_seconds` limits
admission of new calls per invocation; it does not preempt a synchronous worker already
running. Real T03 adapters must implement their own bounded inference/request timeouts.
On budget exhaustion, remaining inputs are pending, exports are saved and the CLI exits
with code 2. Increase the relevant budget in config and rerun to continue. Successful
calls are reused. A crash after a model call but before its result is saved may consume
the one remaining attempt; exactly-once external billing is not promised.

An abrupt process termination can leave `.writer.lock`. Confirm no worker for that run
is active before removing that stale lock. The runner never removes another writer's
lock automatically. Reports are rebuilt from checksummed role state, not appended as
duplicate decisions. `decisions.json` is the atomic checkpoint; JSONL/Markdown exports
are regenerated on resume and can be checked against `artifact_checksums.json`.

## Outputs

Under the configured ignored output root, each run has its own digest-named directory:

| Artifact | Contents |
|---|---|
| `input_manifest.json` | Input/source/prompt/run hashes, source IDs and partition metadata |
| `raw/<role-key>.json` | Checksummed role result, model provenance, attempt history and timings |
| `decisions.json`, `decisions.jsonl` | Every input's accepted/quarantined/pending disposition |
| `quarantine.jsonl` | Excluded examples and unresolved-field reasons |
| `simulated_labels.jsonl` | Fake accepted records, always marked ineligible for training |
| `labels.jsonl` | Intentionally empty for the fake backend |
| `run_manifest.json`, `annotation_report.md` | Counts, evidence tiers, field unknowns and coverage by category/split |
| `artifact_checksums.json` | Checksums of exported reports; raw role files have individual checksums |

Every fake decision carries `is_synthetic: true` and `training_eligible: false`. Neither
successful fake reports nor the 60-training-group gate should be used to claim training
readiness. Token/GPU metrics remain null; billed API cost is zero for these fixtures.

Run `uv run pytest -q`, `uv run ruff check .` and `uv run ruff format --check .` to
validate locally. If Windows restricted-runner permissions prevent atomic replacement
of temporary test files, use a normal local terminal; the production atomic-save path
must not be weakened to work around that environment restriction.
