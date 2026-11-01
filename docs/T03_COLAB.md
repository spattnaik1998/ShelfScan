# T03: run the notebook without mounted storage

The notebook runs directly in Colab. **No Drive/OneDrive mount, repository clone, or
uploaded code archive is required.** Supporting Python code is embedded in the setup
cell and extracted automatically into `/content/shelfscan_t03/`.

## Run

1. Open `notebooks/00_pilot.ipynb` in Colab and select a GPU runtime (T4 16GB or better).
2. Add `HF_TOKEN` in Colab Secrets and enable notebook access. A read token is enough.
3. Run the cells in order. When prompted, upload only `artifacts/pilot_inputs.zip` from
   your computer. This is the 12-image dataset package, not project code.
4. If dependency installation asks for a session restart, restart and rerun the cells.
5. The final cells download a small `pilot_reports.zip` and a full results ZIP. Save both.
   The full ZIP includes `smoke_adapter/` with model adapter weights, processor/tokenizer
   and base-model revision, plus annotation progress, decisions and runtime reports.
6. Return `pilot_reports.zip` to Codex before proceeding to T04.

Results are saved under `/content/shelfscan_results/<RUN_ID>/`. This is temporary runtime
storage; download your results before ending the runtime. Completed calls are cached
within that runtime. Nothing is automatically synced to cloud storage or published to HF.
The saved adapter references the pinned HF base model instead of duplicating its weights.
Private installation/gateway logs are excluded from the downloadable results ZIP.

## What runs

- Checksummed supporting code setup and pilot image upload/verification.
- Pinned GPU dependency installation; HF authentication and GPU/precision preflight.
- One-time resolution of immutable HF model revisions.
- CPU EasyOCR over four image rotations, followed by sequential independent VLM passes.
- Evidence verification, selective adjudication and cached replay check.
- Student baseline, two-step LoRA smoke training and fresh adapter/processor reload.
- Authenticated loopback model endpoint and LiteLLM gateway check, with shutdown.
- Report and complete-results downloads, including partial results when a stage fails.

Teacher A/adjudicator: `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit. Teacher B:
`HuggingFaceM4/Idefics3-8B-Llama3`, 4-bit. Student: `Qwen/Qwen2.5-VL-3B-Instruct`,
4-bit. The HF token authenticates downloads; inference/training uses the Colab GPU,
without paid HF hosted inference. EasyOCR downloads its public OCR weights separately.
Allow tens of GB of model/package downloads and disk space. Read model terms on HF if
access approval is required. Model licenses are captured rather than assumed.

The candidate GPU environment is pinned in `requirements/colab.lock.txt`, resolved for
Linux Python 3.13 together with the project constraints in `pyproject.toml`; its actual
GPU compatibility must be established by your execution. Local checks do not validate
CUDA kernels, real OCR quality or model training. The gate requires all stages, at least
8/12 accepted titles, cached replay, adapter reload and authenticated gateway success.
Complete records are counted separately; unresolved fields exclude a record from training.
A failed stage remains failed in the returned report. No manual annotation or paid fallback
is triggered automatically. This is a feasibility smoke adapter, not a final trained model.

## Rebuild locally

```text
uv run python -m scripts.build_colab_handoff
uv run ruff check notebooks --fix
uv run ruff format notebooks
```

The builder embeds only supporting source/config and split metadata, not image data or
credentials. It pins the embedded source checksum and the separate image-upload checksum.
The canonical notebook is `notebooks/00_pilot.ipynb`.

## Source references

The loader/processor choices follow the versioned [Qwen2.5-VL](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
and [SmolVLM](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/smolvlm) documentation.
The 4-bit adapter preparation follows [PEFT quantization guidance](https://huggingface.co/docs/peft/en/developer_guides/quantization).
The OCR adapter uses the [EasyOCR API](https://www.jaided.ai/easyocr/documentation/), and
gateway aliases/key references use the [LiteLLM config contract](https://docs.litellm.ai/docs/proxy/configs).
These sources establish API patterns, not successful execution of this pilot on Colab.

## Installation conflict fixed

The supplied Colab log identified Pillow 13.0.0 conflicting with ShelfScan's Pillow<13
requirement. The lock now pins Pillow 12.3.0 and includes the project dependency bounds.
Regenerate it with both inputs, never the GPU requirements alone:

```text
uv pip compile requirements/colab.in pyproject.toml --python-version 3.13 --python-platform x86_64-unknown-linux-gnu --only-binary :all: --no-binary docopt --output-file requirements/colab.lock.txt
```

Open the updated notebook and rerun setup to load its corrected embedded environment.
An already uploaded Colab notebook does not update automatically when local files change.
Installation failures now display a redacted pip log tail and save
`install_error.redacted.txt`, rather than hiding the actual error in a private log.


## Corrected pilot rerun

Open the freshly updated `notebooks/00_pilot.ipynb` in a fresh Colab GPU session,
not the previous uploaded copy. Keep its new `RUN_ID = 'pilot-v3'`; run cells from
the top and upload the same pilot_inputs.zip. No Drive mount is needed. T4 now uses
FP16; SmolVLM's required num2words dependency is included. A failed first-image
teacher check stops that teacher pass and saves sanitized diagnostics. Download
pilot_reports.zip and pilot-v3_results.zip at the end, including on failure.


### pilot-v3 teacher checks

Run the updated notebook from the top in a fresh GPU session. After OCR, run the
one-image teacher_a_check and teacher_b_check cells. Both must return True before
running the full teacher passes; successful checks are cached and reused. If either
fails, skip the full passes and use the final export cells to return pilot_reports.zip.
The compact teacher format omits boxes/reasons; Python builds strict claims with
quote evidence and null model boxes. OCR localization remains unchanged. Legacy
model boxes are discarded without guessing coordinates. Non-null unknown values,
malformed JSON and truncated responses remain rejected. Repetition penalty 1.1 and
a 1200-token teacher limit reduce runaway output; GPU behavior still needs validation.


### pilot-v4 parser compatibility

The latest notebook accepts either the fields envelope or exactly the seven named
fields at the top level. No missing/extra keys or contradictory claims are repaired.
Use the rebuilt notebook with pilot-v4 from the top; the new run avoids the previous
terminal failure cache. Both one-image checks still precede the full passes.


### Current handoff: pilot-v5 (field-by-field Teacher B)

Open the freshly rebuilt notebooks/00_pilot.ipynb in a fresh Colab GPU session and
run from the top with RUN_ID pilot-v5. Upload the same pilot_inputs.zip; no Drive
mount or code upload is needed. The previous paused notebook embeds older helpers;
continuing its cells will not load this fix.

Teacher A keeps the strict flat/wrapped parser. Teacher B now requests seven separate
claims with <=384 generated tokens per call, a soft 35-second generation limit,
and at most two attempts per field (initial call plus one validation correction).
The 12-image pilot has a persistent additional ceiling of 168 field generations.
This costs more model calls than the previous single-response approach. Calls are
local GPU inference, not paid HF inference. GPU operation remains user-verified.

Both one-image checks must return True before the full passes. Teacher B prints
field progress and reuses validated field results on rerun. Missing/invalid fields
are never silently changed into absent or unknown labels. The new output directory
avoids terminal errors cached by earlier code. Historical attempt failures stay in
teacher_b_diagnostics.json; teacher_b_field_report.json reports current valid and
unresolved counts. Teacher A diagnostics are separate. Successfully rerun stages
clear their stale stage_errors entry.

If either check fails, skip the full passes and use the final export cells. Download
pilot_reports.zip and pilot-v5_results.zip and return them for review. Full results
include durable teacher_b_fields.json; small reports include role-specific diagnostics
and field counts. No adapter or title-coverage success is implied by parser tests.


### Current handoff: pilot-v6 - replacement Teacher B

Teacher B is HuggingFaceM4/Idefics3-8B-Llama3 (Llama-backed), replacing the failed
SmolVLM default. It uses native Transformers 4.57.1, trust_remote_code=False, NF4
4-bit weights, native precision selection (FP16 on T4), SDPA, and a 728px longest-edge
processor limit to bound image tiles. No new package installation or gated fallback
is added. HF_TOKEN authenticates model downloads; inference uses the Colab GPU.
Allow a new multi-GB weight download and extra runtime disk space. Actual GPU memory
and output quality must be measured in your run; neither is claimed proven locally.

Open the updated notebooks/00_pilot.ipynb in a fresh GPU session with pilot-v6 and run
from the top. Upload the same pilot_inputs.zip. Teacher A retains its one-image check.
The existing teacher_b_check cell now benchmarks three images: all three outputs must
validate and at least two titles must have matching visible quotes supported by OCR.
This is a feasibility proxy, not human-ground-truth accuracy. Both checks must pass
before full teacher passes. All 12 inputs, >=8 accepted titles, student reload and
serving gates still apply afterwards. Successful benchmark calls are cached/reused.

The old fieldwise SmolVLM module remains available for diagnostics/tests but is disabled
in the default model config. The parser still rejects empty evidence and contradictory
claims. No automatic model fallback or manual annotation is introduced. Return
pilot_reports.zip with teacher_b_benchmark.json; download pilot-v6_results.zip as well.
If the new benchmark fails, stop and return reports rather than weakening its gate.

References: [model card](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3),
[pinned native model mapping](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/auto/modeling_auto.py),
[pinned processor](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/idefics3/image_processing_idefics3.py).


### Numeric-year quote correction: resume pilot-v6 or start pilot-v7

Idefics3 returned valid schema on 2/3 images; the third had integer quote 2016 instead
of string "2016". The parser now converts only a present year quote whose integer
value exactly equals an integer year in 1800..2100. It still rejects booleans, floats,
null/empty evidence and mismatched numeric quotes. OCR verification is unchanged.

For the paused runtime, open notebooks/resume_pilot_v6.ipynb and COPY its code cell
into the existing paused Colab notebook. Run it there, where pilot and OUTPUT exist.
It patches the parser in memory and revalidates saved diagnostic responses against
checksummed role caches, current model and image provenance. No new model call is
made by recovery. Recovered results record parser_revalidation provenance and
parser_recovery_report.json. Original generation attempts remain in the ledger.
The cell then reruns teacher_b_check and its OCR benchmark. Only continue when True.
Do not clear caches or rerun setup for this recovery path.

For a new session instead, run updated 00_pilot.ipynb from the top with pilot-v7.
Actual title accuracy and all downstream gates remain pending your returned reports.
