# T03: run the Colab GPU pilot

Status: notebook and helpers prepared; local tests passed. **The real GPU gate is
pending your execution.** No model quality, training success or GPU compatibility has
been claimed from the local checks.

## What you need

- A fresh Google Colab GPU runtime, T4 16GB or better (the preflight requires at least
  14 GiB visible GPU memory). Allow tens of GB of disk/downloads for weights and packages;
  4-bit loading reduces GPU memory, not the original weight-download size.
- A Hugging Face token stored as the Colab Secret `HF_TOKEN`, with notebook access
  enabled. Read permission is sufficient. Accept model access terms if HF requests them.
- The matching `notebooks/00_pilot.ipynb` and `artifacts/t03_colab_inputs.zip` supplied
  by Codex. The archive includes code, pinned requirements, 12 prepared pilot images,
  checksums and their provisional split. No full-dataset upload is required.

The HF token authenticates weight downloads. Models execute on your Colab GPU; this
notebook does not call paid HF Inference Providers. EasyOCR separately downloads its
public recognition/detection weights and runs on CPU. The adapter is saved to Drive;
HF publication is a later task and is not performed automatically.

## Run

1. Upload `t03_colab_inputs.zip` to `MyDrive/shelfscan/t03_colab_inputs.zip` on Google Drive.
2. Open `00_pilot.ipynb` in Colab. Select **Runtime → Change runtime type → GPU**.
3. In Colab's Secrets panel, add `HF_TOKEN` and enable notebook access. Do not paste the
   token into a code cell or into chat.
4. Run the cells in order. The first cells mount Drive and verify the handoff before
   importing its code. Installation is pinned and can take several minutes. If the
   notebook reports stale packages in memory, restart the session and rerun from the top.
5. Keep `RUN_ID = 'pilot-v1'` when resuming unchanged work. A source/model/settings change
   needs a deliberate new run ID; do not delete the old evidence to manufacture a pass.
6. Run the final export cell even if an earlier stage fails. It downloads
   `pilot_reports.zip`. Return that file to Codex for review before T04.

If setup fails before the session object exists, the final cell exports available
sanitized reports and marks setup incomplete. If installation itself fails, inspect
the private installation log for secrets before sharing its relevant error. It is not
included automatically in the report ZIP.

## Cell sequence

1. Drive/input paths; checksummed source/input extraction.
2. Pinned Linux GPU environment installation and stale-module check.
3. HF authentication, GPU/precision preflight and one-time resolution of immutable model
   revisions. The resolved manifest is reused on resume rather than following `main`.
4. Verify all pilot inputs, split membership and runtime/config identity.
5. CPU OCR over four rotations; persist source-frame text boxes.
6. Teacher A on all inputs, unload; teacher B independently on all inputs, unload.
7. Evidence verification and selective adjudication; prove cached replay makes no calls.
8. Base student predictions on the training-only pilot.
9. Two LoRA steps using complete accepted records, assistant-only loss, adapter/processor
   save to Drive, fresh-base reload and decoded-output comparison.
10. Authenticated loopback serving plus LiteLLM: reject missing keys, route one image
    through `shelfscan-local`, validate BookRecord, then shut services down.
11. Gate report and sanitized report export.

This is 22 notebook cells including instructions, with 13 executable cells. No public
tunnel, catalogue application deployment, full-data training or Hub upload is included.
The dependency lock was resolved for Linux Python 3.11; actual Colab Python/package/GPU
versions are recorded for diagnosis. GPU execution remains necessary to establish that
the chosen pinned stack works on your runtime.

## Models and limits

| Role | Candidate | Runtime |
|---|---|---|
| Teacher A | Qwen/Qwen2.5-VL-7B-Instruct | NF4 4-bit, one model at a time |
| Teacher B | HuggingFaceTB/SmolVLM2-2.2B-Instruct | FP16 on T4; BF16 where supported |
| Adjudicator | Same Qwen7B checkpoint as A | Selective, one resolution round |
| Student | Qwen/Qwen2.5-VL-3B-Instruct | NF4 4-bit with tiny LoRA smoke |

The two teachers are different multimodal families. A and the adjudicator share a
checkpoint; this is not an independent judge. The pilot measures whether these choices
work rather than presuming their annotations are correct. Model licenses are captured
from HF metadata, with a model-card pointer if unavailable; no blanket license claim
is made for derived artifacts.

The pilot uses Qwen2.5-VL, not the older plan's provisional Qwen3.5 checkpoint, so its
quantization recipe is specific to this implementation. Candidate GPU requirements are
in `requirements/colab.in`, with resolved transitive versions in `colab.lock.txt`.
Native Transformers loading disables remote model code. Generation uses bounded output
length and a soft time limit; a CUDA operation already running cannot be preempted by
that timer. Budget/technical retry rules persist across interrupted annotation calls.

## Saved outputs

Under `MyDrive/shelfscan/runs/<RUN_ID>/pilot/`:

- `runtime_manifest.json`, `dependency_snapshot.txt`, `models.resolved.yaml`,
  `model_licenses.json` and `input_checksums.json`.
- `annotation/raw/`, `pilot_decisions.json`, teacher/adjudicator reports, and cached
  replay report. These retain evidence/provenance; they are not human gold labels.
- `baseline_predictions.json` and a separate training-only smoke baseline.
- `smoke_adapter/`: adapter weights, processor/tokenizer, base revision and smoke manifest.
- `smoke_reload_report.json`, `gateway_report.json`, `progress.json`, `stage_errors.json`.
- `pilot_report.json/.md` and `pilot_reports.zip`.

Teacher reports capture generated-token counts, inference wall time and peak allocated
VRAM where measured. Billed API cost is zero; Colab GPU billing is not inferred.
Private startup logs, raw responses, model weights and secrets are excluded from the
small return bundle. Keep the adapter on Drive and send only `pilot_reports.zip` first.

Passing requires 12 inputs, completed role passes, at least 8 accepted titles, cached
replay, baseline completion, adapter reload and authenticated gateway success. Complete
records are counted separately: unresolved optional fields exclude an example from the
smoke training set. At least one complete record is necessary for the training smoke.
A failure returns a failed gate with available evidence; it does not trigger manual
annotation, paid-provider fallback or false success claims.

## Source references

The loader/processor choices follow the versioned [Qwen2.5-VL](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2_5_vl)
and [SmolVLM](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/smolvlm) documentation.
The 4-bit adapter preparation follows [PEFT quantization guidance](https://huggingface.co/docs/peft/en/developer_guides/quantization).
The OCR adapter uses the [EasyOCR API](https://www.jaided.ai/easyocr/documentation/), and
gateway aliases/key references use the [LiteLLM config contract](https://docs.litellm.ai/docs/proxy/configs).
These sources establish API patterns, not successful execution of this pilot on Colab.

## Rebuild a matching handoff locally

After changing notebook/helper source, regenerate both handoff artifacts together:

```text
uv run python -m scripts.build_colab_handoff
uv run ruff check notebooks --fix
uv run ruff format notebooks
```

The notebook bootstrap pins the archive SHA-256. Keep that notebook/archive pair
matched; an older archive with a newer notebook is rejected before code execution.
