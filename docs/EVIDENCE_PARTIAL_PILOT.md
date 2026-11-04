# Evidence-first partial-supervision pilot v3

Use `notebooks/evidence_partial_pilot.ipynb`. This is the next correction after the
`pilot-recovery-v2` results. Open this new notebook in Colab and start at the top.
Do not append it to the old notebook or delete earlier results.

## What changed and why

The previous run produced only two accepted titles and no complete records.
Some saved responses contained useful values but empty quotes, while the judge
returned incompatible claim formats. Repeating the same generators did not fix
the supervision bottleneck.

The new replay extracts typed value proposals from saved responses and attaches
actual matching OCR spans. It records the original response hashes, OCR span
indices, rotation and source-image boxes. Model-authored quotes are not evidence.
Matching tolerates case, punctuation and written initials, but does not correct
words, dates or currencies. Conflicting unsupported values remain unknown.

Acceptance requires either agreement between both teachers plus OCR, or a unique
saved judge value corroborated by a teacher plus OCR. Teacher A and the judge
share a model family: their agreement is not independent human verification.
Explicit absence requires both teachers to report absence. Null and unknown
are never converted into absence. If needed, a bounded image selector may choose
only an offered OCR-backed title ID or UNKNOWN; it cannot supply a new value.

This notebook explicitly changes the training eligibility policy. A complete
seven-field record is no longer required for a training-only smoke example.
An accepted title is required; accepted fields and explicit consensus absence
receive loss. Every unknown field's key and value are masked with -100, as are
prompt/image tokens, special tokens and tokens crossing a masked boundary.
Unknown placeholders in the serialized JSON are context, not supervised nulls.
This still permits indirect conditioning on placeholders; it is a feasibility
check, not a claim that partial training produces a production-quality model.

The original full-record gate remains failed in the old run. V3 has a separately
named policy and report; it does not retroactively pass that gate.

## What the local replay found

The latest saved v2 archive contains 12 pilot images. With zero new generations,
the v3 replay yielded 9 accepted titles in 9 distinct groups, and 0 complete
records. The remaining 3 images are excluded from smoke supervision.

| Field | Accepted | Explicit absence | Unknown |
|---|---:|---:|---:|
| Title | 9 | 0 | 3 |
| Authors | 6 | 0 | 6 |
| Publisher | 3 | 0 | 9 |
| Edition | 3 | 2 | 7 |
| Year | 1 | 0 | 11 |
| Price | 0 | 5 | 7 |
| Other text | 5 | 0 | 7 |

These are OCR-backed pseudo-labels across all 12 inputs, not accuracy scores.
The data gate still requires at least 8 accepted titles, now also at least 8
distinct eligible groups. No threshold was lowered to obtain these results.

## Files to have ready

1. The new notebook: `notebooks/evidence_partial_pilot.ipynb`.
2. Latest full source results: `pilot-recovery-v2_results.zip` from ShelfScan.
   Do not substitute the small `pilot_reports.zip` or the older v6 archive.
3. Public images: `artifacts/pilot_inputs.zip`.
4. Hugging Face token in Colab Secrets named `HF_TOKEN`, with notebook access enabled.

Select a GPU runtime (T4 supported by the existing quantized path). The notebook
embeds the updated source. It needs no Drive mount, repository clone, or separate
code upload. Existing matching runtime files are reused; uploads are requested
only when needed. Base-model revisions are taken from the verified source run.
Do not paste your token into a cell or share it in a report.

New project directory: `/content/shelfscan_evidence_v3`.
New output directory: `/content/shelfscan_results/pilot-evidence-v3`.
Earlier output directories are preserved. If installation requests a runtime
restart, restart and rerun from the top. If source or configuration changes after
this run begins, choose a new RUN_ID rather than mixing results.

## Exact execution order

Numbers below count all notebook cells, including Markdown cells, starting at 1.
They are not Colab execution counters such as `[11]`.

| Cell | Action | Expected result |
|---:|---|---|
| 2 | Set isolated paths | New v3 paths printed |
| 4 | Unpack embedded source | Source checksum verification succeeds |
| 6 | Reuse/upload public image ZIP | Input checksum succeeds |
| 8 | Install pinned environment | Installation succeeds; restart only if requested |
| 10 | Read HF secret, import full v2 results, pin model revisions | Runtime preflight succeeds |
| 11 | Create EvidencePilot and import verified cache | Successful source outputs reused |
| 13 | Offline evidence preview | For the reviewed archive: 9 titles, 9 groups, 0 complete records |
| 15 | Optional title selection | Skipped with zero model calls when coverage already passes |
| 17 | Freeze partial examples | Data gate passes with 9 examples for the reviewed archive |
| 19 | Audit masks for every example | Alignment verified, zero unknown-field supervised tokens |
| 21 | Student baseline | Execution/schema report; no accuracy claim |
| 22 | Two-step masked LoRA smoke and clean reload | Finite losses and matching output after adapter reload |
| 24 | Authenticated local serving check | Serving and authentication checks pass |
| 26 | Finish and download small report | Downloads `pilot_reports.zip` |
| 27 | Download full results | Downloads `pilot-evidence-v3_results.zip`, including saved adapter |

Run one code cell at a time from the top. If a stage returns False, stop the later
training/serving cells and run cells 26 and 27. Save both downloads before ending
the runtime, and place the full ZIP in your local ShelfScan folder for inspection.
The same-stage rerun preserves bounded work; restarting the entire pipeline is
not a remedy for a failed evidence or mask gate.

## Reports to inspect

- `evidence_preview.json`: coverage and unresolved field totals.
- `evidence_provenance.json`: accepted value origins and exact OCR support.
- `annotation_coverage.json`: eight-title/group gate and partial policy.
- `partial_mask_audit.json`: processor-only audit before student weights load.
- `smoke_supervision_report.json`: masks checked during actual training steps.
- `smoke_reload_report.json`: finite training/reload outcome and model revision.
- `pilot_report.json`: combined v3 checks, including gateway.
- `stage_errors.json`: reason to inspect if any cell returns False.

The local replay and tokenizer tests pass. The full image-processor audit,
GPU loss steps, adapter reload and gateway must still be verified in your Colab
runtime. Passing them completes this feasibility pilot, not T04 full-scale
annotation, held-out evaluation, or deployment readiness.

## Implementation references

Loss masking uses the ignore-index convention described in
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.8/generated/torch.nn.CrossEntropyLoss.html).
Alignment uses fast-tokenizer offsets and a full token roundtrip, refusing training
when alignment is ambiguous; see the pinned
[Transformers tokenizer interface](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/tokenization_utils_base.py).
This is an implementation correction, not a newly validated annotation method.
