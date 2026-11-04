# Consolidated T03 annotation recovery

Use notebooks/recover_annotations.ipynb. Open it in Colab, select a GPU (T4 or better),
and run from the FIRST cell, one cell at a time. This replaces the earlier manual
patch workflow. No Drive mount or code archive upload is required. HF_TOKEN remains
in Colab Secrets. No model weights or credentials are published automatically.

## Inputs

Keep these files available locally:

- artifacts/pilot_inputs.zip: the same verified 12-image package.
- pilot-v6_results.zip: the full saved results archive, not pilot_reports.zip.

The notebook first tries the existing runtime paths /content/pilot_inputs.zip,
/content/shelfscan_t03/artifacts/pilot_inputs.zip and
/content/shelfscan_results/pilot-v6/. If those are unavailable it prompts for the
image archive and the full result archive. No need to restart the old notebook or
repeat its patch cells. This notebook can also run in a fresh GPU runtime; packages
and model weights may need downloading again.

## What changes

The supporting code is extracted into /content/shelfscan_recovery_v2 and results go
into /content/shelfscan_results/pilot-recovery-v2. Original results stay intact. The
source model revisions are pinned before loading. Verified cached OCR and successful
teacher candidates are imported only after image/model/checksum validation; original
failed attempts are retained in a source snapshot. Old failed states are not reset
or retried in place. In the returned source archive this saves 12 OCR calls, 11
Teacher A candidates and 4 Teacher B candidates. Import makes zero model calls.

A new versioned structured adapter accepts JSON or bounded Python literals using
ast.literal_eval, never eval. It validates each field, preserves good fields, and
retries only invalid fields at most twice per image/role under the new protocol.
Unrepaired fields explicitly abstain; guessed values and empty quotes never become
training targets. The two teachers each have a persistent ceiling of 24 new
structured calls. OCR-assisted title retries have their own 24-call ceiling.

Adjudication uses only disputed fields and bounded relevant OCR hints. The base
prompt is <=4000 characters; a hard 3072-token input check runs before GPU transfer.
Each image has at most two new compact adjudication calls, with at most 24 across
the pilot. GPU memory errors and token-budget failures have explicit recorded codes.
Claims remain pseudo-labels; OCR-assisted retries are not independent reference truth.

## Running and stopping

Check source_import_report.json after setup: expected reused counts are 12 OCR,
11 Teacher A and 4 Teacher B for the supplied archive. The notebook then follows OCR
(reused), title policy, preliminary checks, full teacher passes, annotation, baseline,
smoke training, gateway and downloads. Do not skip the preliminary or full passes.

If a teacher returns False, or annotation reports a failed quality gate, skip the
remaining model/training cells and run the final two DOWNLOAD cells. Teachers may
complete processing with explicit unknown fields; field_claim_statuses and
structured_coverage.json expose those abstentions. This is not a claim of complete
training labels. Annotation must produce >=8 accepted titles AND at least one
complete record before baseline/training is allowed. Unknown fields are never
silently treated as absent. Partial-field training has not been introduced.

If annotation passes, continue through baseline, smoke and gateway in order. Always
download pilot_reports.zip and pilot-recovery-v2_results.zip. Return the small report
and retain the full archive (includes adapter if training succeeded, retry history,
source snapshot and detailed decisions). The final pilot must still pass every T03
gate before T04 can begin. GPU execution and label quality remain user-verified.

## Local verification

Run pytest and Ruff. Tests cover safe literal parsing, preserving a valid title when
an optional claim is invalid, targeted repair ignoring unrequested edits, persistent
retry limits, hard adjudicator prompt bounds, truthful coverage failure, explicit GPU
error codes, archive traversal rejection and notebook bootstrap consistency. The
optional local artifact test imports the actual returned archive without GPU calls.

Rebuild with: python -m scripts.build_annotation_recovery, then Ruff check/format the
notebooks. Canonical fresh-run 00_pilot.ipynb also contains the updated adapters.
