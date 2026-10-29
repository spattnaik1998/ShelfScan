# T02 completion report

Verified locally on 2026-10-29 before the T02 source push. No GPU, real teacher, real
OCR, hosted model API or production deployment was used.

## Delivered

Typed claim/evidence/worker contracts; independent teacher interfaces; OCR interface;
field agreement and evidence verifier; one anonymous selective adjudication round;
whole-example curation; checksummed atomic role cache; persistent attempt/retry budgets;
single-writer protection; resumable pilot/inventory runners; configuration and coverage
reports. Added the annotation package to the installable project.

## Feature verification

| Check | Observed result |
|---|---|
| Automated suite | 51 passed: 25 foundation tests plus 26 annotation tests |
| Lint and formatting | Passed |
| Locked dependency sync | Passed; no new third-party dependency required |
| Installed package imports outside project cwd | Passed |
| 12-input pilot, first final-code run | 48 new attempts, 12 simulated accepted, 0 pending |
| Same pilot, immediate repeat | 0 new attempts, 48 cache hits |
| Full 128-input inventory, first final-code run | 512 new attempts, 128 simulated accepted, 0 pending |
| Same inventory, repeat | 0 new attempts, 512 cache hits |
| Training-eligible examples from fake runs | 0; `labels.jsonl` remains empty |

Tests exercise role schema rejection, invalid evidence coordinates, unknown/absence
separation, OCR agreement versus visual adjudication, unsupported evidence, optional
unknown exclusion, ordered authors, numeric boundaries, anonymous ordering, corrupt
cache rejection, interruption recovery, one-retry persistence, budget stop/resume,
teacher isolation, abstention and competing writer rejection.

The initial restricted Windows test runner denied atomic temporary-file replacement;
the unchanged atomic-save behavior passed in the normal local execution context. The
full inventory smoke test also exposed repeated ledger rescanning; the engine now
loads the durable count once under the single-writer lock and updates it only after
persisting each new attempt. The final first-pass inventory simulation completed in
approximately ten seconds locally. Timings are not predictions for GPU annotation.

## Scope and next handoff

All accepted counts above describe intentionally invented fixture records. They do not
establish label quality, real pilot coverage or full-data training readiness. Generated
traces stay under ignored `data/`; source includes no dataset images, archives, secrets
or model weights. See [engine operations](ANNOTATION_ENGINE.md) for commands, output
contracts, cache semantics and limitations.

Next task: T03. Codex prepares `notebooks/00_pilot.ipynb` and actual model/OCR/LoRA/serving
smoke helpers; the user runs it on Colab and returns the small reports. T02 requires no
Colab action. No web application has been deployed by this task.
