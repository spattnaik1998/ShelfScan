# ShelfScan: Product Requirements Document

Status: Draft v2 — automated annotation, Colab first. Updated 2026-10-27.

## Product and scope

Upload a book-cover photo and receive a structured catalogue card: title, authors, publisher, edition, year, category, price and other visible text. Fine-tune a small open vision-language model with LoRA using automatically generated, evidence-checked labels. All production model requests pass through an LLM gateway; retain a scoped poisoned-cover security demonstration.

The existing dataset contains 128 photos in six folders. This is a small experiment, not sufficient evidence of broad generalization. No manual transcription or mandatory human labeling queue is part of the workflow. Ambiguous examples are automatically retried, partially supervised, or excluded with reasons.

Colab GPU access is the primary compute environment. The Hugging Face token enables authorized model access and artifact upload; hosted inference has separate availability and billing conditions. Local Colab inference is the default for annotators when hosted inference is unavailable. Exact models, licenses, GPU memory and serving compatibility are verified by a pilot before a full run.

## Goals

- Demonstrate an automated annotation pipeline with independent model roles, evidence and reproducible decisions.
- Compare the same small VLM before and after LoRA on identical frozen evaluation inputs.
- Deliver image upload, catalogue storage/search, route/model timing and a gateway guard demonstration.
- Report data coverage, uncertainty, cost and failures alongside performance.

No improvement, attack success, latency or annotation accuracy is assumed in advance. The earlier +20-point gain is an aspiration, not an acceptance requirement.

## Features and priorities

| Priority | Feature |
|---|---|
| Must | Image audit/preparation; duplicate groups; frozen train/validation/test manifest |
| Must | Two independent visual annotators, OCR evidence, selective adjudication, provenance and quarantine |
| Must | Colab baseline, LoRA training, validation-based checkpoint selection, final evaluation |
| Must | One GPU serving path through LiteLLM; upload card; SQLite catalogue/search |
| Must | Rules-based gateway guard, poisoned-image generator and honest direct/gateway comparison |
| Should | Second hosted provider, comparison arena, learned injection classifier |
| Could | Blurbs, CSV export, local Ollama export, additional synthetic training covers |

The multi-agent annotator runs offline during dataset creation. Production scanning uses one student, with fallback for request/schema failures. Valid JSON does not guarantee a correct answer. A blurb chain is optional; the security demo can measure extraction tampering directly.

## Annotation policy

Use distinct teacher model families where feasible. Independent first passes cannot see each other's output or the folder category. An OCR worker supplies text and coordinates; a deterministic verifier checks schema, evidence and agreement; an adjudicator sees disputed fields, source images/crops and evidence. Limit adjudication to one round and one technical retry per failed call. No open-ended agent debates.

Folder categories are supplied by the dataset as targets, not inferred by label annotators and not passed to the student as input. Their accuracy measures reproduction of the dataset taxonomy, not a universal book genre definition.

Accepted records are pseudo-labels, never described as human ground truth. Every input receives a disposition. Unknown and unreadable are distinct. See [the annotation plan](ANNOTATION_PLAN.md).

## Evaluation and success criteria

Freeze approximate 70/15/15 book-group splits before generating training derivatives. Stratify where group constraints permit; publish actual counts. Validation alone controls thresholds, augmentation and checkpoint choice. Final test results are opened once after configuration is frozen.

Report full-schema validity, normalized title/author agreement, category accuracy, visible-field exact match, p50/p95 latency, serving cost and GPU time. Automated reference scores are explicitly pseudo-reference agreement. Report all test inputs, accepted-reference coverage, unresolved counts and denominators; do not hide difficult rejected examples.

Add separately reported programmatically rendered covers with known text/metadata to test transcription and null handling without manual annotations. Generate disjoint books, layouts and seeds for training versus evaluation; verify rendered text fits. This benchmark has a synthetic-domain limitation and cannot certify real-photo accuracy.

Security reports attack success on direct and guarded paths, flag/block rate and clean false positives. Compare the same images, model and decoding configuration. If the unguarded model resists, report that result. Configure guard thresholds on validation, never test. Targets: >=95% schema validity, >=80% title pseudo-reference agreement, >=85% folder category accuracy, >=90% poisoned detection, <=5% clean flags/blocks, cloud p50 <5 seconds. These are goals, not measured claims; always show counts because the test set is tiny.

## Milestones

| Stage | Deliverable and exit gate |
|---|---|
| M0 Feasibility | 12-image pilot (two per category), two teachers + OCR + student inference, memory/time/coverage report, chosen serving path and run budget |
| M1 Data | Full inventory, duplicate groups, frozen splits, automated annotations, provenance, exclusions and coverage report |
| M2 Learning | Baseline, LoRA adapter, validation-selected checkpoint, frozen final evaluation and reproducible manifest |
| M3 Product | Student served from GPU through gateway; scan/store/search work; required checks pass |
| M4 Security | Train/validation/test-separated attacks, deterministic guard, measured direct/gateway outcomes |
| M5 Demo | Results table, attribution, recorded fallback and rehearsed demonstration |

Pilot inputs are assigned to training before inspection. Preserve validation/test isolation. Start implementation with M0; do not build the entire web application first.

## Demo

Show the annotation trace for one cover, the resulting catalogue card, the measured base/student comparison, then the same poisoned cover through direct and guarded routes. Show real outcomes, accepted-label coverage and a failure case. Cached results are marked as cached. Colab disconnects require saved checkpoints and a clearly marked recorded fallback.

## Credits and references

Dataset: Rex Jovakim Joseph, Book Cover Image Dataset (2026), CC BY 4.0. Preserve attribution in data/model cards and the demo. Check model licenses and teacher output terms before distributing derived artifacts. Research rationale and limitations: [RESEARCH.md](RESEARCH.md). Operational details: [SPEC.md](SPEC.md), [ANNOTATION_PLAN.md](ANNOTATION_PLAN.md), [FINETUNE_COLAB.md](FINETUNE_COLAB.md).
