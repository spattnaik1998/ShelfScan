# ShelfScan task-driven implementation specification

Status: T01-T02 verified complete; T03 first GPU run failed, corrected rerun pending; T04-T13 pending. Updated 2026-10-29.

This document defines the execution order and handoffs for [SPEC.md](SPEC.md). Product behavior, annotation policy and research remain defined in [PRD.md](PRD.md), [ANNOTATION_PLAN.md](ANNOTATION_PLAN.md) and [RESEARCH.md](RESEARCH.md). The T01 local foundation and T02 offline annotation engine are implemented; T03 notebook preparation is complete; real GPU execution and application tasks remain pending. See [T01 results](T01_RESULTS.md).

## Working agreement

Codex creates program files and notebooks locally, runs appropriate CPU/offline checks, builds the application, and commits/pushes source to the selected GitHub repository once it is configured. The user runs GPU notebooks in Colab, downloads checkpoints/models from runtime storage, uploads the approved model artifacts to Hugging Face through notebook cells, and returns the small run reports. Codex then consumes those reports and artifact revisions to implement the next dependent task.

A notebook being prepared is not the same as its GPU execution passing. Never mark a GPU task done solely because the notebook parses. No locally available GPU is assumed. Each task follows pending -> implementing -> ready for user run (when applicable) -> verified -> done, or blocked with a concrete reason. Live app integration waits for a working endpoint; local app work with fake model clients can proceed meanwhile.

GitHub status: ShelfScan has a local Git repository on main; origin is https://github.com/spattnaik1998/ShelfScan.git. Repository creation/connection is a separate prerequisite to the first push; local implementation can proceed. Never commit image archives, raw annotation data, credentials, SQLite files, notebook outputs with private data, adapters or model weights. Commit source, sanitized example fixtures, configs and compact reports. Dataset/model artifacts are downloaded locally or stored on HF with hashes and revisions.

## Task index and dependencies

| Task | Deliverable | Execution | Depends on |
|---|---|---|---|
| T01 | Project foundation and prepared pilot inputs | Codex locally | None |
| T02 | Evidence contracts and bounded annotation engine | Codex locally with fakes | T01 |
| T03 | GPU pilot notebook and smoke-training/serving helpers | Codex prepares; user runs Colab | T02 |
| T04 | Full annotation notebook, group refinement and data package | Codex prepares; user runs Colab | T03 pilot passed |
| T05 | Metrics, OCR baseline and exact-label synthetic benchmark | Codex locally | T02; final data for live scoring |
| T06 | Baseline and LoRA training notebook | Codex prepares; user runs Colab | T04 data gate, T05 |
| T07 | Frozen final evaluation notebook and model publication | Codex prepares; user runs Colab | T06 checkpoint frozen |
| T08 | Authenticated GPU serving notebook and gateway config | Codex prepares; user runs Colab; Codex checks locally | T03 skeleton; T06 artifact for student integration |
| T09 | App client, extraction/routing and persistence | Codex locally | T01; mocks until T08 endpoint |
| T10 | Upload/catalogue UI and connected scan | Codex locally | T09; T08 for live gate |
| T11 | Poisoned data, gateway guard and security evaluation | Codex builds; user runs GPU calls if needed | T04 split, T08, T09; validation tuning before T07 final evaluation |
| T12 | Reproducible release, GitHub push and demo | Codex locally; user keeps live Colab running | T07, T10, T11 |
| T13 | Optional arena and secondary integrations | Mixed | Core tasks passed |

T05 and T09 can proceed while a Colab run is pending. T11 validation rule selection must finish before the final test portion of T07; T07 supports a separate security evaluation section after rules are frozen. Never tune from either final test report. First-pass graph: T01 -> T02 -> T03 -> T04 -> T05 -> T06 -> T08 -> T09/T10 -> T11 -> T07 final release evaluation -> T12. T05 tooling and T09 implementation can be built earlier as shown above.

## T01 — Create the local project foundation and pilot data package

Status: verified complete, 2026-10-29. [Measured results and checksums](T01_RESULTS.md).

Owner: Codex. Execution: local CPU. No Colab or API key needed.

Create:

| Files | Responsibility |
|---|---|
| pyproject.toml, uv.lock | Installable project; pinned local dependencies; optional GPU/runtime dependencies kept separate |
| .gitignore, .env.example, README.md | Ignore secrets/data/models; safe environment template; setup and task commands; dataset attribution |
| shelfscan/__init__.py, schemas.py, prompt.py | Shared BookRecord/scan contracts, folder mapping and versioned extraction prompt |
| shelfscan/images.py | Decode, EXIF correction, size safeguards, <=1024px JPEG q85 preparation |
| scripts/__init__.py, prepare_images.py, split.py | Inventory/hashes, prepared copies, initial duplicate groups, deterministic provisional split/pilot reservation |
| configs/data.yaml | Source root relative to project, category map, seed 42, preprocessing and pilot settings |
| tests/test_schemas.py, test_images.py, test_split.py | Meaningful contract, corrupt-image, orientation and group-leakage checks |

Commands to expose and execute from ShelfScan:

```text
uv sync
uv run python -m scripts.prepare_images --config configs/data.yaml
uv run python -m scripts.split --config configs/data.yaml --provisional
uv run pytest tests/test_schemas.py tests/test_images.py tests/test_split.py
```

Outputs: data/inventory.jsonl, data/images_1024/, data/groups.provisional.json, data/split.provisional.json, data/pilot_manifest.json and data/preparation_report.md. Package pilot inputs and manifest as artifacts/pilot_inputs.zip. Pilot includes two valid photos per category where possible, reserves all their groups for training and records exceptions. Split is provisional until OCR/title grouping in T04; do not label it frozen.

Done when: all 128 sources accounted for with explicit decode failures, originals unchanged, prepared images within limits, no group overlap, pilot manifest reproducible, tests pass and pilot archive contains everything needed to load its inputs. Synthetic fixtures are sufficient for unit tests; real data remains ignored. Establish the Git repository in ShelfScan and connect the selected remote before pushing; do not initialize a parent dataset directory as the source repository.

## T02 — Implement automated annotator components

Status: verified complete, 2026-10-29. [Results](T02_RESULTS.md) and [running instructions](ANNOTATION_ENGINE.md). Real adapters and GPU validation remain T03 work.

Owner: Codex; CPU execution using scripted model responses.

Create annotation/__init__.py, contracts.py, cache.py, ocr.py, teachers.py, verify.py, adjudicate.py, curate.py, orchestrator.py; configs/annotation.yaml; scripts/pilot.py, annotate.py; tests/test_annotation_contracts.py, test_verifier.py, test_orchestrator.py, test_cache.py.

Responsibilities: typed role inputs/results and per-field unknown/absent/accepted states; model adapter interface; OCR spans/boxes; independent teacher calls; evidence-based verification; anonymized, seeded candidate order; at most one adjudication; one technical retry; cached resume; budget enforcement; quarantine and coverage reporting. Keep the model loading backend replaceable without changing decision policy. Annotator inputs use opaque IDs and cannot reveal category folders. Runtime adapters are finalized with the pilot, not invented as proven-compatible integrations.

Command: uv run python -m scripts.pilot --config configs/annotation.yaml --backend fake. Done: deterministic fake run produces complete traces, retry limits hold, interrupted jobs resume, and unknown never becomes an absent-field target. MVP excludes partial examples; exact token masks are a later opt-in only after dedicated verification.

## T03 — Prepare and execute the 12-image Colab pilot

Status: first GPU run failed; corrected pilot-v2 ready for user rerun. See T03_RESULTS.md. [Run instructions](T03_COLAB.md) and [preparation report](T03_RESULTS.md).

Codex creates notebooks/00_pilot.ipynb, training/__init__.py, training/runtime.py, training/model_io.py, training/smoke.py, serving/__init__.py, serving/app.py (initial authenticated inference skeleton), configs/models.example.yaml and configs/colab.example.yaml. Real model IDs/revisions are selected from official supported checkpoints at implementation time. GPU environment export is an artifact of the successful run.

Notebook cells, in order:

1. Runtime/GPU capability check, exact code revision or source-archive hash, dependency install, local runtime workspace and HF Secret access without printing it.
2. Import helpers, verify pilot archive hashes and print sanitized model/license/runtime configuration.
3. OCR and teacher A pass; persist and unload model. Teacher B pass independently; persist and unload.
4. Verifier/adjudicator, coverage report and bounded-cache resume check.
5. Student baseline on pilot images, short LoRA smoke step, save adapter/processor, reload in clean model state and infer.
6. Authenticated serving/gateway one-image smoke request; latency/memory logging; shutdown and export reports.

Codex checks notebook structure, helper tests and CPU import paths. User uploads/exposes pilot_inputs.zip and runs cells in Colab. Outputs saved under /content/shelfscan_results/<run_id>/: pilot_report.json/.md, runtime_manifest.json, models.resolved.yaml, dependency snapshot and smoke_reload_report.json; the full results ZIP includes the saved smoke adapter. User returns report files and the model revisions; no token or weights need to be pasted into chat.

Gate: both independent teachers, OCR, adjudication and student reload work; >=8/12 accepted titles; one authenticated serving request succeeds. Failures lead to a corrected notebook/config, not full annotation or automatic paid calls. User must execute any revised GPU cells before the task passes.

## T04 — Full automatic annotation and frozen data artifacts

Codex creates notebooks/01_annotation.ipynb, scripts/package_data.py, annotation/grouping.py and tests/test_grouping.py, test_curation.py, test_package_data.py; extends split.py/annotate.py and finalizes configs from T03 evidence.

Notebook: verify all 128 input hashes -> OCR/title-based refinement of duplicate book groups -> finalize split with seed 42 and pilot groups kept in training -> freeze split -> run/reuse independent annotations -> adjudicate -> curate -> coverage/gate check -> package. Grouping uses only identity evidence, never student scores. Reused cache entries retain provenance but their split membership comes from the final manifest.

User executes and saves data_version directory in runtime storage and downloads it: inventory.jsonl, split.json, annotations/decisions.jsonl and raw role records, labels.train.jsonl, labels.validation.jsonl, references.test.jsonl, quarantine.jsonl, annotation_report.md, run_manifest.json, shelfscan_trainval.zip and shelfscan_eval.zip. Training archive contains only train/validation images and targets plus prompt/config; evaluation archive holds test images/references. Freeze an independent-family reference if feasible and disclose when not available. Save checksums for both archives. Use opaque paths in training input.

Gate: every source has a disposition; all groups/derivatives isolated; >=60 accepted distinct real training groups; no unknown-as-null supervision; accepted/reference coverage published by split/category. User returns compact coverage report/manifests and downloaded artifact files. Keep test outcomes out of checkpoint selection.

## T05 — Implement evaluation and a known-answer benchmark

Codex creates evaluation/__init__.py, metrics.py, references.py, ocr_baseline.py, reports.py; scripts/eval.py, render_benchmark.py; configs/benchmark.yaml; tests/test_metrics.py, test_benchmark.py.

Implement full schema validity, normalized title/author match, exact visible-field match, counts/denominators, category results, book-group bootstrap intervals, p50/p95 timing, API spend versus GPU time and reference disagreement. Evaluate OCR-only parsing as a baseline. Preserve original prediction strings alongside normalized scores.

Generate synthetic covers with exact metadata, font/layout fit checks and disjoint content/templates/seeds across splits. All rendered descendants stay with their book. These are separate-domain tests. Commands: uv run python -m scripts.render_benchmark --config configs/benchmark.yaml; uv run python -m scripts.eval --predictions <file> --references <file> --output <report_dir>. Done: deterministic fixtures reproduce known scores, invalid output counts correctly, quarantined references do not erase all-input schema/latency results.

## T06 — Baseline and full LoRA training

Codex creates notebooks/02_finetune.ipynb; training/dataset.py, collator.py, train.py, checkpointing.py; configs/train.yaml; tests/test_training_data.py, test_supervision.py.

Notebook: verify trainval archive/split hashes -> load pinned student/processor -> save base validation predictions and metrics -> inspect assistant-only supervision using automated assertions -> short training sanity run -> main LoRA training -> validation selection -> save/reload selected checkpoint. Maximum two learning-rate configurations; no test archive loaded. Exclude partial labels in MVP; do not silently synthesize absent fields. Training augmentations restricted to train groups. Use supported precision and measured memory settings from T03.

User executes and saves selected adapter, processor/tokenizer/template, base model revision, training/config manifest, trial history, validation report and reload result as downloadable artifacts. Codex receives the compact report and artifact location to prepare final evaluation/serving. Gate: valid supervision, no test access, reproducible checkpoint selection, successful reload. Run single-teacher ablation on the same group intersection if included; otherwise report it as omitted, not an established ensemble benefit.

## T07 — Final evaluation and publish saved model

Codex creates notebooks/03_evaluate_publish.ipynb, scripts/publish_model.py, templates/model_card.md, tests/test_artifact_manifest.py. Depends on frozen T06 checkpoint; final security section additionally needs frozen T11 rules.

Notebook loads evaluation archive in a separate run, checks frozen code/model/data hashes and computes base/tuned/OCR comparisons, separate synthetic metrics, ensemble versus independent-reference comparisons and coverage. It exposes final test scores only after all choices are frozen. Security section evaluates frozen guard against held-out attacks. Record all test runs and why any rerun occurred; changes after seeing test results invalidate an untouched-test claim.

Separate clearly labeled final cells save and optionally upload the selected adapter and processor to the user's chosen HF repository, with model card, license/attribution and manifest. User executes publishing cells and returns HF repository ID plus immutable revision and eval_summary.md/json. Never infer a repository name or visibility. Merged weights optional. Gate: artifact reloads by revision, final claims distinguish measured outcomes from targets and pseudo-reference agreement from truth. If publication is deferred, a versioned downloaded artifact still enables T08; record publication as pending.

## T08 — Connect authenticated GPU serving to the local gateway

Codex creates notebooks/04_serve.ipynb; finalizes serving/app.py; adds serving/auth.py, loader.py; gateway/litellm_config.yaml; configs/serve.example.yaml; tests/test_serving_contract.py, test_gateway_config.py. Validate the LiteLLM version's actual APIs during implementation.

User runs notebook: load chosen base/adapter revisions (uploaded saved artifacts or HF) -> authenticated /v1/chat/completions and health -> explicit authenticated tunnel start -> display endpoint and keep runtime running. GPU inference is serialized/bounded as appropriate for memory. Secrets remain in Colab; endpoint access token is configured locally through an ignored environment file, never chat/log output. Base and student aliases must be distinguishable for comparison.

Codex runs local gateway and checks auth, bounded timeouts, health, one image response and sanitized logs. Cloud fallback remains explicitly configured; no provider API entitlement is presumed from the HF token. If no external provider exists, a GPU-hosted teacher can supply the fallback alias, with serving-location disclosure and measured latency. Multiple paid providers remain optional. Done: authenticated local app -> gateway -> GPU student works and disconnects produce clear errors. Tunnel URLs are temporary runtime configuration, never committed working defaults.

## T09 — Local app backend

Codex creates shelfscan/config.py, llm.py, extractor.py, store.py, main.py; tests/fakes.py, test_extractor.py, test_routing.py, test_store.py, test_api.py. Extend shared schemas as required by SPEC, without weakening annotation contracts.

Implement multipart input preparation, JSON parsing/full validation, explicit local/cloud/auto routes, no fallback on guard block, attempt timing, SQLite storage/search, consistent errors and health. Inject fake clients for CPU tests. Commands: uv run pytest; uv run uvicorn shelfscan.main:app --reload. Done locally with fakes first; live completion additionally requires one T08 gateway scan and saved searchable record. Provider keys stay outside app configuration, except the explicitly isolated unsafe-demo credential if enabled in T11.

## T10 — Upload and catalogue UI

Codex creates shelfscan/templates/base.html, index.html, books.html, book.html; shelfscan/static/app.js, styles.css; tests/test_ui_routes.py. Provide upload/route selection, card, actual model/location, total timing, attempt errors and guard reasons, plus search/filter. Display pending/error states and safe escaping for model/cover text. Unsafe toggle hidden unless enabled; banner unmistakable when enabled. Done: browser checks with a clean cover, invalid upload, blocked result and disconnected runtime; store/search works through live gateway.

## T11 — Security implementation and measured comparison

Codex creates scripts/make_poisoned.py, eval_security.py; security/__init__.py, rules.py, payloads.py, metrics.py; configs/attacks.yaml; gateway/cover_guard.py; tests/test_guard.py, test_poisoned.py, test_security_flow.py. Optional notebook 05_security.ipynb only if GPU batch execution is needed outside the running serving notebook.

Generate split-separated payloads and retain original records. Implement post-extraction gateway checks and optional writer pre-call checks. Rules in one shared module; version/config hashes in reports. Flags skip writer, blocks never store or retry unguarded. Unsafe direct route is default off and explicitly configured for the same-model comparison. Tune with validation, freeze, then execute final test only in the T07 final evaluation stage. User keeps GPU server running or executes prepared batch notebook; Codex handles local gateway/app testing.

Outputs: poisoned manifest, validation_security_report.md, frozen_guard_config.json, final security results with raw counts, attack success/direct versus guarded, false positives and detection blind spots. Passing does not require a vulnerable base model. Learned guard and blurb chain are optional extensions.

## T12 — Source delivery, reproducibility and demonstration

Codex creates/finishes README.md, docs/RESULTS.md, docs/DEMO.md, docs/RUNBOOK.md and .github/workflows/checks.yml. Record source/data/model revisions, installation, notebook order, credential names, artifact paths, environment checks, launch/shutdown, disconnected-runtime recovery and data/model attribution. CI runs CPU checks only, no paid APIs/GPU secrets. CI configuration is validated once the remote is configured.

Run relevant tests, local browser flow and live gateway scan while user keeps Colab running. Report what was actually verified. Commit/push source to the selected GitHub repository; attach any created PR to this chat. No weights/data/secrets in Git. User saves a backup demo recording; cached results are clearly labeled. Done: push URL/commit provided, reports linked, model revision recorded, clean startup instructions and genuine before/after/security outcomes.

GitHub setup is done when repository owner/name/visibility and remote are known, authentication works, and source scope is verified. Do not let a missing remote block earlier local tasks. Request the destination when preparing the first source push if it has not already been provided.

## T13 — Optional work after core delivery

Create shelfscan/templates/arena.html and API arena support; comparison failures isolated. Optional writer prompt/client, CSV export, second hosted provider, learned guard or Ollama/vLLM export each receives its own compatibility check and measured report. These cannot block T01-T12. Full field masking remains optional unless exclusions prevent the T04 data gate.

## Notebook handoff contract

Every notebook starts with prerequisites, input hashes/paths, selected revisions, expected runtime characteristics and an explicit output directory. Install cells pin the last verified environment; reruns cannot silently pull new model/library revisions. Each phase saves progress in the runtime with downloadable artifacts and has restart/resume instructions. End with a small report bundle containing runtime/config manifest, status, summaries and errors; exclude secrets, large data and weights. Publication and tunnel-start cells clearly explain their side effects and run only when the user executes them.

Codex hands off a notebook path, instructions, required artifact locations, exact cells to run and expected report files. The user runs it, saves artifacts and supplies report files or their accessible paths. Colab runtime files are not automatically visible from this local workspace; copying the compact report here or providing an accessible artifact is necessary. If execution fails, return the failing cell's sanitized error plus runtime report; Codex fixes the notebook and resumes from existing cache/checkpoints.

## Immediate next task

**Execute T03 in Colab.** Open notebooks/00_pilot.ipynb, upload only pilot_inputs.zip when prompted, add HF_TOKEN in Colab Secrets and run the cells. Return pilot_reports.zip. Download the full results ZIP to save the smoke adapter. No storage mount is required. T04 must wait for review of the real pilot gate; local stub tests do not complete that gate. See [T03_COLAB.md](T03_COLAB.md).
