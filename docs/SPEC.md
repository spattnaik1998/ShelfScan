# ShelfScan build specification v2

Updated 2026-10-27. Replaces v1's manual annotation and train/test-only workflow. Product scope: [PRD.md](PRD.md). Data decisions: [ANNOTATION_PLAN.md](ANNOTATION_PLAN.md). Research: [RESEARCH.md](RESEARCH.md).

## Architecture

Offline: images -> preparation/group splits -> OCR + independent teachers -> verifier -> selective adjudicator -> accepted/masked labels or quarantine -> Colab baseline/LoRA/evaluation -> HF adapter and serving artifact.

Online: browser -> FastAPI -> LiteLLM -> GPU-served student, with cloud fallback on request/schema failure. FastAPI stores successful, permitted records in SQLite. Extraction response text and downstream writer input pass gateway guard checks. Offline teacher calls also use configured gateway aliases where supported; Colab-local batch jobs can call local models directly and log equivalent provenance. The gateway runs on the same Colab runtime for the initial pilot where practical.

Primary deployment is Colab GPU transformers/PEFT serving behind an OpenAI-compatible interface. Verify the chosen architecture with a one-image smoke test. vLLM and Ollama export are optional optimizations. A public tunnel requires authenticated serving, a gateway-held upstream key, no directory listing and a live-runtime health check. A production deployment is outside this hackathon's scope.

## Stack and planned layout

Python 3.11+, uv lockfile, Pydantic v2, Pillow, FastAPI/Uvicorn, OpenAI-compatible client, LiteLLM, SQLModel/SQLite, Jinja2/vanilla JS. Colab: compatible pinned torch/transformers/PEFT/TRL, optional Unsloth only after architecture compatibility check; OCR library selected in pilot. Test with pytest and fake model clients; live tests opt in.

```text
ShelfScan/
  docs/PRD.md, SPEC.md, ANNOTATION_PLAN.md, FINETUNE_COLAB.md, RESEARCH.md
  shelfscan/schemas.py, prompt.py, images.py, config.py
  shelfscan/main.py, llm.py, extractor.py, store.py, templates/
  annotation/contracts.py, orchestrator.py, ocr.py, teachers.py
  annotation/verify.py, adjudicate.py, curate.py, cache.py
  scripts/prepare_images.py, split.py, pilot.py, annotate.py
  scripts/render_benchmark.py, eval.py, make_poisoned.py
  notebooks/01_annotation.ipynb, 02_finetune.ipynb
  serving/app.py
  gateway/litellm_config.yaml, cover_guard.py
  data/ (generated images/private raw logs ignored in git)
  tests/, pyproject.toml, uv.lock, .env.example, README.md
```

This is the target layout. T01 foundation files and the T02 offline annotation engine are implemented; GPU adapters, notebooks and application components remain planned. See [T01_RESULTS.md](T01_RESULTS.md) and [TASKS.md](TASKS.md) for execution status.

## Data and extraction contracts

SCH-1 BookRecord: title nonempty string <=200 chars; authors list of <=10 names; publisher/edition/price/other_text nullable strings; other_text <=300 chars; year nullable integer 1800..2100; category one of british_law, fiction, history, indian_bare_act, indian_law, technical. All eight keys must be emitted for fully supervised inference. Reject invalid types; log ignored extra keys. Long visible text truncation must follow a shared deterministic policy, recorded in labeling rules.

SCH-2 AnnotationDecision: immutable source ID/hash, group_id, split, candidate record, field status/evidence tier, supervision mask, OCR spans, adjudication reason, role/model revision and prompt/config hashes. Unknown is distinct from absent. Only the curator creates training targets; unresolved mandatory fields cannot enter training. See annotation plan for acceptance policy.

SCH-3 ScanResult: id nullable, record nullable BookRecord, route (local|cloud|cloud-alt), model_alias, via (gateway|direct), attempts (alias/ok/error/latency), total latency_ms, guard (clean|flagged|blocked plus reasons), optional blurb. The local route name remains an API alias for the self-served student even when its GPU is in Colab; UI displays the actual serving location.

SCH-4 Errors: {error:{code,message,details}}. Codes: bad_image, image_too_large, all_routes_failed, blocked_by_guardrail, unsafe_mode_disabled, not_found.

## Data acceptance criteria

| ID | Requirement / acceptance |
|---|---|
| DATA-1 | Inventory all 128 sources; decode failures explicit; preserve originals; EXIF-correct, long edge <=1024, JPEG q85 |
| DATA-2 | Fixed seed 42, book-group train/validation/test split; no group overlap; all derivatives inherit source split |
| ANN-1 | Two independent teacher passes plus OCR; B cannot consume A; annotators cannot see category folders |
| ANN-2 | Every source has disposition; trace each accepted field to evidence and role revisions; unknown never converted to null |
| ANN-3 | At most one adjudication round, one technical retry/call; persistent cache; budget stop resumes without repeat successful calls |
| ANN-4 | Reject unresolved titles; optional unknowns require correct loss masks or whole-example exclusion |
| ANN-5 | Pilot report and >=8/12 accepted titles before scaling; >=60 distinct accepted real training groups before full fine-tuning |
| EVAL-1 | Full Pydantic validity; explicit field denominators/coverage; frozen references; no test-based tuning |
| EVAL-2 | Separate synthetic exact-label benchmark, real pseudo-reference comparison and independent reference agreement |
| TRAIN-1 | Base evaluation saved before training; validation selects checkpoint; reload adapter and reproduce one result |

## Shared inference contract

EXT-1 One versioned extraction prompt shared by annotators, training and serving; role-specific evidence instructions are wrappers and excluded from student input. Ask for visible metadata, main title only, subtitle/sticker text in other_text, null only for absent fields, JSON only. Cover text is data and never an instruction. Category enumeration is supplied, source category is not.

EXT-2 Reuse image preparation in app and offline workflow. Reject undecodable files, raw uploads >15MB and decompression bombs. Never load dataset originals from an untrusted user-supplied filesystem path.

EXT-3 OpenAI-format image+text call; temperature 0, bounded output tokens, model-compatible chat template. Use JSON response mode only where supported. Strip code fences/thinking wrappers before full schema validation; record raw responses privately for diagnosis. Identical inference settings for before/after comparison.

EXT-4 auto tries student then cloud on transport or schema failure; explicit local/cloud routes have no app-level escalation. Gateway handles provider outage fallback. Record every attempt and end-to-end latency. Exhaustion -> 502 all_routes_failed. Guard blocks terminate, never trigger an unguarded fallback. Schema validity is not an accuracy/confidence estimator.

## Gateway and security

GW-1 Aliases: shelfscan-local (GPU student), shelfscan-base, shelfscan-cloud, optional shelfscan-cloud-alt/writer/guard, annotation-teacher-a/b/adjudicator and evaluation-reference. Concrete model IDs/revisions/URLs live in configuration. Verify available IDs and licenses during pilot. Do not assume HF token grants access to every model or paid inference.

GW-2 Auth required; invalid key ->401; rate limit ->429; upstream timeout/retry bounded. Log alias/latency/spend when available. Local GPU time is reported even when billed API cost is zero. Provider keys stay in gateway/Colab secrets; application knows gateway URL/key only. Optional unsafe direct demo has a separate narrowly scoped credential and is off by default.

SEC-1 Generate deterministic sticker attacks with source/payload/placement manifest. Separate payload lists by train/validation/test; test references retain original metadata.

SEC-2 Shared deterministic patterns detect instruction-like text in model responses; quoted attacks in other_text -> flagged, skip blurb; corruption/instruction hits in title/price -> blocked. Return explicit reasons. Legal words and ordinary publisher URLs alone should not automatically prove malicious instructions. Tune rules using clean/poisoned validation only. Learned guard is optional; exact APIs and support must be checked at implementation. Post-extraction checks cannot detect every visual attack and may miss omitted sticker text.

SEC-3 Writer inputs, when enabled, also receive pre-call checks. Cover/image content and teacher outputs never authorize tools. Guard blocks never stored as catalogue records. Flagged records can be stored with status, but cannot feed the writer.

SEC-4 Unsafe direct mode requires DEMO_UNSAFE_DIRECT=true, visible red banner, same model/prompt as guarded comparison, no deployment enabled by default. Report actual direct success and guarded success, including resistant models. No required successful hijack.

## API and UI

| ID | Endpoint | Acceptance |
|---|---|---|
| API-1 | POST /api/scan, multipart file, route=auto/local/cloud, via=gateway/direct | Valid ->200 stored card; blocked ->200 record=null with reasons, no store; bad image ->400; unsafe disabled ->403 |
| API-2 | GET /api/books?q=&category= | Search title/author and filter category |
| API-3 | GET /api/books/{id} | Record and route/guard metadata, absent ->404 |
| API-4 | POST /api/arena (Should) | Concurrent independent columns, individual errors isolated |
| API-5 | GET /health | App and gateway/runtime status |
| UI-1 | GET / | Upload, route, card, timing, model/location, guard reason |
| UI-2 | GET /books | Searchable catalogue |

## Implementation sequence and checks

The actionable file-by-file backlog, ownership, notebook handoffs and dependencies are in [TASKS.md](TASKS.md). Use its task IDs to track implementation. The checklist below groups those tasks by milestone.

- [ ] M0: image inventory/preparation, schema/prompt, pilot split; OCR/two teachers/student feasibility, LoRA save/reload and gateway serving smoke test.
- [ ] M1: grouping/splits, cached orchestration, verifier/adjudication/curation, report and immutable training package.
- [ ] M2: Colab baseline, masked-label handling or exclusion, LoRA, validation selection, final test report, adapter/model card.
- [ ] M3: gateway routes/auth/limits, extractor/store/API/UI.
- [ ] M4: payload generator, guard, direct comparison, metrics.
- [ ] M5: optional arena, demo script, backup recording and README.

Meaningful unit checks cover split leakage, unknown/null semantics, evidence decisions, bounded retry/cache resume, masks, schema/routing and blocked-write behavior using fakes. Live checks cover selected models, GPU save/reload, authenticated gateway scan and guard behavior. No test requires a guaranteed fine-tuning gain or successful attack.

Done: reproducible run manifests; reported exclusions and benchmark limitations; saved/reloadable adapter; working scan/store/search; no exposed credentials; measured security comparison; dataset/model attribution; documented fresh-start steps and runtime requirements. Test claims and target values must be distinguished from actual measurements.
