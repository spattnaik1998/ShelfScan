# Automated annotation and execution plan

This replaces manual drafting/review in v1. No human annotation is required. This is a research-informed engineering design; the cited papers do not establish its performance on these 128 covers. See [RESEARCH.md](RESEARCH.md).

## 1. First coding task: feasibility pilot

Inventory every original: relative path, folder category, SHA-256, dimensions, EXIF orientation and decode status. Produce separate EXIF-corrected <=1024px JPEG copies at quality 85; retain originals. EXIF correction is distinct from manually rotating a sideways cover.

Identify exact/near duplicates before splitting using hashes and image similarity. Freeze initial book groups and reserve 12 images (two per folder, subject to group availability) as training-only pilot inputs. Finalize approximately 70/15/15 stratified group splits, seed 42; title/OCR similarity can reveal additional book duplicates. Merge such groups and regenerate the split before training, then re-freeze; never move examples based on model scores. Record group evidence and split hash. All transformations inherit their source group.

On Colab, load one teacher at a time if GPU memory is limited. Candidate families: Qwen VL and an independently trained vision family such as InternVL; use current official compatible checkpoints, not assumed version strings. Choose the small student separately. Establish model access, license compatibility, OCR availability, inference, LoRA smoke training, save/reload and one gateway inference. Save exact revisions, packages and GPU type. HF credentials remain in Colab Secrets/environment.

Pilot outputs: schema validity, field agreement, OCR support, accepted/quarantined counts, GPU memory/time and estimated full-run runtime/cost. Set an explicit run budget in config from these measurements. If two teachers cannot run, choose smaller distinct models or a supported paid endpoint; a one-family fallback is an explicit weaker experiment. No automatic paid-provider spending fallback.

Gate: all roles execute and resume from cache, artifacts reload, and at least 8/12 pilot titles receive accepted labels. This coverage threshold is an engineering target, not calibrated accuracy. If it fails, improve crop/orientation handling or choose teachers before scaling. Do not lower acceptance thresholds simply to meet coverage.

## 2. Bounded orchestration

| Role | Input | Output |
|---|---|---|
| OCR worker | Original/prepared image, orientation candidates | Text spans, boxes, OCR scores, crop/rotation provenance |
| Teacher A | Image and fixed extraction prompt | Candidate fields with visible quotes and crop coordinates |
| Teacher B | Same image/prompt, no A output or folder name | Independent candidate and evidence |
| Deterministic verifier | Candidates, OCR, schema | Field agreement, evidence checks, conflicts |
| Adjudicator | Image/crops, OCR, disputed fields and candidates | Evidence-backed resolution or abstention |
| Curator | Verified decisions and source split | Accepted training targets, masks, quarantine and coverage report |

Use a Python state machine with typed artifacts, durable cache and bounded steps. Hide teacher identities from the adjudicator and randomize candidate order with a logged seed to reduce preference bias. A large orchestration framework is unnecessary for 128 images. Roles are jobs, not necessarily separate GPUs. Distinct prompts on the same checkpoint are correlated views, not independent teachers. Raw output and image text are untrusted data; agents have no tools for web browsing, shell execution or credential access.

Two teacher calls per image plus at most one adjudication call: at most 384 successful generative calls for 128 inputs, excluding the separate evaluator. Technical retries are capped at one per failed call, so at most 768 attempts; cache by image hash, model revision, preprocessing and prompt hash. OCR rotation passes are separately counted. Log elapsed time, GPU utilization when available, token usage and billed cost where provided. Exceeding the configured budget stops cleanly and saves a resumable manifest.

## 3. Field acceptance rules

- Category comes from the folder mapping: British_law_books -> british_law; Fiction -> fiction; History -> history; Indian_bare_acts -> indian_bare_act; Indian_law_books -> indian_law; Technical -> technical.
- Main title only; subtitle/tagline goes to other_text. Copy author order, edition and printed price. Year must be visible on the cover; no outside bibliographic lookup or remembered metadata.
- Normalize Unicode, case and whitespace for comparison; preserve a source transcription for final display. Do not erase edition numbers or price symbols. Initial agreement thresholds: title >=95 normalized character similarity, authors matched individually >=95 with consistent order; numeric/year/price fields exact after safe whitespace normalization. Freeze thresholds after pilot/validation.
- Teacher agreement plus matching OCR spans is the strongest acceptance tier. OCR mismatch routes to image-based adjudication; a missing OCR span alone does not prove a hallucination. Quotes/boxes proposed by a VLM are evidence claims, not proof that pixels contain that text.
- A disputed field is accepted only after adjudicator resolution with traceable visible evidence. Store that lower-evidence tier separately. Do not interpret self-reported model confidence as a probability.
- null means confidently not visible/not present, supported by both initial teachers or adjudication. An unreadable or disputed field is unknown, not null. Empty author arrays similarly mean supported absence.
- Unresolved title -> quarantine entire example from supervised training. Unresolved optional field -> omit its loss using explicit token/field masks; if safe masking is not implemented, exclude the entire example. Never train unknown as null.
- Accepted title + schema and known category are necessary; optional partial targets must not silently become defaults. Metadata describing evidence is stored outside BookRecord and never included in student inference input.

No manual queue blocks progress. Quarantine remains in the data report. If fewer than 60 distinct accepted training book groups remain, pause the full fine-tune and improve teachers/evidence handling. Additional synthetic examples cannot count toward this real-book-group gate.

## 4. Artifacts and reproducibility

- data/inventory.jsonl: source/prepared paths, hashes, dimensions, category, decode status, group_id.
- data/split.json: train/validation/test IDs, group membership, seed, actual per-category counts, split hash.
- data/annotations/raw/: per-role responses and OCR artifacts, including errors.
- data/annotations/decisions.jsonl: file, group_id, split, record, per-field status (accepted|absent|unknown), evidence tier, quotes/boxes, disagreement/adjudication reasons, mask and model/prompt revisions.
- data/labels.jsonl: accepted training-compatible records keyed by relative file, with supervision mask; full provenance joins by immutable ID.
- data/quarantine.jsonl and data/annotation_report.md: dispositions for every source, missing fields and coverage per split/category/tier.
- data/run_manifest.json: config, source/split hashes, model revisions/licenses, package versions, seeds, GPU, call counts, timing/cost and artifact hashes. No secrets.

## 5. Evaluation without manual annotation

Freeze labels and configurations before student training. Training consumes train targets only. Validation pseudo-labels select checkpoints; test targets remain inaccessible to the trainer and hyperparameter loop. A third visual model family, excluded from training label construction where feasible, independently annotates test images before seeing student outputs. Freeze those outputs as a separate reference view. If unavailable, disclose the shared-teacher dependency explicitly.

Evaluate base and fine-tuned student with identical preprocessing/prompt/decoding. Include an OCR-only deterministic metadata baseline to show whether fine-tuning adds value over text recognition and simple parsing. Report agreement against the frozen label ensemble and separately against the independent reference on identical applicable records. Publish reference-to-reference agreement, missingness, disputed cases and evaluation denominators; disagreement does not establish which model is correct. Teacher-family correlation and pretraining exposure remain limitations.

Use a separate programmatically rendered, text-fit-checked cover benchmark with exact labels for title/authors/price/year/absence and rotation robustness. Disjoint content/templates/seeds from any synthetic training data. Report synthetic exact-match results separately from real-photo pseudo-reference results. Do not use an LLM judge as the sole accuracy score.

All real test sources receive schema-validity/latency reporting, even if reference labels were quarantined. Field agreement uses explicit eligible-reference denominators. Training augmentation never touches validation/test; fixed evaluation perturbations are separate robustness conditions. Bootstrap uncertainty by book group and show raw correct/eligible counts. Treat small category slices as descriptive.

Minimum ablations: base student vs student trained on single-teacher labels vs student trained on verified ensemble labels. Use the same accepted training intersection and training settings for the controlled single-versus-ensemble comparison; report an additional full-coverage ensemble run separately if desired. If time prevents the extra run, state that the annotation mechanism's benefit has not been isolated.

## 6. Security experiment

Create train, validation and test payload lists with no exact overlap. Augmentation and guard tuning use train/validation only. Save attack corner, payload, image hash and original record. Distinguish extraction corruption from detection. Measure direct attack success, guarded attack success, detection, false positives and abstentions with counts; direct/gateway calls use the same model, prompt and decoding apart from the guard. A failed attack is a valid result. Do not weaken prompts to manufacture a successful attack.

## 7. Build order and stop points

1. Pilot and model/serving feasibility; decide runtime budget and exact checkpoints.
2. Full data orchestration, evidence report and frozen artifacts.
3. Baseline, LoRA smoke run, full training and validation selection.
4. Frozen final evaluation, adapter save/reload and model card.
5. Gateway student route, scan/store/search application.
6. Security measurements and demo; comparison UI only if time remains.

Estimated effort is provisional: pilot/setup 1-2 hours, orchestration implementation 2-4 hours, full annotation dependent on pilot throughput, training/evaluation 1-3 hours, app/security/demo 3-6 hours. These are work estimates, not promised GPU runtimes. Keep checkpoints and logs on Drive; prefer one working GPU serving path over multiple export formats.
