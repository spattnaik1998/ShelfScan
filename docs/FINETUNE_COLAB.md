# ShelfScan Colab workflow v2

Replaces the v1 notebook sketch and mandatory manual-label review. Implement notebooks after the feasibility pilot confirms current model/library support. [ANNOTATION_PLAN.md](ANNOTATION_PLAN.md) defines decisions; [SPEC.md](SPEC.md) defines acceptance criteria.

## 1. Credentials and feasibility

Store HF_TOKEN in Colab Secrets; use least privileges for model download and artifact upload. Never commit tokens or print secret-bearing environment/config values. Access to gated models may require acceptance. HF hosted inference has separate provider/model availability, credits and billing: [official pricing](https://huggingface.co/docs/inference-providers/pricing). Colab GPU inference avoids a hosted-teacher API dependency; measure memory and runtime first.

Select exact teacher A/B checkpoints from distinct VLM families and a small approximately 2B-4B student. Qwen-family small VLMs are candidates, not fixed compatibility promises. Inspect official model cards, licenses and current training examples. Pin model revisions and a known working package set in the notebook/run manifest. Use standard transformers/PEFT/TRL or compatible Unsloth; model-specific documented processor, chat template and trainer arguments take precedence over old sample code.

M0 must exercise OCR, both teacher models, adjudication, base student inference, a short LoRA step, adapter save/reload and authenticated serving through the gateway. Run roles sequentially on one GPU as needed. Do not keep multiple large teachers resident while training.

## 2. Data notebook

Mount Drive; verify inventory/source hashes; run the training-only 12-image pilot. Freeze group splits before producing training derivatives. Cache role results on Drive, keyed by image/config/model/prompt hashes. Run independent annotators, OCR, verifier and selective adjudicator. Export accepted labels with masks, quarantine and coverage report. No required manual transcription step.

Before training, require >=60 distinct accepted real training book groups. Unknown optional fields need correct loss masking; otherwise exclude whole examples. Never fill unknowns with null. Package prepared images, frozen split, decisions/labels, prompt, config and manifest as a versioned training artifact. Keep test targets inaccessible to training code. A separate evaluation job owns test artifacts and releases aggregate final scores after model configuration is frozen.

## 3. Baseline

Evaluate the unmodified student with the fixed extraction prompt and full Pydantic schema. Save outputs and metrics for all eligible validation examples, plus schema validity and latency for every validation input. Final test base inference can be cached by the evaluation job before training, without exposing its scores to the training loop. Do not choose hyperparameters from those test results.

## 4. LoRA experiment

Start conservatively: rank 16, alpha 16, learning rate 1e-4, effective batch 8, maximum 3 epochs, fixed seeds. These are pilot starting values, not research-derived optima. Use bf16 on hardware that supports it; otherwise model-supported fp16/fp32. Quantization, optimizer and sequence length depend on verified architecture/GPU support. Log actual effective batch and trained modules.

Train on accepted train groups only. First adapt language/attention modules; vision adaptation is an optional validation-selected experiment if evidence suggests it helps. Preserve processor-required image tokens. Mask user/system/image/padding tokens and unknown field spans; assert nonzero supervised assistant tokens. If exact unknown-field masking is unsupported, exclude partial examples. Do not let a generic completion trainer accidentally supervise prompts or erase image tokens.

Use training-only rotations and mild photographic perturbations, preserving readable text and attached labels. They do not create independent books. Optional synthetic covers carry provenance and cannot replace the minimum real-group gate. Optional poisoned training uses train-only payloads with unchanged metadata and sticker text as data.

Choose checkpoint using validation title agreement, schema validity and missing-field behavior, with a predefined tie-break (earlier checkpoint). Maximum two learning-rate configurations after the smoke run; record all trials. Save checkpoints to Drive. No inspection of final test failures while making model choices.

## 5. Evaluation and ablations

Freeze checkpoint, thresholds, preprocessing and prompts. Evaluate base and tuned models once on the same final inputs. Report real-photo pseudo-reference agreement and independent-reference agreement separately, reference coverage/disagreement, raw counts by category and book-group uncertainty. Report synthetic exact-label benchmark separately. Full schema validation replaces the v1 two-key validity check.

Run the controlled single-teacher vs verified-ensemble ablation on the same accepted group intersection if time permits. If omitted, state that the causal contribution of multi-agent annotation is untested. GPU cost/time, adapter size and inference latency accompany quality metrics. A negative gain is a valid result.

## 6. Save, reload and serve

Save adapter, base revision, processor/tokenizer/chat template, prompt/config/split hashes, package versions and model card. Reload into a clean process and run a fixed smoke image before declaring export complete. Upload only authorized artifacts to HF, with dataset attribution and pseudo-label limitations. Merged model upload is optional and depends on license/storage requirements.

Primary serving path: a small authenticated OpenAI-compatible service around the supported GPU transformers/PEFT inference pipeline in Colab. Run the gateway locally in the runtime or expose only an authenticated serving endpoint to a laptop gateway. Keep upstream credentials in the gateway. Check health before demo; runtime disconnects end live serving. vLLM and Ollama/GGUF are optional after verified support, not prerequisites.

## Exit checklist

- [ ] Frozen inventory, splits, role decisions, unknown masks/exclusions and coverage report.
- [ ] Reproducible baseline/validation/final reports with pseudo-reference caveat and all denominators.
- [ ] Saved adapter reloads; HF artifact includes processor, manifest, model/data attribution.
- [ ] One image works through gateway as shelfscan-local; base model available for comparison.
- [ ] Drive checkpoints, runtime setup steps and clearly labeled cached/recorded demo fallback.
