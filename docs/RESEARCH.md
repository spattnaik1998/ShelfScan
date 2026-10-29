# ShelfScan automated annotation: research basis

Research checked 2026-10-27. These papers motivate design choices; none establishes an accuracy guarantee for this 128-cover dataset. Proposed thresholds and workflows are engineering choices to test in a pilot.

## Primary literature and implications

| Source | What it supports | ShelfScan application and limit |
|---|---|---|
| Hinton, Vinyals & Dean (2015), [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | Transferring an ensemble's knowledge to a smaller deployable model. | Train a small student from accepted teacher-generated JSON. This is output-level supervised distillation, not the original soft-logit distillation objective. |
| Wang et al. (ACL 2023), [Self-Instruct](https://aclanthology.org/2023.acl-long.754/) | Generate instruction examples, filter invalid or similar examples, then fine-tune. | Automate generation and filtering, retain provenance, avoid treating raw model output as clean labels. Their text instruction results do not establish book-cover extraction accuracy. |
| Liu et al. (2023), [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | Machine-generated instruction data can train multimodal instruction-following systems. | Motivate image-to-JSON supervised examples. The original data generation used a language-only GPT-4 with image descriptions; it was not this proposed multi-VLM labeling pipeline. |
| Singh et al. (CVPR 2021), [TextOCR](https://openaccess.thecvf.com/content/CVPR2021/html/Singh_TextOCR_Towards_Large-Scale_End-to-End_Reasoning_for_Arbitrary-Shaped_Scene_Text_CVPR_2021_paper.html) | Scene-text recognition is a material component of reasoning over images; real-world text is challenging for OCR. | Use OCR strings and boxes as an additional evidence stream for visible metadata. OCR is fallible evidence, not an oracle or mandatory veto for every field. |
| Du et al. (2023), [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) | Multiple model instances proposing and debating answers improved the studied reasoning and factuality tasks. | Use two blind initial extraction passes and selective adjudication. Do not infer that agreement establishes truth; the paper does not validate book metadata annotation. |
| Huang et al. (ICLR 2024), [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798) | Intrinsic self-correction without external feedback can fail or degrade reasoning. | Require fresh image/crop evidence and OCR for adjudication; avoid endless self-critique loops. This is a result about studied models/tasks, not a universal impossibility theorem. |
| Zheng et al. (NeurIPS 2023), [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) | LLM judges can approximate human preference on the studied benchmarks, but exhibit position, verbosity and self-enhancement biases. | Blind judge identities, randomize candidate order, score fields/evidence rather than persuasive explanations, and call automated references “silver.” Preference agreement is not metadata truth. |

## Recommended adaptation

1. Prepare immutable image manifests, detect exact/perceptual duplicates, and group repeated books before assigning train/validation/test. Complete grouping after initial title evidence, before training. Keep every transformed descendant in its source group.
2. Run OCR plus two independent image-reading annotators, preferably from different model families. Each annotator initially sees only the cover and the schema, never the other output. Do not expose folder category to the annotators when measuring category prediction.
3. Require every non-null visible field to include a short evidence quote and optional bounding box/crop reference. Record model revision, prompt hash, image hash, generation settings and raw response. Store evidence outside the student's output schema.
4. Deterministic code checks schema, normalization, category mapping and cross-annotator agreement. OCR support is an evidence signal with known false negatives. Model-reported confidence is not a calibrated correctness probability.
5. Accept clear supported agreements. On conflicts, run one bounded adjudication using the original image, useful higher-resolution crops, OCR, and anonymized candidates. Unresolved required fields cause abstention/quarantine; optional unresolved fields are masked from training or the example is excluded, rather than labeling “unreadable” as “absent.”
6. Generate training augmentation only after splitting, preserving labels only for transformations that do not erase metadata. Synthetic covers with programmatically rendered metadata offer exact provenance but test a different image distribution.
7. Measure coverage as well as agreement: count accepted, retried and quarantined examples by category and field. Otherwise filtering could make results appear better by discarding difficult covers.
8. Evaluate the base student, fine-tuned student and OCR baseline on frozen real-cover silver references. Report schema validity separately from semantic correctness, per-field match, uncertainty and sample counts. Category labels have folder provenance; visible text labels have automated provenance. Do not name silver-reference agreement “verified accuracy.”

## Evaluation without a manual annotation dependency

Use two complementary tracks: (a) frozen real-cover silver references created before student training, and (b) a separately held-out set of procedurally rendered covers with known metadata and non-overlapping books/templates, scored exactly. A separate family of image-reading verifier can report support/contradiction/uncertain for student fields, but its judgments remain model-based. Real-cover truth is not established by consensus alone. A small optional visual audit can later increase confidence, but it is not a required production labeling step. Never pick hyperparameters, labeling thresholds or guard thresholds from final test results.

## Compute and access

The user's Colab GPU can host annotators sequentially, cache results, and later train the student; simultaneous multi-agent GPU loading is unnecessary. Choose sizes after inspecting actual runtime memory. Hugging Face tokens authenticate model downloads/uploads and authorized inference calls. They do not by themselves provide unlimited hosted inference: provider usage has its own credits/billing, and not every Hub model has a hosted provider. See official [Inference Providers](https://huggingface.co/docs/hub/en/models-inference) and [pricing documentation](https://huggingface.co/docs/inference-providers/main/pricing). No paid provider is required if the selected teachers fit the available Colab GPU.

## Claims to avoid

- “Multiple agents agree, so the label is correct.” Models can share errors.
- “128 images guarantee a 20-point improvement.” Treat improvement as an experiment.
- “Rotated copies add independent training examples.” They add augmentation, not new books.
- “An HF key pays for all teacher API calls.” Authentication and compute entitlement differ.
- “A guard detects all text embedded in an image.” A text guard only sees text actually extracted; omitted malicious text is a detection blind spot.
