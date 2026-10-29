# ShelfScan

Turn a book-cover photo into a catalogue card using a small fine-tuned vision-language
model. An offline pipeline will generate evidence-checked training labels with OCR,
two independent vision teachers and bounded adjudication. A gateway-backed web app
and Colab training notebooks follow in later tasks.

## Current implementation: T01 and T02

Available now: installable Python project, strict shared output schemas, versioned
extraction prompt, bounded image preprocessing, inventory/hashes, provisional visual
grouping and group-aware splits, and a self-contained 12-image pilot archive.

T02 adds the bounded annotation engine with typed evidence, independent teacher/OCR
interfaces, selective adjudication, quarantine, durable retries, caching and reports.
The current CLI uses fake models only and never emits training-eligible labels.

There is no application, model training or executable Colab notebook yet. Next is
T03 (`00_pilot.ipynb`, prepared locally and executed by the user in Colab).
See [the task backlog](docs/TASKS.md) and [T02 verification](docs/T02_RESULTS.md).

Measured T01 results: all 128 images prepared without decode failures; 425.61 MiB of
originals became 17.92 MiB of copies. Provisional split: 90 train / 18 validation /
20 test. The 12-image pilot archive is 1.66 MiB. See [verification details](docs/T01_RESULTS.md).

## Local setup and execution

Requires Python 3.11+ and `uv`. Keep the original dataset adjacent to this directory:

```text
Kaggle_Image_DS/
  Book_Cover_Image_Dataset/
  ShelfScan/
```

From `ShelfScan`, run:

```powershell
uv sync --locked
uv run python -m scripts.prepare_images --config configs/data.yaml
uv run python -m scripts.split --config configs/data.yaml --provisional
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

No API key, GPU, model download or paid API is needed. `uv sync` installs the local
CPU dependencies and development tools recorded in `uv.lock`. GPU requirements will
be resolved independently during the Colab pilot. YAML paths resolve relative to the
project containing `configs/data.yaml`, not the terminal's working directory. Change
`source_root` for a different dataset location.

## Run the annotation engine demonstration

After preparing T01 inputs:

```powershell
uv run python -m scripts.pilot --config configs/annotation.yaml --backend fake
```

Repeat to verify zero new worker calls. This is a synthetic control-flow check, not
real book annotation. The fake backend keeps `labels.jsonl` empty and marks every
record ineligible for training. See [annotation engine operations](docs/ANNOTATION_ENGINE.md)
for inventory mode, persistent budgets, outputs and resume behavior.

## Generated artifacts

| Path | Contents |
|---|---|
| `data/inventory.jsonl` | Every image's source path/hash, category, decode status, dimensions and prepared hash |
| `data/images_1024/` | Separate JPEG copies, long edge <=1024px, quality 85, EXIF orientation corrected |
| `data/preparation_report.md` | Counts, sizes and explicit failures; JSON counterpart includes settings and library version |
| `data/groups.provisional.json` | Initial visual groups, exact/near-match evidence and unresolved category conflicts |
| `data/split.provisional.json` | Seed-42 train/validation/test membership, pilot group reservations and category counts |
| `data/pilot_manifest.json` | Selected pilot inputs, hashes, curation categories and prompt/split versions |
| `data/split_report.json` | Split counts, exceptions and archive checksum |
| `artifacts/pilot_inputs.zip` | 12 opaque-named images, manifest, extraction prompt, checksums and attribution |

The originals are read only. Decode failures remain visible in the inventory/report;
the preparation command exits unsuccessfully when there are failures or an unexpected
input count. Fix the cause before treating preparation as complete. Output images are
content-addressed; the current inventory identifies active copies if reruns leave older
copies in the ignored output directory. No recursive cleanup is performed.

Group ratios target 70/15/15 by category at **group** level, so image ratios vary. Near
visual matches are kept together conservatively and marked unverified; similar designs
can over-group books, while alternate photos can escape visual matching. T04 will use
OCR/title evidence to refine groups and freeze the final split before training. All
pilot groups stay in train, including unselected photos belonging to those groups.
Images with conflicting category labels in one visual group stay in train pending
review. Small-category allocation exceptions are recorded rather than hidden.

The pilot package contains category metadata for curation; future teacher adapters
must receive only opaque image paths and prompts, never that category-bearing manifest.
This package contains **inputs**, not labels or trained models. No Colab execution is
required until T03 supplies the notebook. Archives are reproducible within the locked
environment and carry SHA-256 checksums; recompression can differ across Pillow versions.

## Source and data handling

Source belongs in this directory's Git repository. Data, image archives, credentials,
model weights, local database files and generated runtime outputs are ignored. Never
add those with `git add -f`. Notebook outputs must be sanitized before future commits.
The source repository is [spattnaik1998/ShelfScan](https://github.com/spattnaik1998/ShelfScan),
configured locally as `origin`. Store the future HF token in Colab Secrets, not source files.

## Plans

- [Task-driven implementation specification](docs/TASKS.md)
- [Product requirements](docs/PRD.md)
- [Build contracts](docs/SPEC.md)
- [Automated annotation policy](docs/ANNOTATION_PLAN.md)
- [Colab workflow](docs/FINETUNE_COLAB.md)
- [Research rationale](docs/RESEARCH.md)

## Dataset attribution

Rex Jovakim Joseph, *Book Cover Image Dataset* (2026), 128 images across six categories,
licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The local dataset README and LICENSE identify this attribution. Prepared images apply
EXIF orientation correction, resizing and JPEG recompression. Attribution is included in
the pilot archive. Model labels and training artifacts do not yet exist; their eventual
cards must retain dataset attribution and document automated-reference limitations.
