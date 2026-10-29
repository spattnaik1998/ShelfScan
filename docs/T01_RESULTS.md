# T01 completion report

Verified locally on 2026-10-29. No GPU or model API was used.

## Delivered

Installable Python package and dependency lock; strict BookRecord/scan/error schemas;
shared category mapping and versioned prompt; guarded image preparation; complete
inventory; provisional visual grouping and seed-42 group splits; self-contained pilot
archive; local tests and setup instructions. Git initialized on `main` inside ShelfScan.
No GitHub remote has been supplied, so no source push has occurred.

## Dataset results

| Measure | Result |
|---|---:|
| Original photos accounted for | 128 |
| Successfully prepared | 128 |
| Decode/category failures | 0 |
| Original bytes | 446,282,533 (425.61 MiB) |
| Prepared bytes | 18,789,842 (17.92 MiB) |
| Largest prepared image | 254,871 bytes |
| Provisional visual groups | 124 |
| Visual similarity links | 4 |
| Train / validation / test photos | 90 / 18 / 20 |
| Pilot photos | 12; two per category |
| Pilot archive | 1,741,183 bytes; 17 members |

| Category | Total | Train | Validation | Test | Pilot |
|---|---:|---:|---:|---:|---:|
| British law | 7 | 5 | 1 | 1 | 2 |
| Fiction | 23 | 16 | 3 | 4 | 2 |
| History | 22 | 16 | 3 | 3 | 2 |
| Indian bare acts | 29 | 20 | 4 | 5 | 2 |
| Indian law | 36 | 26 | 5 | 5 | 2 |
| Technical | 11 | 7 | 2 | 2 | 2 |

These are visual candidate groups, not confirmed duplicate books. OCR/title evidence
must refine grouping before T04 freezes the final split. No split exceptions were
needed; every selected pilot group is assigned entirely to training.

## Verification

- 25 automated tests passed, covering strict schema/null contracts, orientation,
  resize limits, corrupt/oversized inputs, original preservation, group isolation,
  tampered-input rejection and deterministic archive generation.
- Ruff lint and formatting checks passed.
- Independently checked all 128 original hashes against inventory after preparation.
- Verified all prepared hashes, JPEG formats, <=1024px long edges, stripped orientation
  metadata and <400KiB output sizes.
- Verified all three partitions are disjoint and cover all 128 images.
- Verified archive CRCs, every member checksum and pilot training reservation.
- Rebuilt the real-data archive and confirmed byte-identical SHA-256.
- Confirmed Git ignores data, archives, secrets and weights.

Environment: Python 3.11.0; Pillow 12.3.0; Pydantic 2.14.2; pytest 9.2.0;
Ruff 0.17.3. Full dependency versions and distributions are pinned in `uv.lock`.

## Artifact checksums

```text
inventory.jsonl
48b74b15c02d0df73845fd81122d7ba9172692c178d4d359d3e937b1788bc4f8

split.provisional.json (canonical payload excluding split_sha256 itself)
b2ba90197e00da4e5335dcef63efeff8d8585ff58d51197767f2fe034ea25478

pilot_inputs.zip
5c10a52c1d8f398d2077aa9f7c316ef8891ad383fc51dcc25293d24f40c52498
```

Generated files live under `data/` and `artifacts/` and are intentionally ignored by
Git. `artifacts/pilot_inputs.zip` contains opaque-named images, manifest, extraction
prompt, attribution, README and checksums. It contains no labels, model or notebook.

Next: T02, evidence contracts and bounded annotation engine with fake-model tests.
The first user-run Colab notebook is T03; no Colab action is needed now.
