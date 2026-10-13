# ABCGMP Multimodal Disease Diagnosis

A research-grade, reproducible pipeline for fruit and leaf disease diagnosis on the
**ABCGMP Fruit and Leaf Disease Six Crop Dataset**, built for journal submission with an
explicit focus on scientific validity, leakage prevention, and honest reporting of what
the data can and cannot support.

**Status: Phase 1–11** (repository scaffolding, dataset audit, duplicate detection,
class-mapping-driven data loading, leakage-safe manifest generation, preprocessing, an
image-only baseline classifier, rigorous evaluation, a fully-built-but-correctly-
disabled multimodal path, image-only cross-crop generalization, Grad-CAM explainability,
and an explainable diagnosis assistant — see [Section 18](#18-diagnosis-assistant-chatbot)).
Phases 12 and 14 (multi-task learning, publication figures) are **not implemented** —
see [Development order](#development-order).

**Read this before doing anything else with the data:** Phase 3's duplicate detection
found that **332 images (676 in the `banana.cordana` class alone) are byte-for-byte
identical to images filed under `apple.healthy_fruit` / `apple.disease_fruit`** — real
apple stock photographs were copied into the banana-disease folder in the source
dataset. This is a raw-dataset labeling defect, verified directly against the original
Mendeley zip archives, not an artifact of this pipeline. **Do not train or report on
`banana.cordana` until this is resolved.** Full detail:
[`results/duplicate_detection_summary.md`](results/duplicate_detection_summary.md).

**Resolution (2026-10-02):** the contaminated cordana images were already excluded from
the manifests (all 338 apple-duplicate cordana images in `dataset_main`). The remaining
400 cordana images were reviewed by eye and with perceptual hashing: they are real
banana leaves, stored as ~4 augmented variants per source photo, and the group-aware
split keeps those variants together (0 cross-split near-duplicates at pHash distance
<= 10). One further apple stock photo (`download (3).jpeg`) escaped exact-hash matching
and is now excluded via `configs/manual_exclusions.csv`
(`data.manual_exclusions_csv`). `banana__cordana` is therefore usable, but it contains
only ~100 distinct source photos, so its 100% test score is optimistic. The existing
baseline was trained before this exclusion; retrain for final numbers.

---

## 1. Project objective

Investigate whether incorporating agronomic metadata alongside RGB image information
improves fruit/leaf disease diagnosis compared with an image-only model, and whether
such multimodal information improves generalization to crops unseen during training.
Secondary questions (metadata contribution, cross-crop generalization, joint
fruit/leaf modeling, interpretability) are listed in full in the project's original
specification; **no novelty claim is made here** — novelty must be established through
literature review, not asserted by this codebase.

**Current, audit-derived caveat:** as detailed in [Section 19](#19-research-validity-safeguards)
below, the downloaded dataset contains **no structured metadata files at all**. The
primary multimodal research question therefore cannot yet be answered with this data
release — see `results/dataset_summary.md` for the full explanation and what would be
needed to unblock it.

## 2. Dataset source

- **Name:** ABCGMP Fruit and Leaf Disease Six Crop Dataset
- **Source:** Mendeley Data — <https://data.mendeley.com/datasets/fhbvmpcyy2/2>
- **DOI:** [10.17632/fhbvmpcyy2.2](https://doi.org/10.17632/fhbvmpcyy2.2)
- **Crops:** Apple, Banana, Citrus, Guava, Mango, Papaya
- **Contributors:** Anand Kumar Jain, Neeta Nain, Anadi Jain (Malaviya National
  Institute of Technology Jaipur; Government Women Engineering College Ajmer,
  Bikaner Technical University)

## 3. Dataset license / citation

**License:** CC BY 4.0.

```
Jain, Anand kumar; Nain, Neeta; Jain, Anadi (2026), "ABCGMP Fruit and Leaf Disease
Six Crop Dataset", Mendeley Data, V2, doi: 10.17632/fhbvmpcyy2.2
```

Cite this dataset in any publication derived from this codebase, alongside this
repository.

## 4. Expected directory structure

```
abcgmp-multimodal-disease-diagnosis/
├── data/
│   ├── raw/            # gitignored — extracted dataset archives (see below)
│   ├── processed/       # gitignored — reserved for Phase 6+ preprocessing outputs
│   └── manifests/       # tracked — split manifests (Phase 5+), small CSVs
├── configs/              # YAML configuration, no hard-coded experiment parameters
├── src/                  # all pipeline code
├── experiments/           # per-experiment-type run folders (Phase 7+)
├── notebooks/             # exploratory notebooks (Phase 14)
├── results/               # audit outputs, metrics, figures, reports
├── checkpoints/           # gitignored — model weights
├── logs/                  # gitignored — experiment tracker CSV/JSON
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

`data/raw/`, `data/processed/`, and `checkpoints/` are gitignored — the raw dataset is
~374 MB and must never be committed.

**Actual layout of `data/raw/` as downloaded** (see `results/dataset_summary.md`
"Provenance note" for why there are two collections):

```
data/raw/
├── _archives/                     # original .zip files, preserved untouched
│   ├── dataset_main.zip
│   └── dataset_v1.zip
├── dataset_main/                  # extracted "... Dataset.zip"
│   ├── Apple___Black_rot/*.jpg
│   ├── Apple___healthy/*.jpg
│   ├── Disease Apple Fruit/*.jpg
│   ├── Healthy Apple Fruit/*.jpg
│   ├── Banana cordana/*.jpg
│   ├── ... (24 class folders total, one level deep, no crop/organ subfolders)
└── dataset_v1/                    # extracted "... Dataset (version 1).zip"
    └── (same 24 class-folder names, minor spelling variant on one Citrus folder)
```

There is **no crop-level or organ-level subdirectory** — every class folder mixes
crop, organ, and disease status into a single (inconsistently formatted) folder name.
This is why `configs/classes.yaml` exists: it is the single source of truth mapping
each actual folder name to crop/organ/disease_status/disease_name, populated by
directly inspecting the download (never guessed by string heuristics in code).

## 5. Installation

Requires Python 3.10 or 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Works with a CUDA GPU if available (`torch.cuda.is_available()`); falls back to CPU
otherwise. No cloud services are required.

## 6. Dataset placement

1. Download both archives from
   <https://data.mendeley.com/datasets/fhbvmpcyy2/2> (or the "Download All" bundle).
2. Extract each into its own subdirectory under `data/raw/` so the two collections
   don't collide on their identical top-level folder name — e.g.:
   ```bash
   mkdir -p data/raw/dataset_main data/raw/dataset_v1
   unzip "6 indian Fruit crops Fruit and Leaf Disease Dataset.zip" -d /tmp/main && \
     mv "/tmp/main/6  indian Fruit crops Fruit and Leaf Disease Dataset"/* data/raw/dataset_main/
   unzip "6 indian Fruit crops Fruit and Leaf Disease Dataset (version 1).zip" -d /tmp/v1 && \
     mv "/tmp/v1/6  indian Fruit crops Fruit and Leaf Disease Dataset"/* data/raw/dataset_v1/
   ```
3. Never edit files under `data/raw/` in place — it is treated as read-only. All
   downstream processing writes to `data/processed/` or `data/manifests/`.

## 7. Dataset audit

Run before anything else, and re-run after any change to the raw data or to
`configs/classes.yaml`:

```bash
python -m src.dataset_audit --data-root ./data/raw
```

or, equivalently, driven entirely from config:

```bash
python -m src.dataset_audit --config configs/dataset.yaml
```

Outputs (all in `results/`):

- `dataset_audit.json` — full machine-readable audit (image inventory, per-crop /
  per-class / per-organ counts, dimensions, corruption, exact + preliminary
  near-duplicate groups, metadata audit, list of every skipped file and why).
- `dataset_audit.csv` — one row per discovered image (path, crop, organ, disease
  status, class, hash, dimensions, corruption flag, ...).
- `dataset_summary.md` — human-readable version of the above, with an explicit
  **"Next steps / open ambiguities"** section.
- `metadata_audit.json`, `metadata_completeness.csv` — structured-metadata-file
  profiling (empty for the current dataset release — see below).

The audit **never guesses** crop/organ/class for a folder it doesn't recognize —
unmapped folders are reported, not silently assigned.

### What the audit found for this dataset (DOI 10.17632/fhbvmpcyy2.2)

- 19,679 images across the two overlapping source collections (`dataset_main`:
  9,858; `dataset_v1`: 9,821), all RGB JPEG/PNG, 0 corrupted.
- 24 crop-specific classes (6 crops × 4 categories each), all successfully mapped
  via `configs/classes.yaml`.
- **8,506 exact-duplicate-content groups covering 19,555 of the 19,679 images** —
  because `dataset_main` and `dataset_v1` are two highly overlapping releases of
  essentially the same photo collection. **This must be resolved before any
  train/val/test split is generated** (Phase 3, `src/duplicate_detection.py`);
  splitting now would put the same physical photograph in multiple partitions.
- **Zero structured metadata files** (no CSV/JSON/XLSX/XLS) anywhere in either
  archive, despite the Mendeley description mentioning soil health, humidity,
  nutrient deficiency, and pathological report annotations. **`METADATA_PAIRING_STATUS
  = NOT_AVAILABLE`.** Every metadata-dependent experiment in this project (image +
  metadata fusion, metadata-only baseline, metadata ablation) is therefore disabled
  until such a file is located in a future dataset version and its per-image/
  per-sample linkage is independently verified — see [Section 19](#19-research-validity-safeguards).
- Organ (`fruit` vs `leaf`) is genuinely **unresolved** for two classes —
  `mango.bacterial_canker` and `papaya.ring_spot` — because neither the folder name
  nor the dataset description states which organ was photographed. Confirm visually
  or with a domain expert before running organ-specific experiments on these classes.

Full detail: [`results/dataset_summary.md`](results/dataset_summary.md).

## 7b. Duplicate detection

```bash
python -m src.duplicate_detection --data-root ./data/raw
```

Requires `results/dataset_audit.csv` to already exist (reuses its per-image inventory
rather than re-scanning). Combines four signals into leakage-safe duplicate groups via
union-find: SHA-256 exact match, and near-duplicate agreement across at least 2 of 3
independent perceptual hashes (aHash/dHash/pHash), each computed with per-crop LSH
candidate bucketing so a false positive on one hash can't chain unrelated images
together across crops (see the module docstring for the earlier, corrected mistake
where it briefly did exactly that). An optional `--enable-semantic` flag adds
pretrained-CNN embedding cosine similarity, off by default (compute-heavy).

Outputs: `results/duplicate_report.json` (full group detail + methodology),
`results/duplicate_detection_summary.md` (human-readable), and
`data/manifests/duplicate_groups.csv` (one row per image, tagged with its group —
Phase 5 splitting must keep every group inside a single train/val/test partition).

### What it found

- **8,269 duplicate groups covering 19,560 of 19,679 images.** 8,266 of those groups
  span both `dataset_main` and `dataset_v1`, confirming they are the same underlying
  photo collection released twice — use only one collection (`dataset_main`
  recommended) for any actual experiment.
- **332 images are byte-identical across the `apple` and `banana` crop labels**
  (1,517 rows once both collections are counted) — see the callout at the top of this
  README. This is the single most important finding of Phase 1–3 and blocks any
  apple/banana experiment until resolved.
- 366 groups contain images from more than one class within the same crop (e.g. an
  image duplicated across a healthy and a diseased folder) — a labeling ambiguity in
  its own right, listed in `results/duplicate_report.json`.

## 8. Class mapping

`configs/classes.yaml` maps every actual downloaded folder name to
`{crop, organ, organ_confidence, disease_status, disease_name, canonical_class}`.
Edit this file — never `src/dataset_audit.py` — if:

- a new dataset version adds/renames folders (re-run the audit first to see what's
  unmapped),
- you want to resolve one of the `organ_confidence: unresolved` entries after manual
  inspection.

## 9. Metadata mapping

Not yet applicable: `configs/dataset.yaml` points to `configs/metadata_mapping.yaml`,
but that file does not exist because **no metadata was found in the downloaded
archives** (see Section 7). Do not create this file with invented fields — if a future
dataset version ships real metadata, run the audit again, confirm
`METADATA_PAIRING_STATUS` is `VALID` or `PARTIAL`, and only then define the mapping
against the actual columns present.

### 7c. Near-duplicate augmented copies (found late; fixed 2026-10-11)

```bash
python -m src.near_duplicates            # writes data/manifests/robust_groups.csv
```

The Phase 3 detector only catches identical or near-identical files. Many folders also contain
**augmented copies of the same photograph** — flipped, rotated or cropped (the PlantVillage-style
apple folders, banana leaf folders stored as 3–4 variants per photo, and others) — which it
missed. The first baseline's split therefore leaked: **372 of its 1,341 test images (27.7%; 76% of
`apple__black_rot`) had a copy in train**, the model scored 100% on those and **0.929 accuracy on
the 969 clean test images** (vs the reported 0.949; macro F1 was unchanged at ~0.87 because the rare
classes were unaffected). The validation set was equally leaky (30%), so early stopping was tuned
on contaminated data.

`src/near_duplicates.py` describes each image with a spatial ResNet18 layer3 feature map (7x7
grid, per-channel mean removed) and links images of the same label whose cosine similarity,
maximised over the 8 rotations/flips, is >= 0.75. The threshold was validated by eye on sampled
pairs (pairs from the lowest band were overwhelmingly the same photo reoriented); connected
components, merged with the Phase 3 groups, are the new splitting unit: 6,171 groups, 1,533 with
2+ images covering 4,298 of 8,936 images (largest 60). After re-splitting, no group spans two
partitions (`results/near_duplicate_report.json`). The baseline (Section 12) and the assistant (Section 18) were regenerated on this split. The 3-seed
cross-crop study (Section 14) was **not** re-run: held-out-crop scores are unaffected by within-crop
leakage, but its *seen-crop* test scores (0.976) used the earlier split and are probably slightly
optimistic. Limits: it is a similarity heuristic, so it can still miss heavily
altered copies and may merge some genuinely distinct similar-looking photos (which only makes
groups larger, not the evaluation optimistic).

## 10. Creating splits

```bash
python -m src.data_loader --config configs/baseline.yaml --make-splits
```

Consumes `results/dataset_audit.csv` (Phase 2) and `data/manifests/duplicate_groups.csv`
(Phase 3), plus `data/manifests/robust_groups.csv` (Section 7c) when `data.robust_groups_csv` is set. The splitting **unit is the duplicate group**, never the individual image —
every member of a duplicate group is guaranteed to land in the same train/val/test
partition, so no exact- or near-duplicate pair can leak across splits. Groups whose
members carry more than one distinct crop or class label (i.e. the apple/banana defect
from Section 7b, plus same-crop cross-class duplicates) are excluded entirely by
default (`data.exclude_label_conflicting_duplicate_groups: true` in
`configs/baseline.yaml`) — their true label is unknown, so training on them under any
single label would be fabricating ground truth.

Other defaults baked into `configs/baseline.yaml`'s `data:` block:
- `source_collections: [dataset_main]` — uses only one of the two overlapping releases
  (Section 7b), so images aren't silently double-counted.
- `task: full_class` — the target is `{crop}__{canonical_class}` (23 realized labels,
  since `apple__healthy_fruit` was entirely excluded — see below), i.e. the actual
  per-crop diagnosis task. Set `task: disease_status` for a simpler binary healthy/
  diseased target instead (used later for cross-crop generalization, Phase 11).

Outputs: `data/manifests/{train,val,test}.csv`, `data/manifests/label_map.json`, and
`data/manifests/split_report.{json,md}` — the latter includes a **per-label
contamination table** showing what fraction of each class's images were excluded for
carrying a conflicting label. For this dataset: `apple__healthy_fruit` is **100%**
contaminated (all 338 of its `dataset_main` images are duplicates of images filed
under other labels) and is entirely absent from the resulting splits;
`apple__disease_fruit` (57.3%) and `banana__cordana` (45.7%) are heavily but not
completely contaminated and retain their clean subset. See
[`data/manifests/split_report.md`](data/manifests/split_report.md) for the full table.

Achieved split sizes for this run: 6,249 train / 1,347 val / 1,341 test images across
23 labels (~70/15/15, stratified at the group level).

## 11. Running the baseline

```bash
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/baseline.yaml --epochs 5 --backbone resnet50  # CLI overrides YAML
```

Trains `ImageClassifier` (`src/models.py`): a configurable pretrained backbone
(`resnet18` default; `resnet50`/`efficientnet_b0`/`efficientnet_b2` also supported)
with the backbone frozen except its last block by default, dropout + linear head,
class-weighted cross-entropy loss (configurable to weighted sampling, focal loss, or
none — see `training.class_imbalance_strategy`), `ReduceLROnPlateau` scheduling on
validation macro F1, and early stopping. Device is auto-selected (`cuda` > `mps` >
`cpu`; this was developed and run on Apple Silicon via MPS).

Each run creates `results/experiments/<run_id>/` containing `config.yaml` (exact copy
of what was used), `preprocessing_config.json`, `training_history.csv`,
`training_curve.png`, and `run_metadata.json` (seed, package versions, git commit,
hardware info, dataset split sizes — Section 23 reproducibility requirements). The
best-val-macro-F1 checkpoint is saved to `checkpoints/<run_id>.pt`. A summary row is
also appended to `logs/experiments.csv` (Section 23/24 lightweight tracker).

## 12. Running the evaluation

```bash
python -m src.evaluate --config configs/baseline.yaml            # most recent checkpoint
python -m src.evaluate --config configs/baseline.yaml --run-id <run_id>
```

Runs the checkpoint once on the held-out test split (never touched during training or
model selection) and writes `results/experiments/<run_id>/metrics.json` (full metric
suite + bootstrap 95% CIs for accuracy/balanced accuracy/macro F1),
`classification_report.csv` (per-class precision/recall/F1/support), and
`confusion_matrix.png` (also copied to `results/confusion_matrices/`). Updates the
matching row in `logs/experiments.csv` with test-set metrics.

### Results (clean split, 3 seeds: 42, 1, 2)

ResNet18, backbone frozen except `layer4`, class-weighted loss, 23-class `full_class` task,
`dataset_main` only, split by the transform-robust near-duplicate groups (Section 7c):
6,248 train / 1,347 val / 1,341 test images, 20 epochs (~75 s/epoch on an Apple M4 via MPS).
Runs: `2026-10-11T122729Z_..._seed42`, `2026-10-11T164124Z_..._seed1`, `2026-10-11T170715Z_..._seed2`.

| Metric (test) | seed 42 | seed 1 | seed 2 | mean ± std |
|---|---|---|---|---|
| Accuracy | 0.9456 | 0.9538 | 0.9478 | **0.949 ± 0.004** |
| Balanced accuracy | 0.8775 | 0.8834 | 0.8840 | **0.882 ± 0.004** |
| Macro F1 | 0.8721 | 0.8880 | 0.8800 | **0.880 ± 0.008** |
| Weighted F1 | 0.9457 | 0.9542 | 0.9478 | 0.949 ± 0.004 |

Single-run bootstrap 95% CIs (in each `metrics.json`) are about ±0.01 for accuracy and ±0.03 for
macro F1. Balanced accuracy and macro F1 are the metrics that matter (Section 16c): several
classes have only 8–25 test images. Mean per-class F1 (3 seeds) is **1.00** on `apple__healthy_leaf`,
`mango__bacterial_canker`, `mango__healthy_leaf` (apple black rot 0.999) and lowest on
`apple__disease_fruit` (0.66 ± 0.10, n=18), `banana__disease_fruit` (0.67, n=11),
`citrus__healthy_leaf` (0.72), `citrus__healthy_fruit` (0.72) and `guava__disease_fruit` (0.75);
errors concentrate among visually similar same-crop fruit classes. `banana__cordana` is **0.93**
here (it scored a perfect 1.00 on the earlier leaky split, where augmented copies of its ~100 source
photos sat on both sides of the split).

**Effect of the leak fix.** The earlier split (4 seeds) gave accuracy 0.949 ± 0.002, balanced accuracy
0.873 and macro F1 0.872 — so after fixing it the aggregate numbers did not drop (balanced accuracy and
macro F1 are slightly higher, within noise). The fix mattered for validity, not for the headline: the
earlier model scored 100% on the 372 test images that had a train copy and 0.929 on the clean
remainder, and classes made of augmented copies (banana cordana) were visibly over-credited. Treat the
~0.88 macro F1 as the honest estimate; seed-to-seed variation (std ~0.008) is small next to the
per-class noise from tiny test classes.

## 13. Running the multimodal experiment (Phase 9 / Phase 10)

```bash
python -m src.train --config configs/multimodal.yaml
```

**Implemented, configured, and refuses to run — deliberately.** `src/models.py`
implements `MultimodalFusionModel` (image CNN → embedding, metadata → MLP →
embedding, concat → fusion MLP → classification head, per Section 14 of the project
spec) and passes a shape-correctness smoke test (`python -m src.models`).
`configs/multimodal.yaml` exists with the full architecture/training configuration.
But `src/train.py::check_metadata_feasibility` reads
`results/dataset_audit.json`'s `METADATA_PAIRING_STATUS` before touching any data,
and — since it is `NOT_AVAILABLE` for this dataset release — the command above exits
immediately with:

```
MULTIMODAL TRAINING REFUSED
- METADATA_PAIRING_STATUS is 'NOT_AVAILABLE', but this config requires one of ['PARTIAL', 'VALID'].
- configs/metadata_mapping.yaml does not exist -- there is no metadata field mapping to train against.
```

Fabricating metadata or silently training an image-only model under a multimodal
config name would misrepresent the experiment, so it refuses instead. Full
explanation: [`results/reports/multimodal_and_ablation_not_supported.md`](results/reports/multimodal_and_ablation_not_supported.md)
(this doubles as the Phase 10 / metadata-ablation write-up — ablation requires
*multiple* verified metadata fields, a strict superset of Phase 9's blocker, so
there is nothing separate to implement). If a future dataset version ships real,
verified per-image metadata: re-run `python -m src.dataset_audit`, confirm the
pairing status is `VALID` or `PARTIAL`, define `configs/metadata_mapping.yaml`
against the actual columns present, and the command above will then proceed.

## 14. Running the cross-crop experiment

```bash
python -m src.cross_crop --config configs/cross_crop.yaml                      # all 15 combinations
python -m src.cross_crop --config configs/cross_crop.yaml --held-out apple banana   # one combination
python -m src.cross_crop --config configs/cross_crop.yaml --seed 1             # another seed
python -m src.cross_crop --config configs/cross_crop_cleansplit.yaml --seed 42  # re-run on the clean split (not yet run; ~1 h)
```

**Protocol.** For each of the C(6,2) = 15 pairs of held-out crops, a binary
healthy-vs-diseased image-only ResNet18 (same frozen-backbone setup as the baseline) is
trained on the other four crops' `train` split. Early stopping and LR scheduling use
only the `val` split of those four seen crops, so no held-out-crop data touches model
selection (asserted in code). The model is then tested on **every image** of the two
held-out crops, and also on the seen crops' `test` split so the generalization gap is
explicit. The task is binary because crop-specific class names cannot be predicted for
a crop the model never saw. Each combination writes
`results/cross_crop/seed<N>/heldout_<cropA>_<cropB>.json`; finished combinations are
skipped, so an interrupted run resumes where it stopped.

**Results (3 seeds: 42, 1, 2; 15 combinations each; 10 epochs max with early stopping).**

| | Seen crops (test) | Held-out crops |
|---|---|---|
| Mean macro F1 over 15 combinations (mean ± std across seeds) | **0.976 ± 0.001** | **0.761 ± 0.019** |

Average gap: **0.22 macro F1**. Per-seed held-out means were 0.783, 0.747 and 0.753. Mean held-out
macro F1 / ROC-AUC per crop (averaged over the 5 combinations in which it is held out, then over seeds):

| Held-out crop | Macro F1 (per seed: 42 / 1 / 2) | ROC-AUC |
|---|---|---|
| papaya | **0.514** (0.54 / 0.48 / 0.52) | **0.591** |
| apple | 0.767 (0.82 / 0.74 / 0.74) | 0.847 |
| banana | 0.811 (0.84 / 0.81 / 0.79) | 0.914 |
| guava | 0.840 (0.83 / 0.84 / 0.85) | 0.931 |
| citrus | 0.844 (0.86 / 0.83 / 0.84) | 0.932 |
| mango | 0.854 (0.85 / 0.84 / 0.87) | 0.932 |

Best and worst combinations (mean over seeds): guava+mango 0.900 held-out F1; guava+papaya 0.591,
citrus+papaya 0.628, mango+papaya 0.640 and banana+papaya 0.680 are the four worst, in every seed.
Full per-combination JSON is in `results/cross_crop/seed{42,1,2}/`.

Interpretation: models transfer moderately to banana, guava, citrus and mango (AUC 0.91–0.93, but
10–15 F1 points below in-crop performance), less well to apple (0.77, and the most seed-dependent),
and **essentially do not transfer to papaya** (AUC 0.59, near chance) — consistent across all three
seeds. High same-crop accuracy therefore says little about performance on a new crop.

**Caveats — read before citing these numbers.**
- Three seeds and a single frozen-backbone architecture. The papaya failure and the overall
  gap are stable across seeds; individual combinations still vary (std up to 0.09, e.g.
  apple+mango 0.64–0.82), so the ordering among the non-papaya crops is only approximate.
- Apple is ~97% leaf images (`apple__healthy_fruit` was entirely excluded, Section 7b),
  so any combination involving apple confounds crop with organ.
- Papaya has a different class balance (66% diseased vs ~50% elsewhere), which may
  contribute to the failure independently of visual transfer.
- Several combinations early-stopped at epoch 2–3; a longer schedule might shift
  numbers somewhat.
- This is the **image-only** generalization result. Whether metadata improves it
  cannot be tested: no metadata exists in this dataset release (Section 13).

## 15. Running the multi-task experiment

Not supported by this dataset release (Phase 12). The pairing check found no shared
identifiers between fruit and leaf folders, no usable EXIF (13% of images carry any, none with a
timestamp or GPS), and 12.5% of images with unknown organ, so fruit/leaf pairs of the same plant
cannot be established. See [`results/reports/multitask_pairing_not_supported.md`](results/reports/multitask_pairing_not_supported.md).

## 16. Running explainability

Not implemented yet (Phase 13).

## 15b. Deeper fine-tuning (`baseline_ft2`)

The fruit classes are the weakest (Section 12), so `configs/baseline_ft2.yaml` unfreezes `layer3` + `layer4`
(10.5M trainable parameters instead of 8.4M) with a lower learning rate (5e-4):

```bash
python -m src.train --config configs/baseline_ft2.yaml --seed 42   # ~28 min on an M4
python -m src.evaluate --config configs/baseline_ft2.yaml
```

**Result over 3 seeds (42, 1, 2; same split and seeds as the baseline): no measurable test improvement.**
Runs: `2026-10-12T071530Z_..._ft2_seed42`, `T074338Z_..._ft2_seed1`, `T081140Z_..._ft2_seed2`.

| Metric | baseline (seeds 42 / 1 / 2) | mean ± std | ft2 (seeds 42 / 1 / 2) | mean ± std | difference |
|---|---|---|---|---|---|
| Best validation macro F1 | 0.881 / 0.883 / 0.892 | 0.885 ± 0.006 | 0.905 / 0.908 / 0.910 | **0.908 ± 0.003** | +0.023 |
| Test accuracy | 0.9456 / 0.9538 / 0.9478 | 0.949 ± 0.004 | 0.9523 / 0.9478 / 0.9456 | 0.949 ± 0.003 | -0.0005 |
| Test balanced accuracy | 0.8775 / 0.8834 / 0.8840 | 0.882 ± 0.004 | 0.8908 / 0.8831 / 0.8683 | 0.881 ± 0.011 | -0.001 |
| Test macro F1 | 0.8721 / 0.8880 / 0.8800 | 0.880 ± 0.008 | 0.8877 / 0.8799 / 0.8707 | 0.879 ± 0.009 | -0.001 |

- Validation scores are consistently higher with `ft2` (all three seeds), but test scores are identical within noise. The
  validation set is used for early stopping and for choosing between models, so a gain there that does not appear on test
  should not be trusted as a real improvement.
- Per class (mean of 3 seeds, test F1): 5 classes better, 5 worse, 13 unchanged (±0.01). Examples: `apple__disease_fruit`
  0.66 -> 0.84, but `papaya__disease_fruit` 0.86 -> 0.73, `banana__disease_fruit` 0.67 -> 0.58, `citrus__healthy_fruit`
  0.72 -> 0.68 (classes with only 11-25 test images).
- **The fruit classes are not solved.** My first run (seed 42) looked like a gain (test macro F1 0.872 -> 0.888) but that was the
  best of the three `ft2` seeds on test; the other two seeds show none.

**Which model the assistant uses.** The website and assistant use `ft2` seed 42. It was adopted after seed 42 alone, on the
rule "higher validation macro F1" fixed before running it, and before seeds 1 and 2 existed. It happens to be the best `ft2`
seed on test, so assistant-level numbers in Section 18 (tier accuracies, advice precision) are probably *slightly optimistic*
relative to a typical `ft2` model. It does classify the six dropdown samples that were originally reported as wrong correctly,
but across all 115 dropdown samples accuracy is the same as with the baseline (88% vs 89%). Switching back to the baseline
checkpoint would not change the conclusions, only which individual images are right.

## 16b. Reproducing results

Every run is fully pinned: `results/experiments/<run_id>/config.yaml` is the exact
config used, `run_metadata.json` carries the seed, git commit, package versions, and
hardware info, and `data/manifests/split_report.json` records the exact split (with
its own seed) the run trained on. To reproduce a specific run: check out the recorded
git commit (once this becomes a git repo), restore the same `data/raw/` (or re-download
from the pinned DOI), re-run `dataset_audit` → `duplicate_detection` →
`data_loader --make-splits` → `train --config results/experiments/<run_id>/config.yaml`.
All four steps are seeded (`seed: 42` throughout by default) and none of them touch the
test set except the final `evaluate.py` call.

## 16c. Interpreting metrics

- **Never read accuracy alone** on this dataset — classes range from 8 to ~1,400
  training images, so a model that ignores rare classes can still post high accuracy.
  Always check **balanced accuracy** and **macro F1** (both weight every class equally
  regardless of size) alongside it.
- **Weighted F1** is accuracy-like (dominated by large classes) — useful as a sanity
  check against macro F1, not a replacement for it.
- **Per-class precision/recall/F1** (`classification_report.csv`) is what actually
  tells you which diseases the model struggles with — check it before citing any
  aggregate number.
- **Bootstrap 95% CIs** (`metrics.json` → `bootstrap_confidence_intervals`) describe
  sampling uncertainty on *this* test set for *this* model — they are not, by
  themselves, a comparison against any other model. A paired bootstrap (see
  `src.metrics.compute_paired_bootstrap_diff`) is required to claim one model beats
  another, and is not currently invoked anywhere because there is no second model to
  compare against (multimodal fusion is blocked — Section 13).
- Do not compare `full_class` (23-way) macro F1 against a future `disease_status`
  (binary) macro F1 as if they were the same metric — they are different tasks with
  different chance baselines.

## 17. Limitations

- **332 images are byte-identical across the `apple` and `banana` crop labels** (real
  apple stock photos filed under `banana.cordana`) — a raw-dataset labeling defect,
  not a leakage or code issue. See Section 7b. Blocks apple/banana experiments until
  resolved (manual relabeling, or contacting the dataset authors).
- The dataset ships two overlapping "releases" bundled as one Mendeley version with no
  documented relationship between them; this repository extracted both rather than
  guessing which one is authoritative, and flags the resulting duplication risk
  loudly rather than silently merging them. 8,266 of 8,269 detected duplicate groups
  span both releases.
- No structured metadata exists in this dataset release — the project's primary
  research question (does metadata improve diagnosis?) is currently unanswerable with
  this data and is documented as such rather than worked around with synthetic data.
- Organ is unresolved for 2 of 24 classes (11.4% of images, 2,242 of 19,679) pending
  manual/expert confirmation.
- Two apple classes (`Apple___healthy`, `Apple___Black_rot`) use PlantVillage-style
  folder naming rather than this dataset's own naming convention, raising an open
  provenance question worth resolving before publication.
- 366 duplicate groups contain images spanning more than one class within the same
  crop (e.g. duplicated across a healthy and a diseased folder) — a further labeling
  ambiguity to resolve before trusting any class boundary at face value.
- **`apple__healthy_fruit` has zero trustworthy images in `dataset_main`** and is
  entirely absent from the baseline's 23-label task — not a bug, a direct consequence
  of the contamination above (Section 10). Any future comparison across all "24
  classes" needs to explicitly account for this missing class.
- The baseline uses only `dataset_main` (not the deduplicated union of both
  collections) — simpler and defensible given the near-total overlap, but it does mean
  roughly half of the raw download's images are unused rather than merged in after
  deduplication.
- The baseline backbone is frozen except its last block by default (fast, avoids
  overfitting a small dataset with a fully fine-tuned ImageNet backbone) — this trades
  away some accuracy a fully fine-tuned model might achieve; `freeze_backbone: false`
  in `configs/baseline.yaml` is available but not benchmarked here.
- Only a single seed (42) has been run for the baseline — the bootstrap CIs in
  `metrics.json` describe sampling uncertainty on the fixed test set, not
  seed-to-seed training variance. A multi-seed comparison would be needed to
  characterize the latter.

## 18. Research-validity safeguards

See `configs/classes.yaml` (organ-confidence fields), `src/dataset_audit.py`
(unmapped-folder reporting, never-silent-skip logging, exact-duplicate detection), and
`results/dataset_summary.md` (explicit `METADATA_PAIRING_STATUS`). Full enumerated
safeguards are carried in the project specification this repository implements
against; the ones already enforced by the current code are:

1. Class folder → crop/organ/class mapping is entirely config-driven
   (`configs/classes.yaml`), never inferred by string-matching in `src/`.
2. No metadata is fabricated, duplicated, or inferred from disease/crop names —
   absence of metadata is reported (`METADATA_PAIRING_STATUS = NOT_AVAILABLE`), not
   worked around.
3. Every file the audit skips is logged with a reason
   (`results/dataset_audit.json` → `skipped_files`); nothing is silently dropped.
4. Exact-duplicate detection (SHA-256) runs on every audit; near-duplicate detection
   (aHash/dHash/pHash, per-crop bucketed, multi-signal agreement) runs in Phase 3 —
   both feed the leakage-safe, group-aware splitting in Phase 5.
5. `data/raw/` is treated as read-only; all outputs are written elsewhere.
6. Splitting is at the duplicate-group level, not the image level (`src/data_loader.py`)
   — no exact- or near-duplicate pair can be split across train/val/test.
7. Any duplicate group with more than one distinct crop or class label is excluded
   from training entirely (`exclude_label_conflicting_duplicate_groups: true`), not
   assigned an arbitrary label — see the per-label contamination table in
   `data/manifests/split_report.md`.
8. Preprocessing (normalization/augmentation) never touches val/test with stochastic
   transforms; there is no tabular fitting to leak, since `TabularPreprocessor` is
   deliberately unimplemented (`src/preprocessing.py`) pending real metadata.
9. Reporting never uses accuracy alone: `src/metrics.py` always returns balanced
   accuracy, macro F1, and per-class metrics alongside it, plus bootstrap CIs.
10. `MultimodalFusionModel` (`src/models.py`) is implemented but deliberately not
    wired into any training script or config — building a runnable multimodal
    pipeline against non-existent metadata would mean fabricating input features,
    which Section 5 of the project's original instructions explicitly prohibits.
11. Every training/evaluation run is fully pinned for reproducibility (seed, config
    snapshot, package versions, git commit, hardware info) — see Section 16b.

---

## Development order

This repository is built in phases; do not assume any phase beyond what's listed as
done actually exists yet.

- [x] **Phase 1** — repository structure, configs, README, requirements.
- [x] **Phase 2** — `src/dataset_audit.py`.
- [x] **Phase 3** — duplicate detection (`src/duplicate_detection.py`); found a
      critical apple/banana labeling defect (see Section 7b).
- [x] **Phase 4** — configurable class mapping consumption + `src/data_loader.py`.
- [x] **Phase 5** — manifest generation (`data/manifests/{train,val,test}.csv`);
      group-aware, label-conflict-excluding, with a per-label contamination report.
- [x] **Phase 6** — preprocessing (`src/preprocessing.py`); tabular preprocessing
      deliberately unimplemented (raises, explains why — no metadata exists).
- [x] **Phase 7** — image-only baseline (`src/train.py`, `src/models.py`,
      `configs/baseline.yaml`); ResNet18, frozen backbone + last block, class-weighted
      loss, early stopping.
- [x] **Phase 8** — evaluation + statistics (`src/evaluate.py`, `src/metrics.py`,
      `src/experiment_tracker.py`); bootstrap CIs, full per-class metrics, confusion
      matrix, lightweight experiment log.
- [x] **Phase 9** — multimodal fusion. Architecture (`MultimodalFusionModel`),
      config (`configs/multimodal.yaml`), and a fail-fast feasibility gate
      (`src/train.py::check_metadata_feasibility`) are all implemented; the gate
      correctly refuses to run because `METADATA_PAIRING_STATUS = NOT_AVAILABLE`
      (see Section 13 and `results/reports/multimodal_and_ablation_not_supported.md`).
      This is the complete, honest version of Phase 9 given this dataset — not a
      placeholder.
- [x] **Phase 10** — metadata ablation. Formally blocked for the same reason as
      Phase 9 (a strict superset — ablation needs multiple verified fields); documented
      in the same report rather than separately implemented, since there is nothing to
      ablate.
- [x] **Phase 11** — cross-crop generalization (`src/cross_crop.py`,
      `configs/cross_crop.yaml`); image-only, binary, all 15 held-out-pair combinations
      run for seeds 42, 1, 2 (Section 14). Multimodal comparison remains blocked (no metadata).
- [x] **Phase 12** — multi-task fruit/leaf learning. Formally blocked: no shared IDs, no usable
      EXIF, and 12.5% of images have no organ label, so fruit/leaf pairs of the same plant cannot be
      established. Documented in `results/reports/multitask_pairing_not_supported.md` rather than
      faked (same approach as Phases 9-10).
- [x] **Phase 13** — explainability: Grad-CAM (`src/explain.py`; all 23 classes in
      `results/gradcam/gradcam_all_classes.png`). Tabular attribution N/A (no metadata).
- [x] **Assistant** — calibrated, gated, Grad-CAM diagnosis chatbot with sourced advice
      (`src/assistant/`, `app.py`, Section 18).
- [ ] Phase 14 — publication-ready tables/figures, notebooks.

Phases 1–11 form a complete, honestly-evaluated image-only pipeline (including
cross-crop generalization) with the multimodal path fully built and correctly disabled
pending real metadata. What's left (12–14) is conditional multi-task learning,
explainability, and publication figures.

## 18. Diagnosis assistant (chatbot)

```bash
python -m src.assistant.calibration      # fit temperature + confidence tiers (validation only)
python -m src.assistant.ood              # fit the out-of-scope (feature-distance) detector
ollama serve                              # local LLM for the chat (llama3 installed via Ollama); skip for the rule-based fallback
python server.py                          # local website: http://127.0.0.1:8000 (upload a photo or pick a sample)
streamlit run app.py                      # optional older Streamlit prototype of the same assistant
python -m pytest tests -q                 # 43 tests: evidence, retrieval, guardrails, LLM grounding checks, tiers, OOD, near-duplicates, web API
```

**What it does.** Photo → image model (the fine-tuned clean-split seed-42 model, Section 15b) → calibrated top-3 + Grad-CAM heatmap + a confidence tier → optional questions → sourced
advice and free-form follow-ups. The classifier decides the label; the text layer only
phrases it.

**Chat engine (`src/assistant/llm.py`).** `make_explainer()` picks, in order: Claude (only if
`anthropic` is installed and `ANTHROPIC_API_KEY` is set), a **local LLM through Ollama** (default: the
first installed model, here `llama3`; override with `OLLAMA_MODEL`), then a deterministic template engine.
`LLM_DISABLE=1` forces the template engine (the test suite does this). The local model runs entirely on
this computer; nothing is sent to the cloud. For a follow-up question the LLM receives only the source
passages for the relevant disease, the diagnosis context and the last few turns, and is told to answer
from the passages and cite source numbers. **Every LLM answer is then checked in code**
(`validate_llm_answer`): it must cite at least one valid source number, and every number in it must appear in
the sources or in the user's question (so a dose or schedule cannot be invented). A failing answer, or an
LLM/network error, falls back to the plain sourced passage text; if the model says the sources do not
answer the question, the assistant declines. Users can ask about any covered disease by name even when the
photo shows something else, "why / how sure / what crop" questions explain the model's result for the photo,
"what is this disease?" names the disease with the model's confidence before its sourced symptoms, and a
healthy photo gets a clear "nothing to treat" reply instead of a stock line. If the model refuses a question
that is clearly about symptoms, spread or management (e.g. "how to cure it?"), the matching sourced passages
are shown instead of a refusal; dose/concentration questions get an explicit "my sources give no doses" reply.

| Part | Files | Notes |
|---|---|---|
| Calibration + tiers | `src/assistant/calibration.py`, `results/assistant/calibration.json` | Temperature T=1.57 lowers test ECE 0.030 → 0.016. Thresholds come from validation only: *confident* ≥ 0.854, *refuse* < 0.589. On test (confidence only): confident = 90.7% of images at 98.2% accuracy; refuse = 3.9% at 54% accuracy. |
| Out-of-scope check | `src/assistant/ood.py`, `results/assistant/ood.json` | Min class-conditional Mahalanobis distance on the 512-d features (fit on train). **Separate thresholds for fruit and non-fruit predictions** (95th percentile of validation within each group): a single global threshold flagged 24.5% of fruit but only 1.6% of leaf test photos, which wrongly downgraded correct fruit diagnoses. A flagged photo drops one confidence tier with an explanation. False-flag rate on normal test images: 5.1% (fruit) / 6.0% (other). Flagged test images are 88.5% accurate vs 95.6% for the rest. It flags only **9.4%** of the 339 studio/stock apple photos mis-filed in the dataset (AUROC 0.885; the per-group threshold trades away detection of fruit-style stock photos: it was 37.5% with the global threshold) and 82% of synthetic noise/blank images. With it, the final tiers on test are: confident 86.7% of images at 98.2% accuracy, uncertain 8.0% at 86.9%, refuse 5.4% at 59.7%. |
| Grad-CAM | `src/explain.py` | Plain-PyTorch Grad-CAM on `layer4`. |
| Inference core | `src/assistant/pipeline.py` | `Diagnoser.diagnose(image)`. Refuse tier names no diagnosis and shows no heatmap. |
| Sourced advice | `src/assistant/knowledge.py`, `data/knowledge/knowledge_base.json` | 18 short passages for the six named diseases, each with source URLs (verified against full pages); TF-IDF + question-intent retrieval; declines when nothing relevant is found. |
| Conversational metadata | `src/assistant/evidence.py` | Questions (crop, damage look, weather) re-rank the model's probabilities with fixed likelihood ratios; every adjustment is shown to the user. |
| Chat + website | `src/assistant/chat.py`, `src/assistant/llm.py`, `server.py`, `web/` | Local website (FastAPI + plain HTML/CSS/JS, no external assets) listening on 127.0.0.1 only; photos are processed in memory and never stored. Palette: Forest `#163A24`, Leaf `#2E7D46`, Mint `#8BCB88`, Canvas `#F4F7EF`, White `#FFFFFF`. Guardrails below. |

**End-to-end evaluation** (`python -m src.assistant.evaluate_assistant`, `results/assistant/assistant_eval.json`,
no LLM involved) over the 1,341 test images: 86.7% get a *confident* tier, 8.0% *uncertain*, 5.4% *refuse*.
Disease-specific cited advice was produced for 557 images and **556 of them (99.8%) were for the true
disease**, all with citations; 8.0% of non-refused answers were the generic "fruit disease" fallback and
48.1% "healthy". Retrieval returned the right passage topic for 54/54 intent questions and declined 6/6
off-topic questions. *Caveats:* the question set was written by the author with the retrieval keywords in
mind, so it is a sanity check, not independent validation; advice precision only covers the six diseases
that have specific advice, on dataset photos; and the assistant uses the fine-tuned seed-42 model (adopted for its higher
validation macro F1, not selected on test).

**Grounding ablation** (`python -m src.assistant.ablation`, `--stage2`; `results/assistant/ablation.json`;
full write-up and caveats in [`docs/novelty.md`](docs/novelty.md)). Same local LLM, 44 questions + 12
prompt-injection prompts: without sources the model gave answers containing invented numbers or pesticide
names for 32 of 50 questions (including made-up fungicide concentrations) and never declined a question the
sources do not cover; with source-grounding it invented none (0/50), answered all 24 answerable questions
with citations, and declined all 12 prompt-injection attempts. The extra answer *validator* was never
triggered (the model obeyed the rules), so its own benefit is not demonstrated. Metrics are automatic proxies
on author-written questions with one small model.

**Guardrails (tested).** Refuse tier → no diagnosis, no advice. Advice only from retrieved
passages, with numbered citations. Classes the dataset labels only as generic "fruit
disease" get no disease-specific treatment. Healthy → "no disease detected". A stated crop
that conflicts with the model is disclosed. Off-topic questions get "I don't have a sourced
answer". Every reply ends with an expert-consultation disclaimer.

**Limitations — read before relying on it.**
- **Not the Phase 9 fusion model.** The dataset still has no metadata, so the user's answers
  are unvalidated, hand-set reweighting rules (x3 symptom match, x1.3 wet weather), not a
  trained or evaluated multimodal model. The multimodal research question remains open.
- **Knowledge base provenance.** The 18 passages are short paraphrases written from full reads
  of the cited pages on 2026-10-11 (university extension, regional agriculture fact sheets, one
  company page). An earlier draft contained details the pages do not support, which were
  removed; four sources that blocked retrieval (WVU Extension, Northern Territory DAF, APS
  Education Center, Hawaii Dept. of Agriculture) were dropped. The passages are not
  expert-reviewed; dosages and product schedules are deliberately omitted. Open the source
  before acting.
- Only six named diseases have specific advice; the six crops' generic "fruit disease"
  classes and all healthy classes do not.
- **Out-of-scope images.** The confidence gate alone missed a studio stock photo of apples
  (99% "apple disease fruit"). The feature-distance check now downgrades it to *uncertain*, but
  it is a partial detector: it now flags only ~9% of such stock photos (a deliberate trade-off against over-flagging fruit photos) and its proxies for
  "out of scope" are the dataset's own mis-filed photos plus synthetic images, not a real
  collection of non-plant or other-crop photos. Tiers were calibrated on dataset photos only.
- **Where the model looks.** Grad-CAM can land on background (seen on a mango canker sample).
  A masking probe (`results/assistant/shortcut_probe.json`) found test accuracy 0.949 on full
  images, 0.429 with the central 50% of the area greyed out, and 0.727 with only the centre
  visible: the model mainly uses the object, but background carries some signal (43% vs 22%
  majority-class baseline), so treat heatmaps as indicative, not proof of what the model reasons about.
- Weak classes (citrus healthy fruit, banana/apple/guava disease fruit) will often land in
  the uncertain or refuse tiers; that is intended behaviour.
- The LLM is a small local model (llama3 8B) and can still phrase things imperfectly or answer
  somewhat narrowly; the grounding check limits what it can assert but does not prove every sentence is
  faithful. It was exercised by hand on ~10 questions plus unit tests with a scripted model, not
  evaluated at scale. The Claude path is implemented but untested (no API key). Answers take roughly
  1-6 s on an Apple M4.
- No user study; the end-to-end behaviour is covered by unit tests and a UI smoke test only.
