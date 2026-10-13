# Dataset Audit Summary

- **Data root:** `data/raw`
- **Classes config:** `configs/classes.yaml`
- **Generated:** 2026-08-18T12:52:36.457520+00:00

## Provenance note

The Mendeley dataset page (DOI 10.17632/fhbvmpcyy2.2) offers two separate archives under the same dataset version: `... Dataset.zip` and `... Dataset (version 1).zip`. Both were downloaded and extracted (`data/raw/dataset_main/` and `data/raw/dataset_v1/` respectively) because they are NOT identical (see per-source-collection counts below) and neither is documented as a strict superset of the other. They are highly overlapping collections, so cross-collection duplicate/near-duplicate leakage across train/val/test is a live risk -- see the near-duplicate section below and src/duplicate_detection.py (Phase 3) before generating any split.

## Image inventory

- **Total images discovered:** 19679
- **Corrupted / unreadable images:** 0
- **Unmapped class folders:** 0 (0 images)

### Images per source collection

| Source collection | Images |
|---|---|
| dataset_main | 9858 |
| dataset_v1 | 9821 |

### Images per class folder (raw, as downloaded)

| Class folder | Images | Mapped? |
|---|---|---|
| Apple___Black_rot | 3974 | yes |
| Apple___healthy | 4016 | yes |
| Banana cordana | 1478 | yes |
| Citrus Cancker Leaf | 210 | yes |
| Citrus CanckerLeaf | 210 | yes |
| Citrus Fruit disease | 316 | yes |
| Citrus Healthy Fruit | 380 | yes |
| Citrus Healthy Leaf | 384 | yes |
| Disease Apple Fruit | 572 | yes |
| Disease Banana Fruit | 148 | yes |
| Disease Mango Fruit | 426 | yes |
| Guava Disease Fruit | 320 | yes |
| Guava Healthly Fruit | 100 | yes |
| Guava Healthly Leaf | 300 | yes |
| Guava Red Rust | 180 | yes |
| Healthy Apple Fruit | 639 | yes |
| Healthy Banana Fruit | 266 | yes |
| Healthy Banana Leaf | 1070 | yes |
| Healthy Mango Fruit | 696 | yes |
| Mang Healthy Leaf | 616 | yes |
| Mango Bacterial Canker | 808 | yes |
| Papaya Fruit Disease | 254 | yes |
| Papaya Fruit Healthy | 286 | yes |
| Papaya Healthy leaf | 596 | yes |
| Papaya RingSpot | 1434 | yes |

### Images per crop (mapped only)

| Crop | Images |
|---|---|
| apple | 9201 |
| banana | 2962 |
| citrus | 1500 |
| guava | 900 |
| mango | 2546 |
| papaya | 2570 |

### Images per organ (mapped only)

| Organ | Images |
|---|---|
| fruit | 4403 |
| leaf | 13034 |
| unknown | 2242 |

### Organ-confidence breakdown (mapped only)

`unresolved` classes have organ genuinely unknown -- see `configs/classes.yaml` notes for `mango.bacterial_canker` and `papaya.ring_spot`.

| Confidence | Images |
|---|---|
| high | 7789 |
| low | 9648 |
| unresolved | 2242 |

### Images per canonical class (mapped only)

| Crop | Organ | Class | Images |
|---|---|---|---|
| apple | fruit | disease_fruit | 572 |
| apple | fruit | healthy_fruit | 639 |
| apple | leaf | black_rot | 3974 |
| apple | leaf | healthy_leaf | 4016 |
| banana | fruit | disease_fruit | 148 |
| banana | fruit | healthy_fruit | 266 |
| banana | leaf | cordana | 1478 |
| banana | leaf | healthy_leaf | 1070 |
| citrus | fruit | disease_fruit | 316 |
| citrus | fruit | healthy_fruit | 380 |
| citrus | leaf | canker_leaf | 420 |
| citrus | leaf | healthy_leaf | 384 |
| guava | fruit | disease_fruit | 320 |
| guava | fruit | healthy_fruit | 100 |
| guava | leaf | healthy_leaf | 300 |
| guava | leaf | red_rust | 180 |
| mango | fruit | disease_fruit | 426 |
| mango | fruit | healthy_fruit | 696 |
| mango | leaf | healthy_leaf | 616 |
| mango | unknown | bacterial_canker | 808 |
| papaya | fruit | disease_fruit | 254 |
| papaya | fruit | healthy_fruit | 286 |
| papaya | leaf | healthy_leaf | 596 |
| papaya | unknown | ring_spot | 1434 |

### Image dimensions / channels

- Width: min=194, max=300, mean=299.99, median=300.0
- Height: min=259, max=300, mean=300.0, median=300.0

Top resolutions:

| Resolution | Count |
|---|---|
| 300x300 | 19677 |
| 194x259 | 2 |

Mode / channel breakdown:

| Mode | Count |
|---|---|
| RGB | 19679 |

### File extensions

| Extension | Count |
|---|---|
| .jpg | 14454 |
| .jpeg | 4797 |
| .png | 428 |

### Corrupted / unreadable images

None detected.

### Duplicate filenames (basename collisions)

- **9491** basenames appear more than once (this alone does not imply duplicate content -- see exact/near-duplicate sections below).

### Exact duplicate files (SHA-256)

- **8506** duplicate-content groups, covering **19555** images.
> These images are byte-identical. If any group's members end up split across train/val/test, that IS data leakage. This must be resolved before splitting -- see src/duplicate_detection.py (Phase 3).

### Preliminary near-duplicate groups (average-hash)

- **8445** groups share an identical 8x8 average hash.
> Computed via a naive 8x8 average-hash exact-match grouping over the audit's per-image scan. This is a coarse first-pass indicator only. Authoritative near-duplicate detection with configurable pHash/dHash similarity thresholds and leakage-safe train/val/test grouping belongs to src/duplicate_detection.py (project Phase 3) and MUST be run before any split is trusted for training.

## Metadata audit

- **Metadata files found:** 0

### METADATA_PAIRING_STATUS = `NOT_AVAILABLE`

No structured metadata files (csv/json/xlsx/xls) were found under the data root. The dataset description mentions expert annotations (soil health, humidity, nutrient deficiency, pathological reports) but these are not present in the downloaded archive contents. Multimodal experiments that require per-image or per-sample metadata are DISABLED until such a file is located and its linkage to images is verified.

## Skipped files

- **0** files were skipped during the audit (unsupported extension, unreadable, or unparseable). Every skip is logged with a reason in `results/dataset_audit.json` -> `skipped_files` (rule: never silently skip a sample).

## Next steps / open ambiguities

1. **METADATA_PAIRING_STATUS is `NOT_AVAILABLE`** (as of this audit run): no structured metadata exists in the downloaded archives despite the dataset description mentioning soil health / humidity / nutrient deficiency / pathological reports. Metadata-dependent experiments (multimodal fusion, metadata-only baseline, metadata ablation) are disabled until a metadata source is located and independently verified to link to images.
2. **Two overlapping source collections** (`dataset_main`, `dataset_v1`) must be deduplicated (exact + near-duplicate) before any split is generated, or the same physical photograph could land in both train and test.
3. **Organ is `unresolved`** for `mango.bacterial_canker` and `papaya.ring_spot` in `configs/classes.yaml` -- confirm visually or with a domain expert before running organ-specific (fruit-only / leaf-only) experiments on these classes.
4. Folders using PlantVillage-style naming (`Apple___healthy`, `Apple___Black_rot`) were assigned `organ_confidence: low` -- worth confirming these are genuinely this dataset's own field-collected apple leaf photos and not a repackaged public benchmark, for provenance disclosure in any publication.
