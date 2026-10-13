# Duplicate Detection Summary (Phase 3)

- **Images considered:** 19679
- **Duplicate groups:** 8269
- **Images in duplicate groups:** 19560

## CRITICAL FINDING: byte-identical images under contradictory crop labels

**332 distinct images** (byte-for-byte, verified directly against the original downloaded .zip archives, not an artifact of this pipeline's extraction) appear under **more than one crop label**, covering **1517 rows** across the two source collections. This is a labeling defect in the raw dataset, not a splitting/leakage concern -- the same physical photograph cannot honestly carry two different crop/disease labels, so at least one label for every affected image is simply wrong.

**Crop pairs involved:**

- apple <-> banana: 332 shared images

**Canonical-class pairs involved:**

- cordana <-> healthy_fruit: 168 shared images
- cordana <-> disease_fruit <-> healthy_fruit: 164 shared images

**Affected image count by (crop, class):**

| Crop | Class | Affected images |
|---|---|---|
| apple | disease_fruit | 328 |
| apple | healthy_fruit | 513 |
| banana | cordana | 676 |

Manual inspection of two examples confirms these are real stock photographs of apples (one healthy, one diseased-on-branch, dreamstime-watermarked) that were also copied into the `Banana cordana` folder verbatim -- not a hashing coincidence. This affects roughly half of the `banana.cordana` class.

**Recommendation:** do not train or report `banana.cordana` results until this is resolved. Options, in order of rigor: (1) contact the dataset authors for a corrected release; (2) manually re-verify and relabel every `banana.cordana` image; (3) at minimum, exclude every image listed in `data/manifests/duplicate_groups.csv` with `spans_multiple_crops=True` from any experiment involving banana or apple. Do not simply keep the larger side and discard the smaller side of each pair -- the correct label for a shared image is not implied by which folder has more images.

Example groups (first 10):

- `00cf042768230420...` — crops=['apple', 'banana'], classes=['cordana', 'healthy_fruit']
  - `dataset_v1/Banana cordana/download (2).jpeg`
  - `dataset_v1/Healthy Apple Fruit/HAF (338).jpeg`
  - `dataset_main/Banana cordana/download (2).jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (338).jpeg`
- `03f814aefd5c8931...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (152).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T171329.094.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (152).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T171329.094.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (213).jpeg`
- `0548c804989724b6...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (107).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T170557.622.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (107).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T170557.622.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (168).jpeg`
- `0548d163c4e9348b...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (130).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T171003.219.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (130).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T171003.219.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (191).jpeg`
- `05b5969867be301c...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (45).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T165503.720.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (45).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T165503.720.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (106).jpeg`
- `05f6ececcb0969b5...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (100).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T170506.140.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (100).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T170506.140.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (161).jpeg`
- `062be922f9a0ecef...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (120).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T170821.677.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (120).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T170821.677.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (181).jpeg`
- `065125da89b52810...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (108).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T170604.253.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (108).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T170604.253.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (169).jpeg`
- `067747394612c25d...` — crops=['apple', 'banana'], classes=['cordana', 'disease_fruit', 'healthy_fruit']
  - `dataset_v1/Disease Apple Fruit/DAF (58).jpeg`
  - `dataset_v1/Banana cordana/images - 2026-02-16T165706.550.jpeg`
  - `dataset_main/Disease Apple Fruit/DAF (58).jpeg`
  - `dataset_main/Banana cordana/images - 2026-02-16T165706.550.jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (119).jpeg`
- `077bcd9a5d4c40ca...` — crops=['apple', 'banana'], classes=['cordana', 'healthy_fruit']
  - `dataset_v1/Banana cordana/images (8).jpeg`
  - `dataset_v1/Healthy Apple Fruit/HAF (243).jpeg`
  - `dataset_main/Banana cordana/images (8).jpeg`
  - `dataset_main/Healthy Apple Fruit/HAF (243).jpeg`

## Overlap between the two source collections

- **8266** of 8269 duplicate groups span both `dataset_main` and `dataset_v1` -- confirming the audit's provenance note that these are two overlapping releases of the same underlying photo collection, not independent data. For any experiment, use images from only ONE of the two collections (recommend `dataset_main`, the non-versioned release) unless a reason is documented for using both.

## Within-crop, cross-class duplicates

- **366** duplicate groups contain images from more than one canonical class within the same crop (e.g. an image appearing in both a healthy and a diseased folder for the same crop). Each such group is a labeling ambiguity in its own right -- see `results/duplicate_report.json` -> `groups` (`spans_multiple_classes: true`) for the full list before deciding how to resolve it.

## Methodology

Per-crop LSH banding (candidate generation, bucketed by crop so a genuine near-duplicate pair can never span crops) + exact Hamming-distance verification per candidate pair, for aHash/dHash/pHash independently. A pair is only unioned into a group if at least 2 of the 3 hash algorithms agree within their own threshold (reduces single-hash false positives on this dataset's largely uniform 300x300 images); the agreeing signal names and the minimum confirmed Hamming distance are recorded per group. Unioned with SHA-256 exact-match groups. Recall for the per-crop LSH candidate step is guaranteed-complete for Hamming thresholds strictly less than lsh_bands (see lsh_candidate_pairs docstring); configured thresholds >= lsh_bands trade recall for tractability. Buckets larger than 4000 images were skipped for pairwise verification (see oversized_lsh_buckets) as they are almost certainly already covered by SHA-256 exact grouping. NOTE: an earlier version of this pipeline ran LSH globally (not per-crop) and required only 1 signal, which produced a single spurious 8,872-image group spanning all 6 crops via hash-collision chaining through union-find -- this methodology was corrected in response and is documented here for transparency.

Parameters used for this run:

- `phash_threshold`: 8
- `dhash_threshold`: 8
- `ahash_threshold`: 8
- `min_hash_signal_agreement`: 2
- `lsh_bands`: 4
- `lsh_band_bits`: 16
- `enable_semantic`: False
- `semantic_threshold`: None

## Next steps

1. Resolve the cross-crop mislabeling above before any apple/banana experiment.
2. Phase 5 (manifest generation) MUST consume `data/manifests/duplicate_groups.csv` and keep every group's members inside a single train/val/test partition.
3. Decide whether to use `dataset_main` only, or a deduplicated union of both collections, as the canonical image pool going forward -- using both without deduplication (as this audit did, deliberately, to surface the issues above) would silently double-count nearly every photograph.
