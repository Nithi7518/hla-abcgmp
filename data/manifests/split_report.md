# Split Report

- **Task:** `full_class`
- **Source collections used:** ['dataset_main']
- **Label-conflicting duplicate groups excluded:** True
- **Split seed:** 42

## Exclusions (nothing silently dropped)

| Reason | Count |
|---|---|
| corrupted | 0 |
| unmapped_class_folder | 0 |
| excluded_source_collection | 9821 |
| manual_exclusion | 1 |
| label_conflicting_duplicate_group | 921 |
| final_usable_images | 8936 |
| total_images_in_audit | 19679 |

## Per-label contamination rate (before exclusion)

What fraction of each label's images (in the source collection(s) used, before the label-conflicting-duplicate-group exclusion above) turned out to be part of a group whose members carry more than one distinct crop or class label. A rate of 100% means the label has ZERO trustworthy images left and will not appear in any split below -- see `results/duplicate_detection_summary.md` for the underlying cause (apple stock photos duplicated into unrelated folders).

| Label | Total images | Conflicting images | % conflicting |
|---|---|---|---|
| apple__healthy_fruit | 338 | 338 | 100.0% **← entire class excluded** |
| apple__disease_fruit | 286 | 164 | 57.3% |
| banana__cordana | 739 | 338 | 45.7% |
| citrus__healthy_leaf | 192 | 24 | 12.5% |
| citrus__healthy_fruit | 190 | 23 | 12.1% |
| papaya__disease_fruit | 127 | 4 | 3.1% |
| papaya__healthy_fruit | 143 | 4 | 2.8% |
| banana__disease_fruit | 74 | 1 | 1.4% |
| citrus__canker_leaf | 210 | 3 | 1.4% |
| citrus__disease_fruit | 158 | 2 | 1.3% |
| mango__healthy_leaf | 308 | 3 | 1.0% |
| banana__healthy_fruit | 133 | 1 | 0.8% |
| mango__bacterial_canker | 404 | 2 | 0.5% |
| mango__disease_fruit | 213 | 1 | 0.5% |
| mango__healthy_fruit | 348 | 1 | 0.3% |
| apple__healthy_leaf | 2008 | 6 | 0.3% |
| apple__black_rot | 1987 | 6 | 0.3% |
| papaya__healthy_leaf | 298 | 0 | 0.0% |
| guava__disease_fruit | 160 | 0 | 0.0% |
| guava__red_rust | 90 | 0 | 0.0% |
| guava__healthy_leaf | 150 | 0 | 0.0% |
| guava__healthy_fruit | 50 | 0 | 0.0% |
| banana__healthy_leaf | 535 | 0 | 0.0% |
| papaya__ring_spot | 717 | 0 | 0.0% |

## Images per split

| Split | Images |
|---|---|
| train | 6248 |
| val | 1347 |
| test | 1341 |

## Label distribution per split (23 labels)

| Label | test | train | val |
|---|---|---|---|
| apple__black_rot | 297 | 1387 | 297 |
| apple__disease_fruit | 18 | 85 | 19 |
| apple__healthy_leaf | 300 | 1401 | 301 |
| banana__cordana | 60 | 280 | 60 |
| banana__disease_fruit | 11 | 51 | 11 |
| banana__healthy_fruit | 20 | 92 | 20 |
| banana__healthy_leaf | 80 | 375 | 80 |
| citrus__canker_leaf | 31 | 145 | 31 |
| citrus__disease_fruit | 23 | 109 | 24 |
| citrus__healthy_fruit | 25 | 117 | 25 |
| citrus__healthy_leaf | 25 | 117 | 26 |
| guava__disease_fruit | 24 | 112 | 24 |
| guava__healthy_fruit | 8 | 34 | 8 |
| guava__healthy_leaf | 23 | 104 | 23 |
| guava__red_rust | 14 | 62 | 14 |
| mango__bacterial_canker | 60 | 281 | 61 |
| mango__disease_fruit | 32 | 148 | 32 |
| mango__healthy_fruit | 52 | 243 | 52 |
| mango__healthy_leaf | 46 | 213 | 46 |
| papaya__disease_fruit | 18 | 86 | 19 |
| papaya__healthy_fruit | 21 | 97 | 21 |
| papaya__healthy_leaf | 45 | 208 | 45 |
| papaya__ring_spot | 108 | 501 | 108 |

Splitting unit is the duplicate group (min size 1, max 60, mean 1.448, 4638 singleton groups) -- no group's members are split across partitions.
