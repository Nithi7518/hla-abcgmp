# Phase 9 (Multimodal Fusion) and Phase 10 (Metadata Ablation): Not Supported

**Status:** Both experiments are formally disabled for this dataset release. This is
not a gap in the implementation — the architecture and configuration for both exist
(`src/models.py::MultimodalFusionModel`, `configs/multimodal.yaml`) and
`python -m src.train --config configs/multimodal.yaml` will refuse to run and print
this same explanation. This document records the finding for the record and for
anyone reviewing the project's phase completion.

## Why

The project's primary research question — does agronomic metadata improve disease
diagnosis over an image-only model? — requires per-image or per-sample metadata
(soil health, humidity, nutrient deficiency, pathological reports, per the Mendeley
dataset description) that can be reliably linked to specific images.

`python -m src.dataset_audit` (Phase 2), run against the actual downloaded archives
from <https://data.mendeley.com/datasets/fhbvmpcyy2/2> (DOI 10.17632/fhbvmpcyy2.2),
found:

```
metadata_files_found: []
n_metadata_files: 0
METADATA_PAIRING_STATUS: NOT_AVAILABLE
```

**No CSV, JSON, XLSX, or XLS file exists anywhere in either downloaded archive.** The
dataset's own description text mentions expert ground-truth annotations, but no such
file is present in the actual download — only image files (`.jpg`/`.jpeg`/`.png`)
across 24 class folders. This was independently re-confirmed while building Phase
5–8 (nothing discovered since Phase 2 changed this finding).

See [`results/dataset_summary.md`](../dataset_summary.md) → "Metadata audit" for the
full audit output.

## What was built anyway, and why

Per the project's own instructions (Section 17: *"if a valid fruit-leaf pairing does
not exist, skip the multi-task experiment and explain why"* — the same principle
applies here) and the general research-validity safeguards (Section 28: *"if an
experiment is scientifically invalid because of the dataset structure, explicitly
disable it and explain why"*), the correct response to a blocked experiment is not to
silently omit it, nor to fabricate the missing input, but to:

1. **Implement the architecture** so it is ready the moment real metadata exists.
   `MultimodalFusionModel` (`src/models.py`) implements the exact design from the
   project spec — image CNN → embedding, metadata → MLP → embedding, concatenate →
   fusion MLP → classification head — and passes a shape-correctness smoke test
   (`python -m src.models`).
2. **Implement the configuration** (`configs/multimodal.yaml`) with the fields a real
   run would need (backbone, embedding dimensions, fusion hidden size, which metadata
   fields to use), so wiring up a real run later is a config edit, not new code.
3. **Fail loudly and specifically**, not silently. `src/train.py::check_metadata_feasibility`
   reads `results/dataset_audit.json`'s `METADATA_PAIRING_STATUS` before touching any
   data, and if it isn't `VALID` or `PARTIAL`, exits immediately with the exact
   reason and a pointer back to this document — instead of either crashing
   confusingly deep in a data-loading step, or (far worse) silently training an
   image-only model while claiming to be the multimodal experiment.
4. **Never fabricate the missing input.** No synthetic metadata, no metadata inferred
   from the crop or disease label (which would leak the target into the "feature"),
   no metadata duplicated arbitrarily across images. All of these are explicitly
   prohibited by the project's safeguards, and all of them would produce a number
   that looks like a real multimodal result but isn't one.

## Phase 10 (metadata ablation) specifically

Phase 10 (image + humidity, image + soil health, image + all metadata, etc.) is a
strict superset of the Phase 9 blocker: it requires not just *some* metadata, but
several *distinct, individually verified* metadata fields to ablate between. Since
zero metadata fields exist, Phase 10 is blocked for the same reason and is not
separately implemented — there is nothing to ablate.

## What would unblock this

1. A future version of the ABCGMP dataset (or a related dataset) that ships an actual
   structured metadata file alongside the images.
2. Re-run `python -m src.dataset_audit --data-root <new data root>` and confirm
   `metadata_summary.metadata_pairing_status` is `VALID` or `PARTIAL` — not just that
   a metadata file exists, but that it can be verifiably linked to specific images or
   samples (see `results/metadata_audit.json` → `candidate_id_columns` for what the
   audit looks for).
3. Create `configs/metadata_mapping.yaml` describing the actual columns present
   (never populated speculatively ahead of time).
4. Update `configs/multimodal.yaml`'s `metadata.numeric_fields` /
   `categorical_fields` / `text_fields` to match, noting Section 15's caveat that raw
   pathological-report text must not be fed directly into the metadata MLP, and must
   never be derived from the disease label itself (that would be target leakage, not
   metadata).
5. Only then run `python -m src.train --config configs/multimodal.yaml`, followed by
   the controlled A/B/C comparison (image-only vs. metadata-only vs. image+metadata)
   the project spec requires before any claim about metadata's contribution.

## Bottom line

Phases 9 and 10 are complete in the only sense that is scientifically honest given
this dataset: implemented, configured, and correctly refusing to run, with the exact
reason recorded. Reporting a multimodal result from this dataset release would be
reporting a number that does not measure what it claims to measure.
