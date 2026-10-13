# Phase 12 (Multi-task fruit/leaf learning): Not Supported

**Status:** formally blocked for this dataset release, for the same reason in spirit as Phases 9-10:
the experiment's premise (a *paired* fruit-and-leaf observation of the same plant) cannot be
established from the data, and inventing a pairing would fabricate ground truth.

## What was checked (2026-10-11)

The multi-task question needs fruit and leaf images that can be linked to the **same plant or
sample**, so a model can learn jointly from both organs of one specimen. We looked for any
linkage in `data/manifests/*.csv` (8,936 usable images):

1. **Shared identifiers.** Every class folder uses its own, unrelated naming scheme
   (e.g. `DAF (n).jpeg` apple fruit, `ABR (n).JPG` apple leaf, `HBF (n)` vs `fk/ft/fresh` banana,
   `IMG_x_y` guava, `PFD`/`PHL` papaya). No filename stem, index or ID appears in both a fruit
   and a leaf folder of the same crop that would identify a plant. Index numbers restart in each
   folder and carry no cross-folder meaning.
2. **Embedded metadata.** In a random sample of 400 images only 53 (13%) carry any EXIF block, and
   none of those carry a capture time or GPS position, so photos cannot be linked by when or
   where they were taken.
3. **Organ labels themselves are incomplete.** 1,119 images (12.5%) have `organ = unknown`
   (all of `mango__bacterial_canker` and `papaya__ring_spot`, where the dataset description does
   not say which organ is photographed), and several other organ labels are only low-confidence
   (see `configs/classes.yaml`). The class mix is also heavily skewed: 6,136 leaf vs 1,681 fruit
   images, with apple 97% leaf.
4. **Image similarity.** Visually linking fruit and leaf photos of one plant is not feasible: a
   leaf and a fruit of the same tree share no reliable appearance cue, and the only
   same-photograph relationships we can detect (see `src/near_duplicates.py`) are augmented copies
   *within* a folder, not across organs.

## Consequence

A paired multi-task model (fruit + leaf of one plant -> joint diagnosis) is **not trainable or
evaluable** with this release. An *unpaired* variant (a shared backbone with separate crop /
organ / disease heads on independent photos) is possible, but it is a different experiment: it
tests auxiliary-task regularisation, not joint fruit/leaf reasoning, and organ labels are
missing for 12.5% of images, so we did not substitute it for Phase 12 and make no claim about it.

## What would unblock it

A dataset version that records a plant/sample/tree identifier (or capture metadata) per image, or
a re-collection in which fruit and leaf photos of each plant are captured together. Then:
re-run `python -m src.dataset_audit`, build a pairing table keyed by that identifier, and add a
paired data loader before any model code.
