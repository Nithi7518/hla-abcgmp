"""Preprocessing (Phase 6): image transforms + a documented-disabled tabular stub.

Image transforms are entirely config-driven (configs/baseline.yaml `image:` block) --
resolution, normalization, and train-time augmentation are never hard-coded here.
Train-time augmentation is deliberately mild (small rotation, mild color jitter,
horizontal flip): aggressive augmentation risks destroying disease morphology (lesion
shape/color/texture) that the model needs to learn. Validation/test transforms are
always deterministic (resize + normalize only, no stochastic augmentation).

Tabular preprocessing (imputation/encoding/scaling) is NOT implemented: this dataset's
METADATA_PAIRING_STATUS is NOT_AVAILABLE (Phase 2 audit -- no structured metadata
files exist in the downloaded archives). `TabularPreprocessor` below exists only to
fail loudly and explain why, rather than being silently absent, so nothing downstream
can accidentally construct fabricated tabular features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transforms(image_cfg: dict[str, Any]) -> dict[str, T.Compose]:
    """Returns {'train': ..., 'eval': ...} torchvision transform pipelines built from
    configs/baseline.yaml's `image:` block. `eval` is used for both validation and
    test -- there is no stochastic augmentation outside of `train`."""
    size = image_cfg.get("size", 224)
    normalization = image_cfg.get("normalization", "imagenet")

    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "none":
        mean, std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    else:
        raise ValueError(f"Unknown image.normalization: {normalization}")

    eval_transform = T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    aug_cfg = image_cfg.get("train_augmentation", {})
    train_ops: list = [T.Resize((size, size))]
    if aug_cfg.get("horizontal_flip", False):
        train_ops.append(T.RandomHorizontalFlip(p=0.5))
    if aug_cfg.get("vertical_flip", False):
        train_ops.append(T.RandomVerticalFlip(p=0.5))
    rotation = aug_cfg.get("rotation_degrees", 0)
    if rotation:
        train_ops.append(T.RandomRotation(degrees=rotation))
    jitter = aug_cfg.get("color_jitter", 0)
    if jitter:
        train_ops.append(T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=jitter / 2))
    train_ops += [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    if aug_cfg.get("random_erasing", 0):
        train_ops.append(T.RandomErasing(p=aug_cfg["random_erasing"], scale=(0.02, 0.2)))
    train_transform = T.Compose(train_ops)

    return {"train": train_transform, "eval": eval_transform}


def save_preprocessing_config(image_cfg: dict[str, Any], output_path: str | Path) -> None:
    """Persists the exact image-preprocessing configuration used for a run, for
    reproducibility (Section 23) -- so a saved checkpoint can always be paired with the
    transforms it was trained under."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "size": image_cfg.get("size", 224),
                "normalization": image_cfg.get("normalization", "imagenet"),
                "mean": IMAGENET_MEAN if image_cfg.get("normalization", "imagenet") == "imagenet" else None,
                "std": IMAGENET_STD if image_cfg.get("normalization", "imagenet") == "imagenet" else None,
                "train_augmentation": image_cfg.get("train_augmentation", {}),
            },
            f,
            indent=2,
        )


class TabularPreprocessor:
    """Deliberately unimplemented.

    This dataset's METADATA_PAIRING_STATUS is NOT_AVAILABLE (see
    results/dataset_audit.json -> metadata_summary, results/dataset_summary.md).
    No structured metadata file exists in the downloaded ABCGMP archives, so there is
    nothing to impute, encode, or scale. Constructing this class raises immediately
    rather than silently returning an empty/no-op transform, so that any future code
    that tries to build a multimodal pipeline fails fast and points back to the actual
    blocker (find real metadata and re-run the Phase 2 audit) instead of quietly
    training on zero-value or fabricated features.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "TabularPreprocessor is intentionally unimplemented: this dataset's "
            "METADATA_PAIRING_STATUS is NOT_AVAILABLE (no structured metadata files "
            "were found under data/raw -- see results/dataset_summary.md). Fitting "
            "imputation/encoding/scaling on non-existent metadata would require "
            "fabricating features, which is explicitly prohibited by this project's "
            "research-validity safeguards (see README section 18). If a future "
            "dataset version ships real metadata, re-run `python -m src.dataset_audit`, "
            "confirm METADATA_PAIRING_STATUS is VALID or PARTIAL, define "
            "configs/metadata_mapping.yaml against the actual columns present, and "
            "only then implement this class."
        )
