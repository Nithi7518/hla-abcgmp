"""Manifest generation (Phase 5) + PyTorch Dataset (Phase 4) for the ABCGMP dataset.

Manifest generation consumes:
  - results/dataset_audit.csv       (Phase 2: per-image inventory + crop/organ/class)
  - data/manifests/duplicate_groups.csv (Phase 3: leakage-safe grouping units)
and produces:
  - data/manifests/train.csv, val.csv, test.csv
  - data/manifests/split_report.json / .md

The splitting UNIT is the duplicate group, never the individual image -- every member
of a duplicate group is guaranteed to land in the same partition. Groups whose members
carry more than one distinct label (crop OR class) have no trustworthy ground truth and
are excluded entirely by default (see configs/baseline.yaml `exclude_label_conflicting_
duplicate_groups`), not silently kept under one arbitrary label.

Usage:
    python -m src.data_loader --config configs/baseline.yaml --make-splits
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from torch.utils.data import Dataset

DEFAULT_SEED = 42


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


# --------------------------------------------------------------------------------------
# Manifest building
# --------------------------------------------------------------------------------------


def build_labeled_frame(config: dict) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load the audit + duplicate-group outputs and produce one row per usable image,
    with a `group_id` (splitting unit) and `label` (task target) column. Returns the
    frame, an exclusion log (never a silent drop), and a per-label contamination
    report (see inline comment below for why the latter matters)."""
    data_cfg = config["data"]
    audit_df = pd.read_csv(data_cfg["audit_csv"])
    dup_df = pd.read_csv(data_cfg["duplicate_groups_csv"]) if Path(data_cfg["duplicate_groups_csv"]).exists() else pd.DataFrame(
        columns=["group_id", "image_path", "spans_multiple_crops", "spans_multiple_classes"]
    )

    exclusions: dict[str, int] = defaultdict(int)
    total_start = len(audit_df)

    df = audit_df.copy()

    df = df[~df["corrupted"]]
    exclusions["corrupted"] = total_start - len(df)

    before = len(df)
    df = df[df["mapped"]]
    exclusions["unmapped_class_folder"] = before - len(df)

    before = len(df)
    allowed_collections = set(data_cfg.get("source_collections", []))
    if allowed_collections:
        df = df[df["source_collection"].isin(allowed_collections)]
    exclusions["excluded_source_collection"] = before - len(df)

    # Hand-reviewed exclusions: images with a known-wrong label that exact/perceptual
    # duplicate detection cannot catch (every row carries a written reason).
    manual_csv = data_cfg.get("manual_exclusions_csv")
    if manual_csv and Path(manual_csv).exists():
        manual_paths = set(pd.read_csv(manual_csv)["image_path"])
        before = len(df)
        df = df[~df["image_path"].isin(manual_paths)]
        exclusions["manual_exclusion"] = before - len(df)

    # Attach group_id: duplicate-group id where the image belongs to a detected group,
    # else the image's own path (singleton group of size 1).
    group_map = dup_df.set_index("image_path")["group_id"].to_dict() if len(dup_df) else {}
    df = df.copy()
    df["group_id"] = df["image_path"].map(group_map).fillna(df["image_path"])

    if data_cfg.get("exclude_label_conflicting_duplicate_groups", True) and len(dup_df):
        conflicting_groups = set(
            dup_df.loc[
                dup_df["spans_multiple_crops"] | dup_df["spans_multiple_classes"], "group_id"
            ].unique()
        )
        before = len(df)
        df = df[~df["group_id"].isin(conflicting_groups)]
        exclusions["label_conflicting_duplicate_group"] = before - len(df)

    # Transform-robust near-duplicate groups (src/near_duplicates.py): catches flipped /
    # rotated / cropped copies of the same photograph that the hash-based Phase 3 groups
    # miss. They already include the Phase 3 groups, so they replace them as the SPLITTING
    # unit (the conflict exclusion above still uses the original groups).
    robust_csv = data_cfg.get("robust_groups_csv")
    if robust_csv and Path(robust_csv).exists():
        robust_map = pd.read_csv(robust_csv).set_index("image_path")["group_id"].to_dict()
        df["group_id"] = df["image_path"].map(robust_map).fillna(df["group_id"])

    task = data_cfg.get("task", "full_class")
    if task == "full_class":
        df["label"] = df["crop"].astype(str) + "__" + df["canonical_class"].astype(str)
    elif task == "disease_status":
        df["label"] = df["disease_status"].astype(str)
    else:
        raise ValueError(f"Unknown data.task: {task}")

    # Per-label contamination rate: computed on the pre-conflict-exclusion population
    # (i.e. after source-collection filtering but before the exclusion above), so a
    # class that loses ALL of its images to the exclusion (see apple__healthy_fruit --
    # 100% of its 338 dataset_main images turned out to be duplicates of other,
    # differently-labeled images) is visible and explained rather than just vanishing
    # from the manifest with no trace.
    pre_exclusion_df = audit_df[~audit_df["corrupted"] & audit_df["mapped"]]
    if allowed_collections:
        pre_exclusion_df = pre_exclusion_df[pre_exclusion_df["source_collection"].isin(allowed_collections)]
    pre_exclusion_df = pre_exclusion_df.copy()
    if task == "full_class":
        pre_exclusion_df["label"] = (
            pre_exclusion_df["crop"].astype(str) + "__" + pre_exclusion_df["canonical_class"].astype(str)
        )
    else:
        pre_exclusion_df["label"] = pre_exclusion_df["disease_status"].astype(str)
    if len(dup_df):
        conflicting_group_ids = set(
            dup_df.loc[dup_df["spans_multiple_crops"] | dup_df["spans_multiple_classes"], "group_id"].unique()
        )
        pre_group_map = dup_df.set_index("image_path")["group_id"].to_dict()
        pre_exclusion_df["group_id"] = pre_exclusion_df["image_path"].map(pre_group_map).fillna(
            pre_exclusion_df["image_path"]
        )
        pre_exclusion_df["conflicting"] = pre_exclusion_df["group_id"].isin(conflicting_group_ids)
    else:
        pre_exclusion_df["conflicting"] = False
    contamination = (
        pre_exclusion_df.groupby("label")
        .agg(total_images=("image_path", "size"), conflicting_images=("conflicting", "sum"))
    )
    contamination["pct_conflicting"] = (
        100 * contamination["conflicting_images"] / contamination["total_images"]
    ).round(1)
    contamination_report = contamination.sort_values("pct_conflicting", ascending=False).to_dict(orient="index")

    exclusions["final_usable_images"] = len(df)
    exclusions["total_images_in_audit"] = total_start

    return df.reset_index(drop=True), dict(exclusions), contamination_report


# --------------------------------------------------------------------------------------
# Group-aware stratified splitting
# --------------------------------------------------------------------------------------


def stratified_group_split(
    df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, str]:
    """Greedy per-class group assignment: the hard constraint (every duplicate group
    stays in exactly one partition) makes exact proportional stratification generally
    impossible, so groups are assigned split-by-split, largest first, always to
    whichever partition is currently furthest below its target share for that group's
    label. Returns {group_id: split_name}."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.default_rng(seed)
    group_label = df.groupby("group_id")["label"].agg(lambda s: s.value_counts().idxmax())
    group_size = df.groupby("group_id").size()

    targets = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    assigned_count: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    split_of_group: dict[str, str] = {}

    for label, groups_in_label in group_label.groupby(group_label):
        group_ids = list(groups_in_label.index)
        # Shuffle deterministically, then sort largest-first for better bin-packing.
        rng.shuffle(group_ids)
        group_ids.sort(key=lambda g: -group_size[g])

        for g in group_ids:
            counts = assigned_count[label]
            total_for_label = sum(counts.values())
            g_size = group_size[g]

            def deficit(split_name: str) -> float:
                current_share = counts[split_name] / total_for_label if total_for_label else 0.0
                return targets[split_name] - current_share

            best_split = max(targets.keys(), key=deficit)
            split_of_group[g] = best_split
            assigned_count[label][best_split] += g_size

    return split_of_group


def generate_manifests(config: dict) -> dict[str, Any]:
    df, exclusions, contamination_report = build_labeled_frame(config)

    if len(df) == 0:
        raise RuntimeError(
            "No usable images remain after exclusions -- check data.source_collections "
            "and data.exclude_label_conflicting_duplicate_groups in the config."
        )

    split_cfg = config["splitting"]
    split_of_group = stratified_group_split(
        df,
        split_cfg["train_ratio"],
        split_cfg["val_ratio"],
        split_cfg["test_ratio"],
        split_cfg.get("seed", DEFAULT_SEED),
    )
    df["split"] = df["group_id"].map(split_of_group)

    manifests_dir = Path(config["data"]["manifests_dir"])
    manifests_dir.mkdir(parents=True, exist_ok=True)

    output_columns = [
        "image_path",
        "crop",
        "organ",
        "canonical_class",
        "disease_status",
        "disease_name",
        "label",
        "group_id",
        "source_collection",
        "split",
    ]
    manifest_df = df[output_columns].rename(
        columns={
            "image_path": "image_path",
            "canonical_class": "class",
            "disease_status": "binary_disease_status",
            "group_id": "sample_id",
            "source_collection": "source_id",
        }
    )
    manifest_df["metadata_reference"] = None  # METADATA_PAIRING_STATUS = NOT_AVAILABLE (Phase 2)

    for split_name in ["train", "val", "test"]:
        split_df = manifest_df[manifest_df["split"] == split_name].drop(columns=["split"])
        split_df.to_csv(manifests_dir / f"{split_name}.csv", index=False)

    # Report: achieved per-split per-label distribution, exclusions, group stats.
    dist = (
        df.groupby(["split", "label"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    n_groups = df["group_id"].nunique()
    group_sizes = df.groupby("group_id").size()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": config["data"].get("task", "full_class"),
        "source_collections_used": config["data"].get("source_collections", []),
        "exclude_label_conflicting_duplicate_groups": config["data"].get(
            "exclude_label_conflicting_duplicate_groups", True
        ),
        "exclusions": exclusions,
        "n_images_total": len(df),
        "n_groups_total": int(n_groups),
        "group_size_distribution": {
            "min": int(group_sizes.min()),
            "max": int(group_sizes.max()),
            "mean": round(float(group_sizes.mean()), 3),
            "n_singleton_groups": int((group_sizes == 1).sum()),
        },
        "split_ratios_config": {
            "train": split_cfg["train_ratio"],
            "val": split_cfg["val_ratio"],
            "test": split_cfg["test_ratio"],
        },
        "split_seed": split_cfg.get("seed", DEFAULT_SEED),
        "n_images_per_split": df["split"].value_counts().to_dict(),
        "n_labels": int(df["label"].nunique()),
        "label_distribution_per_split": dist,
        "per_label_contamination": contamination_report,
    }

    with open(manifests_dir / "split_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    write_split_report_md(manifests_dir / "split_report.md", report)

    return report


def write_split_report_md(path: Path, report: dict) -> None:
    lines = []
    lines.append("# Split Report")
    lines.append("")
    lines.append(f"- **Task:** `{report['task']}`")
    lines.append(f"- **Source collections used:** {report['source_collections_used']}")
    lines.append(
        f"- **Label-conflicting duplicate groups excluded:** "
        f"{report['exclude_label_conflicting_duplicate_groups']}"
    )
    lines.append(f"- **Split seed:** {report['split_seed']}")
    lines.append("")
    lines.append("## Exclusions (nothing silently dropped)")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|---|---|")
    for reason, count in report["exclusions"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")
    lines.append("## Per-label contamination rate (before exclusion)")
    lines.append("")
    lines.append(
        "What fraction of each label's images (in the source collection(s) used, before "
        "the label-conflicting-duplicate-group exclusion above) turned out to be part of "
        "a group whose members carry more than one distinct crop or class label. A rate "
        "of 100% means the label has ZERO trustworthy images left and will not appear in "
        "any split below -- see `results/duplicate_detection_summary.md` for the "
        "underlying cause (apple stock photos duplicated into unrelated folders)."
    )
    lines.append("")
    lines.append("| Label | Total images | Conflicting images | % conflicting |")
    lines.append("|---|---|---|---|")
    for label, stats in report["per_label_contamination"].items():
        flag = " **← entire class excluded**" if stats["pct_conflicting"] == 100.0 else ""
        lines.append(
            f"| {label} | {int(stats['total_images'])} | {int(stats['conflicting_images'])} | "
            f"{stats['pct_conflicting']}%{flag} |"
        )
    lines.append("")

    lines.append("## Images per split")
    lines.append("")
    lines.append("| Split | Images |")
    lines.append("|---|---|")
    for split_name, count in report["n_images_per_split"].items():
        lines.append(f"| {split_name} | {count} |")
    lines.append("")
    lines.append(f"## Label distribution per split ({report['n_labels']} labels)")
    lines.append("")
    labels = sorted({lbl for split_counts in report["label_distribution_per_split"].values() for lbl in split_counts})
    header = "| Label | " + " | ".join(report["label_distribution_per_split"].keys()) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(report["label_distribution_per_split"]) + 1))
    for lbl in labels:
        row = [str(report["label_distribution_per_split"][s].get(lbl, 0)) for s in report["label_distribution_per_split"]]
        lines.append(f"| {lbl} | " + " | ".join(row) + " |")
    lines.append("")
    lines.append(
        f"Splitting unit is the duplicate group (min size {report['group_size_distribution']['min']}, "
        f"max {report['group_size_distribution']['max']}, mean "
        f"{report['group_size_distribution']['mean']}, "
        f"{report['group_size_distribution']['n_singleton_groups']} singleton groups) -- "
        "no group's members are split across partitions."
    )
    lines.append("")
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# PyTorch Dataset
# --------------------------------------------------------------------------------------


class ABCGMPImageDataset(Dataset):
    """Reads a manifest CSV (data/manifests/{train,val,test}.csv) and serves
    (image_tensor, label_idx) pairs. Metadata is intentionally not attached --
    METADATA_PAIRING_STATUS = NOT_AVAILABLE for this dataset release (Phase 2)."""

    def __init__(
        self,
        manifest_csv: str | Path,
        data_root: str | Path,
        label_to_idx: dict[str, int],
        transform=None,
        cache_in_memory: bool = True,
    ):
        self.manifest = pd.read_csv(manifest_csv)
        self.data_root = Path(data_root)
        self.label_to_idx = label_to_idx
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self._cache: dict[int, Image.Image] = {} if cache_in_memory else {}

        self.image_paths = [self.data_root / p for p in self.manifest["image_path"]]
        self.labels = [self.label_to_idx[l] for l in self.manifest["label"]]

        unknown = set(self.manifest["label"].unique()) - set(label_to_idx.keys())
        if unknown:
            raise ValueError(
                f"Manifest {manifest_csv} contains labels not in label_to_idx: {unknown}. "
                "Regenerate the label map or the manifest."
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.cache_in_memory and idx in self._cache:
            image = self._cache[idx].copy()
        else:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            if self.cache_in_memory:
                self._cache[idx] = image.copy()

        if self.transform is not None:
            image = self.transform(image)
        label_idx = self.labels[idx]
        return image, label_idx


def build_label_map(manifests_dir: str | Path) -> dict[str, int]:
    """Label indices are derived from the union of labels actually present across all
    three splits (deterministic, sorted) -- not "fit" on train the way normalization
    statistics are, since class identity carries no leakage risk. Saved for
    reproducibility."""
    manifests_dir = Path(manifests_dir)
    labels: set[str] = set()
    for split_name in ["train", "val", "test"]:
        path = manifests_dir / f"{split_name}.csv"
        if path.exists():
            labels |= set(pd.read_csv(path)["label"].unique())
    label_to_idx = {label: i for i, label in enumerate(sorted(labels))}
    with open(manifests_dir / "label_map.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    return label_to_idx


def load_label_map(manifests_dir: str | Path) -> dict[str, int]:
    path = Path(manifests_dir) / "label_map.json"
    if not path.exists():
        return build_label_map(manifests_dir)
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument(
        "--make-splits", action="store_true", help="Generate data/manifests/{train,val,test}.csv."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(Path(args.config))

    if not args.make_splits:
        print("Nothing to do -- pass --make-splits to generate manifests.", file=sys.stderr)
        sys.exit(1)

    for required in ["audit_csv", "duplicate_groups_csv"]:
        p = Path(config["data"][required])
        if not p.exists():
            print(f"ERROR: {p} not found. Run the earlier phases first.", file=sys.stderr)
            sys.exit(1)

    report = generate_manifests(config)
    build_label_map(config["data"]["manifests_dir"])

    print("SPLIT GENERATION COMPLETE")
    print()
    print(f"Task:            {report['task']}")
    print(f"Total images:    {report['n_images_total']}")
    print(f"Total groups:    {report['n_groups_total']}")
    print(f"Labels:          {report['n_labels']}")
    print(f"Images per split: {report['n_images_per_split']}")
    print()
    print(f"Manifests written to: {config['data']['manifests_dir']}/{{train,val,test}}.csv")
    print(f"Report written to:    {config['data']['manifests_dir']}/split_report.md")


if __name__ == "__main__":
    main()
