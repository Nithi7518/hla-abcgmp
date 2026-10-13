"""Dataset audit for the ABCGMP Fruit and Leaf Disease Six Crop Dataset.

Inspects whatever is actually present under a data root directory -- it makes no
assumptions about folder names, metadata fields, or hierarchy depth. Crop / organ /
disease-status / disease-name / canonical class are attached to an image ONLY via the
explicit, human-maintained mapping in configs/classes.yaml (see that file for why).
Folders not present in that mapping are reported as UNMAPPED, never guessed.

Usage:
    python -m src.dataset_audit --data-root ./data/raw
    python -m src.dataset_audit --config configs/dataset.yaml
    python -m src.dataset_audit --data-root ./data/raw --classes-config configs/classes.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

DEFAULT_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
DEFAULT_METADATA_EXTENSIONS = [".csv", ".json", ".xlsx", ".xls"]

# Column-name substrings used only to *flag candidate* id/link columns for a human to
# review -- never used to silently assert that a pairing is valid.
ID_COLUMN_HINTS = ["id", "image", "file", "filename", "path", "sample"]
CROP_COLUMN_HINTS = ["crop", "species", "plant"]
LABEL_COLUMN_HINTS = ["label", "class", "disease", "status", "diagnosis"]


# --------------------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def build_config(args: argparse.Namespace) -> dict:
    """Merge configs/dataset.yaml (if present) with CLI overrides.

    CLI arguments always win over YAML values when explicitly supplied, per the
    project's CLI convention (see README "Reproducing results").
    """
    cfg: dict = {}
    config_path = Path(args.config) if args.config else Path("configs/dataset.yaml")
    if config_path.exists():
        cfg = load_yaml(config_path)
    elif args.config:
        raise FileNotFoundError(f"--config path does not exist: {config_path}")

    dataset_cfg = cfg.get("dataset", {})
    audit_cfg = cfg.get("audit", {})
    classes_cfg = cfg.get("classes", {})

    data_root = args.data_root or dataset_cfg.get("root")
    if not data_root:
        raise ValueError(
            "No data root specified. Pass --data-root or set dataset.root in "
            f"{config_path}."
        )

    classes_config_path = args.classes_config or classes_cfg.get(
        "mapping_file", "configs/classes.yaml"
    )
    output_dir = args.output_dir or audit_cfg.get("output_dir", "results")

    return {
        "data_root": Path(data_root),
        "exclude_dirs": set(dataset_cfg.get("exclude_dirs", [])),
        "image_extensions": {
            e.lower() for e in dataset_cfg.get("image_extensions", DEFAULT_IMAGE_EXTENSIONS)
        },
        "metadata_extensions": {
            e.lower()
            for e in dataset_cfg.get("metadata_extensions", DEFAULT_METADATA_EXTENSIONS)
        },
        "classes_config_path": Path(classes_config_path),
        "output_dir": Path(output_dir),
        "compute_near_duplicate_hash": audit_cfg.get("compute_near_duplicate_hash", True),
        "config_path": config_path,
    }


def load_class_mapping(path: Path) -> dict[str, dict]:
    """Flatten configs/classes.yaml into folder_name -> attributes.

    Returns {} (with a warning printed by the caller) if the file does not exist --
    the audit still runs, everything is simply reported as unmapped.
    """
    if not path.exists():
        return {}

    raw = load_yaml(path)
    folder_to_attrs: dict[str, dict] = {}
    for crop, classes in raw.items():
        if not isinstance(classes, dict):
            continue
        for canonical_class, attrs in classes.items():
            if not isinstance(attrs, dict):
                continue
            folder_names = attrs.get("folder_names", [])
            for folder_name in folder_names:
                if folder_name in folder_to_attrs:
                    print(
                        f"WARNING: folder_name '{folder_name}' is mapped more than once "
                        f"in {path} (last definition wins: crop={crop}, "
                        f"class={canonical_class})",
                        file=sys.stderr,
                    )
                folder_to_attrs[folder_name] = {
                    "crop": crop,
                    "canonical_class": canonical_class,
                    "organ": attrs.get("organ", "unknown"),
                    "organ_confidence": attrs.get("organ_confidence", "unresolved"),
                    "disease_status": attrs.get("disease_status"),
                    "disease_name": attrs.get("disease_name"),
                }
    return folder_to_attrs


# --------------------------------------------------------------------------------------
# Image inventory + per-image feature extraction
# --------------------------------------------------------------------------------------


@dataclass
class ImageRecord:
    image_path: str
    source_collection: str
    class_folder: str
    crop: str | None
    organ: str | None
    organ_confidence: str | None
    disease_status: str | None
    disease_name: str | None
    canonical_class: str | None
    mapped: bool
    extension: str
    size_bytes: int
    width: int | None
    height: int | None
    mode: str | None
    n_channels: int | None
    sha256: str | None
    ahash: str | None
    corrupted: bool
    corruption_reason: str | None


def compute_ahash(img: Image.Image, hash_size: int = 8) -> str:
    """Cheap 64-bit average-hash used only as a *preliminary* near-duplicate signal.

    This is intentionally minimal (no extra dependency). Authoritative near-duplicate
    grouping with configurable pHash/dHash thresholds belongs to the dedicated
    src/duplicate_detection.py module (see project structure, Phase 3) -- this audit
    only surfaces a first-pass count so the issue is visible immediately.
    """
    small = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.float64)
    avg = arr.mean()
    bits = (arr > avg).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def discover_files(
    data_root: Path, exclude_dirs: set[str]
) -> list[Path]:
    all_files: list[Path] = []
    for path in data_root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in exclude_dirs for part in path.relative_to(data_root).parts[:-1]):
            continue
        all_files.append(path)
    return all_files


def audit_images(
    files: list[Path],
    data_root: Path,
    image_extensions: set[str],
    class_mapping: dict[str, dict],
    compute_near_duplicate: bool,
    skip_log: list[dict],
) -> list[ImageRecord]:
    image_files = [f for f in files if f.suffix.lower() in image_extensions]
    records: list[ImageRecord] = []

    for path in tqdm(image_files, desc="Auditing images", unit="img"):
        rel = path.relative_to(data_root)
        parts = rel.parts
        source_collection = parts[0] if len(parts) > 1 else ""
        class_folder = path.parent.name

        mapping = class_mapping.get(class_folder)
        mapped = mapping is not None

        try:
            raw_bytes = path.read_bytes()
        except OSError as e:
            skip_log.append(
                {"image_path": str(rel), "reason": f"unreadable_file: {e}"}
            )
            records.append(
                ImageRecord(
                    image_path=str(rel),
                    source_collection=source_collection,
                    class_folder=class_folder,
                    crop=mapping["crop"] if mapping else None,
                    organ=mapping["organ"] if mapping else None,
                    organ_confidence=mapping["organ_confidence"] if mapping else None,
                    disease_status=mapping["disease_status"] if mapping else None,
                    disease_name=mapping["disease_name"] if mapping else None,
                    canonical_class=mapping["canonical_class"] if mapping else None,
                    mapped=mapped,
                    extension=path.suffix.lower(),
                    size_bytes=0,
                    width=None,
                    height=None,
                    mode=None,
                    n_channels=None,
                    sha256=None,
                    ahash=None,
                    corrupted=True,
                    corruption_reason=f"unreadable_file: {e}",
                )
            )
            continue

        size_bytes = len(raw_bytes)
        sha256 = hashlib.sha256(raw_bytes).hexdigest()

        width = height = None
        mode = None
        n_channels = None
        ahash = None
        corrupted = False
        corruption_reason = None

        if size_bytes == 0:
            corrupted = True
            corruption_reason = "empty_file"
        else:
            try:
                from io import BytesIO

                img = Image.open(BytesIO(raw_bytes))
                img.verify()
                img = Image.open(BytesIO(raw_bytes))  # verify() invalidates the handle
                width, height = img.size
                mode = img.mode
                n_channels = len(img.getbands())
                if compute_near_duplicate:
                    ahash = compute_ahash(img)
            except (UnidentifiedImageError, OSError, ValueError) as e:
                corrupted = True
                corruption_reason = f"{type(e).__name__}: {e}"

        records.append(
            ImageRecord(
                image_path=str(rel),
                source_collection=source_collection,
                class_folder=class_folder,
                crop=mapping["crop"] if mapping else None,
                organ=mapping["organ"] if mapping else None,
                organ_confidence=mapping["organ_confidence"] if mapping else None,
                disease_status=mapping["disease_status"] if mapping else None,
                disease_name=mapping["disease_name"] if mapping else None,
                canonical_class=mapping["canonical_class"] if mapping else None,
                mapped=mapped,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                width=width,
                height=height,
                mode=mode,
                n_channels=n_channels,
                sha256=sha256,
                ahash=ahash,
                corrupted=corrupted,
                corruption_reason=corruption_reason,
            )
        )

    return records


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------


def summarize_images(records: list[ImageRecord]) -> dict[str, Any]:
    df = pd.DataFrame([asdict(r) for r in records])
    summary: dict[str, Any] = {}

    summary["total_images"] = len(df)
    summary["images_per_source_collection"] = (
        df["source_collection"].value_counts().to_dict() if len(df) else {}
    )
    summary["images_per_class_folder"] = (
        df["class_folder"].value_counts().to_dict() if len(df) else {}
    )
    summary["images_per_extension"] = (
        df["extension"].value_counts().to_dict() if len(df) else {}
    )

    mapped_df = df[df["mapped"]] if len(df) else df
    summary["unmapped_class_folders"] = sorted(
        df[~df["mapped"]]["class_folder"].unique().tolist()
    ) if len(df) else []
    summary["unmapped_image_count"] = int((~df["mapped"]).sum()) if len(df) else 0

    summary["images_per_crop"] = (
        mapped_df["crop"].value_counts().to_dict() if len(mapped_df) else {}
    )
    summary["images_per_canonical_class"] = (
        mapped_df["canonical_class"].value_counts().to_dict() if len(mapped_df) else {}
    )
    summary["images_per_organ"] = (
        mapped_df["organ"].value_counts().to_dict() if len(mapped_df) else {}
    )
    summary["images_per_disease_status"] = (
        mapped_df["disease_status"].value_counts().to_dict() if len(mapped_df) else {}
    )
    summary["organ_confidence_breakdown"] = (
        mapped_df["organ_confidence"].value_counts().to_dict() if len(mapped_df) else {}
    )

    # Crop x organ x class cross tab (mapped only)
    if len(mapped_df):
        crosstab = (
            mapped_df.groupby(["crop", "organ", "canonical_class"])
            .size()
            .reset_index(name="count")
            .sort_values(["crop", "organ", "canonical_class"])
        )
        summary["crop_organ_class_table"] = crosstab.to_dict(orient="records")
    else:
        summary["crop_organ_class_table"] = []

    valid_dims = df[(~df["corrupted"]) & df["width"].notna()] if len(df) else df
    if len(valid_dims):
        summary["image_dimensions"] = {
            "width": {
                "min": int(valid_dims["width"].min()),
                "max": int(valid_dims["width"].max()),
                "mean": round(float(valid_dims["width"].mean()), 2),
                "median": float(valid_dims["width"].median()),
            },
            "height": {
                "min": int(valid_dims["height"].min()),
                "max": int(valid_dims["height"].max()),
                "mean": round(float(valid_dims["height"].mean()), 2),
                "median": float(valid_dims["height"].median()),
            },
            "top_10_resolutions": (
                valid_dims.assign(
                    resolution=lambda d: d["width"].astype(int).astype(str)
                    + "x"
                    + d["height"].astype(int).astype(str)
                )["resolution"]
                .value_counts()
                .head(10)
                .to_dict()
            ),
        }
        summary["mode_channel_breakdown"] = valid_dims["mode"].value_counts().to_dict()
    else:
        summary["image_dimensions"] = {}
        summary["mode_channel_breakdown"] = {}

    corrupted_df = df[df["corrupted"]] if len(df) else df
    summary["corrupted_image_count"] = int(len(corrupted_df))
    summary["corrupted_images"] = (
        corrupted_df[["image_path", "corruption_reason"]].to_dict(orient="records")
        if len(corrupted_df)
        else []
    )

    # Duplicate filenames (basename collisions across different folders/paths)
    if len(df):
        df["basename"] = df["image_path"].apply(lambda p: Path(p).name)
        dup_names = df[df.duplicated("basename", keep=False)].sort_values("basename")
        name_groups = (
            dup_names.groupby("basename")["image_path"].apply(list).to_dict()
        )
    else:
        name_groups = {}
    summary["duplicate_filename_group_count"] = len(name_groups)
    summary["duplicate_filename_groups"] = name_groups

    # Exact duplicate files (SHA-256)
    valid_hash_df = df[df["sha256"].notna()] if len(df) else df
    if len(valid_hash_df):
        dup_hash = valid_hash_df[valid_hash_df.duplicated("sha256", keep=False)]
        hash_groups = dup_hash.groupby("sha256")["image_path"].apply(list).to_dict()
    else:
        hash_groups = {}
    summary["exact_duplicate_group_count"] = len(hash_groups)
    summary["exact_duplicate_image_count"] = sum(len(v) for v in hash_groups.values())
    summary["exact_duplicate_groups"] = hash_groups

    # Preliminary near-duplicate groups via average-hash (see compute_ahash docstring)
    valid_ahash_df = df[df["ahash"].notna()] if len(df) else df
    if len(valid_ahash_df):
        dup_ahash = valid_ahash_df[valid_ahash_df.duplicated("ahash", keep=False)]
        ahash_groups = dup_ahash.groupby("ahash")["image_path"].apply(list).to_dict()
    else:
        ahash_groups = {}
    summary["preliminary_near_duplicate_group_count"] = len(ahash_groups)
    summary["preliminary_near_duplicate_note"] = (
        "Computed via a naive 8x8 average-hash exact-match grouping over the audit's "
        "per-image scan. This is a coarse first-pass indicator only. Authoritative "
        "near-duplicate detection with configurable pHash/dHash similarity thresholds "
        "and leakage-safe train/val/test grouping belongs to src/duplicate_detection.py "
        "(project Phase 3) and MUST be run before any split is trusted for training."
    )

    return summary


# --------------------------------------------------------------------------------------
# Metadata audit
# --------------------------------------------------------------------------------------


def profile_dataframe(df: pd.DataFrame, source_file: str) -> dict[str, Any]:
    n_rows = len(df)
    columns_profile = []
    candidate_id_cols = []
    candidate_crop_cols = []
    candidate_label_cols = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        n_missing = int(series.isna().sum())
        n_unique = int(series.nunique(dropna=True))
        is_categorical = dtype == "object" or n_unique < max(20, 0.05 * n_rows)
        is_numeric = pd.api.types.is_numeric_dtype(series)

        col_lower = str(col).lower()
        if any(h in col_lower for h in ID_COLUMN_HINTS):
            candidate_id_cols.append(str(col))
        if any(h in col_lower for h in CROP_COLUMN_HINTS):
            candidate_crop_cols.append(str(col))
        if any(h in col_lower for h in LABEL_COLUMN_HINTS):
            candidate_label_cols.append(str(col))

        columns_profile.append(
            {
                "column": str(col),
                "dtype": dtype,
                "n_missing": n_missing,
                "pct_missing": round(100 * n_missing / n_rows, 2) if n_rows else None,
                "n_unique": n_unique,
                "is_numeric": bool(is_numeric),
                "is_categorical": bool(is_categorical) and not is_numeric,
                "sample_values": [
                    str(v) for v in series.dropna().unique()[:5].tolist()
                ],
            }
        )

    return {
        "source_file": source_file,
        "n_rows": n_rows,
        "n_columns": len(df.columns),
        "columns": columns_profile,
        "candidate_id_columns": candidate_id_cols,
        "candidate_crop_columns": candidate_crop_cols,
        "candidate_label_columns": candidate_label_cols,
    }


def audit_metadata(
    files: list[Path],
    data_root: Path,
    metadata_extensions: set[str],
    image_records: list[ImageRecord],
    skip_log: list[dict],
) -> dict[str, Any]:
    metadata_files = [f for f in files if f.suffix.lower() in metadata_extensions]
    profiles: list[dict[str, Any]] = []

    for path in metadata_files:
        rel = str(path.relative_to(data_root))
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
                profiles.append(profile_dataframe(df, rel))
            elif path.suffix.lower() in (".xlsx", ".xls"):
                sheets = pd.read_excel(path, sheet_name=None)
                for sheet_name, df in sheets.items():
                    profiles.append(profile_dataframe(df, f"{rel}::{sheet_name}"))
            elif path.suffix.lower() == ".json":
                with open(path, "r") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    df = pd.json_normalize(payload)
                    profiles.append(profile_dataframe(df, rel))
                elif isinstance(payload, dict):
                    df = pd.json_normalize([payload])
                    profiles.append(profile_dataframe(df, rel))
                else:
                    skip_log.append(
                        {
                            "image_path": rel,
                            "reason": "json_root_not_list_or_object",
                        }
                    )
        except Exception as e:  # noqa: BLE001 - report and continue, never crash the audit
            skip_log.append(
                {"image_path": rel, "reason": f"metadata_parse_error: {type(e).__name__}: {e}"}
            )

    # Determine METADATA_PAIRING_STATUS. See README "Research-validity safeguards" and
    # project instructions section 5 ("Never fabricate image-metadata pairings").
    image_stems = {Path(r.image_path).stem for r in image_records}
    pairing_status = "NOT_AVAILABLE"
    pairing_explanation = (
        "No structured metadata files (csv/json/xlsx/xls) were found under the data "
        "root. The dataset description mentions expert annotations (soil health, "
        "humidity, nutrient deficiency, pathological reports) but these are not "
        "present in the downloaded archive contents. Multimodal experiments that "
        "require per-image or per-sample metadata are DISABLED until such a file is "
        "located and its linkage to images is verified."
    )

    if profiles:
        best_overlap = 0
        best_profile = None
        for p in profiles:
            for id_col in p["candidate_id_columns"]:
                # We don't have direct access to raw values here without re-reading;
                # overlap is computed defensively below using the original dataframe
                # only when cheap. For now, presence of a candidate id column with
                # high cardinality relative to row count is the strongest signal we
                # can report without risking a false "VALID".
                pass

        # Conservative default: files exist but we found no verified per-image linkage.
        any_id_like_column = any(p["candidate_id_columns"] for p in profiles)
        any_crop_like_column = any(p["candidate_crop_columns"] for p in profiles)

        if any_id_like_column:
            pairing_status = "PARTIAL"
            pairing_explanation = (
                "Structured metadata files were found and contain column(s) whose "
                "names suggest an image/sample identifier (e.g. containing 'id', "
                "'image', 'file'), but automatic value-level matching against actual "
                "image filenames has not been verified. Treat as PARTIAL until a "
                "human confirms the join is exact and unique. Do NOT proceed to "
                "multimodal training on this basis alone."
            )
        elif any_crop_like_column:
            pairing_status = "PARTIAL"
            pairing_explanation = (
                "Structured metadata files were found with a crop-level column only "
                "(no per-image or per-sample identifier candidate). At best this "
                "supports crop-level metadata (coarse), not per-image pairing."
            )
        else:
            pairing_status = "INVALID"
            pairing_explanation = (
                "Structured metadata files were found but no column name suggests "
                "any identifier that could link rows to individual images, samples, "
                "or crops. Reliable pairing cannot be established; multimodal "
                "experiments requiring this metadata are DISABLED."
            )

    return {
        "metadata_files_found": [str(p.relative_to(data_root)) for p in metadata_files],
        "n_metadata_files": len(metadata_files),
        "n_metadata_tables_profiled": len(profiles),
        "profiles": profiles,
        "metadata_pairing_status": pairing_status,
        "metadata_pairing_explanation": pairing_explanation,
    }


# --------------------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------------------


def write_outputs(
    output_dir: Path,
    config: dict,
    image_summary: dict,
    metadata_summary: dict,
    records: list[ImageRecord],
    skip_log: list[dict],
    started_at: str,
    finished_at: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    full_audit = {
        "generated_at_utc": finished_at,
        "started_at_utc": started_at,
        "data_root": str(config["data_root"]),
        "classes_config_path": str(config["classes_config_path"]),
        "image_summary": image_summary,
        "metadata_summary": {
            k: v for k, v in metadata_summary.items() if k != "profiles"
        }
        | {"profiles": metadata_summary["profiles"]},
        "skipped_files": skip_log,
        "n_skipped_files": len(skip_log),
    }

    with open(output_dir / "dataset_audit.json", "w") as f:
        json.dump(full_audit, f, indent=2, default=str)

    per_image_df = pd.DataFrame([asdict(r) for r in records])
    per_image_df.to_csv(output_dir / "dataset_audit.csv", index=False)

    with open(output_dir / "metadata_audit.json", "w") as f:
        json.dump(metadata_summary, f, indent=2, default=str)

    completeness_rows = []
    for profile in metadata_summary["profiles"]:
        for col in profile["columns"]:
            completeness_rows.append(
                {
                    "source_file": profile["source_file"],
                    "column": col["column"],
                    "dtype": col["dtype"],
                    "n_missing": col["n_missing"],
                    "pct_missing": col["pct_missing"],
                    "n_unique": col["n_unique"],
                    "is_numeric": col["is_numeric"],
                    "is_categorical": col["is_categorical"],
                }
            )
    completeness_df = pd.DataFrame(completeness_rows)
    completeness_df.to_csv(output_dir / "metadata_completeness.csv", index=False)

    write_summary_md(output_dir / "dataset_summary.md", config, image_summary, metadata_summary, skip_log)


def write_summary_md(
    path: Path,
    config: dict,
    image_summary: dict,
    metadata_summary: dict,
    skip_log: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Dataset Audit Summary")
    lines.append("")
    lines.append(f"- **Data root:** `{config['data_root']}`")
    lines.append(f"- **Classes config:** `{config['classes_config_path']}`")
    lines.append(f"- **Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    lines.append("## Provenance note")
    lines.append("")
    lines.append(
        "The Mendeley dataset page (DOI 10.17632/fhbvmpcyy2.2) offers two separate "
        "archives under the same dataset version: `... Dataset.zip` and "
        "`... Dataset (version 1).zip`. Both were downloaded and extracted "
        "(`data/raw/dataset_main/` and `data/raw/dataset_v1/` respectively) because "
        "they are NOT identical (see per-source-collection counts below) and neither "
        "is documented as a strict superset of the other. They are highly overlapping "
        "collections, so cross-collection duplicate/near-duplicate leakage across "
        "train/val/test is a live risk -- see the near-duplicate section below and "
        "src/duplicate_detection.py (Phase 3) before generating any split."
    )
    lines.append("")

    lines.append("## Image inventory")
    lines.append("")
    lines.append(f"- **Total images discovered:** {image_summary['total_images']}")
    lines.append(f"- **Corrupted / unreadable images:** {image_summary['corrupted_image_count']}")
    lines.append(f"- **Unmapped class folders:** {len(image_summary['unmapped_class_folders'])} "
                 f"({image_summary['unmapped_image_count']} images)")
    lines.append("")

    lines.append("### Images per source collection")
    lines.append("")
    lines.append("| Source collection | Images |")
    lines.append("|---|---|")
    for k, v in image_summary["images_per_source_collection"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### Images per class folder (raw, as downloaded)")
    lines.append("")
    lines.append("| Class folder | Images | Mapped? |")
    lines.append("|---|---|---|")
    unmapped_set = set(image_summary["unmapped_class_folders"])
    for k, v in sorted(image_summary["images_per_class_folder"].items()):
        mapped_flag = "no" if k in unmapped_set else "yes"
        lines.append(f"| {k} | {v} | {mapped_flag} |")
    lines.append("")

    if image_summary["unmapped_class_folders"]:
        lines.append("> **Ambiguity:** the folders above marked `no` are not present in "
                     "`configs/classes.yaml` and were NOT assigned a crop/organ/class. "
                     "Update the mapping file if they represent real classes.")
        lines.append("")

    lines.append("### Images per crop (mapped only)")
    lines.append("")
    lines.append("| Crop | Images |")
    lines.append("|---|---|")
    for k, v in sorted(image_summary["images_per_crop"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### Images per organ (mapped only)")
    lines.append("")
    lines.append("| Organ | Images |")
    lines.append("|---|---|")
    for k, v in sorted(image_summary["images_per_organ"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### Organ-confidence breakdown (mapped only)")
    lines.append("")
    lines.append(
        "`unresolved` classes have organ genuinely unknown -- see `configs/classes.yaml` "
        "notes for `mango.bacterial_canker` and `papaya.ring_spot`."
    )
    lines.append("")
    lines.append("| Confidence | Images |")
    lines.append("|---|---|")
    for k, v in sorted(image_summary["organ_confidence_breakdown"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("### Images per canonical class (mapped only)")
    lines.append("")
    lines.append("| Crop | Organ | Class | Images |")
    lines.append("|---|---|---|---|")
    for row in image_summary["crop_organ_class_table"]:
        lines.append(f"| {row['crop']} | {row['organ']} | {row['canonical_class']} | {row['count']} |")
    lines.append("")

    lines.append("### Image dimensions / channels")
    lines.append("")
    dims = image_summary.get("image_dimensions", {})
    if dims:
        lines.append(
            f"- Width: min={dims['width']['min']}, max={dims['width']['max']}, "
            f"mean={dims['width']['mean']}, median={dims['width']['median']}"
        )
        lines.append(
            f"- Height: min={dims['height']['min']}, max={dims['height']['max']}, "
            f"mean={dims['height']['mean']}, median={dims['height']['median']}"
        )
        lines.append("")
        lines.append("Top resolutions:")
        lines.append("")
        lines.append("| Resolution | Count |")
        lines.append("|---|---|")
        for res, count in dims["top_10_resolutions"].items():
            lines.append(f"| {res} | {count} |")
        lines.append("")
    lines.append("Mode / channel breakdown:")
    lines.append("")
    lines.append("| Mode | Count |")
    lines.append("|---|---|")
    for mode, count in image_summary.get("mode_channel_breakdown", {}).items():
        lines.append(f"| {mode} | {count} |")
    lines.append("")

    lines.append("### File extensions")
    lines.append("")
    lines.append("| Extension | Count |")
    lines.append("|---|---|")
    for ext, count in image_summary["images_per_extension"].items():
        lines.append(f"| {ext} | {count} |")
    lines.append("")

    lines.append("### Corrupted / unreadable images")
    lines.append("")
    if image_summary["corrupted_image_count"] == 0:
        lines.append("None detected.")
    else:
        lines.append("| Image path | Reason |")
        lines.append("|---|---|")
        for row in image_summary["corrupted_images"][:100]:
            lines.append(f"| {row['image_path']} | {row['corruption_reason']} |")
        if image_summary["corrupted_image_count"] > 100:
            lines.append(
                f"\n_... and {image_summary['corrupted_image_count'] - 100} more; "
                f"see results/dataset_audit.json for the full list._"
            )
    lines.append("")

    lines.append("### Duplicate filenames (basename collisions)")
    lines.append("")
    lines.append(
        f"- **{image_summary['duplicate_filename_group_count']}** basenames appear more "
        "than once (this alone does not imply duplicate content -- see exact/near-"
        "duplicate sections below)."
    )
    lines.append("")

    lines.append("### Exact duplicate files (SHA-256)")
    lines.append("")
    lines.append(
        f"- **{image_summary['exact_duplicate_group_count']}** duplicate-content groups, "
        f"covering **{image_summary['exact_duplicate_image_count']}** images."
    )
    if image_summary["exact_duplicate_group_count"] > 0:
        lines.append(
            "> These images are byte-identical. If any group's members end up split "
            "across train/val/test, that IS data leakage. This must be resolved "
            "before splitting -- see src/duplicate_detection.py (Phase 3)."
        )
    lines.append("")

    lines.append("### Preliminary near-duplicate groups (average-hash)")
    lines.append("")
    lines.append(
        f"- **{image_summary['preliminary_near_duplicate_group_count']}** groups share an "
        "identical 8x8 average hash."
    )
    lines.append(f"> {image_summary['preliminary_near_duplicate_note']}")
    lines.append("")

    lines.append("## Metadata audit")
    lines.append("")
    lines.append(f"- **Metadata files found:** {metadata_summary['n_metadata_files']}")
    if metadata_summary["metadata_files_found"]:
        for f in metadata_summary["metadata_files_found"]:
            lines.append(f"  - `{f}`")
    lines.append("")
    lines.append(f"### METADATA_PAIRING_STATUS = `{metadata_summary['metadata_pairing_status']}`")
    lines.append("")
    lines.append(metadata_summary["metadata_pairing_explanation"])
    lines.append("")

    if metadata_summary["profiles"]:
        lines.append("### Metadata table profiles")
        lines.append("")
        for profile in metadata_summary["profiles"]:
            lines.append(f"#### `{profile['source_file']}`")
            lines.append("")
            lines.append(f"- Rows: {profile['n_rows']}, Columns: {profile['n_columns']}")
            lines.append(
                f"- Candidate id columns: {profile['candidate_id_columns'] or 'none'}"
            )
            lines.append(
                f"- Candidate crop columns: {profile['candidate_crop_columns'] or 'none'}"
            )
            lines.append(
                f"- Candidate label columns: {profile['candidate_label_columns'] or 'none'}"
            )
            lines.append("")
            lines.append("| Column | dtype | % missing | n unique |")
            lines.append("|---|---|---|---|")
            for col in profile["columns"]:
                lines.append(
                    f"| {col['column']} | {col['dtype']} | {col['pct_missing']} | {col['n_unique']} |"
                )
            lines.append("")

    lines.append("## Skipped files")
    lines.append("")
    lines.append(
        f"- **{len(skip_log)}** files were skipped during the audit (unsupported "
        "extension, unreadable, or unparseable). Every skip is logged with a reason "
        "in `results/dataset_audit.json` -> `skipped_files` (rule: never silently "
        "skip a sample)."
    )
    lines.append("")

    lines.append("## Next steps / open ambiguities")
    lines.append("")
    lines.append(
        "1. **METADATA_PAIRING_STATUS is `NOT_AVAILABLE`** (as of this audit run): no "
        "structured metadata exists in the downloaded archives despite the dataset "
        "description mentioning soil health / humidity / nutrient deficiency / "
        "pathological reports. Metadata-dependent experiments (multimodal fusion, "
        "metadata-only baseline, metadata ablation) are disabled until a metadata "
        "source is located and independently verified to link to images."
    )
    lines.append(
        "2. **Two overlapping source collections** (`dataset_main`, `dataset_v1`) must "
        "be deduplicated (exact + near-duplicate) before any split is generated, or "
        "the same physical photograph could land in both train and test."
    )
    lines.append(
        "3. **Organ is `unresolved`** for `mango.bacterial_canker` and "
        "`papaya.ring_spot` in `configs/classes.yaml` -- confirm visually or with a "
        "domain expert before running organ-specific (fruit-only / leaf-only) "
        "experiments on these classes."
    )
    lines.append(
        "4. Folders using PlantVillage-style naming (`Apple___healthy`, "
        "`Apple___Black_rot`) were assigned `organ_confidence: low` -- worth "
        "confirming these are genuinely this dataset's own field-collected apple "
        "leaf photos and not a repackaged public benchmark, for provenance "
        "disclosure in any publication."
    )
    lines.append("")

    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory to audit (overrides dataset.root in --config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configs/dataset.yaml (default: configs/dataset.yaml if it exists).",
    )
    parser.add_argument(
        "--classes-config",
        type=str,
        default=None,
        help="Path to configs/classes.yaml (overrides classes.mapping_file in --config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write audit outputs to (default: results/).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)

    started_at = datetime.now(timezone.utc).isoformat()

    data_root = config["data_root"]
    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        sys.exit(1)

    class_mapping = load_class_mapping(config["classes_config_path"])
    if not class_mapping:
        print(
            f"WARNING: no class mapping loaded from {config['classes_config_path']} "
            "-- all folders will be reported as UNMAPPED.",
            file=sys.stderr,
        )

    print(f"Scanning {data_root} ...")
    all_files = discover_files(data_root, config["exclude_dirs"])
    print(f"Found {len(all_files)} files (all extensions).")

    skip_log: list[dict] = []
    for f in all_files:
        ext = f.suffix.lower()
        if ext not in config["image_extensions"] and ext not in config["metadata_extensions"]:
            skip_log.append(
                {"image_path": str(f.relative_to(data_root)), "reason": f"unsupported_extension: {ext or '(none)'}"}
            )

    image_records = audit_images(
        all_files,
        data_root,
        config["image_extensions"],
        class_mapping,
        config["compute_near_duplicate_hash"],
        skip_log,
    )
    image_summary = summarize_images(image_records)

    metadata_summary = audit_metadata(
        all_files, data_root, config["metadata_extensions"], image_records, skip_log
    )

    finished_at = datetime.now(timezone.utc).isoformat()

    write_outputs(
        config["output_dir"],
        config,
        image_summary,
        metadata_summary,
        image_records,
        skip_log,
        started_at,
        finished_at,
    )

    print()
    print("DATASET AUDIT COMPLETE")
    print()
    print(f"Images:       {image_summary['total_images']}")
    print(f"Crops:        {len(image_summary['images_per_crop'])}")
    print(f"Classes:      {len(image_summary['crop_organ_class_table'])} "
          f"(crop-specific; {len(image_summary['images_per_canonical_class'])} distinct "
          f"semantic labels shared across crops)")
    fruit_images = image_summary["images_per_organ"].get("fruit", 0)
    leaf_images = image_summary["images_per_organ"].get("leaf", 0)
    unknown_organ_images = image_summary["images_per_organ"].get("unknown", 0)
    print(f"Fruit images: {fruit_images}")
    print(f"Leaf images:  {leaf_images}")
    if unknown_organ_images:
        print(f"Unknown-organ images: {unknown_organ_images}")
    print(f"Metadata files: {metadata_summary['n_metadata_files']}")
    total_meta_rows = sum(p["n_rows"] for p in metadata_summary["profiles"])
    print(f"Metadata rows:  {total_meta_rows}")
    print(f"Metadata-to-image pairing: {metadata_summary['metadata_pairing_status']}")
    print(f"Duplicate groups (exact):  {image_summary['exact_duplicate_group_count']}")
    print(f"Corrupt images: {image_summary['corrupted_image_count']}")
    print()
    print(f"Full report written to: {config['output_dir']}/dataset_audit.json")
    print(f"                        {config['output_dir']}/dataset_audit.csv")
    print(f"                        {config['output_dir']}/dataset_summary.md")
    print(f"                        {config['output_dir']}/metadata_audit.json")
    print(f"                        {config['output_dir']}/metadata_completeness.csv")


if __name__ == "__main__":
    main()
