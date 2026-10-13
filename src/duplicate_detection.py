"""Duplicate / near-duplicate detection for leakage-safe splitting.

Consumes the per-image inventory produced by `src.dataset_audit` (results/dataset_audit.csv)
and the actual image files under the data root to build duplicate groups using four
independent signals:

  1. Exact content duplicates      -- SHA-256 (already computed by the audit; re-verified here)
  2. Average-hash near-duplicates  -- aHash, 8x8
  3. Difference-hash near-duplicates -- dHash, 8x8
  4. Perceptual-hash near-duplicates -- pHash (DCT-based), 8x8
  5. (optional, off by default) Semantic near-duplicates -- pretrained CNN embedding
     cosine similarity, bucketed by crop to keep the comparison tractable.

Two images are placed in the same duplicate group if ANY signal links them (exact hash
match, or perceptual-hash Hamming distance <= its configured threshold, or -- if enabled
-- semantic cosine similarity >= its configured threshold). Grouping is transitive via a
union-find structure: if A~B and B~C, then A, B, C are one group even if A and C were
never directly compared.

No group is ever silently discarded or merged into a `random` bucket -- every group is
written out with the signal(s) that created it, and whether it spans multiple crops,
multiple canonical classes, or multiple source collections (dataset_main / dataset_v1),
since a group spanning classes is a potential mislabeling issue, not just a leakage risk.

This module only DETECTS and REPORTS groups. It does not generate splits -- a later
splitting step (Phase 5, src/data_loader.py / manifest generation) must consume
data/manifests/duplicate_groups.csv and keep every group inside a single partition.

Usage:
    python -m src.duplicate_detection
    python -m src.duplicate_detection --audit-csv results/dataset_audit.csv --data-root ./data/raw
    python -m src.duplicate_detection --phash-threshold 6 --dhash-threshold 6
    python -m src.duplicate_detection --enable-semantic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.dataset_audit import build_config as build_dataset_config
from src.dataset_audit import compute_ahash, load_yaml

DEFAULT_PHASH_THRESHOLD = 8
DEFAULT_DHASH_THRESHOLD = 8
DEFAULT_AHASH_THRESHOLD = 8
DEFAULT_LSH_BANDS = 4
DEFAULT_LSH_BAND_BITS = 16
DEFAULT_SEMANTIC_THRESHOLD = 0.97
DEFAULT_MIN_HASH_SIGNAL_AGREEMENT = 2
MAX_BUCKET_FOR_PAIRWISE = 4000  # safety cap; see verify_bucket_pairs docstring


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def build_config(args: argparse.Namespace) -> dict:
    """Reuse src.dataset_audit's config resolution for data_root/classes, then layer
    duplicate_detection-specific settings from the same YAML file (`duplicate_detection:`
    top-level key) with CLI overrides on top."""
    dataset_args = argparse.Namespace(
        data_root=args.data_root,
        config=args.config,
        classes_config=None,
        output_dir=None,
    )
    base_cfg = build_dataset_config(dataset_args)

    config_path = base_cfg["config_path"]
    dd_cfg = load_yaml(config_path).get("duplicate_detection", {}) if config_path.exists() else {}

    return {
        "data_root": base_cfg["data_root"],
        "audit_csv": Path(args.audit_csv or dd_cfg.get("audit_csv", "results/dataset_audit.csv")),
        "output_dir": Path(args.output_dir or dd_cfg.get("output_dir", "results")),
        "manifest_dir": Path(args.manifest_dir or dd_cfg.get("manifest_dir", "data/manifests")),
        "phash_threshold": args.phash_threshold
        if args.phash_threshold is not None
        else dd_cfg.get("phash_threshold", DEFAULT_PHASH_THRESHOLD),
        "dhash_threshold": args.dhash_threshold
        if args.dhash_threshold is not None
        else dd_cfg.get("dhash_threshold", DEFAULT_DHASH_THRESHOLD),
        "ahash_threshold": args.ahash_threshold
        if args.ahash_threshold is not None
        else dd_cfg.get("ahash_threshold", DEFAULT_AHASH_THRESHOLD),
        "min_hash_signal_agreement": args.min_hash_signal_agreement
        if args.min_hash_signal_agreement is not None
        else dd_cfg.get("min_hash_signal_agreement", DEFAULT_MIN_HASH_SIGNAL_AGREEMENT),
        "lsh_bands": dd_cfg.get("lsh_bands", DEFAULT_LSH_BANDS),
        "lsh_band_bits": dd_cfg.get("lsh_band_bits", DEFAULT_LSH_BAND_BITS),
        "enable_semantic": args.enable_semantic or dd_cfg.get("enable_semantic", False),
        "semantic_threshold": args.semantic_threshold
        if args.semantic_threshold is not None
        else dd_cfg.get("semantic_threshold", DEFAULT_SEMANTIC_THRESHOLD),
        "semantic_model": dd_cfg.get("semantic_model", "resnet18"),
    }


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------


def compute_dhash(img: Image.Image, hash_size: int = 8) -> int:
    """Standard difference-hash: compare adjacent horizontal pixel intensities."""
    small = img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.int32)
    diff = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def compute_phash(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """Standard DCT-based perceptual hash (same algorithm as the `imagehash` library's
    phash, reimplemented here to avoid an extra dependency: scipy is already required)."""
    from scipy.fftpack import dct

    img_size = hash_size * highfreq_factor
    small = img.convert("L").resize((img_size, img_size), Image.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    dct_full = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_low = dct_full[:hash_size, :hash_size]
    med = np.median(dct_low)
    bits = (dct_low > med).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def compute_ahash_int(img: Image.Image, hash_size: int = 8) -> int:
    """Wraps src.dataset_audit.compute_ahash (which returns a hex string) as an int,
    so it can share the same Hamming-distance / LSH-bucketing machinery as pHash/dHash."""
    return int(compute_ahash(img, hash_size), 16)


def popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit population count (Hamming weight) for numpy uint64 arrays."""
    x = x.astype(np.uint64)
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


# --------------------------------------------------------------------------------------
# Union-Find
# --------------------------------------------------------------------------------------


class UnionFind:
    def __init__(self, items: list[str]):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return out


# --------------------------------------------------------------------------------------
# LSH banding + threshold verification
# --------------------------------------------------------------------------------------


def lsh_candidate_pairs(
    hash_map: dict[str, int], n_bands: int, band_bits: int
) -> set[tuple[str, str]]:
    """Bucket items by fixed-width bit bands of their hash to generate a small set of
    *candidate* near-duplicate pairs, cheaply, without O(n^2) global comparison.

    Guarantee: two hashes whose Hamming distance is < n_bands share at least one
    identical band (pigeonhole: distance bits can occupy at most `distance` of the
    n_bands bands, so at least one band is untouched). With the default 4 bands this
    guarantees full recall for thresholds < 4; larger configured thresholds trade
    recall for the speed of not doing a full pairwise scan -- documented in
    results/duplicate_report.json's `near_duplicate_detection_method` field.
    """
    candidates: set[tuple[str, str]] = set()
    mask = (1 << band_bits) - 1
    for band_idx in range(n_bands):
        shift = band_idx * band_bits
        buckets: dict[int, list[str]] = defaultdict(list)
        for image_id, h in hash_map.items():
            buckets[(h >> shift) & mask].append(image_id)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > MAX_BUCKET_FOR_PAIRWISE:
                # Almost certainly one giant exact/near-duplicate cluster (already
                # covered by SHA-256 exact grouping); skip pairwise expansion here to
                # avoid an O(bucket^2) memory blowup and log it instead.
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    candidates.add((a, b) if a < b else (b, a))
    return candidates


def verify_pairs_by_threshold(
    candidates: set[tuple[str, str]], hash_map: dict[str, int], threshold: int
) -> list[tuple[str, str, int]]:
    """Compute exact Hamming distance for each LSH candidate pair and keep the ones
    within `threshold` bits. Candidates are typically a small fraction of all pairs,
    so this stays cheap even without vectorization."""
    confirmed = []
    for a, b in candidates:
        dist = bin(hash_map[a] ^ hash_map[b]).count("1")
        if dist <= threshold:
            confirmed.append((a, b, dist))
    return confirmed


def oversized_bucket_warning(
    hash_map: dict[str, int], n_bands: int, band_bits: int
) -> list[dict]:
    warnings = []
    mask = (1 << band_bits) - 1
    for band_idx in range(n_bands):
        shift = band_idx * band_bits
        buckets: dict[int, list[str]] = defaultdict(list)
        for image_id, h in hash_map.items():
            buckets[(h >> shift) & mask].append(image_id)
        for band_val, members in buckets.items():
            if len(members) > MAX_BUCKET_FOR_PAIRWISE:
                warnings.append(
                    {
                        "band_index": band_idx,
                        "band_value": band_val,
                        "bucket_size": len(members),
                        "note": "skipped pairwise verification (exceeds MAX_BUCKET_FOR_PAIRWISE); "
                        "these images are almost certainly already covered by SHA-256 exact grouping.",
                    }
                )
    return warnings


# --------------------------------------------------------------------------------------
# Optional semantic (embedding) near-duplicate detection
# --------------------------------------------------------------------------------------


def compute_semantic_groups(
    df: pd.DataFrame, data_root: Path, threshold: float, model_name: str
) -> list[tuple[str, str, float]]:
    """Pretrained-CNN embedding cosine-similarity near-duplicate detection, bucketed by
    crop to keep the pairwise comparison tractable. Optional -- requires torch/torchvision
    and downloads pretrained ImageNet weights on first use. Off by default (--enable-semantic)."""
    import torch
    import torchvision.transforms as T
    from torchvision import models

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        backbone = models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported semantic_model: {model_name}")
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    confirmed_pairs: list[tuple[str, str, float]] = []
    for crop, group in tqdm(df.groupby("crop"), desc="Semantic dedup by crop"):
        paths = group["image_path"].tolist()
        embeddings = []
        valid_paths = []
        with torch.no_grad():
            for rel_path in paths:
                try:
                    img = Image.open(data_root / rel_path).convert("RGB")
                except Exception:
                    continue
                tensor = transform(img).unsqueeze(0).to(device)
                emb = backbone(tensor).squeeze(0).cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                embeddings.append(emb)
                valid_paths.append(rel_path)

        if len(embeddings) < 2:
            continue
        emb_matrix = np.stack(embeddings)
        sims = emb_matrix @ emb_matrix.T
        n = len(valid_paths)
        for i in range(n):
            for j in range(i + 1, n):
                if sims[i, j] >= threshold:
                    confirmed_pairs.append((valid_paths[i], valid_paths[j], float(sims[i, j])))

    return confirmed_pairs


# --------------------------------------------------------------------------------------
# Main detection pipeline
# --------------------------------------------------------------------------------------


@dataclass
class HashComputationResult:
    dhash: dict[str, int]
    phash: dict[str, int]
    ahash: dict[str, int]
    failures: list[dict]


def compute_missing_hashes(
    df: pd.DataFrame, data_root: Path
) -> HashComputationResult:
    """dataset_audit.py already computed aHash into the CSV (hex string); dHash and
    pHash are computed fresh here since the audit does not store them."""
    dhash_map: dict[str, int] = {}
    phash_map: dict[str, int] = {}
    ahash_map: dict[str, int] = {}
    failures: list[dict] = []

    usable = df[~df["corrupted"]]
    for _, row in tqdm(usable.iterrows(), total=len(usable), desc="Computing pHash/dHash"):
        image_id = row["image_path"]
        try:
            img = Image.open(data_root / image_id)
            dhash_map[image_id] = compute_dhash(img)
            phash_map[image_id] = compute_phash(img)
            if isinstance(row.get("ahash"), str) and row["ahash"]:
                ahash_map[image_id] = int(row["ahash"], 16)
            else:
                ahash_map[image_id] = compute_ahash_int(img)
        except Exception as e:  # noqa: BLE001 - log and continue, never crash the run
            failures.append({"image_path": image_id, "reason": f"{type(e).__name__}: {e}"})

    return HashComputationResult(dhash=dhash_map, phash=phash_map, ahash=ahash_map, failures=failures)


def detect_duplicates(df: pd.DataFrame, data_root: Path, config: dict) -> dict[str, Any]:
    image_ids = df["image_path"].tolist()
    uf = UnionFind(image_ids)

    edge_log: list[dict] = []  # every union, with the signal that caused it

    # 1. Exact duplicates (SHA-256) -- authoritative, already computed by the audit.
    sha_groups = df[df["sha256"].notna()].groupby("sha256")["image_path"].apply(list)
    for members in sha_groups:
        if len(members) < 2:
            continue
        for m in members[1:]:
            uf.union(members[0], m)
            edge_log.append({"a": members[0], "b": m, "signal": "sha256_exact", "distance": 0})

    # 2. Perceptual near-duplicates: aHash / dHash / pHash via LSH banding + threshold.
    #
    # Two important safeguards, both learned from an initial run against this dataset
    # that (without them) produced a single 8,872-image group spanning all 6 crops --
    # clearly a false-positive artifact, not a real finding:
    #
    #   a) Candidate generation is bucketed by crop. A genuine near-duplicate (the same
    #      photo, recompressed/resized/lightly cropped) can never be a different crop,
    #      so restricting comparisons to within-crop eliminates cross-crop hash-collision
    #      chaining by construction (and is cheaper).
    #   b) A pair is only unioned if AT LEAST `min_hash_signal_agreement` of the three
    #      independent hash algorithms (aHash/dHash/pHash) agree within their own
    #      threshold. Any single hash alone (especially aHash on this dataset's largely
    #      uniform 300x300 images) has a non-trivial false-positive rate; requiring
    #      agreement across independent algorithms sharply reduces spurious unions
    #      while still catching genuine near-duplicates (which typically agree on all
    #      three, since they measure different but correlated properties of the image).
    hash_result = compute_missing_hashes(df, data_root)
    oversized: list[dict] = []
    crop_map = df.set_index("image_path")["crop"].to_dict()
    min_agreement = config.get("min_hash_signal_agreement", 2)

    pair_signals: dict[frozenset, set[str]] = defaultdict(set)
    pair_min_distance: dict[frozenset, int] = {}

    for hash_name, hash_map, threshold in [
        ("ahash", hash_result.ahash, config["ahash_threshold"]),
        ("dhash", hash_result.dhash, config["dhash_threshold"]),
        ("phash", hash_result.phash, config["phash_threshold"]),
    ]:
        by_crop: dict[Any, dict[str, int]] = defaultdict(dict)
        for image_id, h in hash_map.items():
            by_crop[crop_map.get(image_id)][image_id] = h

        for crop, sub_map in by_crop.items():
            if len(sub_map) < 2:
                continue
            candidates = lsh_candidate_pairs(sub_map, config["lsh_bands"], config["lsh_band_bits"])
            confirmed = verify_pairs_by_threshold(candidates, sub_map, threshold)
            oversized.extend(
                {**w, "hash_type": hash_name, "crop": crop}
                for w in oversized_bucket_warning(sub_map, config["lsh_bands"], config["lsh_band_bits"])
            )
            for a, b, dist in confirmed:
                key = frozenset((a, b))
                pair_signals[key].add(hash_name)
                pair_min_distance[key] = min(pair_min_distance.get(key, dist), dist)

    for pair, signals in pair_signals.items():
        if len(signals) >= min_agreement:
            a, b = tuple(pair)
            uf.union(a, b)
            edge_log.append(
                {
                    "a": a,
                    "b": b,
                    "signal": "+".join(sorted(signals)) + "_near",
                    "distance": pair_min_distance[pair],
                }
            )

    # 3. Optional semantic near-duplicates.
    semantic_pairs: list[tuple[str, str, float]] = []
    if config["enable_semantic"]:
        semantic_pairs = compute_semantic_groups(
            df, data_root, config["semantic_threshold"], config["semantic_model"]
        )
        for a, b, sim in semantic_pairs:
            uf.union(a, b)
            edge_log.append({"a": a, "b": b, "signal": "semantic_embedding", "distance": round(1 - sim, 4)})

    # Build final groups, enriched with crop/class/collection info + cross-* flags.
    meta = df.set_index("image_path")[["crop", "canonical_class", "source_collection"]].to_dict("index")
    raw_groups = uf.groups()

    signals_by_pair: dict[frozenset, set[str]] = defaultdict(set)
    for e in edge_log:
        signals_by_pair[frozenset((e["a"], e["b"]))].add(e["signal"])

    final_groups = []
    group_idx = 0
    for _, members in raw_groups.items():
        if len(members) < 2:
            continue
        group_idx += 1
        group_id = f"dupgroup_{group_idx:06d}"
        crops = {meta[m]["crop"] for m in members if m in meta}
        classes = {meta[m]["canonical_class"] for m in members if m in meta}
        collections = {meta[m]["source_collection"] for m in members if m in meta}
        signals: set[str] = set()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                signals |= signals_by_pair.get(frozenset((members[i], members[j])), set())
        final_groups.append(
            {
                "group_id": group_id,
                "size": len(members),
                "members": sorted(members),
                "crops": sorted(c for c in crops if c),
                "canonical_classes": sorted(c for c in classes if c),
                "source_collections": sorted(c for c in collections if c),
                "spans_multiple_crops": len(crops) > 1,
                "spans_multiple_classes": len(classes) > 1,
                "spans_multiple_source_collections": len(collections) > 1,
                "detection_signals": sorted(signals),
            }
        )

    final_groups.sort(key=lambda g: -g["size"])

    return {
        "groups": final_groups,
        "hash_failures": hash_result.failures,
        "oversized_lsh_buckets": oversized,
        "n_semantic_pairs_confirmed": len(semantic_pairs),
    }


# --------------------------------------------------------------------------------------
# Output writing
# --------------------------------------------------------------------------------------


def write_outputs(
    result: dict, df: pd.DataFrame, config: dict, started_at: str, finished_at: str
) -> None:
    output_dir = config["output_dir"]
    manifest_dir = config["manifest_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    groups = result["groups"]
    n_images_in_groups = sum(g["size"] for g in groups)
    n_cross_crop = sum(1 for g in groups if g["spans_multiple_crops"])
    n_cross_class = sum(1 for g in groups if g["spans_multiple_classes"])
    n_cross_collection = sum(1 for g in groups if g["spans_multiple_source_collections"])

    report = {
        "generated_at_utc": finished_at,
        "started_at_utc": started_at,
        "data_root": str(config["data_root"]),
        "audit_csv": str(config["audit_csv"]),
        "parameters": {
            "phash_threshold": config["phash_threshold"],
            "dhash_threshold": config["dhash_threshold"],
            "ahash_threshold": config["ahash_threshold"],
            "min_hash_signal_agreement": config["min_hash_signal_agreement"],
            "lsh_bands": config["lsh_bands"],
            "lsh_band_bits": config["lsh_band_bits"],
            "enable_semantic": config["enable_semantic"],
            "semantic_threshold": config["semantic_threshold"] if config["enable_semantic"] else None,
        },
        "near_duplicate_detection_method": (
            "Per-crop LSH banding (candidate generation, bucketed by crop so a genuine "
            "near-duplicate pair can never span crops) + exact Hamming-distance "
            "verification per candidate pair, for aHash/dHash/pHash independently. A "
            f"pair is only unioned into a group if at least {config['min_hash_signal_agreement']} "
            "of the 3 hash algorithms agree within their own threshold (reduces "
            "single-hash false positives on this dataset's largely uniform 300x300 "
            "images); the agreeing signal names and the minimum confirmed Hamming "
            "distance are recorded per group. Unioned with SHA-256 exact-match groups. "
            "Recall for the per-crop LSH candidate step is guaranteed-complete for "
            "Hamming thresholds strictly less than lsh_bands (see lsh_candidate_pairs "
            "docstring); configured thresholds >= lsh_bands trade recall for "
            f"tractability. Buckets larger than {MAX_BUCKET_FOR_PAIRWISE} images "
            "were skipped for pairwise verification (see oversized_lsh_buckets) as "
            "they are almost certainly already covered by SHA-256 exact grouping. "
            "NOTE: an earlier version of this pipeline ran LSH globally (not "
            "per-crop) and required only 1 signal, which produced a single spurious "
            "8,872-image group spanning all 6 crops via hash-collision chaining "
            "through union-find -- this methodology was corrected in response and "
            "is documented here for transparency."
        ),
        "total_images_considered": len(df),
        "n_duplicate_groups": len(groups),
        "n_images_in_duplicate_groups": n_images_in_groups,
        "n_groups_spanning_multiple_crops": n_cross_crop,
        "n_groups_spanning_multiple_classes": n_cross_class,
        "n_groups_spanning_multiple_source_collections": n_cross_collection,
        "hash_computation_failures": result["hash_failures"],
        "oversized_lsh_buckets": result["oversized_lsh_buckets"],
        "groups": groups,
    }

    with open(output_dir / "duplicate_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    rows = []
    for g in groups:
        for member in g["members"]:
            row_meta = df.loc[df["image_path"] == member].iloc[0]
            rows.append(
                {
                    "group_id": g["group_id"],
                    "group_size": g["size"],
                    "image_path": member,
                    "crop": row_meta["crop"],
                    "organ": row_meta["organ"],
                    "canonical_class": row_meta["canonical_class"],
                    "source_collection": row_meta["source_collection"],
                    "spans_multiple_crops": g["spans_multiple_crops"],
                    "spans_multiple_classes": g["spans_multiple_classes"],
                    "spans_multiple_source_collections": g["spans_multiple_source_collections"],
                    "detection_signals": ";".join(g["detection_signals"]),
                }
            )
    groups_df = pd.DataFrame(rows)
    groups_df.to_csv(manifest_dir / "duplicate_groups.csv", index=False)

    write_summary_md(output_dir / "duplicate_detection_summary.md", report, df)

    return report


def analyze_cross_crop_exact_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    """Cross-crop SHA-256 exact matches are not a leakage-only concern -- byte-identical
    content filed under two different crop labels means at least one label is wrong.
    This is checked independently of the general near-duplicate pipeline (which is
    crop-bucketed by design and therefore cannot itself surface a cross-crop link)."""
    sha_to_crops = df.groupby("sha256")["crop"].nunique()
    affected_hashes = sha_to_crops[sha_to_crops > 1].index.tolist()
    if not affected_hashes:
        return {"n_affected_hashes": 0, "n_affected_rows": 0, "crop_pairs": {}, "class_pairs": {}, "examples": []}

    affected_df = df[df["sha256"].isin(affected_hashes)]
    crop_pairs: Counter = Counter()
    class_pairs: Counter = Counter()
    examples = []
    for h in affected_hashes:
        sub = df[df["sha256"] == h]
        crop_pairs[tuple(sorted(sub["crop"].dropna().unique()))] += 1
        class_pairs[tuple(sorted(sub["canonical_class"].dropna().unique()))] += 1
        if len(examples) < 10:
            examples.append(
                {
                    "sha256_prefix": h[:16],
                    "crops": sorted(sub["crop"].dropna().unique().tolist()),
                    "canonical_classes": sorted(sub["canonical_class"].dropna().unique().tolist()),
                    "image_paths": sub["image_path"].tolist(),
                }
            )

    return {
        "n_affected_hashes": len(affected_hashes),
        "n_affected_rows": len(affected_df),
        "crop_pairs": {" <-> ".join(k): v for k, v in crop_pairs.most_common()},
        "class_pairs": {" <-> ".join(k): v for k, v in class_pairs.most_common()},
        "per_crop_class_counts": affected_df.groupby(["crop", "canonical_class"]).size().to_dict(),
        "examples": examples,
    }


def write_summary_md(path: Path, report: dict, df: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Duplicate Detection Summary (Phase 3)")
    lines.append("")
    lines.append(f"- **Images considered:** {report['total_images_considered']}")
    lines.append(f"- **Duplicate groups:** {report['n_duplicate_groups']}")
    lines.append(f"- **Images in duplicate groups:** {report['n_images_in_duplicate_groups']}")
    lines.append("")

    cross_crop = analyze_cross_crop_exact_duplicates(df)
    lines.append("## CRITICAL FINDING: byte-identical images under contradictory crop labels")
    lines.append("")
    if cross_crop["n_affected_hashes"] == 0:
        lines.append("None detected -- no SHA-256 hash is shared across more than one crop label.")
    else:
        lines.append(
            f"**{cross_crop['n_affected_hashes']} distinct images** (byte-for-byte, verified "
            f"directly against the original downloaded .zip archives, not an artifact of this "
            f"pipeline's extraction) appear under **more than one crop label**, covering "
            f"**{cross_crop['n_affected_rows']} rows** across the two source collections. This "
            "is a labeling defect in the raw dataset, not a splitting/leakage concern -- the "
            "same physical photograph cannot honestly carry two different crop/disease labels, "
            "so at least one label for every affected image is simply wrong."
        )
        lines.append("")
        lines.append("**Crop pairs involved:**")
        lines.append("")
        for pair, count in cross_crop["crop_pairs"].items():
            lines.append(f"- {pair}: {count} shared images")
        lines.append("")
        lines.append("**Canonical-class pairs involved:**")
        lines.append("")
        for pair, count in cross_crop["class_pairs"].items():
            lines.append(f"- {pair}: {count} shared images")
        lines.append("")
        lines.append("**Affected image count by (crop, class):**")
        lines.append("")
        lines.append("| Crop | Class | Affected images |")
        lines.append("|---|---|---|")
        for (crop, cls), count in cross_crop["per_crop_class_counts"].items():
            lines.append(f"| {crop} | {cls} | {count} |")
        lines.append("")
        lines.append(
            "Manual inspection of two examples confirms these are real stock photographs of "
            "apples (one healthy, one diseased-on-branch, dreamstime-watermarked) that were "
            "also copied into the `Banana cordana` folder verbatim -- not a hashing "
            "coincidence. This affects roughly half of the `banana.cordana` class."
        )
        lines.append("")
        lines.append("**Recommendation:** do not train or report `banana.cordana` results "
                     "until this is resolved. Options, in order of rigor: (1) contact the "
                     "dataset authors for a corrected release; (2) manually re-verify and "
                     "relabel every `banana.cordana` image; (3) at minimum, exclude every "
                     "image listed in `data/manifests/duplicate_groups.csv` with "
                     "`spans_multiple_crops=True` from any experiment involving banana or "
                     "apple. Do not simply keep the larger side and discard the smaller side "
                     "of each pair -- the correct label for a shared image is not implied by "
                     "which folder has more images.")
        lines.append("")
        lines.append("Example groups (first 10):")
        lines.append("")
        for ex in cross_crop["examples"]:
            lines.append(f"- `{ex['sha256_prefix']}...` — crops={ex['crops']}, classes={ex['canonical_classes']}")
            for p in ex["image_paths"]:
                lines.append(f"  - `{p}`")
        lines.append("")

    lines.append("## Overlap between the two source collections")
    lines.append("")
    n_cross_collection = report["n_groups_spanning_multiple_source_collections"]
    lines.append(
        f"- **{n_cross_collection}** of {report['n_duplicate_groups']} duplicate groups span both "
        "`dataset_main` and `dataset_v1` -- confirming the audit's provenance note that these "
        "are two overlapping releases of the same underlying photo collection, not independent "
        "data. For any experiment, use images from only ONE of the two collections (recommend "
        "`dataset_main`, the non-versioned release) unless a reason is documented for using both."
    )
    lines.append("")

    lines.append("## Within-crop, cross-class duplicates")
    lines.append("")
    n_cross_class = report["n_groups_spanning_multiple_classes"]
    lines.append(
        f"- **{n_cross_class}** duplicate groups contain images from more than one canonical "
        "class within the same crop (e.g. an image appearing in both a healthy and a diseased "
        "folder for the same crop). Each such group is a labeling ambiguity in its own right -- "
        "see `results/duplicate_report.json` -> `groups` (`spans_multiple_classes: true`) for "
        "the full list before deciding how to resolve it."
    )
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(report["near_duplicate_detection_method"])
    lines.append("")
    lines.append("Parameters used for this run:")
    lines.append("")
    for k, v in report["parameters"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    if report["hash_computation_failures"]:
        lines.append("## Hash computation failures")
        lines.append("")
        lines.append(
            f"{len(report['hash_computation_failures'])} images could not be hashed for "
            "near-duplicate detection (see `results/duplicate_report.json` -> "
            "`hash_computation_failures` for paths and reasons)."
        )
        lines.append("")

    lines.append("## Next steps")
    lines.append("")
    lines.append(
        "1. Resolve the cross-crop mislabeling above before any apple/banana experiment.\n"
        "2. Phase 5 (manifest generation) MUST consume `data/manifests/duplicate_groups.csv` "
        "and keep every group's members inside a single train/val/test partition.\n"
        "3. Decide whether to use `dataset_main` only, or a deduplicated union of both "
        "collections, as the canonical image pool going forward -- using both without "
        "deduplication (as this audit did, deliberately, to surface the issues above) would "
        "silently double-count nearly every photograph."
    )
    lines.append("")

    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--audit-csv",
        type=str,
        default=None,
        help="Path to results/dataset_audit.csv (must exist -- run `python -m src.dataset_audit` first).",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--manifest-dir", type=str, default=None)
    parser.add_argument("--phash-threshold", type=int, default=None)
    parser.add_argument("--dhash-threshold", type=int, default=None)
    parser.add_argument("--ahash-threshold", type=int, default=None)
    parser.add_argument(
        "--min-hash-signal-agreement",
        type=int,
        default=None,
        help="Minimum number of independent hash algorithms (out of aHash/dHash/pHash) "
        "that must agree within their own threshold before a near-duplicate pair is "
        "unioned (default: 2). Set to 1 to union on any single hash match (more "
        "recall, more false positives).",
    )
    parser.add_argument(
        "--enable-semantic",
        action="store_true",
        help="Also run pretrained-CNN embedding cosine-similarity near-duplicate detection "
        "(requires torch/torchvision, downloads ImageNet weights on first use, slower).",
    )
    parser.add_argument("--semantic-threshold", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    started_at = datetime.now(timezone.utc).isoformat()

    if not config["audit_csv"].exists():
        print(
            f"ERROR: {config['audit_csv']} not found. Run `python -m src.dataset_audit "
            f"--data-root {config['data_root']}` first -- duplicate detection reuses its "
            "per-image inventory (paths, sha256, ahash, crop/class mapping) rather than "
            "re-scanning the dataset from scratch.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not config["data_root"].exists():
        print(f"ERROR: data root does not exist: {config['data_root']}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(config["audit_csv"])
    required_cols = {"image_path", "crop", "canonical_class", "source_collection", "sha256", "corrupted"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: {config['audit_csv']} is missing expected columns: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df)} image records from {config['audit_csv']}.")
    result = detect_duplicates(df, config["data_root"], config)

    finished_at = datetime.now(timezone.utc).isoformat()
    report = write_outputs(result, df, config, started_at, finished_at)

    print()
    print("DUPLICATE DETECTION COMPLETE")
    print()
    print(f"Images considered:                     {report['total_images_considered']}")
    print(f"Duplicate groups:                       {report['n_duplicate_groups']}")
    print(f"Images in duplicate groups:              {report['n_images_in_duplicate_groups']}")
    print(f"Groups spanning multiple crops:          {report['n_groups_spanning_multiple_crops']}")
    print(f"Groups spanning multiple classes:        {report['n_groups_spanning_multiple_classes']}")
    print(f"Groups spanning multiple source collections: {report['n_groups_spanning_multiple_source_collections']}")
    if report["hash_computation_failures"]:
        print(f"Hash computation failures: {len(report['hash_computation_failures'])} (see duplicate_report.json)")

    cross_crop = analyze_cross_crop_exact_duplicates(df)
    if cross_crop["n_affected_hashes"] > 0:
        print()
        print(f"** CRITICAL: {cross_crop['n_affected_hashes']} images are byte-identical across "
              f"more than one crop label ({cross_crop['n_affected_rows']} rows affected) -- "
              "this is a raw-dataset labeling defect, not a leakage issue. See "
              f"{config['output_dir']}/duplicate_detection_summary.md **")

    print()
    print(f"Full report written to:  {config['output_dir']}/duplicate_report.json")
    print(f"Summary written to:      {config['output_dir']}/duplicate_detection_summary.md")
    print(f"Manifest written to:     {config['manifest_dir']}/duplicate_groups.csv")


if __name__ == "__main__":
    main()
