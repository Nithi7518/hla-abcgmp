"""Feature-space out-of-scope detector (class-conditional Mahalanobis distance).

Usage:
    python -m src.assistant.ood

Fits per-class means and a shared covariance on TRAIN features of the baseline's 512-d
penultimate layer, picks a rejection threshold from VALIDATION only (95th percentile of
in-distribution distances), then reports how well it separates in-distribution test images
from proxies for out-of-scope input:
  - the 338 studio/stock apple photos mis-filed under banana cordana (excluded from training)
  - synthetic noise / blank / gradient images
Output: results/assistant/ood.json (+ ood_stats.npz used by the assistant).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.assistant.model_io import load_model

STATS = Path("results/assistant/ood_stats.npz")
OUT = Path("results/assistant/ood.json")


@torch.no_grad()
def features(model, tf, dev, images) -> np.ndarray:
    out = []
    for s in range(0, len(images), 64):
        x = torch.stack([tf(im.convert("RGB")) for im in images[s:s + 64]]).to(dev)
        out.append(model.backbone(x).cpu().numpy())
    return np.concatenate(out)


def load_images(paths, root="data/raw"):
    return [Image.open(Path(root) / p) for p in paths]


def fit(feats: np.ndarray, y: np.ndarray, n_classes: int, ridge: float = 1e-3):
    means = np.stack([feats[y == c].mean(0) for c in range(n_classes)])
    centered = feats - means[y]
    cov = centered.T @ centered / len(feats) + ridge * np.eye(feats.shape[1])
    return means, np.linalg.inv(cov)


def min_mahalanobis(feats: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = [np.einsum("ij,jk,ik->i", feats - m, prec, feats - m) for m in means]
    return np.sqrt(np.min(d, axis=0))


def organ_group(label: str) -> str:
    """'fruit' for *_fruit classes, 'other' (leaf / leaf-disease) otherwise. Fruit photos vary far more in
    background and framing, so they get their own threshold (a single global one flagged ~20% of fruit photos
    but ~2% of leaf photos)."""
    return "fruit" if "fruit" in label.split("__", 1)[-1] else "other"


def synthetic_images(n: int = 60, size: int = 300) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    imgs = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        elif kind == 1:
            a = np.full((size, size, 3), rng.integers(0, 255, 3), dtype=np.uint8)
        else:
            g = np.linspace(0, 255, size, dtype=np.uint8)
            a = np.stack([np.tile(g, (size, 1)), np.tile(g[::-1], (size, 1)), np.full((size, size), rng.integers(0, 255), np.uint8)], -1)
        imgs.append(Image.fromarray(a))
    return imgs


def main() -> None:
    model, labels, tf, dev = load_model()
    idx = {n: i for i, n in enumerate(labels)}
    tr, va, te = (pd.read_csv(f"data/manifests/{s}.csv") for s in ("train", "val", "test"))
    f_tr = features(model, tf, dev, load_images(tr.image_path))
    y_tr = tr.label.map(idx).to_numpy()
    means, prec = fit(f_tr, y_tr, len(labels))
    f_val = features(model, tf, dev, load_images(va.image_path))
    f_te = features(model, tf, dev, load_images(te.image_path))
    d_val, d_te = min_mahalanobis(f_val, means, prec), min_mahalanobis(f_te, means, prec)
    thr = float(np.percentile(d_val, 95))                       # single global threshold (kept for comparison)

    def groups(feats):                                          # group by the CLASSIFIER's predicted label
        with torch.no_grad():
            t_feats = torch.tensor(feats, dtype=torch.float32).to(dev)
            if hasattr(model, "image_proj"):
                meta = model.default_neutral_metadata(t_feats.shape[0], dev)
                img_emb = model.image_proj(t_feats)
                meta_emb = model.metadata_mlp(meta)
                fused = model.fusion(img_emb, meta_emb)
                pred = model.classifier(fused).argmax(1).cpu().numpy()
            else:
                pred = model.classifier(t_feats).argmax(1).cpu().numpy()
        return np.array([organ_group(labels[i]) for i in pred])

    g_val, g_te = groups(f_val), groups(f_te)
    thr_g = {g: float(np.percentile(d_val[g_val == g], 95)) for g in ("fruit", "other")}
    flag_te = d_te > np.array([thr_g[g] for g in g_te])

    dup = pd.read_csv("data/manifests/duplicate_groups.csv")
    stock = dup[(dup.source_collection == "dataset_main") & dup.spans_multiple_crops & (dup.crop == "banana")].image_path.tolist()
    stock += ["dataset_main/Banana cordana/download (3).jpeg"]
    stock = sorted(set(stock))
    f_stock, f_syn = features(model, tf, dev, load_images(stock)), features(model, tf, dev, synthetic_images())
    d_stock, d_syn = min_mahalanobis(f_stock, means, prec), min_mahalanobis(f_syn, means, prec)
    flag_stock = d_stock > np.array([thr_g[g] for g in groups(f_stock)])
    flag_syn = d_syn > np.array([thr_g[g] for g in groups(f_syn)])

    def auroc(d_ood):
        return float(roc_auc_score(np.r_[np.zeros(len(d_te)), np.ones(len(d_ood))], np.r_[d_te, d_ood]))

    result = {
        "threshold_95th_pct_of_val": thr,
        "in_distribution_test_rejected": float((d_te > thr).mean()),
        "stock_apple_photos": {"n": len(stock), "auroc_vs_test": auroc(d_stock), "rejected": float((d_stock > thr).mean())},
        "synthetic": {"n": len(d_syn), "auroc_vs_test": auroc(d_syn), "rejected": float((d_syn > thr).mean())},
        "per_group_threshold": {
            "thresholds": thr_g,
            "in_distribution_test_flagged": {g: float(flag_te[g_te == g].mean()) for g in ("fruit", "other")},
            "in_distribution_test_flagged_overall": float(flag_te.mean()),
            "stock_apple_photos_flagged": float(flag_stock.mean()),
            "synthetic_flagged": float(flag_syn.mean()),
        },
        "global_threshold_flag_rate_by_group_test": {g: float((d_te[g_te == g] > thr).mean()) for g in ("fruit", "other")},
        "median_distance": {"test": float(np.median(d_te)), "stock": float(np.median(d_stock)), "synthetic": float(np.median(d_syn))},
    }
    OUT.write_text(json.dumps(result, indent=2))
    np.savez_compressed(STATS, means=means.astype(np.float32), prec=prec.astype(np.float32), threshold=thr,
                        thr_fruit=thr_g["fruit"], thr_other=thr_g["other"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
