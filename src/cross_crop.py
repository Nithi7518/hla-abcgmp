"""Phase 11: cross-crop generalization (binary healthy vs diseased).

Usage:
    python -m src.cross_crop --config configs/cross_crop.yaml            # all 15 combos
    python -m src.cross_crop --config configs/cross_crop.yaml --held-out apple banana
    python -m src.cross_crop --config configs/cross_crop.yaml --epochs 1 --held-out apple banana

For each combination of held-out crops the model trains on the remaining crops'
train split, early-stops on those crops' val split, and is tested on ALL images of the
held-out crops. Combinations whose result file already exists are skipped, so an
interrupted run can be resumed. See configs/cross_crop.yaml for the protocol.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loader import ABCGMPImageDataset
from src.metrics import compute_binary_metrics, compute_classification_metrics
from src.models import build_model
from src.preprocessing import build_image_transforms
from src.train import (
    build_criterion,
    build_train_sampler,
    resolve_device,
    run_epoch_eval,
    run_epoch_train,
    set_seed,
)

LABELS = {"healthy": 0, "diseased": 1}
LABEL_NAMES = ["healthy", "diseased"]


def make_dataset(manifests_dir: Path, splits: list[str], crops: list[str], data_root, transform):
    """Dataset over the given splits restricted to `crops`, labelled healthy/diseased."""
    frames = [pd.read_csv(manifests_dir / f"{s}.csv") for s in splits]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["crop"].isin(crops)].reset_index(drop=True)
    df["label"] = df["binary_disease_status"]
    ds = ABCGMPImageDataset.__new__(ABCGMPImageDataset)
    ds.manifest = df
    ds.data_root = Path(data_root)
    ds.label_to_idx = LABELS
    ds.transform = transform
    return ds


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def score(y_true: np.ndarray, p_diseased: np.ndarray) -> dict:
    y_pred = (p_diseased >= 0.5).astype(int)
    m = compute_classification_metrics(y_true, y_pred, LABEL_NAMES)
    out = {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")}
    out["n_images"] = int(len(y_true))
    out["fraction_diseased"] = float(y_true.mean())
    b = compute_binary_metrics(y_true, p_diseased)
    out["roc_auc"] = None if b is None else b.get("roc_auc")
    return out


def run_combo(config: dict, held_out: list[str], device: torch.device, seed: int) -> dict:
    cfg_d, cfg_t = config["data"], config["training"]
    manifests_dir = Path(cfg_d["manifests_dir"])
    seen = [c for c in cfg_d["crops"] if c not in held_out]
    tf = build_image_transforms(config["image"])
    root = cfg_d["data_root"]

    train_ds = make_dataset(manifests_dir, ["train"], seen, root, tf["train"])
    val_ds = make_dataset(manifests_dir, ["val"], seen, root, tf["eval"])
    seen_test_ds = make_dataset(manifests_dir, ["test"], seen, root, tf["eval"])
    unseen_ds = make_dataset(manifests_dir, ["train", "val", "test"], held_out, root, tf["eval"])
    assert not set(seen_test_ds.manifest["crop"]) & set(held_out)
    assert not set(train_ds.manifest["crop"]) & set(held_out)

    set_seed(seed)
    labels = train_ds.manifest["label"].map(LABELS).to_numpy()
    counts = torch.tensor([(labels == i).sum() for i in range(2)], dtype=torch.float32)
    strategy = cfg_t.get("class_imbalance_strategy", "none")
    sampler = build_train_sampler(strategy, labels, counts)
    bs, nw = cfg_t["batch_size"], cfg_t.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, shuffle=sampler is None, num_workers=nw)
    mk = lambda ds: DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
    val_loader, seen_test_loader, unseen_loader = mk(val_ds), mk(seen_test_ds), mk(unseen_ds)

    model = build_model(config["model"], 2).to(device)
    criterion = build_criterion(strategy, counts, device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg_t["learning_rate"], weight_decay=cfg_t.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg_t.get("lr_scheduler_factor", 0.5), patience=cfg_t.get("lr_scheduler_patience", 1)
    )

    metric_key = cfg_t.get("early_stopping_metric", "val_macro_f1").replace("val_", "")
    best, best_epoch, patience, best_state = -np.inf, 0, 0, None
    for epoch in range(1, cfg_t["epochs"] + 1):
        t0 = time.time()
        tr_loss = run_epoch_train(model, train_loader, criterion, optimizer, device)
        val = run_epoch_eval(model, val_loader, criterion, device, LABEL_NAMES)
        scheduler.step(val[metric_key])
        print(f"  [{'+'.join(held_out)}] epoch {epoch}/{cfg_t['epochs']} train_loss={tr_loss:.4f} "
              f"val_{metric_key}={val[metric_key]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if val[metric_key] > best:
            best, best_epoch, patience = val[metric_key], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg_t.get("early_stopping_patience", 3):
                break

    model.load_state_dict(best_state)
    seen_score = score(*predict_probs(model, seen_test_loader, device))
    y_u, p_u = predict_probs(model, unseen_loader, device)
    unseen_score = score(y_u, p_u)
    per_crop = {}
    for crop in held_out:
        mask = (unseen_ds.manifest["crop"] == crop).to_numpy()
        per_crop[crop] = score(y_u[mask], p_u[mask])

    return {
        "held_out": held_out,
        "seen_crops": seen,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metric": float(best),
        "n_train": len(train_ds),
        "seen_crops_test": seen_score,
        "held_out_crops": unseen_score,
        "held_out_per_crop": per_crop,
        "generalization_gap_macro_f1": seen_score["macro_f1"] - unseen_score["macro_f1"],
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/cross_crop.yaml")
    ap.add_argument("--held-out", nargs="+", default=None, help="run a single combination")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    seed = args.seed if args.seed is not None else config["experiment"]["seed"]
    device = resolve_device(args.device or config["experiment"].get("device", "auto"))

    crops = config["data"]["crops"]
    combos = [sorted(args.held_out)] if args.held_out else [
        list(c) for c in itertools.combinations(crops, config["data"]["n_held_out"])
    ]
    out_dir = Path(config["output"]["results_dir"]) / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | {len(combos)} combination(s) | results -> {out_dir}")

    for i, held_out in enumerate(combos, 1):
        path = out_dir / f"heldout_{'_'.join(held_out)}.json"
        if path.exists():
            print(f"[{i}/{len(combos)}] {held_out}: already done, skipping")
            continue
        print(f"[{i}/{len(combos)}] held out: {held_out}", flush=True)
        result = run_combo(config, held_out, device, seed)
        path.write_text(json.dumps(result, indent=2))
        print(f"  -> seen-crop macro F1 {result['seen_crops_test']['macro_f1']:.3f} | "
              f"held-out macro F1 {result['held_out_crops']['macro_f1']:.3f} "
              f"(gap {result['generalization_gap_macro_f1']:+.3f})", flush=True)


if __name__ == "__main__":
    main()
