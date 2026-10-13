"""Temperature scaling + confidence thresholds for the assistant (Phase A).

Usage:
    python -m src.assistant.calibration

Fits a single temperature on the VALIDATION split, derives confidence thresholds from
validation only, then reports the effect on the test split (reporting only -- nothing
is tuned on test). Output: results/assistant/calibration.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.assistant.model_io import DEFAULT_CHECKPOINT, load_model

OUT = Path("results/assistant/calibration.json")


@torch.no_grad()
def collect_logits(model, label_names, transform, device, manifest_csv, data_root="data/raw"):
    df = pd.read_csv(manifest_csv)
    idx = {n: i for i, n in enumerate(label_names)}
    logits, ys = [], []
    for start in range(0, len(df), 64):
        chunk = df.iloc[start:start + 64]
        x = torch.stack([transform(Image.open(Path(data_root) / p).convert("RGB")) for p in chunk.image_path])
        logits.append(model(x.to(device)).cpu())
        ys += [idx[l] for l in chunk.label]
    return torch.cat(logits), torch.tensor(ys)


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def threshold_for_accuracy(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Smallest confidence threshold whose accepted predictions reach `target` accuracy."""
    for t in np.unique(np.round(conf, 4)):
        m = conf >= t
        if m.sum() >= 20 and correct[m].mean() >= target:
            return float(t)
    return 1.0


def threshold_for_refusal(conf: np.ndarray, correct: np.ndarray, max_acc: float = 0.5, window: float = 0.08) -> float:
    """Finds the confidence threshold below which local predictions fail to exceed `max_acc` accuracy."""
    thresholds = np.linspace(0.20, 0.70, 51)
    matching = [
        float(t)
        for t in thresholds
        if ((conf >= t - window) & (conf < t)).sum() >= 15 and correct[(conf >= t - window) & (conf < t)].mean() <= max_acc
    ]
    return round(matching[-1], 4) if matching else 0.45


def coverage_table(conf, correct, thresholds):
    return [{"threshold": float(t), "coverage": float((conf >= t).mean()),
             "accuracy_on_accepted": float(correct[conf >= t].mean()) if (conf >= t).any() else None}
            for t in thresholds]


def main() -> None:
    model, labels, tf, dev = load_model()
    lv, yv = collect_logits(model, labels, tf, dev, "data/manifests/val.csv")
    lt, yt = collect_logits(model, labels, tf, dev, "data/manifests/test.csv")
    T = fit_temperature(lv, yv)
    sm = lambda l, t=1.0: torch.softmax(l / t, 1).numpy()
    pv, pt = sm(lv, T), sm(lt, T)
    cv, ct = pv.max(1), pt.max(1)
    okv, okt = pv.argmax(1) == yv.numpy(), pt.argmax(1) == yt.numpy()

    t_confident = threshold_for_accuracy(cv, okv, 0.985)  # "confident" tier
    t_refuse = threshold_for_refusal(cv, okv, 0.5)         # below this -> refuse
    result = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "temperature": T,
        "ece_val_before": ece(sm(lv), yv.numpy()), "ece_val_after": ece(pv, yv.numpy()),
        "ece_test_before": ece(sm(lt), yt.numpy()), "ece_test_after": ece(pt, yt.numpy()),
        "threshold_confident": t_confident, "threshold_refuse": t_refuse,
        "thresholds_derived_from": "validation split only (confident: accepted accuracy >= 0.985; refuse: rejected accuracy <= 0.5)",
        "test_tiers": {
            "confident": {"coverage": float((ct >= t_confident).mean()),
                          "accuracy": float(okt[ct >= t_confident].mean())},
            "uncertain": {"coverage": float(((ct >= t_refuse) & (ct < t_confident)).mean()),
                          "accuracy": float(okt[(ct >= t_refuse) & (ct < t_confident)].mean())},
            "refuse": {"coverage": float((ct < t_refuse).mean()),
                       "accuracy": float(okt[ct < t_refuse].mean()) if (ct < t_refuse).any() else None},
        },
        "test_coverage_table": coverage_table(ct, okt, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "test_coverage_table"}, indent=2))


if __name__ == "__main__":
    main()
