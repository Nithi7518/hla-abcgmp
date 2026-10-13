"""Metric computation shared by src.train (per-epoch validation) and src.evaluate
(final test-set reporting + statistics, Phase 8).

Never reports accuracy alone on an imbalanced dataset (Section 20/21 safeguard) --
compute_classification_metrics always returns balanced accuracy and macro F1 alongside
plain accuracy, plus full per-class breakdown.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.metrics import auc as sk_auc


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> dict[str, Any]:
    """Full multi-class metric suite. label_names must be index-ordered (label_names[i]
    is the name of class i) -- see data_loader.build_label_map."""
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=list(range(len(label_names)))
    )

    per_class = {
        label_names[i]: {
            "precision": round(float(per_class_p[i]), 4),
            "recall": round(float(per_class_r[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(per_class_support[i]),
        }
        for i in range(len(label_names))
    }

    return {
        "accuracy": round(float(accuracy), 4),
        "balanced_accuracy": round(float(balanced_accuracy), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_precision": round(float(weighted_p), 4),
        "weighted_recall": round(float(weighted_r), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "n_samples": int(len(y_true)),
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_labels)))


def compute_binary_metrics(y_true: np.ndarray, y_prob_positive: np.ndarray) -> dict[str, Any] | None:
    """Sensitivity/specificity/ROC-AUC/PR-AUC for a binary task (disease_status).
    Returns None if the task is not binary (2 classes) -- these metrics are only
    statistically meaningful for a binary target (Section 20)."""
    unique_labels = np.unique(y_true)
    if len(unique_labels) != 2:
        return None

    y_pred = (y_prob_positive >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    try:
        roc_auc = roc_auc_score(y_true, y_prob_positive)
    except ValueError:
        roc_auc = float("nan")

    precision, recall, _ = precision_recall_curve(y_true, y_prob_positive)
    pr_auc = sk_auc(recall, precision)

    return {
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for a scalar metric (Section 22). Resamples the
    test set with replacement n_bootstrap times; does NOT by itself establish
    statistical significance of a difference between two models -- that requires a
    paired test on the same resampled indices (see compute_paired_bootstrap_diff)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_pred[idx])

    alpha = 1 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return {
        "mean": round(float(scores.mean()), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def compute_paired_bootstrap_diff(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap for metric_fn(model_b) - metric_fn(model_a) on the SAME
    resampled indices each iteration (Section 22: "perform paired statistical testing").
    A confidence interval for the difference that excludes 0 is evidence of a real
    difference; a point-estimate difference alone is NOT sufficient (Section 22)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = metric_fn(y_true[idx], y_pred_b[idx]) - metric_fn(y_true[idx], y_pred_a[idx])

    alpha = 1 - confidence
    lower = float(np.percentile(diffs, 100 * alpha / 2))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "significant_at_confidence": bool(lower > 0 or upper < 0),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def accuracy_metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def balanced_accuracy_metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def macro_f1_metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return float(f1)
