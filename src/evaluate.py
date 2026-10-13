"""Test-set evaluation + statistics (Phase 8).

Loads a checkpoint saved by src.train, runs it once on the held-out test split (never
touched during training or model selection), and writes the full required metric
suite plus bootstrap confidence intervals -- never accuracy alone (Section 20/22).

Usage:
    python -m src.evaluate --config configs/baseline.yaml --run-id <run_id>
    python -m src.evaluate --config configs/baseline.yaml --checkpoint checkpoints/<run_id>.pt
    python -m src.evaluate --config configs/baseline.yaml   # picks the most recent checkpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data_loader import ABCGMPImageDataset
from src.experiment_tracker import update_experiment_record
from src.metrics import (
    accuracy_metric_fn,
    balanced_accuracy_metric_fn,
    bootstrap_ci,
    compute_binary_metrics,
    compute_classification_metrics,
    compute_confusion_matrix,
    macro_f1_metric_fn,
)
from src.models import build_model
from src.preprocessing import build_image_transforms
from src.train import resolve_device


def find_latest_checkpoint(checkpoints_dir: str | Path) -> Path:
    checkpoints_dir = Path(checkpoints_dir)
    candidates = sorted(checkpoints_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoints_dir}. Run `python -m src.train` first."
        )
    return candidates[-1]


def save_confusion_matrix_plot(cm: np.ndarray, label_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(label_names)
    fig_size = max(8, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=90, fontsize=7)
    ax.set_yticklabels(label_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(
                    j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=6, color="white" if cm[i, j] > thresh else "black",
                )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--tta", action="store_true", help="Enable 5-view test-time augmentation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.run_id:
        checkpoint_path = Path(config["output"]["checkpoints_dir"]) / f"{args.run_id}.pt"
    else:
        checkpoint_path = find_latest_checkpoint(config["output"]["checkpoints_dir"])

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device(config["experiment"].get("device", "auto"))
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_id = checkpoint["run_id"]
    label_to_idx = checkpoint["label_to_idx"]
    label_names = [name for name, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    num_classes = checkpoint["num_classes"]

    model = build_model(checkpoint["model_config"], num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transforms = build_image_transforms(config["image"])
    manifests_dir = Path(config["data"]["manifests_dir"])
    is_multimodal = checkpoint.get("model_config", {}).get("type") == "multimodal_fusion"
    if is_multimodal:
        from src.multimodal_dataset import MultimodalABCGMPDataset

        test_ds = MultimodalABCGMPDataset(
            manifests_dir / "test.csv", config["data"]["data_root"], label_to_idx, transforms["eval"], metadata_dropout_prob=0.0
        )
    else:
        test_ds = ABCGMPImageDataset(
            manifests_dir / "test.csv", config["data"]["data_root"], label_to_idx, transforms["eval"]
        )
    test_loader = DataLoader(
        test_ds, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["training"].get("num_workers", 0),
    )

    all_probs, all_preds, all_labels = [], [], []
    use_cuda = device.type == "cuda"
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, metadata, labels = batch
                images = images.to(device)
                metadata = metadata.to(device)
            else:
                images, labels = batch
                images = images.to(device)
                metadata = None

            if args.tta:
                t_variants = [
                    images,
                    torch.flip(images, dims=[3]),  # horizontal flip
                    torch.flip(images, dims=[2]),  # vertical flip
                    torch.flip(images, dims=[2, 3]),  # horizontal + vertical flip
                    torch.rot90(images, k=1, dims=[2, 3]),  # 90-degree rotation
                ]
                batch_probs = torch.zeros(images.size(0), num_classes, device=device)
                for variant in t_variants:
                    if use_cuda:
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(variant, metadata) if metadata is not None else model(variant)
                    else:
                        logits = model(variant, metadata) if metadata is not None else model(variant)
                    batch_probs += torch.softmax(logits, dim=1)
                probs = (batch_probs / len(t_variants)).cpu().numpy()
            else:
                if use_cuda:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        logits = model(images, metadata) if metadata is not None else model(images)
                else:
                    logits = model(images, metadata) if metadata is not None else model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

    y_prob = np.concatenate(all_probs)
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)

    metrics = compute_classification_metrics(y_true, y_pred, label_names)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    binary_metrics = None
    if num_classes == 2:
        positive_idx = label_to_idx.get("diseased", 1)
        binary_metrics = compute_binary_metrics(y_true, y_prob[:, positive_idx])

    print("Computing bootstrap confidence intervals...")
    ci_accuracy = bootstrap_ci(y_true, y_pred, accuracy_metric_fn, n_bootstrap=args.n_bootstrap)
    ci_balanced_accuracy = bootstrap_ci(y_true, y_pred, balanced_accuracy_metric_fn, n_bootstrap=args.n_bootstrap)
    ci_macro_f1 = bootstrap_ci(y_true, y_pred, macro_f1_metric_fn, n_bootstrap=args.n_bootstrap)

    run_dir = Path(config["output"]["experiments_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    full_report = {
        "run_id": run_id,
        "checkpoint_path": str(checkpoint_path),
        "n_test_images": len(test_ds),
        "test_metrics": metrics,
        "binary_metrics": binary_metrics,
        "bootstrap_confidence_intervals": {
            "accuracy": ci_accuracy,
            "balanced_accuracy": ci_balanced_accuracy,
            "macro_f1": ci_macro_f1,
        },
        "note_on_statistical_significance": (
            "These are single-model confidence intervals, not a paired comparison. "
            "This project's primary research question (image-only vs multimodal) "
            "cannot currently be tested this way: METADATA_PAIRING_STATUS is "
            "NOT_AVAILABLE for this dataset release, so no multimodal model exists to "
            "compare against (see results/dataset_summary.md). Do not interpret a "
            "narrow single-model CI as evidence for or against any multimodal claim."
        ),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(full_report, f, indent=2, default=str)

    class_report_df = pd.DataFrame(metrics["per_class"]).T
    class_report_df.index.name = "label"
    class_report_df.to_csv(run_dir / "classification_report.csv")

    save_confusion_matrix_plot(cm, label_names, run_dir / "confusion_matrix.png")
    confusion_matrices_dir = Path("results/confusion_matrices")
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_plot(cm, label_names, confusion_matrices_dir / f"{run_id}.png")

    updated = update_experiment_record(
        config["output"]["logs_dir"] + "/experiments.csv",
        run_id,
        {
            "test_macro_f1": metrics["macro_f1"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_accuracy": metrics["accuracy"],
        },
    )

    print()
    print("EVALUATION COMPLETE")
    print(f"Run ID:              {run_id}")
    print(f"Test images:         {len(test_ds)}")
    print(f"Accuracy:            {metrics['accuracy']}  (95% CI [{ci_accuracy['ci_lower']}, {ci_accuracy['ci_upper']}])")
    print(f"Balanced accuracy:   {metrics['balanced_accuracy']}  (95% CI [{ci_balanced_accuracy['ci_lower']}, {ci_balanced_accuracy['ci_upper']}])")
    print(f"Macro F1:            {metrics['macro_f1']}  (95% CI [{ci_macro_f1['ci_lower']}, {ci_macro_f1['ci_upper']}])")
    print(f"Weighted F1:         {metrics['weighted_f1']}")
    if binary_metrics:
        print(f"Sensitivity/Specificity/ROC-AUC/PR-AUC: {binary_metrics}")
    print()
    print(f"Full metrics:        {run_dir}/metrics.json")
    print(f"Classification report: {run_dir}/classification_report.csv")
    print(f"Confusion matrix:    {run_dir}/confusion_matrix.png")
    if updated:
        print(f"logs/experiments.csv row updated with test metrics.")
    else:
        print(f"WARNING: run_id {run_id} not found in logs/experiments.csv -- test metrics not logged there.")


if __name__ == "__main__":
    main()
