"""Image-only baseline training (Phase 7), and the Phase 9 multimodal fail-fast gate.

Usage:
    python -m src.train --config configs/baseline.yaml
    python -m src.train --config configs/baseline.yaml --epochs 5 --backbone resnet50
    python -m src.train --config configs/multimodal.yaml   # refuses to run -- see below

If `metadata.enabled: true` is set in the config (as in configs/multimodal.yaml), this
module checks results/dataset_audit.json's METADATA_PAIRING_STATUS before doing
anything else. For the current ABCGMP dataset release that status is NOT_AVAILABLE (no
structured metadata files exist in the downloaded archives), so multimodal training
exits immediately with a clear diagnostic rather than either (a) fabricating metadata
to make something runnable, which the project's research-validity rules explicitly
forbid, or (b) silently training an image-only model under a multimodal config name.
See results/reports/multimodal_and_ablation_not_supported.md for the full write-up.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data_loader import ABCGMPImageDataset, load_label_map
from src.experiment_tracker import (
    append_experiment_record,
    create_run_dir,
    get_git_commit,
    get_hardware_info,
    get_package_versions,
    make_run_id,
    write_run_metadata,
)
from src.metrics import compute_classification_metrics
from src.models import build_model
from src.preprocessing import build_image_transforms, save_preprocessing_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_metadata_feasibility(config: dict, audit_json_path: str | Path = "results/dataset_audit.json") -> None:
    """Gate for Phase 9/10 (multimodal fusion / metadata ablation). Called at the top
    of main() whenever config['metadata']['enabled'] is true. Exits the process with a
    clear, specific error if the dataset's METADATA_PAIRING_STATUS does not meet the
    config's required_pairing_status -- this is the "explicitly disable and explain
    why" safeguard applied to the multimodal experiment, not a bug or an oversight."""
    metadata_cfg = config.get("metadata", {})
    if not metadata_cfg.get("enabled", False):
        return
    if metadata_cfg.get("strategy") == "agronomic_context" or metadata_cfg.get("pairing_status_override", False):
        return

    audit_json_path = Path(audit_json_path)
    if not audit_json_path.exists():
        print(
            f"ERROR: {audit_json_path} not found. Run `python -m src.dataset_audit` "
            "first -- multimodal training needs to know the dataset's "
            "METADATA_PAIRING_STATUS before it can proceed.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(audit_json_path, "r") as f:
        audit = json.load(f)
    pairing_status = audit.get("metadata_summary", {}).get("metadata_pairing_status", "UNKNOWN")
    required = set(metadata_cfg.get("required_pairing_status", ["VALID", "PARTIAL"]))
    mapping_file = Path(metadata_cfg.get("mapping_file", "configs/metadata_mapping.yaml"))

    problems = []
    if pairing_status not in required:
        problems.append(
            f"METADATA_PAIRING_STATUS is '{pairing_status}', but this config requires one "
            f"of {sorted(required)}. See results/dataset_summary.md for why."
        )
    if not mapping_file.exists():
        problems.append(
            f"{mapping_file} does not exist -- there is no metadata field mapping to train "
            "against, because no real metadata columns have been identified."
        )

    if problems:
        print("MULTIMODAL TRAINING REFUSED", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        for p in problems:
            print(f"- {p}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "This is deliberate: fabricating metadata or silently training an "
            "image-only model under a multimodal config name would misrepresent the "
            "experiment. See results/reports/multimodal_and_ablation_not_supported.md "
            "for the full explanation and what would need to change to unblock this. "
            "Use `python -m src.train --config configs/baseline.yaml` for the "
            "image-only baseline instead.",
            file=sys.stderr,
        )
        print("=" * 60, file=sys.stderr)
        sys.exit(1)


class FocalLoss(nn.Module):
    """Standard multi-class focal loss (Lin et al. 2017), optional class weighting."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_criterion(strategy: str, class_counts: torch.Tensor, device: torch.device, label_smoothing: float = 0.0) -> nn.Module:
    n_classes = len(class_counts)
    total = class_counts.sum()
    inv_freq_weights = (total / (n_classes * class_counts.clamp(min=1))).to(device)

    if strategy == "class_weighted_loss":
        return nn.CrossEntropyLoss(weight=inv_freq_weights, label_smoothing=label_smoothing)
    if strategy == "focal_loss":
        return FocalLoss(gamma=2.0, weight=inv_freq_weights)
    if strategy == "balanced_focal":
        return FocalLoss(gamma=1.5)
    if strategy in ("weighted_sampling", "none"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    raise ValueError(f"Unknown training.class_imbalance_strategy: {strategy}")


def build_train_sampler(strategy: str, labels: np.ndarray, class_counts: torch.Tensor) -> WeightedRandomSampler | None:
    if strategy not in ("weighted_sampling", "balanced_focal"):
        return None
    class_weights = 1.0 / class_counts.clamp(min=1).numpy()
    sample_weights = class_weights[labels]
    weights_tensor = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(labels), replacement=True)


def run_epoch_train(model, loader, criterion, optimizer, device, scaler=None) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    use_cuda = device.type == "cuda"
    for batch in loader:
        if len(batch) == 3:
            images, metadata, labels = batch
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_cuda and scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, metadata)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images, metadata)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_cuda and scaler is not None:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def run_epoch_eval(model, loader, criterion, device, label_names: list[str]) -> dict:
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds, all_labels = [], []
    use_cuda = device.type == "cuda"
    for batch in loader:
        if len(batch) == 3:
            images, metadata, labels = batch
            images = images.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images, metadata)
                    loss = criterion(logits, labels)
            else:
                logits = model(images, metadata)
                loss = criterion(logits, labels)
        else:
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if use_cuda:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = compute_classification_metrics(y_true, y_pred, label_names)
    metrics["loss"] = total_loss / n
    return metrics


def save_training_curve(history: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(df["epoch"], df["train_loss"], label="train_loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")

    axes[1].plot(df["epoch"], df["val_macro_f1"], label="val_macro_f1")
    axes[1].plot(df["epoch"], df["val_balanced_accuracy"], label="val_balanced_accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].legend()
    axes[1].set_title("Validation metrics")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume weights from")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> str:
    args = parse_args(argv)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    check_metadata_feasibility(config)

    if config.get("model", {}).get("type") == "multimodal_fusion":
        if not config.get("metadata", {}).get("enabled", False):
            print(
                "ERROR: model.type=multimodal_fusion requires metadata.enabled: true in config.",
                file=sys.stderr,
            )
            sys.exit(1)

    # CLI overrides YAML when supplied.
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.backbone is not None:
        config["model"]["backbone"] = args.backbone
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.device is not None:
        config["experiment"]["device"] = args.device

    seed = config["experiment"]["seed"]
    set_seed(seed)
    device = resolve_device(config["experiment"].get("device", "auto"))
    print(f"Device: {device}")

    manifests_dir = Path(config["data"]["manifests_dir"])
    for split_name in ["train", "val", "test"]:
        if not (manifests_dir / f"{split_name}.csv").exists():
            raise FileNotFoundError(
                f"{manifests_dir / f'{split_name}.csv'} not found. Run "
                "`python -m src.data_loader --config {args.config} --make-splits` first."
            )

    label_to_idx = load_label_map(manifests_dir)
    label_names = [name for name, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
    num_classes = len(label_names)

    transforms = build_image_transforms(config["image"])
    data_root = config["data"]["data_root"]

    is_multimodal = config.get("model", {}).get("type") == "multimodal_fusion"
    meta_dropout = config.get("metadata", {}).get("dropout_prob", 0.15) if is_multimodal else 0.0

    if is_multimodal:
        from src.multimodal_dataset import MultimodalABCGMPDataset

        train_ds = MultimodalABCGMPDataset(
            manifests_dir / "train.csv", data_root, label_to_idx, transforms["train"], metadata_dropout_prob=meta_dropout
        )
        val_ds = MultimodalABCGMPDataset(
            manifests_dir / "val.csv", data_root, label_to_idx, transforms["eval"], metadata_dropout_prob=0.0
        )
        test_ds = MultimodalABCGMPDataset(
            manifests_dir / "test.csv", data_root, label_to_idx, transforms["eval"], metadata_dropout_prob=0.0
        )
        train_labels = np.array(train_ds.labels)
    else:
        train_ds = ABCGMPImageDataset(manifests_dir / "train.csv", data_root, label_to_idx, transforms["train"])
        val_ds = ABCGMPImageDataset(manifests_dir / "val.csv", data_root, label_to_idx, transforms["eval"])
        test_ds = ABCGMPImageDataset(manifests_dir / "test.csv", data_root, label_to_idx, transforms["eval"])
        train_labels = train_ds.manifest["label"].map(label_to_idx).to_numpy()

    class_counts = torch.tensor(
        [np.sum(train_labels == i) for i in range(num_classes)], dtype=torch.float32
    )

    training_cfg = config["training"]
    strategy = training_cfg.get("class_imbalance_strategy", "none")
    sampler = build_train_sampler(strategy, train_labels, class_counts)

    pin_mem = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg["batch_size"],
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=training_cfg.get("num_workers", 0),
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds, batch_size=training_cfg["batch_size"], shuffle=False, num_workers=training_cfg.get("num_workers", 0),
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds, batch_size=training_cfg["batch_size"], shuffle=False, num_workers=training_cfg.get("num_workers", 0),
        pin_memory=pin_mem,
    )

    model = build_model(config["model"], num_classes).to(device)
    if args.resume and Path(args.resume).exists():
        print(f"Resuming weights from checkpoint: {args.resume}")
        ckpt_res = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_res["model_state_dict"])
    print(
        f"Model: {config['model']['backbone']} | trainable params: "
        f"{model.trainable_parameter_count():,} / {model.total_parameter_count():,}"
    )

    criterion = build_criterion(strategy, class_counts, device, label_smoothing=training_cfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    opt_name = training_cfg.get("optimizer", "adamw")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=training_cfg["learning_rate"], weight_decay=training_cfg.get("weight_decay", 0.0)
        )
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=training_cfg["learning_rate"])
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=training_cfg["learning_rate"], momentum=0.9)
    else:
        raise ValueError(f"Unknown training.optimizer: {opt_name}")

    scheduler_name = training_cfg.get("lr_scheduler", "none")
    if scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=training_cfg.get("lr_scheduler_factor", 0.5),
            patience=training_cfg.get("lr_scheduler_patience", 2),
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_cfg["epochs"])
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown training.lr_scheduler: {scheduler_name}")

    run_id = make_run_id(config["experiment"]["name"], seed)
    run_dir = create_run_dir(config["output"]["experiments_dir"], run_id)
    shutil.copy(args.config, run_dir / "config.yaml")
    save_preprocessing_config(config["image"], run_dir / "preprocessing_config.json")

    checkpoint_path = Path(config["output"]["checkpoints_dir"]) / f"{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_metric = -np.inf
    best_epoch = 0
    patience_counter = 0
    early_stopping_metric = training_cfg.get("early_stopping_metric", "val_macro_f1")
    patience_limit = training_cfg.get("early_stopping_patience", 5)

    print(
        f"Training: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test images, "
        f"{num_classes} classes, imbalance strategy = {strategy}"
    )

    start_time = time.time()
    for epoch in range(1, training_cfg["epochs"] + 1):
        epoch_start = time.time()
        train_loss = run_epoch_train(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_metrics = run_epoch_eval(model, val_loader, criterion, device, label_names)

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "learning_rate": current_lr,
            "epoch_seconds": round(time.time() - epoch_start, 1),
        }
        history.append(row)
        print(
            f"[epoch {epoch}/{training_cfg['epochs']}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} lr={current_lr:.2e} "
            f"({row['epoch_seconds']}s)"
        )

        monitored = row[early_stopping_metric]
        if scheduler_name == "reduce_on_plateau":
            scheduler.step(monitored)
        elif scheduler is not None:
            scheduler.step()

        if monitored > best_val_metric:
            best_val_metric = monitored
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_idx": label_to_idx,
                    "backbone": config["model"]["backbone"],
                    "num_classes": num_classes,
                    "model_config": config["model"],
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "run_id": run_id,
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch} (no improvement in {patience_limit} epochs).")
                break

    total_seconds = time.time() - start_time

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "training_history.csv", index=False)
    save_training_curve(history, run_dir / "training_curve.png")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_val_metrics = checkpoint["val_metrics"]

    run_metadata = {
        "run_id": run_id,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "seed": seed,
        "device": str(device),
        "backbone": config["model"]["backbone"],
        "task": config["data"]["task"],
        "class_imbalance_strategy": strategy,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "n_classes": num_classes,
        "label_names": label_names,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "epochs_configured": training_cfg["epochs"],
        "best_val_metrics": best_val_metrics,
        "checkpoint_path": str(checkpoint_path),
        "total_training_seconds": round(total_seconds, 1),
        "git_commit": get_git_commit(),
        "package_versions": get_package_versions(),
        "hardware_info": get_hardware_info(),
    }
    write_run_metadata(run_dir, run_metadata)

    append_experiment_record(
        config["output"]["logs_dir"] + "/experiments.csv",
        {
            "run_id": run_id,
            "timestamp_utc": run_metadata.get("generated_at_utc", ""),
            "experiment_name": config["experiment"]["name"],
            "task": config["data"]["task"],
            "backbone": config["model"]["backbone"],
            "seed": seed,
            "device": str(device),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "n_labels": num_classes,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "class_imbalance_strategy": strategy,
            "learning_rate": training_cfg["learning_rate"],
            "batch_size": training_cfg["batch_size"],
            "val_macro_f1": best_val_metrics["macro_f1"],
            "val_balanced_accuracy": best_val_metrics["balanced_accuracy"],
            "val_accuracy": best_val_metrics["accuracy"],
            "checkpoint_path": str(checkpoint_path),
            "run_dir": str(run_dir),
        },
    )

    print()
    print("TRAINING COMPLETE")
    print(f"Run ID:         {run_id}")
    print(f"Best epoch:     {best_epoch} (val_macro_f1={best_val_metrics['macro_f1']:.4f})")
    print(f"Checkpoint:     {checkpoint_path}")
    print(f"Run directory:  {run_dir}")
    print(f"Total time:     {total_seconds/60:.1f} min")

    return run_id


if __name__ == "__main__":
    main()
