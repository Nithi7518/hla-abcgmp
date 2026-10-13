"""Lightweight CSV experiment tracker (Section 23/24) -- no external service required.

Every training run appends one row to logs/experiments.csv (created with a header on
first use) and writes a per-run folder under results/experiments/<run_id>/ containing
everything needed to reproduce it: config, seed, split manifest reference, final
metrics, package versions, hardware info.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENTS_CSV_COLUMNS = [
    "run_id",
    "timestamp_utc",
    "experiment_name",
    "task",
    "backbone",
    "seed",
    "device",
    "n_train",
    "n_val",
    "n_test",
    "n_labels",
    "epochs_run",
    "best_epoch",
    "class_imbalance_strategy",
    "learning_rate",
    "batch_size",
    "val_macro_f1",
    "val_balanced_accuracy",
    "val_accuracy",
    "test_macro_f1",
    "test_balanced_accuracy",
    "test_accuracy",
    "checkpoint_path",
    "run_dir",
]


def get_package_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for pkg in ["torch", "torchvision", "numpy", "pandas", "sklearn", "PIL", "scipy"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
    return versions


def get_hardware_info() -> dict[str, Any]:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        pass
    return info


def get_git_commit(repo_root: str | Path = ".") -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return None


def make_run_id(experiment_name: str, seed: int) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    return f"{timestamp}_{experiment_name}_seed{seed}"


def create_run_dir(experiments_dir: str | Path, run_id: str) -> Path:
    run_dir = Path(experiments_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_run_metadata(run_dir: Path, metadata: dict[str, Any]) -> None:
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def append_experiment_record(logs_csv_path: str | Path, record: dict[str, Any]) -> None:
    logs_csv_path = Path(logs_csv_path)
    logs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = logs_csv_path.exists()

    row = {col: record.get(col, "") for col in EXPERIMENTS_CSV_COLUMNS}

    with open(logs_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_experiment_record(logs_csv_path: str | Path, run_id: str, updates: dict[str, Any]) -> bool:
    """Updates the row matching `run_id` in place (e.g. src.evaluate filling in
    test_* columns after src.train wrote the initial row with val-only metrics).
    Returns False (no-op) if run_id is not found rather than silently appending a
    duplicate row."""
    logs_csv_path = Path(logs_csv_path)
    if not logs_csv_path.exists():
        return False

    with open(logs_csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    found = False
    for row in rows:
        if row["run_id"] == run_id:
            row.update({k: v for k, v in updates.items() if k in EXPERIMENTS_CSV_COLUMNS})
            found = True
            break

    if not found:
        return False

    with open(logs_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENTS_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return True
