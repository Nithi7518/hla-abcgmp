"""Load a trained baseline checkpoint for inference."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from src.models import build_model
from src.preprocessing import build_image_transforms

# Chosen by best VALIDATION macro F1 across the four baseline seeds (never by test).
DEFAULT_CHECKPOINT = "checkpoints/2026-10-12T183341Z_multimodal_hlanet_fusion_seed42.pt"
DEFAULT_CONFIG = "configs/multimodal_hlanet.yaml"


def split_label(label: str) -> tuple[str, str]:
    """'apple__black_rot' -> ('apple', 'black_rot')."""
    crop, _, cls = label.partition("__")
    return crop, cls


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str | Path = DEFAULT_CHECKPOINT, config_path: str | Path = DEFAULT_CONFIG,
               device: str = "auto"):
    """Returns (model in eval mode, label_names indexed by class id, eval transform, device)."""
    ck_path = Path(checkpoint)
    if not ck_path.exists():
        candidates = sorted(Path("checkpoints").glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if candidates:
            ck_path = candidates[-1]

    if ck_path.exists():
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        label_names = [n for n, _ in sorted(ck["label_to_idx"].items(), key=lambda kv: kv[1])]
        model = build_model(ck["model_config"], ck["num_classes"])
        model.load_state_dict(ck["model_state_dict"])
    else:
        from src.data_loader import load_label_map
        label_to_idx = load_label_map("data/manifests")
        label_names = [n for n, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg["model"], len(label_names))

    dev = resolve_device(device)
    model.to(dev).eval()
    with open(config_path) as f:
        image_cfg = yaml.safe_load(f)["image"]
    return model, label_names, build_image_transforms(image_cfg)["eval"], dev
