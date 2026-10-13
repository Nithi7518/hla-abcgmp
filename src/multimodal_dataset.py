"""Multimodal Dataset for Image + Agronomic Metadata Fusion.

Extracts high-resolution plant imagery alongside structured biological,
taxonomic, and environmental metadata variables.
Includes metadata dropout (random masking) during training to ensure
arbitrary internet photos without metadata can still be classified robustly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

CROPS = ["apple", "banana", "citrus", "guava", "mango", "papaya"]
ORGANS = ["leaf", "fruit"]

# Regional mean ambient temperature (°C) normalized to [0, 1] across [15°C, 40°C]
CROP_TEMP_NORM = {
    "apple": 0.20,    # ~20°C temperate host
    "banana": 0.52,   # ~28°C humid tropical
    "citrus": 0.44,   # ~26°C subtropical
    "guava": 0.48,    # ~27°C tropical
    "mango": 0.60,    # ~30°C warm tropical
    "papaya": 0.56,   # ~29°C tropical
}

# Regional rainfall / monsoonal moisture factor
CROP_RAIN_INDEX = {
    "apple": 0.35,
    "banana": 0.85,
    "citrus": 0.55,
    "guava": 0.60,
    "mango": 0.70,
    "papaya": 0.80,
}


def encode_metadata_vector(
    crop: str | None = None,
    organ: str | None = None,
    temp_norm: float | None = None,
    humidity_norm: float | None = None,
    rain_norm: float | None = None,
) -> np.ndarray:
    """Encodes biological context into a normalized 12-dimensional vector.
    If attributes are missing (e.g. arbitrary internet upload), neutral priors are assigned.
    """
    vec = np.zeros(12, dtype=np.float32)

    # 1. Crop One-Hot (dim 0..5)
    crop_str = str(crop).lower().strip() if crop else ""
    if crop_str in CROPS:
        vec[CROPS.index(crop_str)] = 1.0
    else:
        vec[0:6] = 1.0 / len(CROPS)  # Uniform prior over all 6 crops

    # 2. Organ One-Hot (dim 6..7)
    organ_str = str(organ).lower().strip() if organ else ""
    if organ_str in ORGANS:
        vec[6 + ORGANS.index(organ_str)] = 1.0
    else:
        vec[6:8] = 0.5  # Neutral 50/50 prior

    # 3. Reproductive / Fruiting stage indicator (dim 8)
    vec[8] = 1.0 if organ_str == "fruit" else (0.0 if organ_str == "leaf" else 0.5)

    # 4. Ambient Temperature (dim 9)
    if temp_norm is not None:
        vec[9] = float(temp_norm)
    elif crop_str in CROP_TEMP_NORM:
        vec[9] = CROP_TEMP_NORM[crop_str]
    else:
        vec[9] = 0.50

    # 5. Relative Humidity index (dim 10)
    if humidity_norm is not None:
        vec[10] = float(humidity_norm)
    else:
        vec[10] = 0.75  # Default ambient field humidity

    # 6. Rainfall / Moisture condition (dim 11)
    if rain_norm is not None:
        vec[11] = float(rain_norm)
    elif crop_str in CROP_RAIN_INDEX:
        vec[11] = CROP_RAIN_INDEX[crop_str]
    else:
        vec[11] = 0.60

    return vec


class MultimodalABCGMPDataset(Dataset):
    """Dataset yielding (image_tensor, metadata_tensor, label_idx)."""

    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        label_to_idx: dict[str, int],
        transform: Callable | None = None,
        metadata_dropout_prob: float = 0.0,
    ) -> None:
        self.data_root = Path(data_root)
        self.transform = transform
        self.metadata_dropout_prob = float(metadata_dropout_prob)
        self.label_to_idx = dict(label_to_idx)

        df = pd.read_csv(manifest_path)
        self.image_paths = df["image_path"].tolist()
        self.labels = [self.label_to_idx[l] for l in df["label"]]

        # Precompute metadata matrix (N, 12)
        meta_list = []
        for _, row in df.iterrows():
            vec = encode_metadata_vector(
                crop=row.get("crop"),
                organ=row.get("organ"),
            )
            meta_list.append(vec)
        self.metadata_matrix = torch.from_numpy(np.array(meta_list, dtype=np.float32))

        # In-memory RAM cache for ultra-fast GPU training
        self._cache: dict[int, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if idx not in self._cache:
            path = self.data_root / self.image_paths[idx]
            img = Image.open(path).convert("RGB")
            self._cache[idx] = img
        else:
            img = self._cache[idx]

        if self.transform is not None:
            image_tensor = self.transform(img)
        else:
            image_tensor = img

        metadata_tensor = self.metadata_matrix[idx].clone()

        # Random metadata masking during training to make model robust to internet photos
        if self.metadata_dropout_prob > 0.0 and torch.rand(1).item() < self.metadata_dropout_prob:
            metadata_tensor = torch.from_numpy(encode_metadata_vector())  # Neutral prior

        label = self.labels[idx]
        return image_tensor, metadata_tensor, label
