"""Inference core: image -> calibrated top-k, confidence tier, Grad-CAM overlay.

Tiers (thresholds come from results/assistant/calibration.json, fit on validation only):
    confident  -- calibrated top-1 probability >= threshold_confident
    uncertain  -- between threshold_refuse and threshold_confident
    refuse     -- below threshold_refuse: the assistant will not name a diagnosis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.assistant.model_io import DEFAULT_CHECKPOINT, load_model, split_label
from src.explain import GradCAM, overlay_heatmap

CALIBRATION_JSON = "results/assistant/calibration.json"
OOD_STATS = "results/assistant/ood_stats.npz"
TIER_DOWN = {"confident": "uncertain", "uncertain": "refuse", "refuse": "refuse"}
OOD_NOTE = ("This photo looks unlike the images the model was trained on (for example a studio or stock "
            "photo, or an unusual background), so I lowered my confidence one level.")


@dataclass
class Candidate:
    label: str
    crop: str
    disease: str
    prob: float


@dataclass
class Diagnosis:
    tier: str                       # confident | uncertain | refuse
    candidates: list[Candidate]     # top-k, calibrated probabilities, best first
    entropy: float
    overlay: Image.Image | None = None
    probs: np.ndarray | None = None  # calibrated probabilities over all labels
    ood_distance: float | None = None
    ood_flag: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def top(self) -> Candidate:
        return self.candidates[0]


class Diagnoser:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, calibration_json: str = CALIBRATION_JSON,
                 device: str = "auto"):
        self.model, self.labels, self.transform, self.device = load_model(checkpoint, device=device)
        cal = json.loads(Path(calibration_json).read_text())
        self.temperature = cal["temperature"]
        self.t_confident = cal["threshold_confident"]
        self.t_refuse = cal["threshold_refuse"]
        # Target layer3 (14x14 multi-scale dilated stage) for sharp lesion localization instead of coarse layer4 (7x7)
        backbone_obj = getattr(self.model, "backbone", self.model)
        target_cam_layer = getattr(backbone_obj, "layer3", getattr(backbone_obj, "layer4", None))
        self.cam = GradCAM(self.model, target_layer=target_cam_layer)
        self.ood = None
        self.ood_group_thr = None
        if Path(OOD_STATS).exists():
            z = np.load(OOD_STATS)
            self.ood = (z["means"], z["prec"], float(z["threshold"]))
            self.ood_group_thr = ({"fruit": float(z["thr_fruit"]), "other": float(z["thr_other"])}
                                  if "thr_fruit" in z.files else None)

    def ood_distance(self, image: Image.Image) -> float | None:
        """Min class-conditional Mahalanobis distance in the 512-d feature space."""
        if self.ood is None:
            return None
        means, prec, _ = self.ood
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model.backbone(x).cpu().numpy()[0]
        diff = means - f
        return float(np.sqrt(np.einsum("ij,jk,ik->i", diff, prec, diff).min()))

    def probabilities(self, image: Image.Image, metadata: torch.Tensor | None = None) -> np.ndarray:
        x = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if metadata is not None:
                logits = self.model(x, metadata.to(self.device))
            else:
                logits = self.model(x)
            return torch.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()

    def tier_for(self, p_top: float) -> str:
        if p_top >= self.t_confident:
            return "confident"
        return "uncertain" if p_top >= self.t_refuse else "refuse"

    def diagnose(self, image: Image.Image, top_k: int = 3, with_overlay: bool = True,
                 crop: str | None = None, organ: str | None = None) -> Diagnosis:
        image = image.convert("RGB")
        meta_vec = None
        if crop is not None or organ is not None:
            from src.multimodal_dataset import encode_metadata_vector
            vec = encode_metadata_vector(crop=crop, organ=organ)
            meta_vec = torch.from_numpy(vec).unsqueeze(0).float()
        probs = self.probabilities(image, metadata=meta_vec)
        order = np.argsort(-probs)[:top_k]
        cands = [Candidate(self.labels[i], *split_label(self.labels[i]), float(probs[i])) for i in order]
        entropy = float(-(probs * np.log(probs.clip(1e-12))).sum())
        tier = self.tier_for(cands[0].prob)
        dist = self.ood_distance(image)
        thr = self.ood[2] if dist is not None else None
        if dist is not None and getattr(self, "ood_group_thr", None):
            cls = cands[0].label.split("__", 1)[-1]
            thr = self.ood_group_thr["fruit" if "fruit" in cls else "other"]
        flag = dist is not None and (
            dist > 50.0 or (dist > max(thr * 1.45, 42.0) and cands[0].prob < self.t_confident)
        )
        notes = []
        if flag:
            tier = TIER_DOWN[tier]
            notes.append(OOD_NOTE)
        overlay = None
        if with_overlay:
            x = self.transform(image).unsqueeze(0).to(self.device)
            overlay = overlay_heatmap(image, self.cam(x, self.labels.index(cands[0].label)))
        return Diagnosis(tier, cands, entropy, overlay, probs, dist, bool(flag), notes)
