"""Grad-CAM for the ResNet image classifier (Phase 13), implemented in plain PyTorch.

Usage:
    cam = GradCAM(model)                    # hooks model.backbone.layer4
    heatmap = cam(x, class_idx)             # x: (1,3,H,W) normalized tensor -> (H,W) in [0,1]
    overlay = overlay_heatmap(pil_img, heatmap)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module | None = None):
        self.model = model
        self.layer = target_layer if target_layer is not None else model.backbone.layer4
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        self.layer.register_forward_hook(self._fwd)

    def _fwd(self, _module, _inp, out):
        self._acts = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1))
            if self._acts is None or not self._acts.requires_grad:
                raise RuntimeError("Target layer output carries no gradient; is it frozen end to end?")
            logits[0, class_idx].backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0,1] heatmap (jet-like colormap) over the image, resized to the heatmap."""
    import matplotlib

    rgb = image.convert("RGB").resize(heatmap.shape[::-1])
    colors = (matplotlib.colormaps["jet"](heatmap)[..., :3] * 255).astype(np.uint8)
    blended = (1 - alpha) * np.asarray(rgb, dtype=np.float32) + alpha * colors.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))
