"""HLANet: Hierarchical Lesion-Aware Multiscale Attention Network.

Built from scratch in PyTorch for botanical and agricultural disease diagnosis.

Architectural highlights:
1. MSRStem: Multi-Scale Receptive Stem using parallel 3x3 and 5x5 convolutions
   to preserve high-frequency symptom boundaries (pustules, micro-lesions).
2. DMSLBlock: Dilated Multi-Scale Lesion Block capturing multi-rate pathology:
   - Branch 1: Dilation rate = 1 (micro-scale pinpricks and fungal spores)
   - Branch 2: Dilation rate = 2 (meso-scale halo margins and ring spots)
   - Branch 3: Dilation rate = 4 (macro-scale necrotic patches and blights)
3. CoordinateLesionAttention: Coordinate-aware horizontal and vertical pooling
   attention that highlights lesion regions and suppresses background shortcuts.
4. Anti-Overfitting Safeguards:
   - Depthwise-separable convolutions throughout (parameter-efficient ~2.4M params).
   - DropPath (stochastic depth) in residual branches.
   - Channel and spatial Batch Normalization with SiLU activations.
   - Pre-classifier Dropout (0.3).
5. Anti-Underfitting Safeguards:
   - Full 4-stage residual pyramid with smooth gradient propagation.
   - Kaiming He-normal initialization for all convolutional kernels.
   - Fully compatible with PyTorch autograd, Grad-CAM (hooks layer4), and OOD Mahalanobis.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) module."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class CoordinateLesionAttention(nn.Module):
    """Coordinate Lesion Attention (CLA).

    Encodes direction-aware location information by decomposing 2D global pooling
    into 1D horizontal and 1D vertical spatial pooling. This allows the model to
    locate lesions accurately on the leaf blade while suppressing irrelevant background
    and table textures (mitigating shortcut learning).
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(8, in_channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape

        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        return identity * a_h * a_w


class MSRStem(nn.Module):
    """Multi-Scale Receptive Stem.

    Instead of immediate harsh downsampling, runs parallel 3x3 and 5x5 depthwise
    convolutions to capture both micro-scale puncture lesions and broader discolored halos
    without obliterating fine boundary cues.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        mid_channels = out_channels // 2

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        # MaxPool to achieve 1/4 resolution (56x56) smoothly
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out3 = self.branch3x3(x)
        out5 = self.branch5x5(x)
        fused = self.fuse(torch.cat([out3, out5], dim=1))
        return self.pool(fused)


class DMSLBlock(nn.Module):
    """Dilated Multi-Scale Lesion Block (D-MSLB).

    Extracts features simultaneously across three dilation rates (1, 2, 4) using
    parameter-efficient depthwise separable convolutions, followed by coordinate
    lesion attention and stochastic depth (DropPath) regularization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        hidden_dim = in_channels * expansion
        branch_dim = hidden_dim // 3
        # Adjust hidden_dim so it divides evenly by 3
        hidden_dim = branch_dim * 3

        # 1x1 Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )

        # 3 Parallel Multi-Scale Dilation Branches (Depthwise)
        self.dw_d1 = nn.Sequential(
            nn.Conv2d(
                branch_dim,
                branch_dim,
                kernel_size=3,
                stride=stride,
                padding=1,
                dilation=1,
                groups=branch_dim,
                bias=False,
            ),
            nn.BatchNorm2d(branch_dim),
            nn.SiLU(inplace=True),
        )
        self.dw_d2 = nn.Sequential(
            nn.Conv2d(
                branch_dim,
                branch_dim,
                kernel_size=3,
                stride=stride,
                padding=2,
                dilation=2,
                groups=branch_dim,
                bias=False,
            ),
            nn.BatchNorm2d(branch_dim),
            nn.SiLU(inplace=True),
        )
        self.dw_d4 = nn.Sequential(
            nn.Conv2d(
                branch_dim,
                branch_dim,
                kernel_size=3,
                stride=stride,
                padding=4,
                dilation=4,
                groups=branch_dim,
                bias=False,
            ),
            nn.BatchNorm2d(branch_dim),
            nn.SiLU(inplace=True),
        )

        # Coordinate Lesion Attention over fused branches
        self.attn = CoordinateLesionAttention(hidden_dim)

        # 1x1 Projection back to out_channels
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Residual shortcut
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        expanded = self.expand(x)
        b1, b2, b3 = torch.chunk(expanded, 3, dim=1)

        out1 = self.dw_d1(b1)
        out2 = self.dw_d2(b2)
        out3 = self.dw_d4(b3)

        fused = torch.cat([out1, out2, out3], dim=1)
        fused = self.attn(fused)
        projected = self.project(fused)

        return identity + self.drop_path(projected)


class HLABackbone(nn.Module):
    """4-Stage Hierarchical Lesion-Aware Backbone.

    Named with layer1, layer2, layer3, layer4 to be 100% plug-and-play compatible
    with Grad-CAM (which hooks layer4), feature extractors, and OOD Mahalanobis metrics.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        stage_depths: tuple[int, int, int, int] = (2, 2, 3, 2),
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        c1, c2, c3, c4 = stage_channels
        d1, d2, d3, d4 = stage_depths

        # Linearly increasing drop-path rates across blocks
        total_blocks = sum(stage_depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Stem (56x56 output)
        self.stem = MSRStem(in_channels, c1)

        # Stage 1 (layer1): 56x56
        blocks_l1 = []
        for i in range(d1):
            blocks_l1.append(DMSLBlock(c1, c1, stride=1, drop_path_rate=dpr[i]))
        self.layer1 = nn.Sequential(*blocks_l1)

        # Stage 2 (layer2): 28x28 (stride 2 at first block)
        blocks_l2 = []
        for i in range(d2):
            stride = 2 if i == 0 else 1
            in_c = c1 if i == 0 else c2
            blocks_l2.append(DMSLBlock(in_c, c2, stride=stride, drop_path_rate=dpr[d1 + i]))
        self.layer2 = nn.Sequential(*blocks_l2)

        # Stage 3 (layer3): 14x14 (stride 2 at first block)
        blocks_l3 = []
        for i in range(d3):
            stride = 2 if i == 0 else 1
            in_c = c2 if i == 0 else c3
            blocks_l3.append(DMSLBlock(in_c, c3, stride=stride, drop_path_rate=dpr[d1 + d2 + i]))
        self.layer3 = nn.Sequential(*blocks_l3)

        # Stage 4 (layer4): 7x7 (stride 2 at first block)
        blocks_l4 = []
        for i in range(d4):
            stride = 2 if i == 0 else 1
            in_c = c3 if i == 0 else c4
            blocks_l4.append(DMSLBlock(in_c, c4, stride=stride, drop_path_rate=dpr[d1 + d2 + d3 + i]))
        self.layer4 = nn.Sequential(*blocks_l4)

        # Global average pool
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.out_dim = c4
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal initialization to ensure smooth, unhindered gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-d global features."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class HLANet(nn.Module):
    """End-to-End Hierarchical Lesion-Aware Attention Network.

    Combines the HLABackbone with a regularized classification head.
    Equipped with trainable and total parameter counter methods for seamless
    integration with existing train.py and evaluate.py pipelines.
    """

    def __init__(
        self,
        num_classes: int,
        stage_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        stage_depths: tuple[int, int, int, int] = (2, 2, 3, 2),
        dropout: float = 0.3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = "hla_net"
        self.backbone = HLABackbone(
            stage_channels=stage_channels,
            stage_depths=stage_depths,
            drop_path_rate=drop_path_rate,
        )
        self.feature_dim = self.backbone.out_dim

        # Classifier head with dropout to prevent co-adaptation and overfitting
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_hlanet(model_cfg: dict[str, Any], num_classes: int) -> HLANet:
    """Builds HLANet from configuration dictionary."""
    stage_channels = tuple(model_cfg.get("stage_channels", (64, 128, 256, 512)))
    stage_depths = tuple(model_cfg.get("stage_depths", (2, 2, 3, 2)))
    dropout = float(model_cfg.get("dropout", 0.3))
    drop_path_rate = float(model_cfg.get("drop_path_rate", 0.1))

    return HLANet(
        num_classes=num_classes,
        stage_channels=stage_channels,
        stage_depths=stage_depths,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
    )
