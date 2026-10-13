"""Model architectures.

`ImageClassifier` / `build_model` are the Phase 7 image-only baseline -- fully
runnable, used by src.train.

`MetadataMLP` / `MultimodalFusionModel` are the Phase 9 multimodal architecture,
implemented so the codebase is ready the moment valid metadata exists, but NOT wired
into src.train and NOT runnable against this dataset release: METADATA_PAIRING_STATUS
is NOT_AVAILABLE (Phase 2 audit -- no structured metadata files were found under
data/raw). Do not construct MultimodalFusionModel with fabricated metadata; wait for a
dataset version where metadata_summary.metadata_pairing_status is VALID or PARTIAL, or
skip and use ImageClassifier alone.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
SUPPORTED_BACKBONES = ["resnet18", "resnet50", "efficientnet_b0", "efficientnet_b2", "hla_net"]


def build_image_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Returns (backbone_with_identity_head, feature_dim)."""
    from torchvision import models
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        feature_dim = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    elif name == "efficientnet_b2":
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b2(weights=weights)
        feature_dim = net.classifier[1].in_features
        net.classifier = nn.Identity()
    elif name == "hla_net":
        from src.hla_net import build_hlanet
        net = build_hlanet({}, num_classes=23)
        return net.backbone, net.feature_dim
    else:
        raise ValueError(f"Unsupported backbone: {name}. Supported: {SUPPORTED_BACKBONES}")
    return net, feature_dim


def freeze_backbone_layers(net: nn.Module, unfreeze_last_n_blocks: int) -> nn.Module:
    """Freezes all backbone parameters, then unfreezes the last `unfreeze_last_n_blocks`
    top-level children THAT ACTUALLY HAVE PARAMETERS (e.g. for resnet, its `layer4`,
    ...) so fine-tuning can touch late, task-specific features while keeping early
    generic ImageNet features fixed. unfreeze_last_n_blocks=0 freezes the entire
    backbone.

    The parameter-count filter matters: build_image_backbone() replaces the
    classification head with nn.Identity() (zero parameters) before this function
    runs, so a naive `list(net.children())[-n:]` would silently select that empty
    placeholder instead of the last real block -- freezing the entire backbone while
    reporting a nonzero unfreeze_last_n_blocks."""
    for p in net.parameters():
        p.requires_grad = False
    if unfreeze_last_n_blocks <= 0:
        return net
    children_with_params = [c for c in net.children() if any(True for _ in c.parameters())]
    for child in children_with_params[-unfreeze_last_n_blocks:]:
        for p in child.parameters():
            p.requires_grad = True
    return net


class ImageClassifier(nn.Module):
    """Phase 7 image-only baseline: pretrained CNN backbone -> dropout -> linear head."""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feature_dim = build_image_backbone(backbone_name, pretrained)
        if freeze_backbone:
            freeze_backbone_layers(self.backbone, unfreeze_last_n_blocks)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(model_cfg: dict[str, Any], num_classes: int) -> nn.Module:
    model_type = model_cfg.get("type", "")
    backbone = model_cfg.get("backbone", "resnet18")

    if model_type == "multimodal_fusion":
        return MultimodalFusionModel(
            backbone_name=backbone,
            num_metadata_features=model_cfg.get("num_metadata_features", 12),
            num_classes=num_classes,
            image_embedding_dim=model_cfg.get("image_embedding_dim", 256),
            metadata_embedding_dim=model_cfg.get("metadata_embedding_dim", 64),
            fusion_hidden_dim=model_cfg.get("fusion_hidden_dim", 128),
            pretrained=model_cfg.get("pretrained", False),
            dropout=model_cfg.get("dropout", 0.3),
        )

    if backbone == "hla_net" or model_type == "hla_net":
        from src.hla_net import build_hlanet

        return build_hlanet(model_cfg, num_classes)
    return ImageClassifier(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        freeze_backbone=model_cfg.get("freeze_backbone", True),
        unfreeze_last_n_blocks=model_cfg.get("unfreeze_last_n_blocks", 1),
        dropout=model_cfg.get("dropout", 0.3),
    )


# --------------------------------------------------------------------------------------
# Multimodal Architecture: Vision (HLA-Net) + Agronomic Context + Gated Fusion
# --------------------------------------------------------------------------------------


class MetadataMLP(nn.Module):
    """Multi-Layer Perceptron embedding for biological/agronomic metadata variables."""

    def __init__(self, input_dim: int = 12, embedding_dim: int = 64, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    """Dynamic Gated Fusion Layer.

    Computes a learned cross-modal gating scalar/vector between visual pathology features
    and agronomic/environmental context.
    When metadata is neutral/missing (e.g. internet uploads), the gate smoothly attenuates
    the metadata pathway and allows pure visual features to drive the diagnosis.
    """

    def __init__(self, visual_dim: int, meta_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.v_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.m_proj = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, v: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        v_h = self.v_proj(v)
        m_h = self.m_proj(m)
        concat = torch.cat([v_h, m_h], dim=1)
        g = self.gate(concat)
        modulated = torch.cat([v_h * g, m_h * (1.0 - g)], dim=1)
        return self.fusion_mlp(modulated)


class MultimodalFusionModel(nn.Module):
    """Multimodal Network: HLA-Net (Visual) + Agronomic Metadata MLP + Gated Fusion."""

    def __init__(
        self,
        backbone_name: str = "hla_net",
        num_metadata_features: int = 12,
        num_classes: int = 23,
        image_embedding_dim: int = 256,
        metadata_embedding_dim: int = 64,
        fusion_hidden_dim: int = 128,
        pretrained: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_metadata_features = num_metadata_features
        self.num_classes = num_classes

        self.image_backbone, feature_dim = build_image_backbone(backbone_name, pretrained)
        self.backbone = self.image_backbone  # For Grad-CAM and Mahalanobis feature extraction compatibility

        self.image_proj = nn.Sequential(
            nn.Linear(feature_dim, image_embedding_dim),
            nn.BatchNorm1d(image_embedding_dim),
            nn.SiLU(),
        )
        self.metadata_mlp = MetadataMLP(num_metadata_features, metadata_embedding_dim, hidden_dim=64, dropout=dropout)
        self.fusion = GatedFusion(image_embedding_dim, metadata_embedding_dim, fusion_hidden_dim, dropout=dropout)
        self.classifier = nn.Linear(fusion_hidden_dim, num_classes)

    def default_neutral_metadata(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Constructs uninformative neutral prior metadata for arbitrary internet uploads."""
        vec = torch.zeros(batch_size, self.num_metadata_features, device=device, dtype=dtype)
        vec[:, :6] = 1.0 / 6.0  # Equal crop prior
        vec[:, 6:8] = 0.5       # Equal organ prior
        vec[:, 8] = 0.5         # Neutral organ indicator
        vec[:, 9] = 0.50        # Mid-range temperature
        vec[:, 10] = 0.75       # Typical field humidity
        vec[:, 11] = 0.60       # Typical seasonal rainfall
        return vec

    def forward(self, image: torch.Tensor, metadata: torch.Tensor | None = None) -> torch.Tensor:
        if metadata is None:
            metadata = self.default_neutral_metadata(image.shape[0], image.device, image.dtype)
        features = self.image_backbone(image)
        image_embedding = self.image_proj(features)
        metadata_embedding = self.metadata_mlp(metadata)
        fused = self.fusion(image_embedding, metadata_embedding)
        return self.classifier(fused)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = MultimodalFusionModel(backbone_name="hla_net", num_metadata_features=12, num_classes=23)
    dummy_image = torch.randn(2, 3, 256, 256)
    dummy_metadata = torch.randn(2, 12)
    out1 = model(dummy_image, dummy_metadata)
    assert out1.shape == (2, 23), out1.shape
    out2 = model(dummy_image)  # Missing metadata fallback
    assert out2.shape == (2, 23), out2.shape
    print(f"MultimodalFusionModel test OK! Total params: {model.total_parameter_count():,}")
