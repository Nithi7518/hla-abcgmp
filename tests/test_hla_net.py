"""Unit tests for HLA-Net architecture and modules."""

import torch

from src.explain import GradCAM
from src.hla_net import CoordinateLesionAttention, DMSLBlock, HLANet, MSRStem, build_hlanet
from src.models import build_model


def test_msr_stem():
    stem = MSRStem(in_channels=3, out_channels=64)
    x = torch.randn(2, 3, 224, 224)
    out = stem(x)
    assert out.shape == (2, 64, 56, 56), f"Unexpected stem output shape: {out.shape}"


def test_coordinate_lesion_attention():
    cla = CoordinateLesionAttention(in_channels=64, reduction=16)
    x = torch.randn(2, 64, 28, 28)
    out = cla(x)
    assert out.shape == x.shape, f"CLA altered tensor shape: {out.shape}"


def test_dmsl_block_stride1():
    block = DMSLBlock(in_channels=64, out_channels=64, stride=1, drop_path_rate=0.1)
    x = torch.randn(2, 64, 28, 28)
    out = block(x)
    assert out.shape == (2, 64, 28, 28)


def test_dmsl_block_stride2():
    block = DMSLBlock(in_channels=64, out_channels=128, stride=2, drop_path_rate=0.1)
    x = torch.randn(2, 64, 28, 28)
    out = block(x)
    assert out.shape == (2, 128, 14, 14)


def test_hlanet_forward_backward():
    net = HLANet(num_classes=23)
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    assert out.shape == (2, 23)

    loss = out.sum()
    loss.backward()
    # Check that stem gradients exist
    assert net.backbone.stem.branch3x3[0].weight.grad is not None


def test_hlanet_parameter_count():
    net = HLANet(num_classes=23)
    params = net.total_parameter_count()
    # Should be compact (~2.5M to 3.5M) to prevent overfitting on ~6k images
    assert 2_000_000 <= params <= 3_500_000, f"Expected 2-3.5M params, got {params}"


def test_build_model_dispatch():
    cfg = {"backbone": "hla_net", "dropout": 0.25}
    model = build_model(cfg, num_classes=23)
    assert isinstance(model, HLANet)
    assert model.backbone_name == "hla_net"


def test_gradcam_compatibility():
    model = HLANet(num_classes=23)
    cam = GradCAM(model)
    x = torch.randn(1, 3, 224, 224)
    heatmap = cam(x)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_ood_backbone_compatibility():
    model = HLANet(num_classes=23)
    x = torch.randn(2, 3, 224, 224)
    features = model.backbone(x)
    assert features.shape == (2, 512)
    logits = model.classifier(features)
    assert logits.shape == (2, 23)
