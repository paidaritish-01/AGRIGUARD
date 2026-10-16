"""
Backbone factory for transfer learning in AgriScan.

Supports:
- Common torchvision architectures (ResNet18, MobileNetV3-Small, EfficientNet-B0).
- Freezing/unfreezing backbone feature extractors for multi-stage fine-tuning.
- Custom classification head with dropout and proper output dimension.
"""

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


SUPPORTED_BACKBONES = ["resnet18", "mobilenet_v3_small", "efficientnet_b0"]


def build_backbone(
    name: str = "resnet18",
    num_classes: int = 4,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float = 0.2,
) -> Tuple[nn.Module, int]:
    """
    Constructs a vision backbone with frozen/unfrozen feature extractor and custom head.
    Returns the initialized model and the in_features dimension of the head.
    """
    name_clean = name.lower().replace("-", "_")

    if name_clean == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        in_features = base_model.fc.in_features

        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        # Ensure new head is trainable
        for param in base_model.fc.parameters():
            param.requires_grad = True

        return base_model, in_features

    elif name_clean == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        in_features = base_model.classifier[0].in_features

        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )
        for param in base_model.classifier.parameters():
            param.requires_grad = True

        return base_model, in_features

    elif name_clean == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        in_features = base_model.classifier[1].in_features

        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        for param in base_model.classifier.parameters():
            param.requires_grad = True

        return base_model, in_features

    else:
        raise ValueError(
            f"Unsupported backbone '{name}'. Supported backbones are: {SUPPORTED_BACKBONES}"
        )


def set_backbone_trainable(model: nn.Module, trainable: bool = True) -> None:
    """
    Unfreeze or freeze all backbone feature layers for fine-tuning.
    Classification head layers remain trainable.
    """
    # For resnet:
    if hasattr(model, "fc"):
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = trainable
    # For mobilenet or efficientnet:
    elif hasattr(model, "classifier"):
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = trainable
