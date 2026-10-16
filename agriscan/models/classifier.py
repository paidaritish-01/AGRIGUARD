"""
Close-up crop disease classifier wrapper.

Encapsulates feature extraction, classification head, checkpoint persistence,
and post-hoc temperature scaling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from agriscan.models.backbone import build_backbone
from agriscan.models.calibration import TemperatureScaler


class AgriCloseUpClassifier(nn.Module):
    """
    Close-up crop pathology classifier for detailed leaf or fruit images.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
        label_to_idx: Optional[Dict[str, int]] = None,
        crop: str = "tomato",
        model_version: str = "1.0.0",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.crop = crop
        self.model_version = model_version

        self.model, self.in_features = build_backbone(
            name=backbone_name,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout=dropout,
        )

        self.label_to_idx = label_to_idx or {}
        self.idx_to_label = {idx: lbl for lbl, idx in self.label_to_idx.items()}
        self.temperature_scaler = TemperatureScaler()
        self.is_calibrated = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward pass returning unscaled logits."""
        return self.model(x)

    def forward_calibrated(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling applied to logits."""
        logits = self.forward(x)
        if self.is_calibrated:
            return self.temperature_scaler(logits)
        return logits

    @torch.no_grad()
    def predict_tensor(
        self, x: torch.Tensor, use_calibration: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Runs inference on preprocessed tensor.
        Returns:
            predicted_indices: [batch_size]
            scores: [batch_size] (between 0.0 and 1.0)
            score_type: "calibrated_probability" or "raw_softmax_confidence"
        """
        self.eval()
        logits = self.forward(x)

        if use_calibration and self.is_calibrated:
            scaled_logits = self.temperature_scaler(logits)
            probs = F.softmax(scaled_logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)
            score_type = "calibrated_probability"
        else:
            probs = F.softmax(logits, dim=-1)
            scores, preds = torch.max(probs, dim=-1)
            score_type = "raw_softmax_confidence"

        return preds, scores, score_type

    def save_checkpoint(
        self,
        filepath: str | Path,
        metrics: Optional[Dict[str, Any]] = None,
        epoch: int = 0,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save weights, architecture specification, and label mappings."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.state_dict(),
            "backbone_name": self.backbone_name,
            "num_classes": self.num_classes,
            "crop": self.crop,
            "model_version": self.model_version,
            "label_to_idx": self.label_to_idx,
            "idx_to_label": self.idx_to_label,
            "is_calibrated": self.is_calibrated,
            "temperature": float(self.temperature_scaler.temperature.item()),
            "metrics": metrics or {},
            "epoch": epoch,
            "config": config_dict or {},
        }
        torch.save(payload, filepath)

    @classmethod
    def load_checkpoint(
        cls,
        filepath: str | Path,
        device: str | torch.device = "cpu",
    ) -> "AgriCloseUpClassifier":
        """Load and instantiate model from saved checkpoint."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        checkpoint = torch.load(filepath, map_location=device)

        instance = cls(
            backbone_name=checkpoint.get("backbone_name", "resnet18"),
            num_classes=checkpoint.get("num_classes", len(checkpoint.get("label_to_idx", {}))),
            pretrained=False,
            freeze_backbone=False,
            label_to_idx=checkpoint.get("label_to_idx", {}),
            crop=checkpoint.get("crop", "unknown_crop"),
            model_version=checkpoint.get("model_version", "1.0.0"),
        )

        instance.load_state_dict(checkpoint["model_state_dict"])
        instance.is_calibrated = checkpoint.get("is_calibrated", False)
        if "temperature" in checkpoint:
            instance.temperature_scaler.temperature.data.fill_(checkpoint["temperature"])

        instance.to(device)
        instance.eval()
        return instance
