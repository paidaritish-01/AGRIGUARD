"""
Aerial analysis adapter and dataset contract for AgriScan.

Fulfills the aerial analysis requirement:
- Slices high-resolution drone imagery into overlapping tiles.
- Classifies visible canopy stress and abnormal void regions.
- Projects tile coordinates back to the global coordinate system and merges via NMS.
- Enforces clear contracts and states that RGB aerial imagery cannot distinguish
  pathological disease from water stress or nutrient deficiency without ground truthing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from agriscan.data.tiling import AerialTilingEngine, DetectionBox
from agriscan.data.transforms import get_eval_transforms
from agriscan.models.backbone import build_backbone


@dataclass
class AerialDatasetContract:
    """
    Contract specifying mandatory requirements for training/evaluating aerial models.
    """
    expected_image_types: List[str] = ("orthomosaic_geotiff", "nadir_drone_rgb_jpeg")
    required_metadata: List[str] = (
        "flight_altitude_m",
        "ground_sampling_distance_cm",
        "capture_date",
        "farm_id",
        "drone_flight_id",
    )
    supported_annotation_types: List[str] = ("bounding_boxes", "tile_level_stress_labels")
    licensing_requirement: str = (
        "Dataset must have a clear commercial or research license (e.g., CC-BY-4.0, MIT, "
        "or written field trial consent). Datasets with restricted or unknown terms cannot be used."
    )


class AgriAerialStressDetector(nn.Module):
    """
    Adapter and model for aerial crop stress analysis.
    Processes full drone images through overlapping tiles and merges stress detections.
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 3,
        classes: Optional[List[str]] = None,
        tile_size: int = 512,
        tile_overlap: float = 0.20,
        nms_iou_threshold: float = 0.40,
        min_anomaly_confidence: float = 0.65,
        model_version: str = "1.0.0",
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.classes = classes or [
            "healthy_canopy",
            "visible_crop_stress",
            "abnormal_canopy_void",
        ]
        self.num_classes = len(self.classes)
        self.label_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.idx_to_label = {i: name for i, name in enumerate(self.classes)}

        self.model, self.in_features = build_backbone(
            name=backbone_name,
            num_classes=self.num_classes,
            pretrained=False,
            freeze_backbone=False,
        )

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.min_anomaly_confidence = min_anomaly_confidence
        self.tiling_engine = AerialTilingEngine(
            tile_size=tile_size,
            overlap=tile_overlap,
            nms_iou_threshold=nms_iou_threshold,
        )
        self.eval_transforms = get_eval_transforms(image_size=224)
        self.device = torch.device(device)
        self.model_version = model_version
        self.is_trained = False
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @torch.no_grad()
    def analyze_aerial_image(
        self,
        image: Image.Image,
        confidence_threshold: Optional[float] = None,
    ) -> Tuple[List[DetectionBox], Dict[str, Any]]:
        """
        Slice aerial image into overlapping tiles, classify canopy condition,
        filter by confidence, and merge detections into original image coordinates.
        """
        threshold = confidence_threshold if confidence_threshold is not None else self.min_anomaly_confidence
        tiles = self.tiling_engine.slice_image(image)

        raw_detections: List[DetectionBox] = []
        tile_stats = {cls_name: 0 for cls_name in self.classes}

        self.eval()

        for tile in tiles:
            tile_tensor = self.eval_transforms(tile.image).unsqueeze(0).to(self.device)
            logits = self.forward(tile_tensor)
            probs = F.softmax(logits, dim=-1)[0]

            score, pred_idx = torch.max(probs, dim=-1)
            score_val = float(score.item())
            pred_label = self.idx_to_label[int(pred_idx.item())]

            tile_stats[pred_label] += 1

            # Only report anomalies / visible stress (not normal healthy canopy)
            if pred_label != "healthy_canopy" and score_val >= threshold:
                # Convert tile-local box to global coordinates
                raw_detections.append(DetectionBox(
                    x_min=float(tile.x_min),
                    y_min=float(tile.y_min),
                    x_max=float(tile.x_max),
                    y_max=float(tile.y_max),
                    confidence=score_val,
                    label=pred_label,
                ))

        merged_detections = self.tiling_engine.merge_detections(raw_detections)

        summary = {
            "total_tiles_analyzed": len(tiles),
            "tile_distribution": tile_stats,
            "anomalies_detected": len(merged_detections),
            "confidence_threshold": threshold,
            "is_trained_model": self.is_trained,
        }

        return merged_detections, summary

    def save_checkpoint(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.state_dict(),
            "classes": self.classes,
            "backbone_name": self.backbone_name,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "is_trained": True,
            "model_version": self.model_version,
        }, filepath)

    @classmethod
    def load_checkpoint(
        cls, filepath: str | Path, device: str | torch.device = "cpu"
    ) -> "AgriAerialStressDetector":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        ckpt = torch.load(filepath, map_location=device)
        instance = cls(
            backbone_name=ckpt.get("backbone_name", "resnet18"),
            classes=ckpt.get("classes"),
            tile_size=ckpt.get("tile_size", 512),
            tile_overlap=ckpt.get("tile_overlap", 0.20),
            model_version=ckpt.get("model_version", "1.0.0"),
            device=device,
        )
        instance.load_state_dict(ckpt["model_state_dict"])
        instance.is_trained = ckpt.get("is_trained", True)
        instance.eval()
        return instance
