"""
Aerial image tiling with configurable overlap and detection merging.

Enables processing of high-resolution drone orthophotos / canopy imagery by slicing into
manageable sub-tiles, preserving global pixel coordinates, and merging overlapping detections
using Non-Maximum Suppression (NMS).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import torch


@dataclass
class Tile:
    tile_id: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    image: Image.Image

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass
class DetectionBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    label: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": [round(self.x_min, 1), round(self.y_min, 1), round(self.x_max, 1), round(self.y_max, 1)],
            "confidence": round(self.confidence, 4),
            "label": self.label,
        }


class AerialTilingEngine:
    """
    Slices large aerial orthophotos into overlapping tiles and merges detections back
    into global image coordinates.
    """

    def __init__(self, tile_size: int = 512, overlap: float = 0.20, nms_iou_threshold: float = 0.40):
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"Overlap must be in [0.0, 1.0), got {overlap}")
        if tile_size <= 0:
            raise ValueError(f"Tile size must be > 0, got {tile_size}")

        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1.0 - overlap))
        if self.stride <= 0:
            self.stride = 1
        self.nms_iou_threshold = nms_iou_threshold

    def slice_image(self, image: Image.Image) -> List[Tile]:
        """
        Slice a PIL Image into overlapping tiles preserving absolute bounding box coordinates.
        """
        img_w, img_h = image.size
        tiles: List[Tile] = []

        if img_w <= self.tile_size and img_h <= self.tile_size:
            # Image is already smaller than or equal to tile size
            return [Tile(tile_id="tile_0_0", x_min=0, y_min=0, x_max=img_w, y_max=img_h, image=image)]

        y = 0
        row = 0
        while y < img_h:
            y_end = min(y + self.tile_size, img_h)
            y_start = max(0, y_end - self.tile_size)

            x = 0
            col = 0
            while x < img_w:
                x_end = min(x + self.tile_size, img_w)
                x_start = max(0, x_end - self.tile_size)

                tile_crop = image.crop((x_start, y_start, x_end, y_end))
                tile_obj = Tile(
                    tile_id=f"tile_{row}_{col}",
                    x_min=x_start,
                    y_min=y_start,
                    x_max=x_end,
                    y_max=y_end,
                    image=tile_crop,
                )
                tiles.append(tile_obj)

                if x_end >= img_w:
                    break
                x += self.stride
                col += 1

            if y_end >= img_h:
                break
            y += self.stride
            row += 1

        return tiles

    @staticmethod
    def compute_iou(box_a: DetectionBox, box_b: DetectionBox) -> float:
        """Compute Intersection over Union between two detection bounding boxes."""
        xA = max(box_a.x_min, box_b.x_min)
        yA = max(box_a.y_min, box_b.y_min)
        xB = min(box_a.x_max, box_b.x_max)
        yB = min(box_a.y_max, box_b.y_max)

        inter_w = max(0.0, xB - xA)
        inter_h = max(0.0, yB - yA)
        inter_area = inter_w * inter_h

        area_a = (box_a.x_max - box_a.x_min) * (box_a.y_max - box_a.y_min)
        area_b = (box_b.x_max - box_b.x_min) * (box_b.y_max - box_b.y_min)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def merge_detections(self, detections: List[DetectionBox]) -> List[DetectionBox]:
        """
        Merge overlapping detections across tiles using class-aware Non-Maximum Suppression (NMS).
        """
        if not detections:
            return []

        # Group by label so NMS does not suppress different classes
        by_class: Dict[str, List[DetectionBox]] = {}
        for d in detections:
            by_class.setdefault(d.label, []).append(d)

        merged: List[DetectionBox] = []

        for label, class_dets in by_class.items():
            # Sort descending by confidence
            sorted_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
            keep: List[DetectionBox] = []

            while sorted_dets:
                best = sorted_dets.pop(0)
                keep.append(best)

                remaining: List[DetectionBox] = []
                for other in sorted_dets:
                    iou = self.compute_iou(best, other)
                    if iou < self.nms_iou_threshold:
                        remaining.append(other)
                sorted_dets = remaining

            merged.extend(keep)

        # Sort all results descending by confidence
        return sorted(merged, key=lambda d: d.confidence, reverse=True)
