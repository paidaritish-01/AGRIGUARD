"""
Configuration module for AgriScan.

Supports loading, validating, and dumping configurations for close-up classification
and aerial stress detection pipelines.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class CloseUpModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = True
    fine_tune_epochs: int = 5
    dropout: float = 0.2
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    early_stopping_patience: int = 3
    metric_for_best_checkpoint: str = "val_macro_f1"  # "val_macro_f1" or "val_loss"
    use_class_weights: bool = True
    abstention_threshold: float = 0.70
    image_size: int = 224


@dataclass
class AerialModelConfig:
    backbone: str = "resnet18"
    tile_size: int = 512
    tile_overlap: float = 0.20
    min_anomaly_confidence: float = 0.65
    nms_iou_threshold: float = 0.40
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 1e-3
    stress_classes: List[str] = field(
        default_factory=lambda: [
            "healthy_canopy",
            "visible_crop_stress",
            "abnormal_canopy_void",
        ]
    )


@dataclass
class DataConfig:
    dataset_dir: str = "data/close_up"
    metadata_file: Optional[str] = None
    split_group_column: str = "farm_id"  # farm_id or session_id to prevent data leakage
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    num_workers: int = 0
    max_duplicate_check_samples: int = 10000


MANGO_CLASSES = [
    "healthy",
    "anthracnose",
    "powdery_mildew",
    "turning_brown",
    "gall_midge",
]


@dataclass
class AgriScanConfig:
    analysis_mode: str = "close_up"  # "close_up" or "aerial"
    supported_crops: List[str] = field(
        default_factory=lambda: ["mango", "tomato", "corn", "wheat", "apple", "general_crop"]
    )
    primary_crop: str = "mango"
    seed: int = 42
    device: str = "auto"  # "cuda", "cpu", or "auto"
    output_dir: str = "checkpoints"
    model_version: str = "1.0.0"
    close_up: CloseUpModelConfig = field(default_factory=CloseUpModelConfig)
    aerial: AerialModelConfig = field(default_factory=AerialModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgriScanConfig":
        close_up_data = data.pop("close_up", {})
        aerial_data = data.pop("aerial", {})
        data_data = data.pop("data", {})

        return cls(
            close_up=CloseUpModelConfig(**close_up_data),
            aerial=AerialModelConfig(**aerial_data),
            data=DataConfig(**data_data),
            **data,
        )

    @classmethod
    def load_yaml(cls, path: str | Path) -> "AgriScanConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
