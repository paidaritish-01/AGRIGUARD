"""
CLI training script for AgriScan close-up pathology classifier.

Executes:
- Group-split generation by farm/session.
- Class-weight computation for imbalanced categories.
- Two-phase transfer learning (frozen backbone -> optional deep fine-tuning).
- Early stopping and validation checkpointing.
- Post-training validation temperature calibration.
- Label mapping persistence alongside checkpoint.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from agriscan.config import AgriScanConfig
from agriscan.data.dataset import (
    load_dataset_records,
    split_records_by_group,
    AgriCloseUpDataset,
)
from agriscan.data.transforms import get_training_transforms, get_eval_transforms
from agriscan.models.classifier import AgriCloseUpClassifier
from agriscan.training.trainer import ModelTrainer, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train AgriScan disease classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/close_up_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device ('cpu', 'cuda', 'auto')",
    )

    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Error: Config file not found: {cfg_path}")
        sys.exit(1)

    config = AgriScanConfig.load_yaml(cfg_path)
    if args.epochs is not None:
        config.close_up.epochs = args.epochs
    if args.batch_size is not None:
        config.close_up.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    seed_everything(config.seed)

    device_str = (
        "cuda"
        if (config.device != "cpu" and torch.cuda.is_available())
        else "cpu"
    )
    device = torch.device(device_str)

    print("\n" + "=" * 80)
    print(f"AGRISCAN MODEL TRAINING ({config.analysis_mode.upper()} MODE)")
    print(f"Target Crop: {config.primary_crop} | Device: {device} | Backbone: {config.close_up.backbone}")
    print("=" * 80)

    # 1. Load dataset records
    data_dir = Path(config.data.dataset_dir)
    meta_file = Path(config.data.metadata_file) if config.data.metadata_file else None

    if not data_dir.exists():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        print("Run 'python scripts/prepare_synthetic_data.py' first to generate a test fixture.")
        sys.exit(1)

    records = load_dataset_records(data_dir, metadata_file=meta_file)
    if not records:
        print("Error: No training records found.")
        sys.exit(1)

    # 2. Split records by farm/session group to prevent data leakage
    train_recs, val_recs, test_recs = split_records_by_group(
        records,
        group_key=config.data.split_group_column,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.seed,
    )

    print(
        f"Dataset Split (by {config.data.split_group_column}): "
        f"Train={len(train_recs)} | Val={len(val_recs)} | Test={len(test_recs)}"
    )

    # 3. Create datasets and transforms
    train_transforms = get_training_transforms(image_size=config.close_up.image_size)
    eval_transforms = get_eval_transforms(image_size=config.close_up.image_size)

    train_dataset = AgriCloseUpDataset(
        train_recs, dataset_dir=data_dir, transform=train_transforms
    )
    # Share label mapping from train across val and test
    val_dataset = AgriCloseUpDataset(
        val_recs,
        dataset_dir=data_dir,
        label_to_idx=train_dataset.label_to_idx,
        transform=eval_transforms,
    )

    # Save label mapping to output dir
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / "label_mapping.json"
    train_dataset.save_label_mapping(mapping_path)
    print(f"Saved label mapping ({len(train_dataset.label_to_idx)} classes) to {mapping_path}")

    # Compute class weights for imbalanced classes
    class_weights = train_dataset.get_class_weights()
    print(f"Computed class weights: {class_weights.numpy().round(3).tolist()}")

    # 4. DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.close_up.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.close_up.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # 5. Initialize model
    model = AgriCloseUpClassifier(
        backbone_name=config.close_up.backbone,
        num_classes=len(train_dataset.label_to_idx),
        pretrained=config.close_up.pretrained,
        freeze_backbone=config.close_up.freeze_backbone,
        dropout=config.close_up.dropout,
        label_to_idx=train_dataset.label_to_idx,
        crop=config.primary_crop,
        model_version=config.model_version,
    )

    # 6. Trainer
    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        class_weights=class_weights,
    )

    # 7. Execute training
    results = trainer.fit()

    print("\n" + "=" * 80)
    print("TRAINING RUN COMPLETE")
    print(f"Best Epoch: {results['best_epoch']} | Best {results['metric_name']}: {results['best_metric']:.4f}")
    print(f"Model Checkpoint: {results['checkpoint_path']}")
    print(f"Learned Calibration Temperature: {results['calibrated_temperature']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
