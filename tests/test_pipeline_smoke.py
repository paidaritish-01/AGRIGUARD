"""
End-to-end pipeline smoke test for AgriScan.

Executes complete lifecycle in isolated temp workspace:
1. Synthetic fixture generation.
2. Dataset validation and zero-leakage split verification.
3. Lightweight 1-epoch model training with transfer learning.
4. Checkpoint saving and label mapping verification.
5. Held-out test evaluation (precision, recall, macro F1, confusion matrix, ECE).
6. Model loading and single image inference execution.
"""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from agriscan.config import AgriScanConfig
from agriscan.data.dataset import (
    load_dataset_records,
    split_records_by_group,
    AgriCloseUpDataset,
)
from agriscan.data.synthetic import generate_synthetic_dataset
from agriscan.data.transforms import get_training_transforms, get_eval_transforms
from agriscan.data.validator import DatasetValidator
from agriscan.evaluation.evaluator import ModelEvaluator
from agriscan.inference.engine import AgriScanInferenceEngine
from agriscan.models.classifier import AgriCloseUpClassifier
from agriscan.training.trainer import ModelTrainer, seed_everything


def test_end_to_end_pipeline_smoke(tmp_path):
    seed_everything(42)

    # 1. Generate mini synthetic dataset
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = generate_synthetic_dataset(
        base_dir=data_dir,
        num_samples_per_class=3,  # minimal for fast smoke test
        num_farms=3,
    )
    assert summary["num_records"] == 12

    # 2. Validate dataset
    close_up_dir = Path(summary["close_up_dir"])
    meta_file = close_up_dir / "metadata.csv"

    validator = DatasetValidator(dataset_dir=close_up_dir, metadata_file=meta_file)
    records = load_dataset_records(close_up_dir, metadata_file=meta_file)
    assert len(records) == 12

    val_res = validator.validate_metadata_records(records)
    assert val_res["valid_records"] == 12
    assert len(val_res["corrupt_images"]) == 0
    assert len(val_res["missing_labels"]) == 0

    # 3. Verify zero leakage split
    train_recs, val_recs, test_recs = split_records_by_group(
        records, group_key="farm_id", train_ratio=0.50, val_ratio=0.25, test_ratio=0.25, seed=42
    )
    assert len(train_recs) > 0
    assert len(val_recs) > 0
    assert len(test_recs) > 0

    leakage_check = validator.validate_split_leakage(train_recs, val_recs, test_recs, group_key="farm_id")
    assert leakage_check["leakage_detected"] is False

    # 4. Create datasets and loaders
    train_transforms = get_training_transforms(image_size=128)
    eval_transforms = get_eval_transforms(image_size=128)

    train_ds = AgriCloseUpDataset(train_recs, dataset_dir=close_up_dir, transform=train_transforms)
    val_ds = AgriCloseUpDataset(val_recs, dataset_dir=close_up_dir, label_to_idx=train_ds.label_to_idx, transform=eval_transforms)
    test_ds = AgriCloseUpDataset(test_recs, dataset_dir=close_up_dir, label_to_idx=train_ds.label_to_idx, transform=eval_transforms)

    # Save label mapping
    mapping_file = output_dir / "label_mapping.json"
    train_ds.save_label_mapping(mapping_file)
    assert mapping_file.exists()

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # 5. Build and train model for 1 epoch
    config = AgriScanConfig(
        output_dir=str(output_dir),
        seed=42,
    )
    config.close_up.epochs = 1
    config.close_up.fine_tune_epochs = 0
    config.close_up.batch_size = 4
    config.close_up.image_size = 128

    model = AgriCloseUpClassifier(
        backbone_name="resnet18",
        num_classes=len(train_ds.label_to_idx),
        pretrained=False,  # faster offline unit test
        freeze_backbone=True,
        label_to_idx=train_ds.label_to_idx,
    )

    trainer = ModelTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device("cpu"),
        class_weights=train_ds.get_class_weights(),
    )

    train_results = trainer.fit()
    ckpt_path = Path(train_results["checkpoint_path"])
    assert ckpt_path.exists()

    # 6. Evaluate on held-out test split
    loaded_model = AgriCloseUpClassifier.load_checkpoint(ckpt_path, device="cpu")
    evaluator = ModelEvaluator(loaded_model, device="cpu")
    test_metrics = evaluator.evaluate(test_loader, abstention_threshold=0.50)

    assert "macro_f1" in test_metrics
    assert "per_class" in test_metrics
    assert "confusion_matrix" in test_metrics
    assert "calibration" in test_metrics

    # 7. Test inference engine on test image
    inference_engine = AgriScanInferenceEngine(
        close_up_checkpoint=ckpt_path,
        default_abstention_threshold=0.50,
        device="cpu",
    )
    assert inference_engine.is_demo_mode is False

    from PIL import Image
    sample_img_path = close_up_dir / train_recs[0]["image_path"]
    with Image.open(sample_img_path) as sample_img:
        pred_result = inference_engine.run_close_up_inference(
            sample_img, crop="tomato"
        )

    assert pred_result["status"] in {"success", "uncertain"}
    assert pred_result["model_version"] == "1.0.0"
    assert pred_result["is_demo_mode"] is False
    assert len(pred_result["relevant_limitations"]) > 0
