"""
CLI evaluation script for AgriScan model checkpoints.

Evaluates on the held-out test set:
- Strict separation: Uses validation set for abstention threshold tuning, then applies to test set.
- Per-class precision, recall, F1, and support.
- Macro F1.
- Full confusion matrix.
- Breakdown by held-out farm / capture session to verify generalization.
- Expected Calibration Error (ECE) and reliability assessment.
"""

import argparse
import json
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
from agriscan.data.transforms import get_eval_transforms
from agriscan.models.classifier import AgriCloseUpClassifier
from agriscan.evaluation.evaluator import ModelEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate AgriScan checkpoint on test split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/close_up_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=None,
        help="Custom abstention threshold (if None, tunes on validation set)",
    )
    parser.add_argument(
        "--use-calibration",
        action="store_true",
        default=True,
        help="Apply post-hoc temperature scaling if available",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="checkpoints/evaluation_report.json",
        help="Path to save evaluation metrics JSON report",
    )

    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Error: Checkpoint file not found: {ckpt_path}")
        sys.exit(1)

    cfg_path = Path(args.config)
    config = AgriScanConfig.load_yaml(cfg_path) if cfg_path.exists() else AgriScanConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL CHECKPOINT: {ckpt_path}")
    print(f"Execution Device: {device}")
    print("=" * 80)

    # 1. Load trained model
    model = AgriCloseUpClassifier.load_checkpoint(ckpt_path, device=device)
    evaluator = ModelEvaluator(model, device=device)

    # 2. Re-create identical group splits
    data_dir = Path(config.data.dataset_dir)
    meta_file = Path(config.data.metadata_file) if config.data.metadata_file else None
    records = load_dataset_records(data_dir, metadata_file=meta_file)

    train_recs, val_recs, test_recs = split_records_by_group(
        records,
        group_key=config.data.split_group_column,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        seed=config.seed,
    )

    eval_transforms = get_eval_transforms(image_size=config.close_up.image_size)
    val_dataset = AgriCloseUpDataset(
        val_recs, dataset_dir=data_dir, label_to_idx=model.label_to_idx, transform=eval_transforms
    )
    test_dataset = AgriCloseUpDataset(
        test_recs, dataset_dir=data_dir, label_to_idx=model.label_to_idx, transform=eval_transforms
    )

    val_loader = DataLoader(val_dataset, batch_size=config.close_up.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.close_up.batch_size, shuffle=False)

    # 3. Determine abstention threshold
    if args.abstention_threshold is not None:
        chosen_threshold = args.abstention_threshold
        print(f"Using explicitly specified abstention threshold: {chosen_threshold:.2f}")
    else:
        print("Tuning abstention threshold on VALIDATION set (held-out test set untouched)...")
        chosen_threshold = evaluator.tune_abstention_threshold_on_val(
            val_evaluator=evaluator,
            val_loader=val_loader,
            min_required_accuracy=0.80,
        )
        print(f"Selected validation-tuned abstention threshold: {chosen_threshold:.2f}")

    # 4. Evaluate on held-out test split
    print(f"\nRunning evaluation on HELD-OUT TEST SET ({len(test_dataset)} samples)...")
    results = evaluator.evaluate(
        dataloader=test_loader,
        abstention_threshold=chosen_threshold,
        use_calibration=args.use_calibration,
    )

    # 5. Display Report
    print("\n" + "-" * 40 + " TEST EVALUATION RESULTS " + "-" * 40)
    print(f"Score Type:             {results['score_type']}")
    print(f"Total Test Samples:     {results['total_samples']}")
    print(f"Retained Predictions:   {results['retained_samples']}")
    print(f"Abstained Predictions:  {results['abstained_samples']} ({results['abstention_rate']*100:.1f}%)")
    print(f"Raw Accuracy:           {results['raw_accuracy']*100:.2f}%")
    print(f"Retained Accuracy:      {results['retained_accuracy']*100:.2f}%")
    print(f"Macro F1:               {results['macro_f1']:.4f}")
    print(f"Expected Calib. Error:  {results['calibration']['expected_calibration_error']:.4f}")

    print("\n--- PER-CLASS METRICS ---")
    print(f"{'Class':<26} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 75)
    for cls_name, m in results["per_class"].items():
        print(
            f"{cls_name:<26} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | "
            f"{m['f1']:<10.4f} | {m['support']:<8}"
        )

    print("\n--- PERFORMANCE BY HELD-OUT FARM ---")
    for farm, fm in results["farm_performance"].items():
        print(
            f"  {farm:<15}: Samples={fm['total_samples']:>2} | "
            f"Accuracy={fm['accuracy']*100:>5.1f}% | "
            f"Abstained={fm['abstained_count']:>2} ({fm['abstention_rate']*100:>4.1f}%)"
        )

    print("\n--- CONFUSION MATRIX ---")
    labels = results["confusion_matrix"]["labels"]
    print("Labels: " + ", ".join([f"[{i}] {name}" for i, name in enumerate(labels)]))
    for row in results["confusion_matrix"]["matrix"]:
        print("  " + "  ".join([f"{val:>4}" for val in row]))

    # Save JSON report
    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved full evaluation report to: {out_json_path.resolve()}")

    print("\nEvaluation completed successfully.\n")


if __name__ == "__main__":
    main()
