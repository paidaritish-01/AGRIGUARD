"""
Evaluation pipeline for AgriScan.

Computes:
- Per-class Precision, Recall, and F1.
- Macro F1.
- Confusion Matrix.
- Abstention metrics and validation-based threshold tuning.
- Expected Calibration Error (ECE) and calibration reliability.
- Performance split by held-out farm/session.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from agriscan.models.classifier import AgriCloseUpClassifier
from agriscan.models.calibration import compute_ece


class ModelEvaluator:
    """
    Evaluates agricultural classification models with support for abstention,
    calibration analysis, and held-out farm evaluation.
    """

    def __init__(
        self,
        model: AgriCloseUpClassifier,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        abstention_threshold: float = 0.70,
        use_calibration: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate dataset and return comprehensive diagnostic metrics.
        """
        all_targets: List[int] = []
        all_preds: List[int] = []
        all_scores: List[float] = []
        all_farms: List[str] = []
        abstained_count = 0
        retained_preds: List[int] = []
        retained_targets: List[int] = []

        score_type_used = "calibrated_probability" if (use_calibration and self.model.is_calibrated) else "raw_softmax_confidence"

        for images, labels, meta in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds, scores, _ = self.model.predict_tensor(images, use_calibration=use_calibration)

            preds_np = preds.cpu().numpy()
            scores_np = scores.cpu().numpy()
            targets_np = labels.cpu().numpy()
            farms = meta.get("farm_id", ["unknown"] * len(labels))

            for p, s, t, f in zip(preds_np, scores_np, targets_np, farms):
                all_targets.append(int(t))
                all_preds.append(int(p))
                all_scores.append(float(s))
                all_farms.append(str(f))

                if float(s) < abstention_threshold:
                    abstained_count += 1
                else:
                    retained_preds.append(int(p))
                    retained_targets.append(int(t))

        total_samples = len(all_targets)
        abstention_rate = abstained_count / max(total_samples, 1)

        # Labels mapping
        idx_to_label = self.model.idx_to_label
        unique_classes = sorted(list(idx_to_label.keys()))
        class_names = [idx_to_label.get(i, f"class_{i}") for i in unique_classes]

        # Overall and retained accuracy
        raw_accuracy = (np.array(all_preds) == np.array(all_targets)).mean() if total_samples > 0 else 0.0
        retained_accuracy = (
            (np.array(retained_preds) == np.array(retained_targets)).mean()
            if retained_preds else 0.0
        )

        # Per-class metrics on retained (actionable) predictions
        eval_preds = retained_preds if retained_preds else all_preds
        eval_targets = retained_targets if retained_targets else all_targets

        precision, recall, f1, support = precision_recall_fscore_support(
            eval_targets, eval_preds, labels=unique_classes, zero_division=0
        )

        per_class_metrics = {}
        for idx, name in enumerate(class_names):
            per_class_metrics[name] = {
                "precision": round(float(precision[idx]), 4),
                "recall": round(float(recall[idx]), 4),
                "f1": round(float(f1[idx]), 4),
                "support": int(support[idx]),
            }

        macro_f1 = float(np.mean(f1))

        # Confusion Matrix
        cm = confusion_matrix(eval_targets, eval_preds, labels=unique_classes)
        confusion_dict = {
            "matrix": cm.tolist(),
            "labels": class_names,
        }

        # Held-out farm breakdown
        farm_metrics: Dict[str, Dict[str, Any]] = {}
        unique_farms = sorted(list(set(all_farms)))
        for farm in unique_farms:
            indices = [i for i, f in enumerate(all_farms) if f == farm]
            farm_targets = [all_targets[i] for i in indices]
            farm_preds = [all_preds[i] for i in indices]
            farm_scores = [all_scores[i] for i in indices]
            farm_acc = (np.array(farm_preds) == np.array(farm_targets)).mean() if farm_targets else 0.0
            farm_abstained = sum(1 for s in farm_scores if s < abstention_threshold)

            farm_metrics[farm] = {
                "total_samples": len(indices),
                "accuracy": round(float(farm_acc), 4),
                "abstained_count": farm_abstained,
                "abstention_rate": round(farm_abstained / max(len(indices), 1), 4),
            }

        # Calibration evaluation
        calibration_results = compute_ece(
            confidences=np.array(all_scores),
            predictions=np.array(all_preds),
            ground_truth=np.array(all_targets),
            n_bins=10,
        )

        return {
            "total_samples": total_samples,
            "retained_samples": len(retained_preds),
            "abstained_samples": abstained_count,
            "abstention_rate": round(abstention_rate, 4),
            "abstention_threshold": abstention_threshold,
            "score_type": score_type_used,
            "raw_accuracy": round(float(raw_accuracy), 4),
            "retained_accuracy": round(float(retained_accuracy), 4),
            "macro_f1": round(macro_f1, 4),
            "per_class": per_class_metrics,
            "confusion_matrix": confusion_dict,
            "farm_performance": farm_metrics,
            "calibration": calibration_results,
            "limitations_notice": (
                "Confidence thresholds help filter low-confidence in-distribution samples, "
                "but cannot reliably identify every unsupported crop or out-of-distribution image. "
                "Raw scores must not be interpreted as calibrated posterior probabilities unless calibrated."
            ),
        }

    @staticmethod
    def tune_abstention_threshold_on_val(
        val_evaluator: "ModelEvaluator",
        val_loader: DataLoader,
        min_required_accuracy: float = 0.85,
        candidate_thresholds: Optional[List[float]] = None,
    ) -> float:
        """
        Select an abstention threshold on validation data (NEVER on test data)
        that satisfies min_required_accuracy while maximizing retained sample count.
        """
        thresholds = candidate_thresholds or [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        best_threshold = 0.70

        for thresh in thresholds:
            res = val_evaluator.evaluate(val_loader, abstention_threshold=thresh)
            if res["retained_accuracy"] >= min_required_accuracy:
                best_threshold = thresh
                break

        return best_threshold
