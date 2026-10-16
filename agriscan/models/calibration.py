"""
Confidence calibration and Expected Calibration Error (ECE) for AgriScan.

Ensures that model confidence scores are not misrepresented as calibrated
probabilities without explicit empirical calibration.

Includes:
- Temperature scaling (Guo et al., 2017) for post-hoc logit calibration.
- Expected Calibration Error (ECE) metric computation across confidence bins.
- Reliability diagram data generation.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """
    Learns a single scalar temperature T > 0 on validation set logits
    to calibrate output probabilities: calibrated_probs = softmax(logits / T).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Prevent division by zero or negative temperatures
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """
        Optimize temperature using NLL on validation logits.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def compute_ece(
    confidences: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute Expected Calibration Error (ECE) and bin details.
    
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.asarray(confidences)
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)

    accuracies = (predictions == ground_truth).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bin_data: List[Dict[str, float]] = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Samples within this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / len(confidences)) * bin_error

            bin_data.append({
                "bin_lower": round(bin_lower, 2),
                "bin_upper": round(bin_upper, 2),
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(bin_error, 4),
            })

    return {
        "expected_calibration_error": round(float(ece), 4),
        "n_bins": n_bins,
        "sample_count": len(confidences),
        "bins": bin_data,
    }
