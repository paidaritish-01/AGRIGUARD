"""
Training pipeline for AgriScan close-up pathology classifier.

Features:
- Deterministic seed initialization for reproducibility.
- Class weighting in CrossEntropyLoss to address dataset imbalance.
- Two-phase training: feature extractor freeze -> optional deep fine-tuning.
- Early stopping tracking validation metrics (val_loss or val_macro_f1).
- Best model checkpointing with embedded metadata and label mappings.
- Post-training validation logit calibration using temperature scaling.
"""

import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from agriscan.config import AgriScanConfig
from agriscan.models.classifier import AgriCloseUpClassifier
from agriscan.models.backbone import set_backbone_trainable


def seed_everything(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ModelTrainer:
    """
    Orchestrates model training, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        config: AgriScanConfig,
        model: AgriCloseUpClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)

        # Loss function with optional class weighting
        if class_weights is not None and config.close_up.use_class_weights:
            weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate_val(self) -> Tuple[float, float, float, torch.Tensor, torch.Tensor]:
        """
        Evaluate on validation set.
        Returns:
            val_loss, accuracy, macro_f1, all_logits, all_labels
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []

        for images, labels, _ in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_logits.append(logits.cpu())

        total = len(all_targets)
        val_loss = total_loss / max(total, 1)
        val_acc = (np.array(all_preds) == np.array(all_targets)).mean() if total > 0 else 0.0
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))

        val_logits_tensor = torch.cat(all_logits, dim=0) if all_logits else torch.empty(0)
        val_targets_tensor = torch.tensor(all_targets, dtype=torch.long)

        return val_loss, float(val_acc), macro_f1, val_logits_tensor, val_targets_tensor

    def fit(self) -> Dict[str, Any]:
        """
        Execute training routine with frozen backbone first,
        followed by optional fine-tuning, early stopping, and checkpointing.
        """
        seed_everything(self.config.seed)
        cfg = self.config.close_up

        # Phase 1: Train classification head with frozen backbone
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        best_metric = -float("inf") if cfg.metric_for_best_checkpoint == "val_macro_f1" else float("inf")
        best_epoch = -1
        patience_counter = 0
        best_val_logits = None
        best_val_targets = None
        history: List[Dict[str, Any]] = []

        checkpoint_path = self.output_dir / "best_model.pth"

        print(f"Starting Phase 1 training ({cfg.epochs} epochs) on {self.device}...")

        for epoch in range(1, cfg.epochs + 1):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(optimizer)
            val_loss, val_acc, val_f1, val_logits, val_targets = self.evaluate_val()
            elapsed = time.time() - start_time

            current_metric = val_f1 if cfg.metric_for_best_checkpoint == "val_macro_f1" else -val_loss
            target_metric = val_f1 if cfg.metric_for_best_checkpoint == "val_macro_f1" else val_loss

            improved = (
                (val_f1 > best_metric) if cfg.metric_for_best_checkpoint == "val_macro_f1"
                else (val_loss < best_metric)
            )

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "val_macro_f1": round(val_f1, 4),
                "elapsed_sec": round(elapsed, 2),
            })

            print(
                f"Epoch {epoch:02d}/{cfg.epochs:02d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | "
                f"Time: {elapsed:.1f}s"
            )

            if improved:
                best_metric = target_metric
                best_epoch = epoch
                patience_counter = 0
                best_val_logits = val_logits
                best_val_targets = val_targets

                # Save best checkpoint
                self.model.save_checkpoint(
                    checkpoint_path,
                    metrics={"best_epoch": best_epoch, "val_loss": val_loss, "val_macro_f1": val_f1},
                    epoch=best_epoch,
                    config_dict=self.config.to_dict(),
                )
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.")
                    break

        # Optional Phase 2: Unfreeze backbone for fine-tuning
        if cfg.fine_tune_epochs > 0 and checkpoint_path.exists():
            print(f"\nStarting Phase 2: Unfreezing backbone for {cfg.fine_tune_epochs} fine-tuning epochs...")
            # Reload best model from Phase 1
            self.model = AgriCloseUpClassifier.load_checkpoint(checkpoint_path, device=self.device)
            set_backbone_trainable(self.model.model, trainable=True)

            ft_optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.fine_tune_lr,
                weight_decay=cfg.weight_decay,
            )

            for ft_epoch in range(1, cfg.fine_tune_epochs + 1):
                cur_epoch = cfg.epochs + ft_epoch
                train_loss, train_acc = self.train_epoch(ft_optimizer)
                val_loss, val_acc, val_f1, val_logits, val_targets = self.evaluate_val()

                improved = (
                    (val_f1 > best_metric) if cfg.metric_for_best_checkpoint == "val_macro_f1"
                    else (val_loss < best_metric)
                )

                if improved:
                    best_metric = val_f1 if cfg.metric_for_best_checkpoint == "val_macro_f1" else val_loss
                    best_epoch = cur_epoch
                    best_val_logits = val_logits
                    best_val_targets = val_targets
                    self.model.save_checkpoint(
                        checkpoint_path,
                        metrics={"best_epoch": best_epoch, "val_loss": val_loss, "val_macro_f1": val_f1},
                        epoch=best_epoch,
                        config_dict=self.config.to_dict(),
                    )
                    print(f"Fine-tune Epoch {ft_epoch}: Improved -> Best metric: {best_metric:.4f}")

        # Post-training Calibration: Fit temperature scaling on best validation logits
        calibrated_temp = 1.0
        if best_val_logits is not None and best_val_targets is not None and len(best_val_targets) > 0:
            print("\nFitting temperature scaling on validation logits...")
            calibrated_temp = self.model.temperature_scaler.fit(
                best_val_logits, best_val_targets
            )
            self.model.is_calibrated = True
            # Update checkpoint with calibration info
            self.model.save_checkpoint(
                checkpoint_path,
                metrics={"best_epoch": best_epoch, "metric": best_metric, "temperature": calibrated_temp},
                epoch=best_epoch,
                config_dict=self.config.to_dict(),
            )
            print(f"Calibration completed. Learned Temperature T = {calibrated_temp:.4f}")

        return {
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "metric_name": cfg.metric_for_best_checkpoint,
            "checkpoint_path": str(checkpoint_path),
            "calibrated_temperature": calibrated_temp,
            "history": history,
        }
