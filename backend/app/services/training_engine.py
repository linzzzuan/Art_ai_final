"""Training engine — full training loop with weighted CE, cosine LR, early stopping."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from app.models.cnn import EmotionCNN
from app.models.geo_encoder import GeoEncoder
from app.models.fusion import FusionClassifier
from app.utils.metrics import compute_metrics, compute_confusion_matrix

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when validation accuracy stops improving."""

    def __init__(self, patience: int = 8):
        self.patience = patience
        self.best_acc = 0.0
        self.counter = 0

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_class_weights(dataset, num_classes: int = 7) -> torch.Tensor:
    """Compute w_k = N / (K * N_k) for weighted cross-entropy."""
    counts = dataset.get_class_counts()
    N = sum(counts.values())
    K = num_classes
    weights = []
    for k in range(K):
        n_k = counts.get(k, 1)  # avoid division by zero
        weights.append(N / (K * n_k))
    return torch.FloatTensor(weights)


class TrainingEngine:
    """Orchestrates the complete training pipeline."""

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        epochs: int = 50,
        lr: float = 0.001,
        class_weights: torch.Tensor | None = None,
        patience: int = 8,
        task_name: str = "default",
        output_dir: str = "experiments",
        checkpoint_dir: str = "checkpoints",
    ):
        self.device = torch.device(device)
        self.epochs = epochs
        self.task_name = task_name

        # Directories
        self.exp_dir = Path(output_dir) / task_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Models
        self.cnn = EmotionCNN().to(self.device)
        self.geo_encoder = GeoEncoder().to(self.device)
        self.classifier = FusionClassifier().to(self.device)

        # All parameters for optimizer
        params = (
            list(self.cnn.parameters())
            + list(self.geo_encoder.parameters())
            + list(self.classifier.parameters())
        )
        self.optimizer = Adam(params, lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=15)

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Early stopping
        self.early_stopping = EarlyStopping(patience=patience)

        # History
        self.history: list[dict] = []
        self.best_val_acc = 0.0

    def _set_train(self):
        self.cnn.train()
        self.geo_encoder.train()
        self.classifier.train()

    def _set_eval(self):
        self.cnn.eval()
        self.geo_encoder.eval()
        self.classifier.eval()

    def _train_one_epoch(self) -> tuple[float, float]:
        """Run one training epoch. Returns (avg_loss, accuracy)."""
        self._set_train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, geos, labels in self.train_loader:
            images = images.to(self.device)
            geos = geos.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            pixel_feat = self.cnn(images)
            geo_feat = self.geo_encoder(geos)
            logits = self.classifier(pixel_feat, geo_feat)

            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc

    @torch.no_grad()
    def _validate(self) -> tuple[float, float, list[int], list[int]]:
        """Run validation. Returns (avg_loss, accuracy, y_true, y_pred)."""
        self._set_eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_true: list[int] = []
        all_pred: list[int] = []

        for images, geos, labels in self.val_loader:
            images = images.to(self.device)
            geos = geos.to(self.device)
            labels = labels.to(self.device)

            pixel_feat = self.cnn(images)
            geo_feat = self.geo_encoder(geos)
            logits = self.classifier(pixel_feat, geo_feat)

            loss = self.criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_true.extend(labels.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        return avg_loss, acc, all_true, all_pred

    def _save_checkpoint(self, path: Path):
        torch.save(
            {
                "cnn": self.cnn.state_dict(),
                "geo_encoder": self.geo_encoder.state_dict(),
                "classifier": self.classifier.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Resume from a previous checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.cnn.load_state_dict(ckpt["cnn"])
        self.geo_encoder.load_state_dict(ckpt["geo_encoder"])
        self.classifier.load_state_dict(ckpt["classifier"])
        logger.info("Resumed from checkpoint: %s", path)

    def train(self) -> dict:
        """Run the full training loop. Returns final performance dict."""
        logger.info(
            "Training started — epochs=%d, device=%s, task=%s",
            self.epochs, self.device, self.task_name,
        )
        start_time = time.time()
        y_true_final: list[int] = []
        y_pred_final: list[int] = []

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, y_true, y_pred = self._validate()
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            self.history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "lr": round(lr, 6),
            })

            logger.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.4f "
                "val_loss=%.4f val_acc=%.4f lr=%.6f (%.1fs)",
                epoch, self.epochs,
                train_loss, train_acc, val_loss, val_acc, lr, epoch_time,
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_path = self.ckpt_dir / "best_model.pth"
                self._save_checkpoint(best_path)
                logger.info("Best model saved (val_acc=%.4f) -> %s", val_acc, best_path)
                y_true_final = y_true
                y_pred_final = y_pred

            # Early stopping
            if self.early_stopping(val_acc):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        total_time = time.time() - start_time
        logger.info("Training finished in %.1f seconds", total_time)

        # Save experiment outputs
        self._save_metrics()
        performance = self._save_performance(y_true_final, y_pred_final)
        self._save_confusion_matrix(y_true_final, y_pred_final)
        self._save_config(total_time)

        return performance

    def _save_metrics(self):
        path = self.exp_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        logger.info("Metrics saved -> %s", path)

    def _save_performance(self, y_true: list[int], y_pred: list[int]) -> dict:
        if not y_true:
            performance = {"metrics": {}, "macro_avg": {}, "weighted_avg": {}}
        else:
            performance = compute_metrics(y_true, y_pred)
        path = self.exp_dir / "performance.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(performance, f, indent=2)
        logger.info("Performance saved -> %s", path)
        return performance

    def _save_confusion_matrix(self, y_true: list[int], y_pred: list[int]):
        if not y_true:
            cm = {"labels": [], "matrix": []}
        else:
            cm = compute_confusion_matrix(y_true, y_pred)
        path = self.exp_dir / "confusion_matrix.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cm, f, indent=2)
        logger.info("Confusion matrix saved -> %s", path)

    def _save_config(self, total_time: float):
        cfg = {
            "task_name": self.task_name,
            "epochs_completed": len(self.history),
            "epochs_total": self.epochs,
            "best_val_acc": round(self.best_val_acc, 4),
            "total_time_seconds": round(total_time, 1),
            "device": str(self.device),
        }
        path = self.exp_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        logger.info("Config saved -> %s", path)
