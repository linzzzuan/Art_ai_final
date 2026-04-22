"""Training CLI entry point.

Usage:
    # Local mock validation
    python train.py --use-mock --epochs 2

    # Cloud full training
    python train.py --dataset-path /mnt/affectnet/affectnet7 --epochs 50 --batch-size 64 --task-name v1

    # Resume from checkpoint
    python train.py --dataset-path /mnt/affectnet/affectnet7 --resume checkpoints/best_model.pth --task-name v1_resume
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Ensure the backend directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from app.config import get_settings
from app.core.logging import setup_logging
from app.utils.image import AffectNetDataset, MockDataset
from app.utils.data_augment import get_train_transform, get_val_transform
from app.services.training_engine import TrainingEngine, compute_class_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Detection Model Training")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to AffectNet-7 dataset root")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--task-name", type=str, default=None, help="Experiment task name")
    parser.add_argument("--use-mock", action="store_true", help="Use mock dataset for local validation")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_settings()
    setup_logging(cfg)

    logger = logging.getLogger("train")

    # Resolve parameters: CLI args > config.yaml
    device = cfg["model"]["device"]
    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    use_mock = args.use_mock or cfg["training"].get("use_mock", False)
    task_name = args.task_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("  Profile:    %s", cfg["_profile"])
    logger.info("  Device:     %s", device)
    logger.info("  Epochs:     %d", epochs)
    logger.info("  Batch size: %d", batch_size)
    logger.info("  Use mock:   %s", use_mock)
    logger.info("  Task name:  %s", task_name)
    logger.info("  LR:         %f", args.lr)
    logger.info("=" * 60)

    # Build datasets
    if use_mock:
        dataset_path = cfg["dataset"].get("mock_data_path", "datasets/mock")
        logger.info("Using mock dataset at: %s", dataset_path)
        if Path(dataset_path).exists() and (Path(dataset_path) / "train").exists():
            train_ds = AffectNetDataset(dataset_path, split="train", transform=get_train_transform())
            val_ds = AffectNetDataset(dataset_path, split="val", transform=get_val_transform())
        else:
            logger.info("Mock directory not found, using random MockDataset")
            train_ds = MockDataset(num_samples=100)
            val_ds = MockDataset(num_samples=20)
    else:
        dataset_path = args.dataset_path or cfg["dataset"].get("affectnet_path")
        if not dataset_path or not Path(dataset_path).exists():
            logger.error("Dataset path does not exist: %s", dataset_path)
            sys.exit(1)
        logger.info("Using AffectNet dataset at: %s", dataset_path)
        train_ds = AffectNetDataset(dataset_path, split="train", transform=get_train_transform())
        val_ds = AffectNetDataset(dataset_path, split="val", transform=get_val_transform())

    logger.info("Train samples: %d, Val samples: %d", len(train_ds), len(val_ds))

    # Class weights
    class_weights = None
    if hasattr(train_ds, "get_class_counts"):
        class_weights = compute_class_weights(train_ds)
        logger.info("Class weights: %s", class_weights.tolist())

    # DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
    )

    # Training engine
    engine = TrainingEngine(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=args.lr,
        class_weights=class_weights,
        patience=8,
        task_name=task_name,
        output_dir="experiments",
        checkpoint_dir="checkpoints",
    )

    # Resume if requested
    if args.resume:
        engine.load_checkpoint(args.resume)

    # Run training
    performance = engine.train()

    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("  Best val acc: %.4f", engine.best_val_acc)
    if performance.get("macro_avg"):
        logger.info("  Macro F1:     %.4f", performance["macro_avg"].get("f1", 0))
        logger.info("  Weighted F1:  %.4f", performance["weighted_avg"].get("f1", 0))
    logger.info("  Outputs:      experiments/%s/", task_name)
    logger.info("  Best model:   checkpoints/best_model.pth")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
