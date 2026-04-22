"""Dataset information routes."""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends

from app.dependencies import get_config
from app.utils.image import CLASS_DIRS, EMOTION_CLASSES

logger = logging.getLogger(__name__)
router = APIRouter()


def _count_images(dataset_path: str | None) -> list[int]:
    """Count images per class in the dataset directory."""
    counts = [0] * 7
    if not dataset_path:
        return counts
    root = Path(dataset_path)
    for split in ("train", "val"):
        for i, cls_dir in enumerate(CLASS_DIRS):
            cls_path = root / split / cls_dir
            if cls_path.exists():
                counts[i] += sum(
                    1 for f in cls_path.iterdir()
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                )
    return counts


@router.get("/api/v1/dataset/info")
async def get_dataset_info(config: dict = Depends(get_config)):
    dataset_cfg = config["dataset"]
    path = dataset_cfg.get("affectnet_path") or dataset_cfg.get("mock_data_path")
    counts = _count_images(path)
    total = sum(counts)
    return {
        "name": "AffectNet-7",
        "path": path,
        "total_samples": total,
        "image_size": "224x224",
        "num_classes": 7,
        "classes": EMOTION_CLASSES,
    }


@router.get("/api/v1/dataset/class-distribution")
async def get_class_distribution(config: dict = Depends(get_config)):
    dataset_cfg = config["dataset"]
    path = dataset_cfg.get("affectnet_path") or dataset_cfg.get("mock_data_path")
    counts = _count_images(path)
    total = max(sum(counts), 1)
    return {
        "classes": [
            {
                "name": EMOTION_CLASSES[i],
                "count": counts[i],
                "percentage": round(counts[i] / total * 100, 1),
            }
            for i in range(7)
        ]
    }
