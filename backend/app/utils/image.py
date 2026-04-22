"""Image processing utilities: datasets, preprocessing, and base64 helpers."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Project naming convention: idx_name
CLASS_DIRS = ["0_angry", "1_disgust", "2_fear", "3_happy", "4_sad", "5_surprise", "6_neutral"]

# AffectNet standard label order → project class index mapping
# AffectNet: 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Anger
# Project:   0=angry,   1=disgust, 2=fear, 3=happy,   4=sad, 5=surprise, 6=neutral
AFFECTNET_TO_PROJECT = {
    0: 6,  # Neutral  → neutral
    1: 3,  # Happy    → happy
    2: 4,  # Sad      → sad
    3: 5,  # Surprise → surprise
    4: 2,  # Fear     → fear
    5: 1,  # Disgust  → disgust
    6: 0,  # Anger    → angry
}
# Contempt (folder 7) is skipped for 7-class classification


def get_inference_transform():
    """Transform for inference / validation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64 string (with or without data URI prefix) into a PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def image_to_tensor(image: Image.Image, transform=None) -> torch.Tensor:
    """Convert a PIL Image to a batched tensor (1, 3, 224, 224)."""
    if transform is None:
        transform = get_inference_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)


class AffectNetDataset(Dataset):
    """AffectNet-7 dataset loader.

    Supports two directory formats (auto-detected):

    Format A — project naming:
        root/train/0_angry/  1_disgust/  ...  6_neutral/
    Format B — Kaggle / AffectNet standard (numeric folders):
        root/train/0/  1/  2/  3/  4/  5/  6/  [7/]
        Where 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Anger
        Folder 7 (Contempt) is automatically skipped.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        geo_features_map: Optional[dict] = None,
    ):
        self.transform = transform or get_inference_transform()
        self.geo_features_map = geo_features_map  # filename -> [5 floats]
        self.samples: list[tuple[Path, int]] = []

        # Resolve root: handle nested dirs like AffectNetCustom/
        resolved_root = self._resolve_root(Path(root))

        # Resolve split: fall back "val" → "test" if needed
        split_path = resolved_root / split
        if not split_path.exists() and split == "val":
            split_path = resolved_root / "test"
        self.root = split_path

        fmt = self._detect_format()

        if fmt == "named":
            # Format A: 0_angry, 1_disgust, ...
            for class_idx, class_dir in enumerate(CLASS_DIRS):
                self._load_class_dir(self.root / class_dir, class_idx)
        else:
            # Format B: 0, 1, 2, ... (AffectNet standard order)
            for affectnet_idx in range(7):
                class_path = self.root / str(affectnet_idx)
                project_idx = AFFECTNET_TO_PROJECT[affectnet_idx]
                self._load_class_dir(class_path, project_idx)

    def _resolve_root(self, root: Path) -> Path:
        """Handle nested dataset dirs (e.g. archive extracts to root/AffectNetCustom/)."""
        if (root / "train").exists() or (root / "test").exists() or (root / "val").exists():
            return root
        # Check for a single subdirectory containing train/test/val
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            sub = subdirs[0]
            if (sub / "train").exists() or (sub / "test").exists() or (sub / "val").exists():
                return sub
        return root

    def _detect_format(self) -> str:
        """Detect whether folders use named ('0_angry') or numeric ('0') convention."""
        if (self.root / CLASS_DIRS[0]).exists():
            return "named"
        if (self.root / "0").exists():
            return "numeric"
        # Fallback: try named first
        return "named"

    def _load_class_dir(self, class_path: Path, class_idx: int):
        if not class_path.exists():
            return
        for img_path in sorted(class_path.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Geometric features: use provided map or zeros as placeholder
        if self.geo_features_map and img_path.name in self.geo_features_map:
            geo = torch.tensor(self.geo_features_map[img_path.name], dtype=torch.float32)
        else:
            geo = torch.zeros(5, dtype=torch.float32)

        return image, geo, label

    def get_class_counts(self) -> dict[int, int]:
        """Return {class_idx: sample_count} for weighted loss computation."""
        counts: dict[int, int] = {}
        for _, label in self.samples:
            counts[label] = counts.get(label, 0) + 1
        return counts


class MockDataset(Dataset):
    """Generate random tensors for local development validation."""

    def __init__(self, num_samples: int = 100, num_classes: int = 7):
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        geo = torch.randn(5)
        label = idx % self.num_classes
        return image, geo, label
