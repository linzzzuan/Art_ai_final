"""Model service: loads models, manages checkpoints, runs inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.models.cnn import EmotionCNN
from app.models.geo_encoder import GeoEncoder
from app.models.fusion import FusionClassifier
from app.utils.image import EMOTION_CLASSES

logger = logging.getLogger(__name__)


class ModelService:
    """Encapsulates the three sub-networks and provides predict / save / load."""

    def __init__(self, config: dict) -> None:
        self.device = torch.device(config["model"]["device"])
        self.cnn = EmotionCNN().to(self.device)
        self.geo_encoder = GeoEncoder().to(self.device)
        self.classifier = FusionClassifier().to(self.device)

        root = Path(config["_root"])
        ckpt_path = root / config["model"]["checkpoint_path"]
        self.loaded = self._load_checkpoint(ckpt_path)

        # Switch to eval mode
        self.cnn.eval()
        self.geo_encoder.eval()
        self.classifier.eval()

        # Warmup (first inference is slower due to lazy initialization)
        self._warmup()

        logger.info(
            "ModelService initialized — device=%s, loaded=%s, params=%d",
            self.device,
            self.loaded,
            self.param_count(),
        )

    def _warmup(self) -> None:
        """Run a dummy inference to warm up the model."""
        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_geo = torch.randn(1, 5).to(self.device)
        self.predict(dummy_img, dummy_geo)

    # ── checkpoint management ──────────────────────────────────────

    def _load_checkpoint(self, path: Path) -> bool:
        if not path.exists():
            logger.warning("Checkpoint not found at %s — using random weights", path)
            return False
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.cnn.load_state_dict(ckpt["cnn"])
            self.geo_encoder.load_state_dict(ckpt["geo_encoder"])
            self.classifier.load_state_dict(ckpt["classifier"])
            logger.info("Checkpoint loaded from %s", path)
            return True
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return False

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cnn": self.cnn.state_dict(),
                "geo_encoder": self.geo_encoder.state_dict(),
                "classifier": self.classifier.state_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved to %s", path)

    # ── inference ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, geo_features: torch.Tensor) -> dict:
        """Run inference on a single sample or batch.

        Args:
            image_tensor: (B, 3, 224, 224) float tensor
            geo_features: (B, 5) float tensor

        Returns:
            dict with emotions, prediction, confidence, inference_time_ms
        """
        image_tensor = image_tensor.to(self.device)
        geo_features = geo_features.to(self.device)

        start = time.perf_counter()

        pixel_feat = self.cnn(image_tensor)          # (B, 128)
        geo_feat = self.geo_encoder(geo_features)    # (B, 64)
        logits = self.classifier(pixel_feat, geo_feat)  # (B, 7)
        probs = F.softmax(logits, dim=1)             # (B, 7)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Take first sample in batch for single-sample response
        probs_np = probs[0].cpu().numpy()
        pred_idx = int(probs_np.argmax())

        emotions = {name: round(float(probs_np[i]), 4) for i, name in enumerate(EMOTION_CLASSES)}
        return {
            "emotions": emotions,
            "prediction": EMOTION_CLASSES[pred_idx],
            "confidence": round(float(probs_np[pred_idx]), 4),
            "inference_time_ms": round(elapsed_ms, 1),
        }

    # ── utilities ──────────────────────────────────────────────────

    def param_count(self) -> int:
        total = 0
        for m in (self.cnn, self.geo_encoder, self.classifier):
            total += sum(p.numel() for p in m.parameters())
        return total
