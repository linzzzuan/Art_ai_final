"""Stats routes — performance, confusion matrix, latency."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_config, get_latency_tracker
from app.utils.latency import LatencyTracker

logger = logging.getLogger(__name__)
router = APIRouter()


def _find_latest_experiment(root: Path) -> Path | None:
    """Find the most recently modified experiment directory."""
    exp_root = root / "experiments"
    if not exp_root.exists():
        return None
    dirs = [d for d in exp_root.iterdir() if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _load_experiment_file(root: Path, filename: str) -> dict | None:
    """Load a JSON file from the latest experiment directory."""
    exp_dir = _find_latest_experiment(root)
    if exp_dir is None:
        return None
    path = exp_dir / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/api/v1/stats/performance")
async def get_performance(config: dict = Depends(get_config)):
    root = Path(config["_root"])
    data = _load_experiment_file(root, "performance.json")
    if data is None:
        raise HTTPException(status_code=404, detail="No performance data available — run training first")
    return data


@router.get("/api/v1/stats/confusion-matrix")
async def get_confusion_matrix(config: dict = Depends(get_config)):
    root = Path(config["_root"])
    data = _load_experiment_file(root, "confusion_matrix.json")
    if data is None:
        raise HTTPException(status_code=404, detail="No confusion matrix data available — run training first")
    return data


@router.get("/api/v1/stats/latency")
async def get_latency(tracker: LatencyTracker = Depends(get_latency_tracker)):
    return tracker.stats()
