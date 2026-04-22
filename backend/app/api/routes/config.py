"""Configuration info endpoint."""

from fastapi import APIRouter

from app.config import get_settings
from app.dependencies import get_model_service

router = APIRouter()


@router.get("/config")
async def get_config_info():
    cfg = get_settings()
    model_svc = get_model_service()
    return {
        "active_profile": cfg["_profile"],
        "model_device": cfg["model"]["device"],
        "checkpoint_path": cfg["model"]["checkpoint_path"],
        "model_loaded": model_svc.loaded if model_svc else False,
    }
