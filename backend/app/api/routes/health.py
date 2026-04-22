"""Health check endpoint."""

import time
from fastapi import APIRouter, Depends, Request

from app.config import get_settings
from app.dependencies import get_model_service

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    cfg = get_settings()
    uptime = int(time.time() - request.app.state.start_time)
    model_svc = get_model_service()
    return {
        "status": "ok",
        "model_loaded": model_svc.loaded if model_svc else False,
        "device": cfg["model"]["device"],
        "uptime_seconds": uptime,
    }
