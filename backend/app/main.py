"""FastAPI application entry point."""

import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import health, config as config_route, inference, dataset, stats
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)

cfg = get_settings()

setup_logging(cfg)

app = FastAPI(
    title="Emotion Detection API",
    description="Multi-modal real-time emotion detection system",
    version="0.1.0",
)

# Store startup time for uptime calculation
app.state.start_time = time.time()

# ── Exception handlers ──

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error("Runtime error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg["app"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(config_route.router, prefix="/api/v1")
app.include_router(inference.router)
app.include_router(dataset.router)
app.include_router(stats.router)
