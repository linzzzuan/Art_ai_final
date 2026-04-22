"""FastAPI dependency injection providers."""

from __future__ import annotations

import logging

from app.config import get_settings
from app.utils.latency import LatencyTracker

logger = logging.getLogger(__name__)

# Singletons
_model_service = None
_latency_tracker = LatencyTracker(capacity=100)


def get_config() -> dict:
    """Provide the current configuration dict."""
    return get_settings()


def get_model_service():
    """Lazy-initialize and return the ModelService singleton."""
    global _model_service
    if _model_service is None:
        from app.services.model_service import ModelService
        _model_service = ModelService(get_settings())
    return _model_service


def get_latency_tracker() -> LatencyTracker:
    """Return the shared latency tracker."""
    return _latency_tracker
