"""Health check response schemas."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: int
