"""Stats response schemas."""

from pydantic import BaseModel


class ClassMetrics(BaseModel):
    precision: float
    recall: float
    f1: float


class PerformanceResponse(BaseModel):
    metrics: dict[str, ClassMetrics]
    macro_avg: ClassMetrics
    weighted_avg: ClassMetrics


class ConfusionMatrixResponse(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class LatencyResponse(BaseModel):
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
