"""Inference request/response schemas."""

from pydantic import BaseModel


class GeoFeatures(BaseModel):
    ear: float
    mar: float
    eyebrow_eye_dist: float
    mouth_curvature: float
    nasolabial_depth: float


class InferenceRequest(BaseModel):
    face_image: str
    geo_features: GeoFeatures


class InferenceResponse(BaseModel):
    emotions: dict[str, float]
    prediction: str
    confidence: float
    inference_time_ms: float
