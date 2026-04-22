"""Inference routes — HTTP POST and WebSocket stream."""

import logging

import torch
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect, HTTPException

from app.api.schemas.inference import InferenceRequest, InferenceResponse, GeoFeatures
from app.dependencies import get_model_service, get_latency_tracker
from app.services.model_service import ModelService
from app.utils.image import decode_base64_image, image_to_tensor
from app.utils.latency import LatencyTracker

logger = logging.getLogger(__name__)
router = APIRouter()


def _prepare_geo_tensor(geo: GeoFeatures) -> torch.Tensor:
    return torch.tensor(
        [[geo.ear, geo.mar, geo.eyebrow_eye_dist, geo.mouth_curvature, geo.nasolabial_depth]],
        dtype=torch.float32,
    )


@router.post("/api/v1/inference/emotion", response_model=InferenceResponse)
async def infer_emotion(
    req: InferenceRequest,
    model_service: ModelService = Depends(get_model_service),
    tracker: LatencyTracker = Depends(get_latency_tracker),
):
    """Single-frame emotion inference."""
    if not model_service:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        image = decode_base64_image(req.face_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid face_image — failed to decode base64")

    image_tensor = image_to_tensor(image)
    geo_tensor = _prepare_geo_tensor(req.geo_features)

    result = model_service.predict(image_tensor, geo_tensor)
    tracker.record(result["inference_time_ms"])

    return InferenceResponse(
        emotions=result["emotions"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        inference_time_ms=result["inference_time_ms"],
    )


@router.websocket("/ws/v1/inference/stream")
async def ws_inference_stream(
    websocket: WebSocket,
    model_service: ModelService = Depends(get_model_service),
    tracker: LatencyTracker = Depends(get_latency_tracker),
):
    """WebSocket streaming inference — one JSON per frame."""
    await websocket.accept()
    logger.info("WebSocket inference stream connected")

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("action") == "stop":
                logger.info("WebSocket client requested stop")
                break

            try:
                face_image = data.get("face_image", "")
                geo_data = data.get("geo_features", {})

                image = decode_base64_image(face_image)
                image_tensor = image_to_tensor(image)
                geo_tensor = torch.tensor(
                    [[
                        geo_data.get("ear", 0),
                        geo_data.get("mar", 0),
                        geo_data.get("eyebrow_eye_dist", 0),
                        geo_data.get("mouth_curvature", 0),
                        geo_data.get("nasolabial_depth", 0),
                    ]],
                    dtype=torch.float32,
                )

                result = model_service.predict(image_tensor, geo_tensor)
                tracker.record(result["inference_time_ms"])

                await websocket.send_json(result)

            except Exception as e:
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket inference stream disconnected")
