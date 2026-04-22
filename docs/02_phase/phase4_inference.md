# Phase 4: 推理服务

## 1. 目标

实现后端推理 API（HTTP + WebSocket），加载训练好的权重提供情绪识别服务。

## 2. 接口实现

### 2.1 情绪推理（HTTP POST）

**文件**: `backend/app/api/routes/inference.py`

```python
@router.post("/api/v1/inference/emotion")
async def infer_emotion(req: InferenceRequest):
    """单帧情绪推理"""
    # 1. 解码 base64 图像
    image_tensor = decode_image(req.face_image)  # → (1, 3, 224, 224)

    # 2. 准备几何特征
    geo = req.geo_features
    geo_tensor = torch.tensor([[geo.ear, geo.mar, geo.eyebrow_eye_dist,
                                 geo.mouth_curvature, geo.nasolabial_depth]])

    # 3. 推理
    start = time.time()
    result = model_service.predict(image_tensor, geo_tensor.to(device))
    elapsed_ms = (time.time() - start) * 1000

    return InferenceResponse(
        emotions=result["emotions"],
        prediction=result["prediction"],
        confidence=result["confidence"],
        inference_time_ms=round(elapsed_ms),
    )
```

### 2.2 情绪推理（WebSocket 流式）

```python
@router.websocket("/ws/v1/inference/stream")
async def ws_infer(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        if data.get("action") == "stop":
            break

        # 同 HTTP 推理逻辑
        result = process_frame(data)
        await websocket.send_json(result)
```

### 2.3 请求/响应 Schema

**文件**: `backend/app/api/schemas/inference.py`

```python
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
    inference_time_ms: int
```

## 3. 图像处理

**文件**: `backend/app/utils/image.py`

```python
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms

def decode_image(base64_str: str) -> torch.Tensor:
    """base64 → (1, 3, 224, 224) tensor"""
    # 移除 data URI 前缀
    if base64_str.startswith("data:"):
        base64_str = base64_str.split(",")[1]

    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  # (1, 3, 224, 224)
```

## 4. 模型服务

**文件**: `backend/app/services/model_service.py`

更新 Phase 2 的 `ModelService`：

```python
class ModelService:
    def __init__(self, config: dict):
        ...
        self.loaded = self._load_checkpoint(checkpoint_path)
        if self.loaded:
            self._warmup()  # 预热模型（首次推理较慢）

    def _warmup(self):
        dummy_img = torch.randn(1, 3, 224, 224).to(self.device)
        dummy_geo = torch.randn(1, 5).to(self.device)
        with torch.no_grad():
            self.predict(dummy_img, dummy_geo)

    def predict(self, image_tensor, geo_tensor) -> dict:
        with torch.no_grad():
            pixel_feat = self.cnn(image_tensor)       # (1, 128)
            geo_feat = self.geo_encoder(geo_tensor)   # (1, 64)
            logits = self.classifier(pixel_feat, geo_feat)  # (1, 7)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        emotions = dict(zip(labels, probs.tolist()))
        pred_idx = torch.argmax(probs).item()

        return {
            "emotions": emotions,
            "prediction": labels[pred_idx],
            "confidence": probs[pred_idx].item(),
        }
```

## 5. 数据集和统计接口

### 5.1 数据集信息

**文件**: `backend/app/api/routes/dataset.py`

```python
@router.get("/api/v1/dataset/info")
def get_dataset_info():
    path = config["dataset"]["affectnet_path"]
    counts = count_images(path)
    return {
        "name": "AffectNet-7",
        "path": path,
        "total_samples": sum(c for _, c in counts),
        "image_size": "224x224",
        "num_classes": 7,
        "classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    }

@router.get("/api/v1/dataset/class-distribution")
def get_class_distribution():
    path = config["dataset"]["affectnet_path"]
    counts = count_images(path)
    total = sum(c for _, c in counts)
    labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    return {
        "classes": [
            {"name": labels[i], "count": counts[i], "percentage": round(counts[i]/total*100, 1)}
            for i in range(7)
        ]
    }
```

### 5.2 统计接口

**文件**: `backend/app/api/routes/stats.py`

```python
@router.get("/api/v1/stats/performance")
def get_performance():
    # 读取 experiments/ 下最新的 performance.json
    perf = load_latest_experiment("performance.json")
    return perf

@router.get("/api/v1/stats/confusion-matrix")
def get_confusion_matrix():
    cm = load_latest_experiment("confusion_matrix.json")
    return cm

@router.get("/api/v1/stats/latency")
def get_latency():
    # 从内存缓存或 SQLite 读取
    return get_latency_stats()
```

## 6. 延迟统计

维护一个内存环形缓冲区，记录最近 100 次推理耗时：

```python
class LatencyTracker:
    def __init__(self, capacity=100):
        self.times = collections.deque(maxlen=capacity)

    def record(self, ms):
        self.times.append(ms)

    def stats(self):
        if not self.times:
            return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "fps": 0}
        arr = sorted(self.times)
        return {
            "avg_ms": round(sum(arr) / len(arr), 1),
            "p50_ms": round(arr[len(arr) // 2], 1),
            "p95_ms": round(arr[int(len(arr) * 0.95)], 1),
            "p99_ms": round(arr[int(len(arr) * 0.99)], 1),
            "fps": round(1000 / (sum(arr) / len(arr)), 1),
        }
```

## 7. 依赖注入

**文件**: `backend/app/dependencies.py`

```python
_model_service = None

def get_model_service():
    global _model_service
    if _model_service is None:
        _model_service = ModelService(get_config())
    return _model_service
```

## 8. 测试

```bash
# 本地测试推理
curl -X POST http://localhost:8000/api/v1/inference/emotion \
  -H "Content-Type: application/json" \
  -d '{"face_image": "...", "geo_features": {...}}'

# 测试 WebSocket（使用 wscat）
wscat -c ws://localhost:8000/ws/v1/inference/stream
> {"face_image": "...", "geo_features": {...}}
```

## 9. 交付物

- [ ] `POST /api/v1/inference/emotion` 接口实现
- [ ] `WS /ws/v1/inference/stream` 接口实现
- [ ] `GET /api/v1/dataset/info` 接口实现
- [ ] `GET /api/v1/dataset/class-distribution` 接口实现
- [ ] `GET /api/v1/stats/performance` 接口实现
- [ ] `GET /api/v1/stats/confusion-matrix` 接口实现
- [ ] `GET /api/v1/stats/latency` 接口实现
- [ ] `GET /api/v1/config` 接口实现
- [ ] 加载云端训练权重，本地 CPU 推理验证通过
- [ ] 延迟统计功能
