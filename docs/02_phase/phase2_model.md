# Phase 2: 模型实现（本地 CPU 开发）

## 1. 目标

在本地 CPU 环境实现完整的 CNN 模型、几何特征编码器、特征融合模块，使用 mock 数据验证数据流正确性。

## 2. 模型定义

### 2.1 CNN 网络

**文件**: `backend/app/models/cnn.py`

```python
import torch.nn as nn

class EmotionCNN(nn.Module):
    """
    Input(224x224x3)
    → Conv2D(32, 3x3, p=1) + BN + ReLU + MaxPool(2x2)  → 112x112x32
    → Conv2D(64, 3x3, p=1) + BN + ReLU + MaxPool(2x2)  → 56x56x64
    → Conv2D(96, 3x3, p=1) + BN + ReLU + MaxPool(2x2)  → 28x28x96
    → Conv2D(128, 3x3, p=1) + BN + ReLU + MaxPool(2x2) → 14x14x128
    → GlobalAvgPool                                     → 128
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)    # (B, 128, 14, 14)
        x = self.gap(x)         # (B, 128, 1, 1)
        return x.squeeze(-1).squeeze(-1)  # (B, 128)
```

### 2.2 几何特征编码器

**文件**: `backend/app/models/geo_encoder.py`

```python
class GeoEncoder(nn.Module):
    """Input(5) → FC(32) + ReLU → FC(64) + ReLU → 64"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)  # (B, 64)
```

### 2.3 融合分类头

**文件**: `backend/app/models/fusion.py`

```python
class FusionClassifier(nn.Module):
    """Concat(128+64)→192 → FC(128) + ReLU + Dropout(0.5) → FC(7)"""
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 7),
        )

    def forward(self, pixel_feat, geo_feat):
        x = torch.cat([pixel_feat, geo_feat], dim=1)  # (B, 192)
        return self.head(x)  # (B, 7)
```

### 2.4 完整模型包装

**文件**: `backend/app/services/model_service.py`

```python
class ModelService:
    def __init__(self, config: dict):
        self.device = torch.device(config["model"]["device"])
        self.cnn = EmotionCNN().to(self.device)
        self.geo_encoder = GeoEncoder().to(self.device)
        self.classifier = FusionClassifier().to(self.device)

        checkpoint_path = config["model"]["checkpoint_path"]
        self.loaded = self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, path: str) -> bool:
        # 如文件存在则加载，否则返回 False（mock 模式）
        ...

    def predict(self, image_tensor, geo_features) -> dict:
        # 推理入口
        ...
```

## 3. 关键点索引常量

**文件**: `backend/app/utils/landmarks.py`

定义 MediaPipe 468 关键点中用于几何特征计算的索引：

```python
# 左眼关键点
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# 右眼关键点
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# 嘴部关键点
MOUTH_INDICES = [61, 291, 0, 17, 78, 308]

# 眉毛关键点
LEFT_EYEBROW = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
```

## 4. 几何特征计算（训练预处理用）

**文件**: `backend/app/services/geo_features.py`

纯 Python 函数，输入 468 个关键点坐标，输出 5 维特征向量：

- `calc_ear(landmarks)` — 眼宽高比
- `calc_mar(landmarks)` — 嘴宽高比
- `calc_eyebrow_eye_dist(landmarks)` — 眉眼间距
- `calc_mouth_curvature(landmarks)` — 嘴角曲率
- `calc_nasolabial_depth(landmarks)` — 鼻唇沟深度

同时提供归一化函数 `normalize_landmarks(landmarks)`：以双眼中心为原点、眼距为缩放因子。

## 5. 数据处理

### 5.1 数据加载器

**文件**: `backend/app/utils/image.py`

- `AffectNetDataset` — 读取目录结构，加载图片 + 标签
- `MockDataset` — 返回随机生成的张量（本地验证用）
- 图像预处理：Resize → ToTensor → Normalize（ImageNet 均值/标准差）

### 5.2 数据增强

**文件**: `backend/app/utils/data_augment.py`

```python
from torchvision import transforms

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
```

## 6. Mock 数据验证

使用 mock 数据集（每类 15-20 张图片）在 CPU 上跑通：

1. 加载模型（随机权重）
2. 输入一张 mock 图片 + mock 几何特征
3. 验证输出形状为 `(1, 7)`，概率和为 1

```bash
cd backend
python -c "from app.services.model_service import ModelService; ..."
```

## 7. 模型权重保存/加载

```python
def save_checkpoint(model, path):
    torch.save({
        'cnn': model.cnn.state_dict(),
        'geo_encoder': model.geo_encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
    }, path)

def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location=model.device)
    model.cnn.load_state_dict(ckpt['cnn'])
    model.geo_encoder.load_state_dict(ckpt['geo_encoder'])
    model.classifier.load_state_dict(ckpt['classifier'])
```

## 8. 交付物

- [ ] `EmotionCNN` 网络定义与论文表 2 一致（总参数量 ~203K）
- [ ] `GeoEncoder` 和 `FusionClassifier` 实现完成
- [ ] `ModelService` 可加载/保存权重
- [ ] 关键点索引常量定义完成
- [ ] 几何特征计算函数实现
- [ ] Mock 数据验证通过，数据流正确
- [ ] `train.py` 训练脚本骨架（Phase 3 填充）
