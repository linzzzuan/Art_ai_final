# 多模态实时情绪检测系统 — 前后端分离架构方案

## 1. 系统概述

基于《基于 MediaPipe 与 CNN 的多模态实时情绪检测系统设计与实现》论文，
构建一个 Web 端前后端分离的实时情绪检测交互系统。

### 1.1 核心功能
- 实时摄像头视频流采集
- 面部 468 关键点检测（MediaPipe Face Mesh）
- 几何特征提取（EAR、MAR、眉眼间距、嘴角曲率、鼻唇沟深度）
- CNN 像素特征提取
- 特征层融合 → 七类情绪分类（愤怒/厌恶/恐惧/高兴/悲伤/惊讶/中性）
- 实时情绪标签叠加可视化
- 训练/推理管理、模型评估、数据统计

### 1.2 性能目标
- 云端 GPU 推理 ≥ 30 FPS
- CPU 推理（本地开发调试）≥ 10 FPS
- 端到端延迟 ≤ 80ms（WebSocket 模式，云端同区域）
- 训练在云端 GPU（如 RTX 4090 / A100）上完成，目标 3 小时内完成 50 epoch

### 1.3 部署模式
- **本地开发**：CPU 推理 + 前端 localhost 联调（无需 GPU）
- **云端训练**：租赁 GPU 服务器，上传 AffectNet 数据集执行训练
- **云端部署**：前端 + 后端统一部署至云服务器，用户通过浏览器访问

---

## 2. 技术选型

| 层次 | 技术 | 说明 |
|------|------|------|
| **前端框架** | React + TypeScript + Vite | 组件化开发，类型安全 |
| **UI 组件库** | Ant Design | 企业级组件，中后台友好 |
| **视频处理** | MediaPipe JS (@mediapipe/face-mesh) | **前端执行**关键点检测，减轻后端压力 |
| **可视化** | ECharts | 训练曲线、混淆矩阵、F1 柱状图 |
| **后端框架** | Python + FastAPI | 高性能异步 API，自动 OpenAPI 文档 |
| **深度学习** | PyTorch | CNN 训练与推理 |
| **数据库** | SQLite → 可迁移 PostgreSQL | 存储训练记录、实验结果、用户上传数据 |
| **ORM** | SQLAlchemy | 数据库操作 |

### 2.1 关键架构决策：MediaPipe 放在前端

论文原版在 Python 端运行 MediaPipe，但在 Web 架构中，将 MediaPipe Face Mesh
部署在**前端浏览器**更优：

- **降低服务器负载**：468 关键点检测在客户端完成，后端只接收特征数据
- **减少网络传输**：无需传输原始视频流到后端，仅传输关键点和裁剪图像
- **更低延迟**：浏览器端实时检测，无网络往返延迟
- **隐私保护**：原始视频帧不离开客户端

前端负责：摄像头采集 → MediaPipe 检测 → 几何特征计算 → 裁剪面部图像
后端负责：CNN 推理 → 特征融合 → 情绪分类 → 返回结果

---

## 3. 系统架构

### 3.1 逻辑架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户浏览器 (客户端)                     │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │ 摄像头采集│──▶│MediaPipe检测 │──▶│ 几何特征计算    │  │
│  │ 视频帧    │   │ 468关键点    │   │ EAR/MAR/...    │  │
│  └──────────┘   └──────────────┘   └────────┬───────┘  │
│                                              │          │
│                    ┌────────────────┐         │          │
│                    │ 面部裁剪图像    │◀────────┘          │
│                    │ 224x224 JPEG   │                    │
│                    └────────┬───────┘                    │
│                             │                           │
│                    ┌────────▼───────┐                    │
│                    │  WebSocket /   │                    │
│                    │  HTTPS POST    │                    │
│                    └────────┬───────┘                    │
└─────────────────────────────┼───────────────────────────┘
                              │ (base64图像 + 5维几何特征)
                              │ 通过互联网 HTTPS/WSS
                              ▼
┌─────────────────────────────────────────────────────────┐
│              云服务器（租赁 GPU 平台）                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Nginx（反向代理 + HTTPS 终端）                     │   │
│  │  /          → 前端静态资源 (React SPA)             │   │
│  │  /api/      → 后端 FastAPI (Gunicorn + Uvicorn)   │   │
│  │  /ws/       → 后端 WebSocket 代理                  │   │
│  └────────┬─────────────────────────────────────────┘   │
│           │                                             │
│           ▼                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────┐  │
│  │ API 路由层    │──▶│ 推理服务层   │──▶│ CNN模型    │  │
│  │ /api/v1/*    │   │ 特征融合+分类 │   │ PyTorch    │  │
│  └──────────────┘   └──────────────┘   └─────┬──────┘  │
│                                               │         │
│                                      ┌────────▼──────┐  │
│                                      │  GPU (CUDA)   │  │
│                                      │  RTX 4090/A10 │  │
│                                      └───────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SQLite / PostgreSQL                  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 开发环境与生产环境差异

| 维度 | 本地开发环境 | 云端生产环境 |
|------|-------------|-------------|
| 配置 profile | `local` | `server` |
| GPU | 无（CPU only） | 有（RTX 4090 / A10 / A100） |
| 推理设备 | CPU | GPU (CUDA) |
| 训练 | 不训练（或 mock 验证） | GPU 全量训练 |
| 前端 | Vite dev server (localhost:5173) | Nginx 托管构建产物 |
| 后端 | `uvicorn --reload` (localhost:8000) | Gunicorn + Uvicorn workers |
| 数据库 | SQLite (本地文件) | SQLite 或 PostgreSQL |
| 协议 | HTTP | HTTPS + WSS（摄像头必须 HTTPS） |
| 数据集 | mock 数据 | 完整 AffectNet 挂载 |

---

## 4. 后端详细设计 (backend/)

### 4.1 目录结构

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 应用入口
│   ├── config.py               # 配置管理（环境变量、路径）
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── inference.py    # 情绪推理接口
│   │   │   ├── training.py     # 模型训练管理
│   │   │   ├── dataset.py      # 数据集管理
│   │   │   └── stats.py        # 统计与可视化数据
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── inference.py    # 请求/响应模型
│   │       ├── training.py
│   │       └── dataset.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_loader.py     # PyTorch 模型加载与管理
│   │   ├── inference.py        # 推理逻辑（特征融合+分类）
│   │   ├── training.py         # 训练流程控制
│   │   └── geo_features.py     # 几何特征计算（用于训练数据预处理）
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py              # CNN 网络架构定义
│   │   ├── fusion.py           # 特征融合模块
│   │   ├── geo_encoder.py      # 几何特征全连接编码器
│   │   └── db_models.py        # SQLAlchemy 数据库模型
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image.py            # 图像处理工具
│   │   ├── landmarks.py        # 关键点归一化、关键点索引常量
│   │   ├── metrics.py          # 评估指标计算
│   │   └── data_augment.py     # 数据增强工具
│   └── core/
│       ├── __init__.py
│       └── logging.py          # 日志配置
├── checkpoints/                # 训练好的模型权重 .pth
├── datasets/                   # AffectNet 数据集路径（挂载或软链接）
├── experiments/                # 训练日志、曲线数据、混淆矩阵
├── data/                       # SQLite 数据库文件
├── docker/
│   ├── Dockerfile              # 后端容器镜像
│   └── nginx.conf              # Nginx 配置
├── requirements.txt
├── requirements-dev.txt        # 开发环境依赖（不含 CUDA）
├── requirements-prod.txt       # 生产环境依赖（含 CUDA/PyTorch GPU）
├── .env.example
└── gunicorn_conf.py            # Gunicorn 生产启动配置
```

### 4.2 部署脚本结构（项目根目录）

```
scripts/
├── deploy_setup.sh            # 脚本一：环境部署（首次执行）
├── run_training.sh            # 脚本二：启动训练
├── start_server.sh            # 脚本三：启动/管理服务
├── config.yaml              # 统一配置（本地/服务器共用，已提交 git）
└── utils/
    ├── check_gpu.sh           # GPU 环境检查
    ├── check_deps.sh          # Python/Node.js 依赖检查
    └── health_check.sh        # 服务健康检查
```

### 4.2 后端目录结构（独立项目）

后端作为独立项目交付，目录结构如下：

```
emotion-backend/                        ← 后端项目根目录
├── app/
│   ├── __init__.py
│   ├── main.py                         # FastAPI 应用入口
│   ├── config.py                       # config.yaml 加载器
│   ├── dependencies.py                 # 依赖注入（模型实例、配置）
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py               # GET /health
│   │   │   ├── inference.py            # POST /api/v1/inference/emotion
│   │   │   │                           # WS  /ws/v1/inference/stream
│   │   │   ├── dataset.py              # 数据集信息
│   │   │   └── stats.py                # 评估结果查询
│   │   │
│   │   └── schemas/                    # Pydantic 请求/响应模型
│   │       ├── __init__.py
│   │       ├── inference.py
│   │       ├── dataset.py
│   │       └── stats.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── model_service.py            # 模型加载、预热、推理
│   │   └── geo_features.py             # 几何特征计算（训练预处理用）
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py                      # CNN 网络定义
│   │   ├── fusion.py                   # 特征融合 + 分类头
│   │   └── geo_encoder.py              # 几何特征编码器
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image.py                    # base64 解码、预处理
│   │   ├── landmarks.py                # 关键点索引常量
│   │   └── metrics.py                  # Precision/Recall/F1/混淆矩阵
│   │
│   └── core/
│       ├── __init__.py
│       └── logging.py
│
├── checkpoints/                        # 模型权重目录
├── experiments/                        # 训练产物
├── data/                               # SQLite 数据库
├── logs/                               # 日志
│
├── train.py                            # 训练入口脚本（被 run_training.sh 调用）
├── config.yaml                         # 统一配置（local/server profiles）
├── requirements.txt                    # 基础依赖
├── requirements-prod.txt               # 生产依赖（含 torch+cuda）
├── gunicorn_conf.py                    # Gunicorn 配置
│
├── scripts/                            # 部署脚本
│   ├── deploy_setup.sh
│   ├── run_training.sh
│   ├── start_server.sh
│   └── utils/
│       ├── check_gpu.sh
│       ├── check_deps.sh
│       └── health_check.sh
│
├── docs/
│   └── api.md                          # 本接口文档（即下文）
│
└── README.md
```

### 4.3 后端 API 接口文档（前端消费）

> 训练通过 `run_training.sh` 命令行执行，前端只消费推理和训练结果数据。
> 以下文档可直接粘贴给 Figma AI / v0 / Lovable 等工具生成前端代码。
> FastAPI 部署后也可通过 `http://host/docs` 访问交互式 Swagger UI。

#### 通用约定

- 所有接口返回 HTTP 200 成功，错误返回对应 HTTP 状态码 + 错误体
- 时间字段均为 ISO 8601 格式：`"2026-04-21T10:30:00"`
- 情绪类别固定 7 种：`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`
- 概率值范围 `[0, 1]`，总和为 1

#### 接口总览

| # | 类别 | 方法 | 路径 | 前端用途 |
|---|------|------|------|---------|
| 1 | 系统 | `GET` | `/health` | 健康检查 |
| 2 | 推理 | `POST` | `/api/v1/inference/emotion` | 单帧情绪识别 |
| 3 | 推理 | `WS` | `/ws/v1/inference/stream` | 实时流式识别 |
| 4 | 数据 | `GET` | `/api/v1/dataset/info` | 数据集概览 |
| 5 | 数据 | `GET` | `/api/v1/dataset/class-distribution` | 类别分布柱状图 |
| 6 | 结果 | `GET` | `/api/v1/stats/performance` | Precision/Recall/F1 表格 |
| 7 | 结果 | `GET` | `/api/v1/stats/confusion-matrix` | 混淆矩阵热力图 |
| 8 | 结果 | `GET` | `/api/v1/stats/latency` | 延迟分布统计 |
| 9 | 配置 | `GET` | `/api/v1/config` | 当前配置信息 |

共 **9 个接口**，3 类：系统、推理、数据/结果。

---

#### 1. 健康检查

```
GET /health
```

**响应 (200)**：
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "uptime_seconds": 3600
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `"ok"` 表示服务正常 |
| `model_loaded` | bool | 模型权重是否已加载 |
| `device` | string | `"cuda"` 或 `"cpu"` |
| `uptime_seconds` | int | 服务运行时长（秒） |

---

#### 2. 情绪推理（HTTP）

```
POST /api/v1/inference/emotion
```

**请求体**：
```json
{
  "face_image": "data:image/jpeg;base64,/9j/4AAQSkZJ...",
  "geo_features": {
    "ear": 0.25,
    "mar": 0.35,
    "eyebrow_eye_dist": 0.18,
    "mouth_curvature": 0.12,
    "nasolabial_depth": 0.95
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `face_image` | string | 是 | base64 编码的 224×224 JPEG（含 data URI 前缀或纯 base64 均可） |
| `geo_features.ear` | float | 是 | 眼宽高比 |
| `geo_features.mar` | float | 是 | 嘴宽高比 |
| `geo_features.eyebrow_eye_dist` | float | 是 | 眉眼间距 |
| `geo_features.mouth_curvature` | float | 是 | 嘴角曲率 |
| `geo_features.nasolabial_depth` | float | 是 | 鼻唇沟深度 |

**响应 (200)**：
```json
{
  "emotions": {
    "angry": 0.01,
    "disgust": 0.01,
    "fear": 0.01,
    "happy": 0.82,
    "sad": 0.02,
    "surprise": 0.05,
    "neutral": 0.08
  },
  "prediction": "happy",
  "confidence": 0.82,
  "inference_time_ms": 45
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `emotions` | object | 七类情绪概率，key 为情绪名，value 为概率值 |
| `prediction` | string | 预测情绪类别 |
| `confidence` | float | 预测置信度（最高概率值） |
| `inference_time_ms` | int | 推理耗时（毫秒） |

**错误响应**：
| 状态码 | 响应体 | 含义 |
|--------|--------|------|
| 400 | `{ "detail": "face_image is required" }` | 缺少必要参数 |
| 503 | `{ "detail": "Model not loaded" }` | 模型未加载 |

---

#### 3. 情绪推理（WebSocket 流式）

```
WS /ws/v1/inference/stream
```

**客户端发送（每帧一条 JSON 字符串）**：
```json
{
  "face_image": "data:image/jpeg;base64,/9j/4AAQSkZJ...",
  "geo_features": {
    "ear": 0.25,
    "mar": 0.35,
    "eyebrow_eye_dist": 0.18,
    "mouth_curvature": 0.12,
    "nasolabial_depth": 0.95
  }
}
```

**服务端返回（每帧一条 JSON 字符串）**：
```json
{
  "emotions": {
    "angry": 0.01,
    "disgust": 0.01,
    "fear": 0.01,
    "happy": 0.82,
    "sad": 0.02,
    "surprise": 0.05,
    "neutral": 0.08
  },
  "prediction": "happy",
  "confidence": 0.82,
  "inference_time_ms": 45
}
```

**连接生命周期**：
- 客户端打开连接后可持续发送帧
- 服务端对每帧返回推理结果
- 客户端关闭连接或发送 `{ "action": "stop" }` 结束

---

#### 4. 数据集概览

```
GET /api/v1/dataset/info
```

**响应 (200)**：
```json
{
  "name": "AffectNet-7",
  "path": "/mnt/affectnet",
  "total_samples": 287568,
  "image_size": "224x224",
  "num_classes": 7,
  "classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
}
```

---

#### 5. 类别分布

```
GET /api/v1/dataset/class-distribution
```

**响应 (200)**：
```json
{
  "classes": [
    { "name": "angry", "count": 24882, "percentage": 8.7 },
    { "name": "disgust", "count": 3803, "percentage": 1.3 },
    { "name": "fear", "count": 6378, "percentage": 2.2 },
    { "name": "happy", "count": 94897, "percentage": 33.0 },
    { "name": "sad", "count": 25459, "percentage": 8.9 },
    { "name": "surprise", "count": 14090, "percentage": 4.9 },
    { "name": "neutral", "count": 75374, "percentage": 26.2 }
  ]
}
```

---

#### 6. 分类性能

```
GET /api/v1/stats/performance
```

**响应 (200)**：
```json
{
  "metrics": {
    "angry":    { "precision": 0.61, "recall": 0.58, "f1": 0.59 },
    "disgust":  { "precision": 0.45, "recall": 0.51, "f1": 0.48 },
    "fear":     { "precision": 0.38, "recall": 0.44, "f1": 0.41 },
    "happy":    { "precision": 0.85, "recall": 0.79, "f1": 0.82 },
    "sad":      { "precision": 0.55, "recall": 0.52, "f1": 0.53 },
    "surprise": { "precision": 0.71, "recall": 0.68, "f1": 0.69 },
    "neutral":  { "precision": 0.67, "recall": 0.63, "f1": 0.65 }
  },
  "macro_avg":    { "precision": 0.60, "recall": 0.59, "f1": 0.60 },
  "weighted_avg": { "precision": 0.68, "recall": 0.68, "f1": 0.68 }
}
```

---

#### 7. 混淆矩阵

```
GET /api/v1/stats/confusion-matrix
```

**响应 (200)**：
```json
{
  "labels": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
  "matrix": [
    [320,  45,  30,  15,  60,  20,  10],
    [ 50, 110,  25,  10,  30,  15,   5],
    [ 40,  30, 140,  20,  35,  25,  10],
    [ 20,  10,  15, 560,  15,  30,  50],
    [ 55,  20,  30,  25, 280,  15,  75],
    [ 25,  15,  35,  40,  20, 380,  15],
    [ 30,  10,  15,  45,  50,  20, 430]
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `labels` | string[] | 类别标签数组（与矩阵行列对应） |
| `matrix` | number[][] | 7×7 混淆矩阵，`matrix[i][j]` 表示真实为 i 但预测为 j 的样本数 |

---

#### 8. 延迟统计

```
GET /api/v1/stats/latency
```

**响应 (200)**：
```json
{
  "avg_ms": 45,
  "p50_ms": 42,
  "p95_ms": 68,
  "p99_ms": 95,
  "fps": 22.2
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `avg_ms` | float | 平均推理延迟 |
| `p50_ms` | float | P50 延迟 |
| `p95_ms` | float | P95 延迟 |
| `p99_ms` | float | P99 延迟 |
| `fps` | float | 等效帧率 |

---

#### 9. 配置信息

```
GET /api/v1/config
```

**响应 (200)**：
```json
{
  "active_profile": "server",
  "model_device": "cuda",
  "checkpoint_path": "backend/checkpoints/best_model.pth",
  "model_loaded": true
}
```

---

#### 错误响应格式

所有错误统一返回：

```json
{ "detail": "Error message here" }
```

| HTTP 状态码 | 含义 |
|-------------|------|
| 400 | 请求参数错误 |
| 503 | 服务不可用（模型未加载） |
| 500 | 服务器内部错误 |

### 4.4 模型定义

#### CNN 网络（与论文一致）
```
Input(224x224x3)
→ Conv2D(32, 3x3) + BN + ReLU + MaxPool(2x2)   → 112x112x32
→ Conv2D(64, 3x3) + BN + ReLU + MaxPool(2x2)   → 56x56x64
→ Conv2D(96, 3x3) + BN + ReLU + MaxPool(2x2)   → 28x28x96
→ Conv2D(128, 3x3) + BN + ReLU + MaxPool(2x2)  → 14x14x128
→ GlobalAvgPool                                 → 128
```

#### 几何特征编码器
```
Input(5) → FC(32) + ReLU → FC(64) + ReLU → 64
```

#### 融合分类头
```
Concat(128 + 64) → 192
→ FC(128) + ReLU + Dropout(0.5) → FC(7) → Softmax
```

总参数量约 203K，极轻量级。

### 4.4 训练流程

1. **数据加载**：AffectNet-7 数据集，按 8:1:1 划分
2. **数据增强**：随机翻转、随机旋转 ±10°、亮度抖动、对比度调整、高斯噪声
3. **加权交叉熵损失**：处理类别不均衡
4. **优化器**：Adam，初始 lr=0.001，余弦衰减
5. **早停**：patience=8，回滚最优权重
6. **训练过程通过 SSE/WebSocket 推送进度到前端**

### 4.5 本地开发与云端训练的分工

由于本地无 GPU 资源，开发过程中训练相关功能按以下策略处理：

- **模型结构验证**：使用极小样本（如 100 张图片）在 CPU 上跑 1-2 个 epoch，验证数据流正确性
- **完整训练**：在云端 GPU 服务器执行，数据集提前上传或挂载
- **模型权重传递**：训练完成后 `.pth` 文件可通过 scp/rsync 下载至本地测试，或直接留在云端
- **推理验证**：本地 CPU 加载训练好的权重进行推理接口测试（模型仅 203K 参数，CPU 推理完全可行）
- **数据集处理**：本地开发使用 mock 数据集结构；云端训练前再上传完整 AffectNet

---

## 5. 前端生成指南 — 基于 API 文档 + Figma AI

> 后端完成后，将本节第 4.3 节的 **API 接口文档** 连同 Figma 设计稿一起
> 提供给 Figma AI（v0 / Lovable / Cursor 等工具）生成前端代码。

### 5.1 前端项目结构

```
frontend/
├── public/
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── pages/
│   │   ├── Detection/           # 实时检测主页
│   │   │   ├── index.tsx
│   │   │   ├── CameraView.tsx   # 摄像头视图
│   │   │   ├── EmotionPanel.tsx # 情绪结果展示
│   │   │   └── StatsBar.tsx     # 实时统计
│   │   ├── Dataset/             # 数据集信息
│   │   │   ├── index.tsx
│   │   │   └── DistributionChart.tsx
│   │   └── Results/             # 实验结果展示
│   │       ├── index.tsx
│   │       ├── ConfusionMatrix.tsx
│   │       └── PerformanceChart.tsx
│   ├── components/              # 通用组件
│   │   ├── Layout/              # 布局组件
│   │   └── Charts/              # 图表封装
│   ├── services/                # API 调用层
│   │   ├── client.ts            # axios 实例 + 拦截器
│   │   ├── inference.ts         # 推理接口
│   │   ├── dataset.ts           # 数据集接口
│   │   └── stats.ts             # 统计接口
│   ├── hooks/                   # 自定义 Hooks
│   │   ├── useCamera.ts         # 摄像头管理
│   │   ├── useMediaPipe.ts      # MediaPipe 检测
│   │   ├── useGeoFeatures.ts    # 几何特征计算
│   │   └── useEmotionInference.ts # 推理请求
│   ├── utils/
│   │   ├── landmarks.ts         # 关键点索引定义、归一化
│   │   ├── geometry.ts          # EAR/MAR/曲率计算
│   │   └── image.ts             # Canvas 图像裁剪
│   ├── types/                   # TypeScript 类型
│   │   └── index.ts
│   └── constants/
│       └── emotions.ts          # 情绪类别定义、颜色映射
├── package.json
├── tsconfig.json
├── vite.config.ts
└── index.html
```

### 5.2 前端 TypeScript 类型定义（由 API 文档生成）

```typescript
// src/types/index.ts

export type EmotionLabel = 'angry' | 'disgust' | 'fear' | 'happy' | 'sad' | 'surprise' | 'neutral';

// 推理请求
export interface InferenceRequest {
  face_image: string;
  geo_features: GeoFeatures;
}

export interface GeoFeatures {
  ear: number;
  mar: number;
  eyebrow_eye_dist: number;
  mouth_curvature: number;
  nasolabial_depth: number;
}

// 推理响应
export interface InferenceResponse {
  emotions: Record<EmotionLabel, number>;
  prediction: EmotionLabel;
  confidence: number;
  inference_time_ms: number;
}

// 数据集信息
export interface DatasetInfo {
  name: string;
  path: string;
  total_samples: number;
  image_size: string;
  num_classes: number;
  classes: EmotionLabel[];
}

export interface ClassDistribution {
  classes: Array<{ name: EmotionLabel; count: number; percentage: number }>;
}

// 分类性能
export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
}

export interface PerformanceStats {
  metrics: Record<EmotionLabel, ClassMetrics>;
  macro_avg: ClassMetrics;
  weighted_avg: ClassMetrics;
}

// 混淆矩阵
export interface ConfusionMatrix {
  labels: EmotionLabel[];
  matrix: number[][];
}

// 延迟统计
export interface LatencyStats {
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  fps: number;
}

// 健康检查
export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  device: string;
  uptime_seconds: number;
}
```

### 5.3 API 服务层（由 API 文档生成）

```typescript
// src/services/client.ts
import axios from 'axios';

export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  timeout: 30000,
});

// src/services/inference.ts
import { apiClient } from './client';
import type { InferenceRequest, InferenceResponse } from '@/types';

export async function predictEmotion(req: InferenceRequest): Promise<InferenceResponse> {
  const { data } = await apiClient.post('/api/v1/inference/emotion', req);
  return data;
}

export function connectInferenceStream() {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  return new WebSocket(`${wsUrl}/ws/v1/inference/stream`);
}

// src/services/dataset.ts
import type { DatasetInfo, ClassDistribution } from '@/types';

export async function getDatasetInfo(): Promise<DatasetInfo> {
  const { data } = await apiClient.get('/api/v1/dataset/info');
  return data;
}

export async function getClassDistribution(): Promise<ClassDistribution> {
  const { data } = await apiClient.get('/api/v1/dataset/class-distribution');
  return data;
}

// src/services/stats.ts
import type { PerformanceStats, ConfusionMatrix, LatencyStats } from '@/types';

export async function getPerformance(): Promise<PerformanceStats> {
  const { data } = await apiClient.get('/api/v1/stats/performance');
  return data;
}

export async function getConfusionMatrix(): Promise<ConfusionMatrix> {
  const { data } = await apiClient.get('/api/v1/stats/confusion-matrix');
  return data;
}

export async function getLatency(): Promise<LatencyStats> {
  const { data } = await apiClient.get('/api/v1/stats/latency');
  return data;
}

// src/services/system.ts
import type { HealthStatus } from '@/types';

export async function getHealth(): Promise<HealthStatus> {
  const { data } = await apiClient.get('/health');
  return data;
}
```

### 5.4 实时检测流程（前端侧）

```
1. useCamera:  navigator.mediaDevices.getUserMedia → 获取视频流
2. useMediaPipe:
   - 初始化 FaceMesh 模型
   - 每帧输入 video 元素 → 输出 468 个关键点
3. useGeoFeatures:
   - 关键点归一化（双眼中心为原点，眼距为缩放因子）
   - 计算 EAR / MAR / 眉眼间距 / 嘴角曲率 / 鼻唇沟深度
4. useEmotionInference:
   - Canvas 裁剪面部区域 → 缩放至 224x224 → JPEG base64
   - 调用 predictEmotion() 或 WebSocket 发送
   - 接收 InferenceResponse
5. CameraView:
   - Canvas 叠加绘制 468 关键点网格
   - 叠加情绪标签、置信度、FPS
6. EmotionPanel:
   - 七类情绪概率柱状图/仪表盘
   - 情绪历史趋势
```

### 5.5 前端页面布局

```
┌─────────────────────────────────────────────────────┐
│  Logo  │  实时检测  │  数据集  │  实验结果            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐ │
│  │                     │  │  情绪概率分布          │ │
│  │  摄像头实时画面      │  │  ████████░ happy 82%  │ │
│  │  + 关键点叠加        │  │  ████░░░░ neutral 8%  │ │
│  │  + 情绪标签叠加      │  │  ...                  │ │
│  │                     │  │                       │ │
│  │  FPS: 15            │  │  ───────────────      │ │
│  │  预测: Happy (82%)  │  │  情绪历史趋势          │ │
│  │                     │  │  (最近30秒折线图)      │ │
│  └─────────────────────┘  └──────────────────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.6 给 Figma AI 的 Prompt 建议

使用 Figma AI / v0 / Lovable 生成前端时，提供以下信息：

1. **API 接口文档**：即本节 4.3 完整文档（请求/响应 JSON 示例）
2. **TypeScript 类型定义**：第 5.2 节 `types/index.ts`
3. **API 服务层**：第 5.3 节 `services/` 代码
4. **页面布局示意**：第 5.5 节的 ASCII 布局图
5. **UI 组件库**：指定使用 Ant Design

---

## 6. 数据库设计

训练通过 CLI 脚本执行，结果直接保存为 JSON 文件，无需复杂的数据库表。
后端读取这些 JSON 文件并通过 API 返回给前端。

### 6.1 实验结果存储

```
experiments/
└── <task_name>/
    ├── config.json           # 训练超参数
    ├── metrics.json          # 每个 epoch 的 loss/acc
    ├── performance.json      # Precision/Recall/F1 per class
    ├── confusion_matrix.json # 混淆矩阵
    └── best_model.pth        # 最优权重（软链接到 checkpoints/）
```

**performance.json 格式**：
```json
{
  "metrics": {
    "angry":    { "precision": 0.61, "recall": 0.58, "f1": 0.59 },
    "disgust":  { "precision": 0.45, "recall": 0.51, "f1": 0.48 },
    "fear":     { "precision": 0.38, "recall": 0.44, "f1": 0.41 },
    "happy":    { "precision": 0.85, "recall": 0.79, "f1": 0.82 },
    "sad":      { "precision": 0.55, "recall": 0.52, "f1": 0.53 },
    "surprise": { "precision": 0.71, "recall": 0.68, "f1": 0.69 },
    "neutral":  { "precision": 0.67, "recall": 0.63, "f1": 0.65 }
  },
  "macro_avg":    { "precision": 0.60, "recall": 0.59, "f1": 0.60 },
  "weighted_avg": { "precision": 0.68, "recall": 0.68, "f1": 0.68 }
}
```

### 6.2 SQLite（可选）

如需记录推理日志、用户会话等运行时数据，可使用 SQLite：

| 表名 | 用途 |
|------|------|
| `inference_log` | 推理请求记录（时间、延迟、预测结果） |
| `latency_stats` | 延迟聚合统计（用于 `/api/v1/stats/latency`） | |

---

## 7. 统一配置管理 — 本地到服务器的无缝切换

### 7.1 设计理念

整个系统使用 **一份代码 + 一份统一配置文件**，通过 `profile` 字段区分
`local` / `server` 两种环境。所有环境差异（路径、设备、端口等）全部收口
到这一个文件中，代码中 **不硬编码任何环境相关的路径或参数**。

### 7.2 配置文件定义

配置文件位于项目根目录：`config.yaml`

```yaml
# ============================================================
# config.yaml — 统一配置文件（本地/服务器共用）
# ============================================================

# 当前激活的环境: "local" | "server"
active_profile: local

profiles:
  # ──────────── 本地开发环境 ────────────
  local:
    app:
      host: "127.0.0.1"
      port: 8000
      debug: true
      cors_origins: ["*"]

    model:
      device: "cpu"                    # 本地无 GPU
      checkpoint_path: "backend/checkpoints/best_model.pth"
      # 如本地暂无权重，可设为 null，推理接口返回 mock 结果
      auto_download: false

    dataset:
      affectnet_path: null             # 本地无完整数据集
      mock_data_path: "backend/datasets/mock"   # 用 mock 数据验证流程

    database:
      url: "sqlite:///./backend/data/emotion.db"

    training:
      # 本地仅做 smoke test
      epochs: 2
      batch_size: 8
      use_mock: true                   # 使用 mock 数据跑通流程

    inference:
      smoothing_window: 5
      max_fps: 10                      # 本地 CPU 降低目标帧率

    logging:
      level: "DEBUG"
      file: null                       # 本地仅输出到控制台

  # ──────────── 服务器生产环境 ────────────
  server:
    app:
      host: "0.0.0.0"
      port: 8000
      debug: false
      cors_origins: ["https://your-domain.com"]

    model:
      device: "cuda"                   # 服务器有 GPU
      checkpoint_path: "backend/checkpoints/best_model.pth"
      auto_download: false

    dataset:
      affectnet_path: "/mnt/affectnet"  # 挂载的数据集目录
      mock_data_path: null

    database:
      url: "sqlite:///./backend/data/emotion.db"

    training:
      epochs: 50
      batch_size: 64
      use_mock: false

    inference:
      smoothing_window: 5
      max_fps: 30

    logging:
      level: "INFO"
      file: "logs/app.log"
```

### 7.3 前端配置 — Vite 环境变量注入

前端配置通过 `deploy_setup.sh` 脚本在构建时自动注入，无需手动修改：

```bash
# scripts/deploy_setup.sh 中的逻辑（概念性）
PROFILE="${DEPLOY_PROFILE:-server}"
if [ "$PROFILE" = "server" ]; then
  # 从 config.yaml 读取服务器域名，写入前端 .env.production
  DOMAIN=$(yq e '.profiles.server.app.cors_origins[0]' config.yaml | sed 's|https://||')
  echo "VITE_API_BASE_URL=https://${DOMAIN}/api"  > frontend/.env.production
  echo "VITE_WS_URL=wss://${DOMAIN}/ws"            >> frontend/.env.production
  echo "VITE_INFERENCE_MODE=websocket"             >> frontend/.env.production
fi
```

本地开发时，前端使用 `.env.development`（已提交到 git），指向 localhost：

```env
# frontend/.env.development（本地开发，已提交到 git）
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_INFERENCE_MODE=http
```

### 7.4 后端配置加载逻辑

```python
# backend/app/config.py (概念性设计)
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # 项目根目录
CONFIG_FILE = ROOT / "config.yaml"

def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    profile_name = cfg["active_profile"]  # "local" 或 "server"
    profile = cfg["profiles"][profile_name]
    return {**profile, "_profile": profile_name, "_root": ROOT}
```

代码中所有路径均通过配置解析：
```python
cfg = load_config()
checkpoint = cfg["_root"] / cfg["model"]["checkpoint_path"]
device = cfg["model"]["device"]  # "cpu" 或 "cuda"
```

### 7.5 脚本如何读取配置

三个部署脚本均读取同一个 `config.yaml`，保持一致性：

```bash
# scripts/start_server.sh 中的逻辑（概念性）
PROFILE=$(yq e '.active_profile' config.yaml)
DEVICE=$(yq e ".profiles.${PROFILE}.model.device" config.yaml)
MODEL_PATH=$(yq e ".profiles.${PROFILE}.model.checkpoint_path" config.yaml)
PORT=$(yq e ".profiles.${PROFILE}.app.port" config.yaml)
EPOCHS=$(yq e ".profiles.${PROFILE}.training.epochs" config.yaml)

# 自动设置环境变量
export DEVICE=$DEVICE
export MODEL_PATH=$MODEL_PATH
```

### 7.6 无缝切换流程

```
本地开发:
  1. 代码写好后，config.yaml 中 active_profile: local
  2. 直接运行后端 uvicorn、前端 vite dev server
  3. 所有路径、设备自动匹配本地环境

部署到服务器:
  1. git push → 服务器 git pull
  2. 修改 config.yaml: active_profile: server
  3. ./scripts/deploy_setup.sh   → 读取 config.yaml 安装环境 + 构建前端
  4. ./scripts/run_training.sh   → 读取 config.yaml 启动训练
  5. ./scripts/start_server.sh   → 读取 config.yaml 启动服务

从服务器下载权重到本地验证:
  1. scp server:backend/checkpoints/best_model.pth backend/checkpoints/
  2. 本地 active_profile 仍为 local，直接加载权重进行 CPU 推理测试
```

### 7.7 配置文件版本管理

```
config.yaml          → 提交到 git（包含 local 和 server 两个 profile 的完整定义）
.env.development     → 提交到 git（前端本地开发用）
.env.production      → 不提交（.gitignore），由 deploy_setup.sh 动态生成
logs/                → 不提交（.gitignore）
data/                → 不提交（.gitignore），SQLite 数据库文件
checkpoints/*.pth    → 不提交（.gitignore），模型权重体积大，独立管理
```

### 7.8 配置热切换

支持运行时通过 API 切换 profile（仅限非训练参数）：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/config` | 查看当前配置 |
| POST | `/api/v1/config/profile` | 切换 profile（重启服务生效） |

### 7.9 依赖工具

- **yq**：脚本中解析 YAML（Ubuntu: `apt install yq` 或下载二进制）
- **PyYAML**：Python 端读取配置（已包含在 requirements 中）

---

## 8. 服务器部署与运维脚本设计

### 8.1 部署理念：一键脚本

系统交付后，在云服务器上的操作应简化为 **三个脚本**：

| 脚本 | 文件名 | 用途 | 执行时机 |
|------|--------|------|---------|
| 环境部署 | `deploy_setup.sh` | 安装依赖、构建前端、配置 Nginx | 首次部署 / 代码更新 |
| 模型训练 | `run_training.sh` | 启动训练任务（支持后台运行） | 需要训练/重新训练时 |
| 服务启动 | `start_server.sh` | 启动前后端服务 | 每次服务器重启后 |

### 8.2 脚本一：`deploy_setup.sh` — 环境部署

**功能**：从零配置服务器运行环境，执行一次后无需重复。

```
deploy_setup.sh 执行流程：

  1. 系统检查
     ├── 检测 OS 版本 (Ubuntu 22.04 LTS)
     ├── 检测 NVIDIA GPU + CUDA 驱动
     ├── 检测 Python 版本 (>= 3.10)
     └── 检测 Node.js 版本 (>= 18)

  2. 后端环境
     ├── 创建 Python venv
     ├── 安装 requirements-prod.txt (PyTorch GPU + FastAPI + Gunicorn)
     └── 创建必要目录 (checkpoints/, data/, experiments/, logs/)

  3. 前端构建
     ├── npm install
     ├── 注入生产环境变量 (VITE_API_BASE_URL / VITE_WS_URL)
     └── npm run build → 产物输出 dist/

  4. Nginx 配置
     ├── 复制 nginx.conf 到 /etc/nginx/sites-available/
     ├── 配置 HTTPS 证书路径 (Let's Encrypt)
     ├── 静态文件指向 → 前端 dist/
     ├── /api/ 代理 → Gunicorn (localhost:8000)
     ├── /ws/ 代理 → WebSocket 到后端
     └── 启动/重载 Nginx

  5. 验证
     ├── curl localhost:8000/health 检查后端
     └── curl localhost/ 检查前端
```

**前置条件**：
- 服务器已安装 Ubuntu 22.04 LTS
- 已安装 NVIDIA 驱动和 CUDA Toolkit
- 代码已部署到服务器（git clone 或 scp 上传）
- 域名已解析到服务器 IP（用于 HTTPS）

### 8.3 脚本二：`run_training.sh` — 启动训练

**功能**：在 GPU 服务器上启动模型训练，支持后台运行和断点续训。

```
run_training.sh 执行流程：

  1. 参数解析
     ├── --dataset-path   : AffectNet 数据集路径 (默认: /mnt/affectnet)
     ├── --epochs         : 训练轮数 (默认: 50)
     ├── --batch-size     : 批次大小 (默认: 64)
     ├── --task-name      : 任务名称 (默认: auto_$(date))
     ├── --resume         : 从检查点恢复训练 (可选)
     └── --dry-run        : 仅验证环境，不实际训练

  2. 环境验证
     ├── 检查 GPU 可用性 (nvidia-smi)
     ├── 检查数据集路径是否存在
     ├── 检查 Python 依赖
     └── 检查磁盘空间

  3. 启动训练
     ├── 激活 Python venv
     ├── nohup 后台运行训练脚本
     │   python -m app.services.training \
     │     --dataset-path $DATASET_PATH \
     │     --epochs $EPOCHS \
     │     --batch-size $BATCH_SIZE \
     │     --task-name $TASK_NAME \
     │     > logs/training_${TASK_NAME}.log 2>&1 &
     └── 记录 PID 到 logs/training_${TASK_NAME}.pid

  4. 状态输出
     ├── 打印训练日志路径
     ├── 打印任务 PID
     └── 提示: tail -f logs/training_xxx.log 实时查看进度
```

**使用示例**：
```bash
# 首次完整训练
./run_training.sh --dataset-path /mnt/affectnet --task-name experiment_v1

# 从断点恢复
./run_training.sh --resume checkpoints/best_model.pth --task-name experiment_v1_resume

# 快速验证环境
./run_training.sh --dry-run
```

**训练监控**：
- 通过 `tail -f logs/training_xxx.log` 查看终端日志
- 通过前端训练管理页面实时查看 Loss/Accuracy 曲线（SSE 推送）
- 训练完成后，最优权重自动保存至 `checkpoints/best_model.pth`

### 8.4 脚本三：`start_server.sh` — 启动服务

**功能**：启动后端 Gunicorn + Nginx 前端服务，支持优雅停止和重启。

```
start_server.sh 执行流程：

  1. 前置检查
     ├── 检查 Python venv 是否存在
     ├── 检查模型权重文件 (checkpoints/best_model.pth)
     ├── 检查前端构建产物 (frontend/dist/)
     └── 检查 8000/443 端口占用

  2. 启动后端
     ├── 激活 Python venv
     ├── 设置环境变量 (DEVICE, MODEL_PATH, DATABASE_URL)
     └── gunicorn -c gunicorn_conf.py app.main:app
         ├── 4 workers (可配置)
         ├── uvicorn.workers.UvicornWorker
         └── 后台运行, PID → logs/gunicorn.pid

  3. 启动前端
     ├── 检查 Nginx 是否已运行
     ├── 如未运行: systemctl start nginx
     └── 如已运行: systemctl reload nginx (更新静态文件)

  4. 健康检查
     ├── 等待 3 秒
     ├── curl http://localhost:8000/health
     └── curl http://localhost/
         └── 打印访问地址: https://<服务器IP>/
```

**子命令**：
```bash
./start_server.sh          # 启动服务
./start_server.sh stop     # 停止服务 (kill gunicorn, 不关闭 nginx)
./start_server.sh restart  # 重启服务
./start_server.sh status   # 查看服务状态 (gunicorn pid, nginx status)
./start_server.sh logs     # 查看后端日志 (tail -f logs/gunicorn.log)
```

### 8.5 项目根目录脚本结构

所有脚本放置在项目根目录 `scripts/` 下：

```
scripts/
├── deploy_setup.sh      # 环境部署（首次执行）
├── run_training.sh      # 启动训练
├── start_server.sh      # 启动/管理 服务
├── config.yaml          # 统一配置（本地/服务器共用，已提交 git）
└── utils/
    ├── check_gpu.sh     # GPU 环境检查
    ├── check_deps.sh    # 依赖检查
    └── health_check.sh  # 服务健康检查
```

所有脚本从 `config.yaml` 读取当前激活的 profile 参数，无需额外配置 env.sh。

### 8.6 配置驱动机制

所有脚本和后端代码通过 `config.yaml` 驱动，实现零配置切换：

```bash
# 本地开发
# config.yaml 中 active_profile: local
# 直接启动，无需额外配置

# 服务器部署
# config.yaml 中 active_profile: server
# 仅需修改 affectnet_path 为实际挂载路径
./scripts/deploy_setup.sh     # 自动读取 server profile 参数
./scripts/run_training.sh     # 自动读取训练参数
./scripts/start_server.sh     # 自动读取模型路径、设备信息
```

修改服务器配置只需编辑 `config.yaml` 中 `profiles.server` 下的对应字段，脚本自动适配。

### 8.7 完整操作流程

#### 首次部署（训练 + 推理）
```
1. git clone 项目到服务器
2. 修改 config.yaml: active_profile: server，设置 affectnet_path
3. ./scripts/deploy_setup.sh          → 读取 config.yaml 安装环境
4. ./scripts/run_training.sh          → 读取 config.yaml 启动训练
5. 等待训练完成（约 1.5-3.5 小时）
6. ./scripts/start_server.sh          → 读取 config.yaml 启动服务
7. 浏览器访问 https://<服务器IP>/     → 开始使用
```

#### 仅更新代码（模型不变）
```
1. git pull
2. ./scripts/deploy_setup.sh          → 重新构建前端 + 更新后端依赖
3. ./scripts/start_server.sh restart  → 重启服务
```

#### 仅重新训练（代码不变）
```
1. ./scripts/run_training.sh --dataset-path /mnt/affectnet --task-name v2
```

#### 日常启停
```
1. ./scripts/start_server.sh          → 启动
2. ./scripts/start_server.sh stop     → 停止
```

### 8.8 租赁平台推荐

| 平台 | 推荐 GPU | 适用场景 | 参考价格 |
|------|---------|---------|---------|
| AutoDL | RTX 4090 / RTX 3090 | 训练 + 部署 | 按小时计费 |
| 恒源云 | RTX 4090 / A10 | 训练 + 部署 | 按小时计费 |
| AWS | g5.xlarge (A10G) | 生产部署 | 按小时计费，较高 |
| 阿里云 | gn7i (A10) | 生产部署 | 按量/包月 |

**建议**：训练阶段租用按量计费的 GPU 实例（如 RTX 4090），训练完成后转为仅 CPU 推理实例或保留 GPU 实例用于实时推理。

### 8.9 推荐服务器配置

#### 训练阶段
- GPU：RTX 4090 24GB 或 A100 40GB
- CPU：8 核以上
- 内存：32GB+
- 存储：500GB+（AffectNet 数据集约 80GB + 训练产物）
- OS：Ubuntu 22.04 LTS
- 预计训练时间：约 3.5 小时（RTX 3060 基准，4090 约 1.5 小时）

#### 推理部署阶段（可选降级）
- GPU：不需要（模型仅 203K 参数，CPU 推理约 15 FPS）
- 如需 GPU：最便宜的 GPU 实例即可（如 T4 / A10）
- CPU：4 核+
- 内存：8GB+
- 存储：50GB（模型权重 + 数据库 + 日志）
- 带宽：5Mbps+（影响前端加载和推理图像传输）

### 8.10 部署架构

```
用户浏览器 ──HTTPS──▶ 云服务器 (公网 IP)
                         │
                    ┌────▼────┐
                    │ Nginx   │ :443 (HTTPS)
                    │         │
                    ├─────────┤
                    │ /       │ → 前端静态文件 (frontend/dist/)
                    │ /api/   │ → Gunicorn+Uvicorn (localhost:8000)
                    │ /ws/    │ → WebSocket 代理到后端
                    └─────────┘
                         │
                    ┌────▼────┐
                    │ FastAPI │ :8000
                    │ PyTorch │ (GPU/CPU)
                    └─────────┘
```

### 8.11 数据集上传策略

AffectNet 数据集约 80GB，上传方式：

1. **云盘挂载**：部分平台支持 OSS/NAS 挂载，直接读取
2. **预装镜像**：选择预装常见数据集的镜像（如 AutoDL 部分镜像已含 AffectNet）
3. **手动上传**：通过 scp 或平台文件管理工具上传（大文件建议压缩后传输）
4. **训练时下载**：如果平台提供了数据集镜像，直接使用

### 8.12 HTTPS 要求

**重要**：浏览器摄像头 API (`getUserMedia`) 必须在 HTTPS 环境下才能使用（localhost 除外）。

- `deploy_setup.sh` 中集成 Let's Encrypt certbot 自动申请证书
- Nginx 配置 HTTPS 终端，后端保持 HTTP 内部通信
- 证书自动续期（cron 定时任务）

### 8.13 成本估算

| 阶段 | 时长 | GPU | 预估成本 |
|------|------|-----|---------|
| 本地开发 | 持续 | 无 | ¥0 |
| 完整训练 | 2-4 小时 | RTX 4090 | ¥10-30 |
| 推理部署 | 按需 | CPU 或低端 GPU | ¥1-5/小时 |

---

## 9. 开发阶段规划

### Phase 0: 开发环境准备 + 数据集
- 本地 Python 虚拟环境配置（CPU only）
- 前端 Node.js 环境配置
- Mock 数据集创建（用于本地开发验证）
- 前后端基础联调通路
- **AffectNet-7 数据集下载与准备**：
  - 推荐从 Kaggle 下载 AffectNet Aligned 子集（~261 MB，已做人脸对齐）
  - 也可注册 AffectNet 官方数据集获取完整数据（~287K 训练集 + ~4K 测试集）
  - 代码自动兼容两种目录格式：数字文件夹（`0/`~`6/`）和命名文件夹（`0_angry/`~`6_neutral/`）
  - 上传至云服务器挂载目录（或通过平台预装镜像获取）
  - 验证数据集完整性与目录结构

### Phase 1: 基础架构搭建
- 后端 FastAPI 项目骨架、路由框架
- 前端 React 项目骨架、路由与布局
- 数据库模型与迁移
- Vite 代理配置（本地开发避免 CORS）
- 前后端联调基础接口

### Phase 2: 模型实现（本地 CPU 开发）
- PyTorch CNN 网络定义（与论文表 2 一致）
- 几何特征编码器
- 特征融合模块
- 模型权重加载/保存
- 使用 Mock 数据（100 张样本）验证数据流

### Phase 3: 训练系统 + 云端训练
- 训练流程实现（`train.py`，加权交叉熵、Adam、余弦衰减、早停）
- 数据增强管线
- 训练结果保存为 JSON（metrics/performance/confusion_matrix）
- `run_training.sh` 脚本编写
- **云端 GPU 环境配置**：租用服务器、上传数据集、部署代码
- **执行完整训练**：在云端 GPU 上完成 50 epoch 训练
- **模型权重**：训练产物自动保存至 `checkpoints/`

### Phase 4: 推理服务
- HTTP 推理 API 实现
- WebSocket 流式推理 API
- 模型热加载
- 推理延迟监控
- 本地 CPU 推理验证（加载云端训练权重）

### Phase 5: 前端实时检测
- 摄像头采集与 MediaPipe Face Mesh 集成
- 几何特征计算（EAR/MAR/眉眼间距/嘴角曲率/鼻唇沟深度）
- 面部裁剪与 base64 编码
- 推理结果接收与展示
- 关键点网格叠加绘制
- 情绪标签实时叠加

### Phase 6: 数据分析与可视化
- 混淆矩阵展示
- 各类别 Precision/Recall/F1 图表
- 消融实验对比
- 延迟分布统计

### Phase 7: 云端部署脚本
- 编写 `deploy_setup.sh`：环境安装、前端构建、Nginx 配置
- 编写 `run_training.sh`：训练启动、后台运行、断点续训
- 编写 `start_server.sh`：服务启动/停止/重启/状态管理
- 编写 `utils/`：健康检查、依赖检查等辅助脚本
- 在云服务器上完整测试三个脚本的执行流程
- HTTPS 证书自动申请与续期配置

### Phase 8: 优化与打磨
- 推理性能优化（模型量化可选）
- 前端性能优化（帧率平滑、请求节流）
- 错误处理与边界情况
- 文档完善

---

## 10. 关键技术难点与解决方案

### 10.1 前端 MediaPipe 性能
- **问题**：浏览器端 MediaPipe 可能不如 Python 版稳定
- **方案**：使用 @mediapipe/tasks-vision 新 API，WASM 后端加速；提供降级方案（后端检测）

### 10.2 实时性保障
- **问题**：HTTP 请求往返延迟影响帧率（尤其跨地域网络）
- **方案**：优先使用 WebSocket 长连接；前端请求节流（target FPS 控制）；结果移动平均平滑；选择与用户地理位置相近的云服务器降低延迟

### 10.3 图像传输优化
- **问题**：base64 编码增加数据量，云服务器带宽有限时可能成为瓶颈
- **方案**：224x224 JPEG 质量设为 0.85 平衡质量与大小（单张约 8-15KB）；WebSocket 模式可尝试二进制传输；根据网络状况动态调整目标 FPS

### 10.4 类别不均衡
- **问题**：AffectNet-7 高兴类占 33%，恐惧/厌恶不足 3%
- **方案**：严格按论文公式（4.2）计算类别权重，使用加权交叉熵损失

### 10.5 HTTPS 与摄像头访问
- **问题**：浏览器 `getUserMedia` 要求 HTTPS 环境，localhost 除外
- **方案**：本地开发使用 localhost + Vite 代理（无需 HTTPS）；云端部署使用 Let's Encrypt 证书 + Nginx HTTPS 终端

### 10.6 无本地 GPU 的开发流程
- **问题**：无法在本地验证训练完整流程
- **方案**：
  - `config.yaml` 中 `active_profile: local`，使用 mock 数据跑通流程
  - 模型仅 203K 参数，CPU 推理完全满足接口测试需求
  - 切换到服务器时仅需改 `config.yaml` 中 `active_profile: server`
  - 所有脚本和后端代码自动适配，无需手动修改路径或参数
  - 训练产物（.pth）直接保存在 `checkpoints/`，`start_server.sh` 自动加载

### 10.7 云服务器成本控制
- **问题**：GPU 实例按小时计费，持续运行成本高
- **方案**：
  - 训练阶段：租用 GPU 实例 2-4 小时完成训练后立即释放
  - 推理阶段：使用 CPU 实例（模型轻量，CPU 推理 15 FPS 满足需求）
  - 如需要 GPU 推理，选择最便宜的 GPU 实例（T4/A10 即可）
  - 前端 MediaPipe 在客户端运行，进一步降低服务端压力

---

## 11. 文件交付清单

| 文件 | 说明 | 提交到 git |
|------|------|-----------|
| `docs/00_setup/ch_RP_updated.pdf` | 原始论文 | 是 |
| `docs/01_architecture_plan.md` | 本架构方案 | 是 |
| `config.yaml` | 统一配置文件（local + server profiles） | 是 |
| `backend/` | 后端代码目录 | 是 |
| `frontend/` | 前端代码目录 | 是 |
| `scripts/` | 部署脚本（3 核心 + utils） | 是 |
| `frontend/.env.development` | 前端本地开发环境变量 | 是 |
| `frontend/.env.production` | 前端生产环境变量 | **否**（构建时动态生成） |
| `checkpoints/*.pth` | 模型权重 | **否**（.gitignore，独立管理） |
| `data/*.db` | SQLite 数据库 | **否**（.gitignore） |
| `logs/` | 日志文件 | **否**（.gitignore） |

### 脚本速查

| 操作 | 命令 |
|------|------|
| **首次部署** | `./scripts/deploy_setup.sh` |
| **启动训练** | `./scripts/run_training.sh --dataset-path /mnt/affectnet --task-name v1` |
| **启动服务** | `./scripts/start_server.sh` |
| **停止服务** | `./scripts/start_server.sh stop` |
| **查看状态** | `./scripts/start_server.sh status` |
| **查看日志** | `./scripts/start_server.sh logs` |
