# Emotion Detection System

基于 MediaPipe + CNN 的多模态实时情绪检测系统。

## 系统架构

- **前端**: React + TypeScript + Vite + Ant Design + ECharts
- **后端**: Python + FastAPI + PyTorch
- **人脸检测**: MediaPipe Face Mesh (468 landmarks, 前端运行)
- **情绪分类**: EmotionCNN + GeoEncoder + FusionClassifier (7类情绪)

## 7 类情绪

| 标签 | 中文 | 颜色 |
|------|------|------|
| angry | 愤怒 | 红色 |
| disgust | 厌恶 | 黄绿 |
| fear | 恐惧 | 紫色 |
| happy | 高兴 | 金色 |
| sad | 悲伤 | 蓝色 |
| surprise | 惊讶 | 青色 |
| neutral | 中性 | 灰色 |

## 快速开始

### 本地开发

```bash
# 后端
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# 前端
cd frontend
npm install
npm run dev
```

### 训练模型

```bash
# 使用 mock 数据（本地测试）
./scripts/run_training.sh --use-mock --task-name test_v1

# 使用 Kaggle AffectNet Aligned 数据集（推荐）
pip install kaggle
kaggle datasets download -d yakhyokhuja/affectnetaligned
unzip affectnetaligned.zip -d backend/datasets/affectnet_aligned
./scripts/run_training.sh --dataset-path backend/datasets/affectnet_aligned --task-name v1

# 使用完整 AffectNet 数据集（需官方申请）
./scripts/run_training.sh --dataset-path /path/to/affectnet7 --task-name v1
```

> 代码自动兼容两种数据集目录格式：Kaggle 数字文件夹（`0/`~`6/`）和项目命名文件夹（`0_angry/`~`6_neutral/`），标签映射自动处理。

### 云端部署

```bash
# 1. 修改 config.yaml: active_profile: server
# 2. 部署环境
./scripts/deploy_setup.sh

# 3. 训练模型
./scripts/run_training.sh --task-name v1

# 4. 启动服务
./scripts/start_server.sh

# 5. 日常管理
./scripts/start_server.sh status   # 查看状态
./scripts/start_server.sh stop     # 停止
./scripts/start_server.sh restart  # 重启
./scripts/start_server.sh logs     # 查看日志
```

## API 接口

启动后端后访问 http://localhost:8000/docs 查看完整 Swagger 文档。

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/api/v1/config` | 获取配置信息 |
| POST | `/api/v1/inference/emotion` | 单帧情绪推理 |
| WS | `/ws/v1/inference/stream` | WebSocket 流式推理 |
| GET | `/api/v1/dataset/info` | 数据集信息 |
| GET | `/api/v1/dataset/class-distribution` | 类别分布 |
| GET | `/api/v1/stats/performance` | 分类性能 |
| GET | `/api/v1/stats/confusion-matrix` | 混淆矩阵 |
| GET | `/api/v1/stats/latency` | 延迟统计 |

## 项目结构

```
config-art-master/
├── backend/
│   ├── app/
│   │   ├── api/routes/       # API 路由
│   │   ├── api/schemas/      # 请求/响应模型
│   │   ├── models/           # CNN + GeoEncoder + Fusion
│   │   ├── services/         # 模型服务、训练引擎
│   │   └── utils/            # 图像处理、特征计算、延迟追踪
│   ├── config.yaml           # 统一配置（local/server）
│   ├── train.py              # 训练入口
│   └── gunicorn_conf.py      # 生产 WSGI 配置
├── frontend/
│   ├── src/
│   │   ├── pages/Detection/  # 实时检测页面
│   │   ├── pages/Dataset/    # 数据集可视化
│   │   ├── pages/Results/    # 实验结果可视化
│   │   ├── hooks/            # useCamera, useMediaPipe, etc.
│   │   ├── services/         # API 客户端
│   │   └── components/       # 布局、错误边界
│   └── vite.config.ts
├── scripts/
│   ├── deploy_setup.sh       # 一键部署
│   ├── run_training.sh       # 训练脚本
│   ├── start_server.sh       # 服务管理
│   └── utils/                # 辅助检查脚本
└── docs/                     # 项目文档
```

## 配置

编辑 `backend/config.yaml` 切换 `active_profile`：

- `local`: 本地开发（CPU, mock 数据, debug 模式）
- `server`: 服务器生产（GPU, AffectNet 数据集, Gunicorn + Nginx）
