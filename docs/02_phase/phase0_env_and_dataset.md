# Phase 0: 开发环境准备 + 数据集

## 1. 目标

搭建本地开发和服务器训练所需的基础环境，下载并准备好 AffectNet-7 数据集。

## 2. 本地环境

### 2.1 Python 环境

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/macOS

# 安装基础依赖（后续创建）
pip install -r requirements.txt
```

**requirements.txt（Phase 0 基础版）**：
```
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.0.0
pyyaml>=6.0
numpy>=1.24.0
opencv-python>=4.8.0
Pillow>=10.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

### 2.2 Node.js 环境

```bash
node -v   # >= 18
npm -v    # >= 9
```

## 3. AffectNet-7 数据集

### 3.1 数据集获取

#### 方式一：Kaggle 下载（推荐）

使用 AffectNet Aligned 子集（已做人脸对齐，约 261 MB）：

```bash
# 安装 Kaggle CLI
pip install kaggle

# 配置凭证（从 https://www.kaggle.com/settings 获取 API Token）
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/kaggle.json

# 下载并解压
kaggle datasets download -d yakhyokhuja/affectnetaligned
unzip affectnetaligned.zip -d backend/datasets/affectnet_aligned
```

> 该数据集使用 AffectNet 标准编号文件夹（`0/`=Neutral, `1/`=Happy, ..., `6/`=Anger），
> 代码会自动识别格式并重映射到项目标签顺序。

#### 方式二：AffectNet 官方申请

1. 访问 http://mohammadmahoor.com/affectnet/ 注册账号
2. 申请 AffectNet-7 子集访问权限
3. 下载链接通过邮件发送

### 3.2 数据集规模

| 数据集 | 训练集 | 验证集 | 大小 | 说明 |
|--------|--------|--------|------|------|
| AffectNet Aligned (Kaggle) | ~25,000 | ~3,500 | 261 MB | 人脸对齐，数字文件夹 |
| AffectNet-7 官方 | 287,568 | 3,999 | ~80 GB | 完整数据集 |

### 3.3 数据集目录结构

代码自动支持两种文件夹命名格式：

**格式 A — 项目命名（mock 数据集使用）：**
```
affectnet7/
├── train/
│   ├── 0_angry/
│   ├── 1_disgust/
│   ├── 2_fear/
│   ├── 3_happy/
│   ├── 4_sad/
│   ├── 5_surprise/
│   └── 6_neutral/
└── val/
    ├── 0_angry/
    └── ...
```

**格式 B — AffectNet 标准编号（Kaggle 数据集使用）：**
```
affectnet_aligned/
├── train/
│   ├── 0/    # Neutral  → 映射到 neutral
│   ├── 1/    # Happy    → 映射到 happy
│   ├── 2/    # Sad      → 映射到 sad
│   ├── 3/    # Surprise → 映射到 surprise
│   ├── 4/    # Fear     → 映射到 fear
│   ├── 5/    # Disgust  → 映射到 disgust
│   ├── 6/    # Anger    → 映射到 angry
│   └── 7/    # Contempt → 自动跳过（7类不需要）
└── val/
    ├── 0/
    └── ...
```

> **标签重映射**由 `backend/app/utils/image.py` 中的 `AFFECTNET_TO_PROJECT` 字典自动处理，
> 无需手动重命名文件夹。

### 3.4 上传到云服务器

```bash
# 方式一：压缩后传输
tar czf affectnet_aligned.tar.gz affectnet_aligned/
scp affectnet_aligned.tar.gz user@server:/mnt/affectnet/
# 服务器端解压
ssh user@server "cd /mnt/affectnet && tar xzf affectnet_aligned.tar.gz"

# 方式二：在服务器上直接下载
ssh user@server
pip install kaggle
kaggle datasets download -d yakhyokhuja/affectnetaligned
unzip affectnetaligned.zip -d /mnt/affectnet/affectnet_aligned

# 方式三：使用平台 NAS/OSS 挂载（推荐，免传输）
# AutoDL 等平台支持将数据集直接挂载到指定目录
```

### 3.5 验证数据集

```bash
# Kaggle 格式（数字文件夹）
for dir in affectnet_aligned/train/*/; do
    echo "$(basename $dir): $(ls $dir | wc -l)"
done

# 项目格式（命名文件夹）
for dir in affectnet7/train/*/; do
    echo "$(basename $dir): $(ls $dir | wc -l)"
done
```

AffectNet Aligned (Kaggle) 预期输出：
```
0: ~7500   # Neutral
1: ~7500   # Happy
2: ~7500   # Sad
3: ~7500   # Surprise
4: ~7500   # Fear
5: ~7500   # Disgust
6: ~7500   # Anger
```

> 注意：Kaggle 子集经过平衡采样，各类数量基本相等。

## 4. Mock 数据集（本地开发用）

本地无完整数据集时，创建小规模 mock 数据验证代码流程：

```
backend/datasets/mock/
├── train/
│   ├── 0_angry/      # 每类放 15-20 张示例图片
│   ├── 1_disgust/
│   ├── 2_fear/
│   ├── 3_happy/
│   ├── 4_sad/
│   ├── 5_surprise/
│   └── 6_neutral/
└── val/
    ├── 0_angry/      # 每类放 5 张
    ├── ...
    └── 6_neutral/
```

可从 AffectNet 中每个类别抽取 20 张作为 mock，或使用任意人脸图片替代。

## 5. config.yaml 初始配置

```yaml
active_profile: local

profiles:
  local:
    app:
      host: "127.0.0.1"
      port: 8000
      debug: true
      cors_origins: ["*"]

    model:
      device: "cpu"
      checkpoint_path: "checkpoints/best_model.pth"
      auto_download: false

    dataset:
      affectnet_path: null          # 本地可设为 "datasets/affectnet_aligned"
      mock_data_path: "datasets/mock"

    database:
      url: "sqlite:///./data/emotion.db"

    training:
      epochs: 2
      batch_size: 8
      use_mock: true

    inference:
      smoothing_window: 5
      max_fps: 10

    logging:
      level: "DEBUG"
      file: null

  server:
    app:
      host: "0.0.0.0"
      port: 8000
      debug: false
      cors_origins: ["https://your-domain.com"]

    model:
      device: "cuda"
      checkpoint_path: "checkpoints/best_model.pth"
      auto_download: false

    dataset:
      affectnet_path: "/mnt/affectnet/affectnet7"
      mock_data_path: null

    database:
      url: "sqlite:///./data/emotion.db"

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

## 6. 服务器环境（Phase 0 预备）

在云服务器上预装：

```bash
# 安装 NVIDIA 驱动 + CUDA（通常平台镜像已包含）
nvidia-smi

# 安装系统依赖
apt update && apt install -y python3.10 python3.10-venv python3-pip nginx git curl yq

# 安装 Node.js（用于前端构建）
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs

# 验证
python3 --version   # >= 3.10
node --version      # >= 18
nvcc --version      # >= 12.0（如有 GPU）
```
