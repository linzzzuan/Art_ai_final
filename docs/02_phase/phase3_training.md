# Phase 3: 训练系统 + 云端训练

## 1. 目标

实现完整训练流程，在云端 GPU 服务器上完成 AffectNet-7 全量训练。

## 2. 训练脚本

### 2.1 `train.py` — 训练入口

```bash
# 本地 mock 验证
python train.py --use-mock --epochs 2

# 云端完整训练
python train.py \
  --dataset-path /mnt/affectnet/affectnet7 \
  --epochs 50 \
  --batch-size 64 \
  --task-name experiment_v1
```

### 2.2 训练主循环

**文件**: `backend/app/services/training_engine.py`

核心流程：
```
1. 加载配置（config.yaml 的 training section）
2. 初始化模型（EmotionCNN + GeoEncoder + FusionClassifier）
3. 创建 DataLoader（train/val 8:1:1 划分）
4. 创建优化器（Adam, lr=0.001）
5. 创建学习率调度器（CosineAnnealingLR, T_max=15）
6. 创建损失函数（加权交叉熵）
7. for epoch in range(epochs):
     train_one_epoch()
     validate()
     save_if_best()
     check_early_stopping()
8. 保存最终结果 JSON 到 experiments/<task_name>/
```

### 2.3 加权交叉熵损失

```python
def compute_class_weights(dataset):
    """w_k = N / (K * N_k)"""
    counts = dataset.get_class_counts()  # {0: 24882, 1: 3803, ...}
    N = sum(counts.values())
    K = len(counts)
    weights = [N / (K * counts[k]) for k in range(K)]
    return torch.FloatTensor(weights)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

### 2.4 早停机制

```python
class EarlyStopping:
    def __init__(self, patience=8):
        self.patience = patience
        self.best_acc = 0
        self.counter = 0

    def __call__(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
            return False  # 不触发
        self.counter += 1
        return self.counter >= self.patience  # 触发早停
```

### 2.5 训练结果保存

训练完成后保存至 `experiments/<task_name>/`：

**`metrics.json`**：
```json
[
  {"epoch": 1, "train_loss": 1.85, "val_loss": 1.92, "train_acc": 0.28, "val_acc": 0.25, "lr": 0.001},
  {"epoch": 2, "train_loss": 1.42, "val_loss": 1.55, "train_acc": 0.42, "val_acc": 0.38, "lr": 0.00098}
]
```

**`performance.json`**：
```json
{
  "metrics": {
    "angry": {"precision": 0.61, "recall": 0.58, "f1": 0.59},
    ...
  },
  "macro_avg": {"precision": 0.60, "recall": 0.59, "f1": 0.60},
  "weighted_avg": {"precision": 0.68, "recall": 0.68, "f1": 0.68}
}
```

**`confusion_matrix.json`**：
```json
{
  "labels": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
  "matrix": [[320, 45, ...], ...]
}
```

## 3. 部署脚本：run_training.sh

**文件**: `scripts/run_training.sh`

```bash
#!/bin/bash
set -e

# 读取 config.yaml
PROFILE=$(yq e '.active_profile' config.yaml)
DATASET_PATH=$(yq e ".profiles.${PROFILE}.dataset.affectnet_path" config.yaml)
EPOCHS=$(yq e ".profiles.${PROFILE}.training.epochs" config.yaml)
BATCH_SIZE=$(yq e ".profiles.${PROFILE}.training.batch_size" config.yaml)
DEVICE=$(yq e ".profiles.${PROFILE}.model.device" config.yaml)

# 默认值覆盖
TASK_NAME="experiment_$(date +%Y%m%d_%H%M)"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --task-name) TASK_NAME="$2"; shift 2;;
    --dataset-path) DATASET_PATH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --dry-run) echo "Environment OK"; exit 0;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done

# 环境检查
source scripts/utils/check_gpu.sh
source scripts/utils/check_deps.sh

# 创建日志目录
mkdir -p logs

# 后台启动训练
export DEVICE
export DATASET_PATH
nohup python train.py \
  --dataset-path "$DATASET_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --task-name "$TASK_NAME" \
  > "logs/training_${TASK_NAME}.log" 2>&1 &

echo $! > "logs/training_${TASK_NAME}.pid"
echo "Training started: $TASK_NAME (PID: $!)"
echo "Log: logs/training_${TASK_NAME}.log"
echo "Tail: tail -f logs/training_${TASK_NAME}.log"
```

## 4. 云端训练流程

### 4.1 服务器准备

```bash
# 1. clone 代码
git clone <repo_url> && cd <repo>

# 2. 切换配置
# 编辑 config.yaml: active_profile: server

# 3. 部署环境
./scripts/deploy_setup.sh

# 4. 验证 GPU
nvidia-smi

# 5. 验证数据集
ls /mnt/affectnet/affectnet7/train/
```

### 4.2 执行训练

```bash
./scripts/run_training.sh --task-name experiment_v1
# 或指定数据集路径
./scripts/run_training.sh --dataset-path /mnt/affectnet/affectnet7 --task-name v1
```

### 4.3 监控训练

```bash
# 方式一：查看终端日志
tail -f logs/training_experiment_v1.log

# 方式二：通过 API（Phase 4 实现后）
curl http://localhost:8000/api/v1/stats/performance
```

### 4.4 训练完成

产物自动保存：
- `checkpoints/best_model.pth` — 最优权重
- `experiments/experiment_v1/metrics.json` — 训练曲线
- `experiments/experiment_v1/performance.json` — 分类性能
- `experiments/experiment_v1/confusion_matrix.json` — 混淆矩阵

## 5. 预期训练指标（参考论文）

- 训练时间：~3.5h (RTX 3060) / ~1.5h (RTX 4090)
- 最佳验证准确率：~65.8%（第 24 epoch）
- 早停触发：~32 epoch（patience=8）
- 加权平均 F1：~0.68

## 6. 交付物

- [ ] `train.py` 训练脚本实现完成
- [ ] 加权交叉熵损失实现
- [ ] 学习率余弦衰减调度
- [ ] 早停机制（patience=8）
- [ ] 训练结果保存为 JSON
- [ ] `run_training.sh` 脚本实现
- [ ] 云端 GPU 训练成功，权重保存至 checkpoints/
- [ ] 本地 CPU mock 训练验证通过
