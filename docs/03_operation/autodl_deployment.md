# AutoDL 服务器部署与运行手册

## 0. 前置条件

| 项目 | 要求 |
|------|------|
| AutoDL 实例 | 推荐 RTX 4090 / A100，镜像选 PyTorch 2.x + CUDA 12.x |
| 本地项目 | `config-art-master/` 目录，含 `backend/datasets/real/`（已准备好的 7 类数据集） |
| 浏览器 | Chrome / Edge（需支持 WebRTC 摄像头访问） |

---

## 1. 上传项目到服务器

### 方式一：打包上传

```bash
# 本地打包（Windows Git Bash 或 WSL）
cd C:\Users\你的用户名\config-art-master
tar -czf emotion-project.tar.gz .

# 通过 scp 上传
scp -P <SSH端口> emotion-project.tar.gz root@<AutoDL-IP>:/root/autodl-tmp/
```

### 方式二：AutoDL 网盘上传

1. AutoDL 控制台 → 文件存储 → 上传 `emotion-project.tar.gz`
2. 或使用 JupyterLab 的上传功能

### 方式三：Git 拉取

```bash
# 如果项目已推送到 Git 仓库
ssh -p <SSH端口> root@<AutoDL-IP>
cd /root/autodl-tmp
git clone <仓库地址> emotion-project
```

---

## 2. 服务器端解压与目录结构

```bash
ssh -p <SSH端口> root@<AutoDL-IP>

cd /root/autodl-tmp
mkdir -p emotion-project && cd emotion-project
tar xzf ../emotion-project.tar.gz
```

确认目录结构：

```
/root/autodl-tmp/emotion-project/
├── backend/
│   ├── app/                    # FastAPI 应用
│   ├── datasets/
│   │   ├── real/               # 7 类数据集（训练用）
│   │   │   ├── train/          # ~33,788 张
│   │   │   └── val/            # ~2,799 张
│   │   └── mock/               # 本地测试用（可忽略）
│   ├── config.yaml             # 配置文件（需修改）
│   ├── train.py                # 训练入口
│   ├── requirements.txt
│   ├── requirements-prod.txt
│   └── gunicorn_conf.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── deploy_setup.sh
│   ├── run_training.sh
│   └── start_server.sh
└── docs/
```

验证数据集：

```bash
echo "=== train ==="
for d in backend/datasets/real/train/*/; do
    echo "$(basename $d): $(ls "$d" | wc -l)"
done
echo "=== val ==="
for d in backend/datasets/real/val/*/; do
    echo "$(basename $d): $(ls "$d" | wc -l)"
done
```

预期输出：

```
=== train ===
0_angry: 4999
1_disgust: 4996
2_fear: 4998
3_happy: 3801
4_sad: 4999
5_surprise: 4996
6_neutral: 4999
=== val ===
0_angry: 400
1_disgust: 399
2_fear: 400
3_happy: 400
4_sad: 400
5_surprise: 400
6_neutral: 400
```

---

## 3. 修改配置文件

编辑 `backend/config.yaml`：

```bash
vi backend/config.yaml
```

修改以下内容：

```yaml
# 第 6 行：切换为 server
active_profile: server

profiles:
  server:
    app:
      host: "0.0.0.0"
      port: 8000                          # 后端端口（Nginx 反代）
      debug: false
      cors_origins: ["*"]                 # AutoDL 动态域名，先用通配符

    model:
      device: "cuda"                      # 使用 GPU
      checkpoint_path: "checkpoints/best_model.pth"

    dataset:
      affectnet_path: "datasets/real"     # 指向准备好的数据集

    training:
      epochs: 50
      batch_size: 64                      # GPU 可用较大 batch
      use_mock: false

    inference:
      smoothing_window: 5
      max_fps: 30

    logging:
      level: "INFO"
      file: "logs/app.log"
```

---

## 4. 安装依赖

### 4.1 Python 依赖

```bash
cd /root/autodl-tmp/emotion-project/backend

# AutoDL 通常预装了 PyTorch + CUDA，验证：
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 安装项目依赖（跳过 torch/torchvision 如果已预装）
pip install fastapi uvicorn pydantic pyyaml opencv-python Pillow \
    scikit-learn matplotlib gunicorn websockets

# 或完整安装：
# pip install -r requirements-prod.txt
```

### 4.2 Node.js 环境

```bash
# 检查是否已安装
node -v

# 如未安装
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
node -v   # 应 >= 20.x
npm -v
```

### 4.3 Nginx 安装

```bash
apt update && apt install -y nginx
```

### 4.4 验证所有依赖

```bash
python -c "
import torch, fastapi, numpy, yaml, sklearn, PIL
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'FastAPI: {fastapi.__version__}')
print(f'NumPy: {numpy.__version__}')
"
node -v
nginx -v
```

---

## 5. 训练模型

### 5.1 启动训练

```bash
cd /root/autodl-tmp/emotion-project/backend

# 创建必要目录
mkdir -p checkpoints experiments logs

# 开始训练
python train.py \
    --dataset-path datasets/real \
    --epochs 50 \
    --batch-size 64 \
    --task-name affectnet_v1 \
    --num-workers 4
```

### 5.2 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset-path` | config.yaml 中配置 | 数据集路径 |
| `--epochs` | 50 | 训练轮数 |
| `--batch-size` | 64 | 批量大小（GPU 显存 24GB 可用 128） |
| `--lr` | 0.001 | 初始学习率 |
| `--task-name` | 自动生成时间戳 | 实验名称 |
| `--num-workers` | 0 | 数据加载线程数（推荐 4） |
| `--resume` | 无 | 从检查点恢复训练 |

### 5.3 后台训练（可选）

如果想断开 SSH 后继续训练：

```bash
nohup python train.py \
    --dataset-path datasets/real \
    --epochs 50 \
    --batch-size 64 \
    --task-name affectnet_v1 \
    --num-workers 4 \
    > logs/training.log 2>&1 &

# 查看训练日志
tail -f logs/training.log
```

或使用 `tmux`：

```bash
tmux new -s train
python train.py --dataset-path datasets/real --epochs 50 --batch-size 64 --task-name affectnet_v1 --num-workers 4
# Ctrl+B, D 脱离 tmux
# tmux attach -t train  重新连接
```

### 5.4 训练预期

| 硬件 | 预计时间 | 说明 |
|------|----------|------|
| RTX 4090 | 30-60 分钟 | batch_size=64, 50 epoch |
| RTX 3090 | 60-90 分钟 | batch_size=64 |
| A100 | 20-40 分钟 | batch_size=128 |

训练日志示例：

```
Epoch 1/50 — train_loss=1.8432 train_acc=0.2156 val_loss=1.6821 val_acc=0.3012 lr=0.001000
Epoch 2/50 — train_loss=1.5123 train_acc=0.3845 val_loss=1.3456 val_acc=0.4523 lr=0.000998
...
Epoch 24/50 — train_loss=0.4521 train_acc=0.8234 val_loss=0.7123 val_acc=0.6580 lr=0.000234
Best model saved (val_acc=0.6580) -> checkpoints/best_model.pth
```

### 5.5 验证训练结果

```bash
# 检查输出文件
ls checkpoints/
# best_model.pth

ls experiments/affectnet_v1/
# config.json  confusion_matrix.json  metrics.json  performance.json

# 查看性能
cat experiments/affectnet_v1/performance.json | python -m json.tool
```

---

## 6. 构建前端

```bash
cd /root/autodl-tmp/emotion-project/frontend

# 创建生产环境变量文件
cat > .env.production <<'EOF'
VITE_API_BASE_URL=
VITE_INFERENCE_MODE=http
EOF

# 安装依赖并构建
npm install
npm run build

# 验证
ls dist/
# index.html  assets/
```

> `VITE_API_BASE_URL` 留空表示使用相对路径，由 Nginx 统一代理。

---

## 7. 配置 Nginx

### 7.1 创建 Nginx 配置

```bash
cat > /etc/nginx/sites-available/emotion-detection <<'NGINX'
server {
    listen 6006;
    server_name _;

    # 前端静态文件
    root /root/autodl-tmp/emotion-project/frontend/dist;
    index index.html;

    # SPA 路由回退
    location / {
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 反向代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
    }

    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }

    # WebSocket 推理
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
    }

    # Gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml;
    gzip_min_length 1000;

    # 请求体大小限制（base64 图片）
    client_max_body_size 10M;
}
NGINX
```

### 7.2 启用配置

```bash
# 创建软链接
ln -sf /etc/nginx/sites-available/emotion-detection /etc/nginx/sites-enabled/

# 移除默认配置
rm -f /etc/nginx/sites-enabled/default

# 测试并重启
nginx -t
systemctl restart nginx
```

> **为什么是 6006 端口？**
> AutoDL 的「自定义服务」功能默认暴露 6006 端口，并提供 HTTPS 访问地址。
> 这是获取摄像头权限（`getUserMedia` 需要安全上下文）的关键。

---

## 8. 启动后端服务

### 8.1 前台运行（调试用）

```bash
cd /root/autodl-tmp/emotion-project/backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 8.2 后台运行（推荐）

```bash
cd /root/autodl-tmp/emotion-project/backend
mkdir -p logs

nohup uvicorn app.main:app \
    --host 127.0.0.1 \
    --port 8000 \
    > logs/server.log 2>&1 &

echo $! > logs/server.pid
echo "Backend started, PID: $(cat logs/server.pid)"
```

### 8.3 验证服务

```bash
# 后端健康检查
curl -s http://127.0.0.1:8000/health | python -m json.tool

# 通过 Nginx 检查
curl -s http://127.0.0.1:6006/health | python -m json.tool

# 检查前端页面
curl -s http://127.0.0.1:6006/ | head -5
```

预期输出：

```json
{
    "status": "ok",
    "model_loaded": true,
    "device": "cuda",
    "uptime_seconds": 5.2
}
```

---

## 9. 浏览器访问

### 9.1 获取访问地址

1. 登录 AutoDL 控制台
2. 找到你的实例 → 点击「自定义服务」
3. 获得 HTTPS 地址，形如：

```
https://u-xxxxx-6006.westc.gpuhub.com
```

### 9.2 打开系统

1. 浏览器打开上述 HTTPS 地址
2. 自动跳转到 `/detection` 实时检测页面
3. 点击「开启摄像头」→ 浏览器弹窗允许摄像头权限
4. 系统开始工作：
   - 浏览器端：摄像头采集 → MediaPipe 人脸检测 → 裁剪人脸 + 提取几何特征
   - 服务器端：接收数据 → CNN 推理 → 返回 7 类情绪概率
   - 浏览器端：显示情绪标签 + 概率条 + FPS

### 9.3 页面导航

| 页面 | 路径 | 功能 |
|------|------|------|
| 实时检测 | `/detection` | 摄像头 + 情绪识别 + 实时统计 |
| 数据集 | `/dataset` | 数据集概览 + 类别分布图 |
| 实验结果 | `/results` | F1 性能图 + 混淆矩阵热力图 |

---

## 10. 常用运维命令

### 服务管理

```bash
# 查看后端是否运行
ps aux | grep uvicorn

# 停止后端
kill $(cat /root/autodl-tmp/emotion-project/backend/logs/server.pid)

# 重启后端
cd /root/autodl-tmp/emotion-project/backend
kill $(cat logs/server.pid) 2>/dev/null
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > logs/server.log 2>&1 &
echo $! > logs/server.pid

# 重启 Nginx
systemctl restart nginx

# 查看后端日志
tail -f /root/autodl-tmp/emotion-project/backend/logs/server.log

# 查看 Nginx 日志
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### API 测试

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 配置信息
curl http://127.0.0.1:8000/api/v1/config

# 数据集信息
curl http://127.0.0.1:8000/api/v1/dataset/info

# 分类性能
curl http://127.0.0.1:8000/api/v1/stats/performance

# 混淆矩阵
curl http://127.0.0.1:8000/api/v1/stats/confusion-matrix

# 延迟统计
curl http://127.0.0.1:8000/api/v1/stats/latency
```

### 重新训练

```bash
cd /root/autodl-tmp/emotion-project/backend

# 新一轮训练
python train.py \
    --dataset-path datasets/real \
    --epochs 50 \
    --batch-size 64 \
    --task-name affectnet_v2

# 从上次最佳模型继续训练
python train.py \
    --dataset-path datasets/real \
    --epochs 30 \
    --batch-size 64 \
    --resume checkpoints/best_model.pth \
    --task-name affectnet_v2_finetune
```

---

## 11. 故障排查

### 问题：摄像头无法访问

- **原因**：浏览器要求 HTTPS 安全上下文才能使用 `getUserMedia()`
- **解决**：必须通过 AutoDL 的自定义服务 HTTPS 地址访问，不能用 HTTP IP

### 问题：模型未加载 (model_loaded: false)

```bash
# 检查模型文件是否存在
ls -lh backend/checkpoints/best_model.pth

# 如果不存在，需要先训练
cd backend && python train.py --dataset-path datasets/real --epochs 50 --task-name v1
```

### 问题：Nginx 502 Bad Gateway

```bash
# 检查后端是否在运行
curl http://127.0.0.1:8000/health

# 如果没有，重启后端
cd /root/autodl-tmp/emotion-project/backend
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > logs/server.log 2>&1 &
```

### 问题：CUDA out of memory

```bash
# 减小 batch_size
python train.py --dataset-path datasets/real --batch-size 32 --task-name v1

# 或清理 GPU 显存
nvidia-smi
kill <占用显存的进程PID>
```

### 问题：端口被占用

```bash
# 查看端口占用
lsof -i :8000
lsof -i :6006

# 杀掉占用进程
kill -9 <PID>
```

### 问题：前端页面空白

```bash
# 确认前端已构建
ls frontend/dist/index.html

# 确认 Nginx 配置的 root 路径正确
grep "root" /etc/nginx/sites-available/emotion-detection

# 重新构建
cd frontend && npm run build
systemctl restart nginx
```

---

## 12. 完整操作速查

从零开始的最简命令序列：

```bash
# ① 上传解压
cd /root/autodl-tmp/emotion-project

# ② 修改配置
vi backend/config.yaml
# active_profile: server, device: cuda, affectnet_path: datasets/real

# ③ 安装依赖
cd backend
pip install fastapi uvicorn pydantic pyyaml opencv-python Pillow scikit-learn matplotlib gunicorn websockets
cd ../frontend
npm install

# ④ 训练
cd ../backend
python train.py --dataset-path datasets/real --epochs 50 --batch-size 64 --task-name v1 --num-workers 4

# ⑤ 构建前端
cd ../frontend
echo -e "VITE_API_BASE_URL=\nVITE_INFERENCE_MODE=http" > .env.production
npm run build

# ⑥ 配置 Nginx（见第 7 节）
# ...

# ⑦ 启动后端
cd ../backend
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 > logs/server.log 2>&1 &

# ⑧ 浏览器打开 AutoDL 自定义服务 HTTPS 地址
```
