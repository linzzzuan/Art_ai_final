# Phase 7: 云端部署脚本

## 1. 目标

编写三个核心部署脚本，实现一键部署、一键训练、一键启停。

## 2. 脚本一：deploy_setup.sh

**文件**: `scripts/deploy_setup.sh`

```bash
#!/bin/bash
set -e

echo "=== 环境部署开始 ==="

PROFILE=$(yq e '.active_profile' config.yaml)
echo "Profile: $PROFILE"

# 1. 系统检查
source scripts/utils/check_gpu.sh
source scripts/utils/check_deps.sh

# 2. 后端环境
echo "[2/5] 配置 Python 环境..."
cd backend
python3 -m venv .venv
source .venv/bin/activate
if [ "$PROFILE" = "server" ]; then
  pip install -r requirements-prod.txt
else
  pip install -r requirements.txt
fi
mkdir -p checkpoints experiments data logs

# 3. 前端构建
echo "[3/5] 构建前端..."
cd ../frontend
if [ "$PROFILE" = "server" ]; then
  # 从 config.yaml 注入生产环境变量
  CORS=$(yq e ".profiles.${PROFILE}.app.cors_origins[0]" ../config.yaml | sed 's|https://||')
  echo "VITE_API_BASE_URL=https://${CORS}/api"  > .env.production
  echo "VITE_WS_URL=wss://${CORS}/ws"           >> .env.production
  echo "VITE_INFERENCE_MODE=websocket"          >> .env.production
  npm ci
  npm run build
  echo "Frontend built: dist/"
fi

# 4. Nginx 配置
echo "[4/5] 配置 Nginx..."
sudo cp backend/docker/nginx.conf /etc/nginx/sites-available/emotion-detection
sudo ln -sf /etc/nginx/sites-available/emotion-detection /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 5. 验证
echo "[5/5] 验证服务..."
sleep 2
curl -sf http://localhost:8000/health > /dev/null && echo "Backend: OK" || echo "Backend: FAIL"
curl -sf http://localhost/ > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"

echo "=== 部署完成 ==="
```

## 3. 脚本二：run_training.sh

**文件**: `scripts/run_training.sh`

```bash
#!/bin/bash
set -e

echo "=== 训练开始 ==="

PROFILE=$(yq e '.active_profile' config.yaml)
DATASET_PATH=$(yq e ".profiles.${PROFILE}.dataset.affectnet_path" config.yaml)
EPOCHS=$(yq e ".profiles.${PROFILE}.training.epochs" config.yaml)
BATCH_SIZE=$(yq e ".profiles.${PROFILE}.training.batch_size" config.yaml)
TASK_NAME="experiment_$(date +%Y%m%d_%H%M)"

# 参数解析
while [[ $# -gt 0 ]]; do
  case $1 in
    --task-name) TASK_NAME="$2"; shift 2;;
    --dataset-path) DATASET_PATH="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --dry-run) echo "Environment verified"; exit 0;;
    *) echo "Unknown: $1"; exit 1;;
  esac
done

source scripts/utils/check_gpu.sh
source scripts/utils/check_deps.sh

mkdir -p logs

# 后台启动
cd backend
source .venv/bin/activate
export DATASET_PATH
nohup python train.py \
  --dataset-path "$DATASET_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --task-name "$TASK_NAME" \
  > "../logs/training_${TASK_NAME}.log" 2>&1 &

echo $! > "../logs/training_${TASK_NAME}.pid"
echo "Training started: $TASK_NAME (PID: $!)"
echo "Log: logs/training_${TASK_NAME}.log"
echo "Tail: tail -f logs/training_${TASK_NAME}.log"
```

## 4. 脚本三：start_server.sh

**文件**: `scripts/start_server.sh`

```bash
#!/bin/bash
set -e

PROFILE=$(yq e '.active_profile' config.yaml)
DEVICE=$(yq e ".profiles.${PROFILE}.model.device" config.yaml)
MODEL_PATH=$(yq e ".profiles.${PROFILE}.model.checkpoint_path" config.yaml)

cd backend

case "${1:-start}" in
  start)
    echo "=== 启动服务 ==="

    # 检查
    [ -f .venv/bin/activate ] || { echo "Run deploy_setup.sh first"; exit 1; }
    [ -f "$MODEL_PATH" ] || { echo "Warning: Model not found: $MODEL_PATH"; }

    # 启动 Gunicorn
    source .venv/bin/activate
    export DEVICE
    export MODEL_PATH

    gunicorn -c gunicorn_conf.py app.main:app --daemon
    echo "Gunicorn started (PID: $(cat ../logs/gunicorn.pid 2>/dev/null || echo 'unknown'))"

    # 启动/重载 Nginx
    sudo systemctl start nginx 2>/dev/null || sudo systemctl reload nginx
    echo "Nginx running"

    sleep 2
    curl -sf http://localhost:8000/health > /dev/null && echo "Backend: OK" || echo "Backend: FAIL"
    curl -sf http://localhost/ > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"
    ;;

  stop)
    echo "=== 停止服务 ==="
    if [ -f ../logs/gunicorn.pid ]; then
      kill $(cat ../logs/gunicorn.pid) 2>/dev/null && echo "Gunicorn stopped"
    else
      pkill -f gunicorn && echo "Gunicorn stopped"
    fi
    ;;

  restart)
    $0 stop
    sleep 1
    $0 start
    ;;

  status)
    echo "=== 服务状态 ==="
    pgrep -f gunicorn > /dev/null && echo "Backend: running" || echo "Backend: stopped"
    sudo systemctl is-active nginx 2>/dev/null || echo "Nginx: unknown"
    ;;

  logs)
    tail -f ../logs/gunicorn.log
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|status|logs}"
    exit 1
    ;;
esac
```

## 5. 辅助脚本

### 5.1 check_gpu.sh

```bash
#!/bin/bash
echo "[check] GPU..."
if ! command -v nvidia-smi &> /dev/null; then
  echo "Warning: nvidia-smi not found, running CPU only"
else
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  echo "GPU Memory: ${GPU_MEM}MB"
fi
```

### 5.2 check_deps.sh

```bash
#!/bin/bash
echo "[check] Dependencies..."
python3 --version || { echo "Python3 not found"; exit 1; }
node --version || { echo "Node.js not found"; exit 1; }
yq --version || { echo "yq not found: apt install yq"; exit 1; }
```

### 5.3 health_check.sh

```bash
#!/bin/bash
echo "Health check..."
curl -sf http://localhost:8000/health | python3 -m json.tool
curl -sf http://localhost/ > /dev/null && echo "Frontend: OK"
```

## 6. Nginx 配置

**文件**: `backend/docker/nginx.conf`

```nginx
server {
    listen 443 ssl;
    server_name _;

    ssl_certificate     /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;

    # 前端静态文件
    root /app/frontend/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    # 后端 API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket 代理
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

server {
    listen 80;
    return 301 https://$host$request_uri;
}
```

## 7. 完整操作流程

### 首次部署
```bash
git clone <repo> && cd <repo>
# 编辑 config.yaml: active_profile: server, 设置 affectnet_path
./scripts/deploy_setup.sh
./scripts/run_training.sh --task-name v1
# 等待训练完成...
./scripts/start_server.sh
# 浏览器访问 https://<server-ip>/
```

### 日常启停
```bash
./scripts/start_server.sh        # 启动
./scripts/start_server.sh stop   # 停止
./scripts/start_server.sh status # 状态
```

## 8. 交付物

- [ ] `deploy_setup.sh` — 环境部署完成
- [ ] `run_training.sh` — 训练脚本完成
- [ ] `start_server.sh` — 服务管理完成
- [ ] `utils/check_gpu.sh` / `check_deps.sh` / `health_check.sh`
- [ ] Nginx 配置 + HTTPS 证书
- [ ] 完整部署流程测试通过
