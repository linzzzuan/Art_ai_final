#!/bin/bash
# ============================================================
# deploy_setup.sh — 一键部署环境配置
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== 环境部署开始 ==="

# Read config
CONFIG="backend/config.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found"
    exit 1
fi

read_config() {
    local key="$1"
    if command -v yq &> /dev/null; then
        yq e "$key" "$CONFIG"
    else
        python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
profile = cfg['active_profile']
key_str = '$key'.replace('.profiles.\${PROFILE}', f'.profiles.{profile}').lstrip('.')
keys = key_str.split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
" 2>/dev/null || echo ""
    fi
}

PROFILE=$(read_config '.active_profile')
echo "Profile: $PROFILE"
echo ""

# ── Step 1: System checks ──
echo "[1/5] 系统检查..."
source "$SCRIPT_DIR/utils/check_gpu.sh" 2>/dev/null || true
echo ""
source "$SCRIPT_DIR/utils/check_deps.sh" 2>/dev/null || true
echo ""

# ── Step 2: Backend environment ──
echo "[2/5] 配置 Python 环境..."
cd "$PROJECT_ROOT/backend"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created virtual environment"
else
    echo "  Virtual environment exists"
fi

source .venv/bin/activate

if [ "$PROFILE" = "server" ]; then
    pip install -r requirements-prod.txt
else
    pip install -r requirements.txt
fi

mkdir -p checkpoints experiments data logs
echo "  Backend directories ready"
echo ""

# ── Step 3: Frontend build ──
echo "[3/5] 构建前端..."
cd "$PROJECT_ROOT/frontend"

if [ "$PROFILE" = "server" ]; then
    CORS=$(read_config ".profiles.${PROFILE}.app.cors_origins[0]" | sed 's|https://||')
    cat > .env.production <<EOF
VITE_API_BASE_URL=https://${CORS}/api
VITE_WS_URL=wss://${CORS}/ws
VITE_INFERENCE_MODE=websocket
EOF
    echo "  Generated .env.production"
    npm ci
    npm run build
    echo "  Frontend built: dist/"
else
    echo "  Local profile — skipping production build"
    echo "  Use 'npm run dev' for development"
fi
echo ""

# ── Step 4: Nginx configuration (server only) ──
echo "[4/5] 配置 Nginx..."
if [ "$PROFILE" = "server" ]; then
    NGINX_CONF="$PROJECT_ROOT/backend/docker/nginx.conf"
    if [ -f "$NGINX_CONF" ]; then
        sudo cp "$NGINX_CONF" /etc/nginx/sites-available/emotion-detection
        sudo ln -sf /etc/nginx/sites-available/emotion-detection /etc/nginx/sites-enabled/
        sudo nginx -t && sudo systemctl reload nginx
        echo "  Nginx configured and reloaded"
    else
        echo "  Warning: $NGINX_CONF not found, skipping Nginx setup"
    fi
else
    echo "  Local profile — skipping Nginx setup"
fi
echo ""

# ── Step 5: Verify ──
echo "[5/5] 验证环境..."
cd "$PROJECT_ROOT"
echo "  Project root: $PROJECT_ROOT"
echo "  Profile: $PROFILE"
echo "  Backend venv: backend/.venv/"

if [ "$PROFILE" = "server" ]; then
    echo "  Frontend dist: frontend/dist/"
    [ -d "frontend/dist" ] && echo "  Frontend build: OK" || echo "  Frontend build: MISSING"
fi

echo ""
echo "=== 部署完成 ==="
echo ""
echo "下一步操作:"
echo "  训练模型: ./scripts/run_training.sh --task-name v1"
echo "  启动服务: ./scripts/start_server.sh"
