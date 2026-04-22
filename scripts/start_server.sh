#!/bin/bash
# ============================================================
# start_server.sh — 服务管理（启动/停止/重启/状态/日志）
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="backend/config.yaml"

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
DEVICE=$(read_config ".profiles.${PROFILE}.model.device")
HOST=$(read_config ".profiles.${PROFILE}.app.host")
PORT=$(read_config ".profiles.${PROFILE}.app.port")

PID_FILE="$PROJECT_ROOT/logs/gunicorn.pid"
LOG_FILE="$PROJECT_ROOT/logs/gunicorn.log"

case "${1:-start}" in
    start)
        echo "=== 启动服务 ==="
        echo "  Profile: $PROFILE"
        echo "  Device:  $DEVICE"
        echo "  Bind:    $HOST:$PORT"
        echo ""

        cd "$PROJECT_ROOT/backend"

        # Check venv
        if [ ! -f ".venv/bin/activate" ]; then
            echo "Error: Virtual environment not found. Run deploy_setup.sh first."
            exit 1
        fi

        # Check model
        MODEL_PATH=$(read_config ".profiles.${PROFILE}.model.checkpoint_path")
        if [ ! -f "$MODEL_PATH" ]; then
            echo "Warning: Model not found at $MODEL_PATH — inference will fail until model is trained"
        fi

        source .venv/bin/activate
        export DEVICE
        mkdir -p "$PROJECT_ROOT/logs"

        if [ "$PROFILE" = "server" ]; then
            # Production: Gunicorn with workers
            gunicorn -c gunicorn_conf.py app.main:app \
                --pid "$PID_FILE" \
                --access-logfile "$LOG_FILE" \
                --error-logfile "$LOG_FILE" \
                --daemon

            echo "Gunicorn started (PID: $(cat "$PID_FILE" 2>/dev/null || echo 'unknown'))"

            # Start / reload Nginx
            sudo systemctl start nginx 2>/dev/null || sudo systemctl reload nginx 2>/dev/null || true
            echo "Nginx running"
        else
            # Development: Uvicorn single process
            echo "Starting Uvicorn (dev mode)..."
            nohup python -m uvicorn app.main:app \
                --host "$HOST" \
                --port "$PORT" \
                --reload \
                > "$LOG_FILE" 2>&1 &

            echo $! > "$PID_FILE"
            echo "Uvicorn started (PID: $!)"
        fi

        # Health check
        sleep 2
        curl -sf "http://localhost:${PORT}/health" > /dev/null && echo "Backend: OK" || echo "Backend: FAIL (may need a moment to start)"

        if [ "$PROFILE" = "server" ]; then
            curl -sf http://localhost/ > /dev/null && echo "Frontend: OK" || echo "Frontend: check Nginx config"
        fi
        ;;

    stop)
        echo "=== 停止服务 ==="
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            kill "$PID" 2>/dev/null && echo "Process stopped (PID: $PID)" || echo "Process not running"
            rm -f "$PID_FILE"
        else
            # Fallback: kill by name
            pkill -f "gunicorn.*app.main:app" 2>/dev/null && echo "Gunicorn stopped" || true
            pkill -f "uvicorn.*app.main:app" 2>/dev/null && echo "Uvicorn stopped" || true
        fi
        ;;

    restart)
        "$0" stop
        sleep 1
        "$0" start
        ;;

    status)
        echo "=== 服务状态 ==="
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                echo "Backend: running (PID: $PID)"
            else
                echo "Backend: stopped (stale PID file)"
            fi
        else
            echo "Backend: stopped"
        fi

        if [ "$PROFILE" = "server" ]; then
            sudo systemctl is-active nginx 2>/dev/null && echo "Nginx: running" || echo "Nginx: stopped"
        fi

        # Quick health check
        PORT=${PORT:-8000}
        curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1 && echo "Health: OK" || echo "Health: unreachable"
        ;;

    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found at $LOG_FILE"
        fi
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
