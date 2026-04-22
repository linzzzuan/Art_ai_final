#!/bin/bash
# ============================================================
# run_training.sh — Start model training
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Read defaults from config.yaml
CONFIG="backend/config.yaml"
if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG not found"
    exit 1
fi

# Try to read from config.yaml (yq or python fallback)
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
# Navigate the key path
keys = '$key'.replace('.profiles.\${PROFILE}', f'.profiles.{profile}').lstrip('.').split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
" 2>/dev/null || echo ""
    fi
}

PROFILE=$(read_config '.active_profile')
DATASET_PATH=$(read_config ".profiles.${PROFILE}.dataset.affectnet_path")
EPOCHS=$(read_config ".profiles.${PROFILE}.training.epochs")
BATCH_SIZE=$(read_config ".profiles.${PROFILE}.training.batch_size")
USE_MOCK=$(read_config ".profiles.${PROFILE}.training.use_mock")
DEVICE=$(read_config ".profiles.${PROFILE}.model.device")

TASK_NAME="experiment_$(date +%Y%m%d_%H%M)"
RESUME=""
DRY_RUN=false

# Parse CLI arguments (override config)
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-name)    TASK_NAME="$2"; shift 2;;
        --dataset-path) DATASET_PATH="$2"; shift 2;;
        --epochs)       EPOCHS="$2"; shift 2;;
        --batch-size)   BATCH_SIZE="$2"; shift 2;;
        --resume)       RESUME="$2"; shift 2;;
        --dry-run)      DRY_RUN=true; shift;;
        --use-mock)     USE_MOCK=true; shift;;
        *)              echo "Unknown option: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  Training Configuration"
echo "  Profile:      $PROFILE"
echo "  Device:       $DEVICE"
echo "  Dataset:      $DATASET_PATH"
echo "  Epochs:       $EPOCHS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Use mock:     $USE_MOCK"
echo "  Task name:    $TASK_NAME"
echo "  Resume:       ${RESUME:-none}"
echo "============================================================"

# Environment checks
echo ""
source "$SCRIPT_DIR/utils/check_gpu.sh" 2>/dev/null || true
echo ""
source "$SCRIPT_DIR/utils/check_deps.sh" 2>/dev/null || true
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete — environment OK"
    exit 0
fi

# Prepare directories
mkdir -p backend/logs backend/checkpoints backend/experiments

# Build training command
cd backend
TRAIN_CMD="python train.py --epochs $EPOCHS --batch-size $BATCH_SIZE --task-name $TASK_NAME"

if [ "$USE_MOCK" = "true" ] || [ "$USE_MOCK" = "True" ]; then
    TRAIN_CMD="$TRAIN_CMD --use-mock"
elif [ -n "$DATASET_PATH" ] && [ "$DATASET_PATH" != "null" ]; then
    TRAIN_CMD="$TRAIN_CMD --dataset-path $DATASET_PATH"
else
    echo "Error: No dataset path configured and use_mock is false"
    exit 1
fi

if [ -n "$RESUME" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
fi

# Start training in background
LOG_FILE="logs/training_${TASK_NAME}.log"
PID_FILE="logs/training_${TASK_NAME}.pid"

echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

nohup $TRAIN_CMD > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "Training started!"
echo "  PID:  $(cat $PID_FILE)"
echo "  Log:  backend/$LOG_FILE"
echo "  Tail: tail -f backend/$LOG_FILE"
