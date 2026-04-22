#!/bin/bash
# ============================================================
# check_gpu.sh — Check GPU availability
# ============================================================
set -e

echo "[GPU Check]"

if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA driver found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "  CUDA available: yes"
else
    echo "  NVIDIA driver not found — CPU mode only"
    echo "  CUDA available: no"
fi

# Check PyTorch CUDA
python3 -c "
import torch
print(f'  PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')
" 2>/dev/null || echo "  PyTorch not installed or failed to import"
