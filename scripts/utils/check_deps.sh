#!/bin/bash
# ============================================================
# check_deps.sh — Check Python and Node.js dependencies
# ============================================================
set -e

echo "[Dependency Check]"

# Python version
PYTHON_VER=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "  Python: $PYTHON_VER"

# Node.js version
NODE_VER=$(node --version 2>/dev/null || echo "not installed")
echo "  Node.js: $NODE_VER"

# Check key Python packages
echo "  Python packages:"
python3 -c "
import torch; print(f'    torch: {torch.__version__}')
import fastapi; print(f'    fastapi: {fastapi.__version__}')
import numpy; print(f'    numpy: {numpy.__version__}')
import yaml; print(f'    pyyaml: {yaml.__version__}')
import sklearn; print(f'    scikit-learn: {sklearn.__version__}')
" 2>/dev/null || echo "    Some packages missing — run pip install -r requirements.txt"

# Check disk space
echo "  Disk space:"
df -h . 2>/dev/null | tail -1 | awk '{print "    Available: "$4}' || echo "    Unable to check"
