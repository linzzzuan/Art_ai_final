#!/bin/bash
# ============================================================
# health_check.sh — Check if backend service is healthy
# ============================================================
set -e

HOST="${1:-localhost}"
PORT="${2:-8000}"

echo "[Health Check] http://${HOST}:${PORT}/health"

RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health" 2>/dev/null)

if [ "$RESPONSE" = "200" ]; then
    echo "  Status: OK"
    curl -s "http://${HOST}:${PORT}/health" | python3 -m json.tool 2>/dev/null || \
        curl -s "http://${HOST}:${PORT}/health"
else
    echo "  Status: FAILED (HTTP $RESPONSE)"
    exit 1
fi
