"""Latency tracker — ring buffer for inference timing statistics."""

from __future__ import annotations

import collections
import threading


class LatencyTracker:
    """Thread-safe ring buffer for recording inference latencies."""

    def __init__(self, capacity: int = 100):
        self._times: collections.deque[float] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, ms: float) -> None:
        with self._lock:
            self._times.append(ms)

    def stats(self) -> dict:
        with self._lock:
            if not self._times:
                return {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0, "fps": 0}
            arr = sorted(self._times)

        n = len(arr)
        avg = sum(arr) / n
        return {
            "avg_ms": round(avg, 1),
            "p50_ms": round(arr[n // 2], 1),
            "p95_ms": round(arr[min(int(n * 0.95), n - 1)], 1),
            "p99_ms": round(arr[min(int(n * 0.99), n - 1)], 1),
            "fps": round(1000 / avg, 1) if avg > 0 else 0,
        }
