import os
from pathlib import Path

# Ensure logs directory exists
_log_dir = Path(__file__).resolve().parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"
workers = int(os.getenv("GUNICORN_WORKERS", "4"))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5
pidfile = str(_log_dir / "gunicorn.pid")
accesslog = str(_log_dir / "gunicorn.log")
errorlog = str(_log_dir / "gunicorn.log")
loglevel = os.getenv("LOG_LEVEL", "info")
