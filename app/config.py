import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_PATH = str(BASE_DIR / "models" / "best.pt")

# Default thresholds
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# CORS configuration for Frontend integration
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "*",
]
