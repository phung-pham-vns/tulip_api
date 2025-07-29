"""
Configuration settings for TULIP API.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "open_clip"

# Model configurations
TULIP_MODELS: Dict[str, Dict[str, Any]] = {
    "TULIP-B-16-224": {
        "model_path": str(MODELS_DIR / "tulip-B-16-224.ckpt"),
        "image_size": 224,
    },
    "TULIP-so400m-14-384": {
        "model_path": str(MODELS_DIR / "tulip-so400m-14-384.ckpt"),
        "image_size": 384,
    },
}

# API settings
DEFAULT_MODEL = "TULIP-so400m-14-384"
API_TITLE = "TULIP Embedding API"
API_DESCRIPTION = "API for generating embeddings from text and images using TULIP models"
API_VERSION = "0.1.0"

# Server settings
HOST = "0.0.0.0"
PORT = 8000
