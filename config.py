"""
Configuration for Classical Tibetan Parser
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
DATABASE_DIR = PROJECT_ROOT / "database"
MODELS_DIR = PROJECT_ROOT / "models"

# TiBERT Model
TIBERT_MODEL_NAME = str(PROJECT_ROOT / "model" / "TiBERT")

# Qwen API Configuration
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
QWEN_MODEL = "qwen-turbo"
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# API settings
API_TIMEOUT = 30
MAX_RETRIES = 3

# UI settings
PAGE_ICON = "📚"
PAGE_TITLE = "古典藏文解析辅助学习系统"
PAGE_LAYOUT = "wide"
