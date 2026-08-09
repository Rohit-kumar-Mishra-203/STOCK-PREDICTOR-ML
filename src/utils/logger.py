"""
Centralized logging setup.

Why loguru instead of print() or raw logging module:
- Zero-config setup (no boilerplate handlers/formatters)
- Automatic log rotation and retention, which matters once this pipeline
  runs daily in production and logs pile up
- Structured, readable output makes debugging a failed pipeline run
  far faster than scrolling through print statements

Why this matters for "production-ready": a pipeline that fails silently
with no logs is not production-ready, no matter how good the model is.
"""

from loguru import logger
import sys
from src.utils.config import PROJECT_ROOT

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Remove default handler to avoid duplicate console logs
logger.remove()

# Console: human-readable, INFO and above
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

# File: DEBUG and above, rotated daily, kept for 14 days
logger.add(
    LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="1 day",
    retention="14 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
)

__all__ = ["logger"]