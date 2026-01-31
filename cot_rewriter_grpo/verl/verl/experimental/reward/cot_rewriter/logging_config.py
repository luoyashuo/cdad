"""
Unified Logging Configuration for CoT Rewriter

Uses loguru for structured logging with file name and line numbers.
Compatible with Ray distributed environment.
"""

import sys
import os
from loguru import logger

# Remove default handler
logger.remove()

# Configure loguru with file name and line number
# Format: timestamp | level | file:line | message
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{file}:{line}</cyan> | "
    "<level>{message}</level>"
)

# Add stderr handler with custom format
logger.add(
    sys.stderr,
    format=log_format,
    level=os.getenv("VERL_LOGGING_LEVEL", "INFO"),
    colorize=True,
    enqueue=True,  # Thread-safe, important for Ray
)

# Export logger for use in other modules
__all__ = ["logger"]
