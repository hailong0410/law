"""
Centralized logging configuration for Law Chatbot Backend.

This module provides a singleton logger that can be imported and used
throughout the application. Logs are saved to storage/logs directory
with automatic rotation.

Author: Law Chatbot Team
Version: 1.0.0
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

# Determine the project root (law-chatbot-backend directory)
# This file is in law-chatbot-backend/config/logging.py
PROJECT_ROOT = Path(__file__).parent.parent

# Create logs directory if it doesn't exist
LOGS_DIR = PROJECT_ROOT / "storage" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file path with date
LOG_FILE = LOGS_DIR / f"app_{datetime.now().strftime('%Y%m%d')}.log"

# Create logger
logger = logging.getLogger("law_chatbot")
logger.setLevel(logging.DEBUG)

# Prevent duplicate handlers if logger is imported multiple times
if not logger.handlers:
    # Console Handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # File Handler (DEBUG level) with rotation
    # Max 10MB per file, keep 5 backup files
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Log initialization
logger.info("=" * 80)
logger.info("Logger initialized")
logger.info(f"Log file: {LOG_FILE}")
logger.info("=" * 80)


def get_logger(name: str = None):
    """
    Get logger instance.
    
    Args:
        name: Optional logger name (for child loggers)
    
    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logger.getChild(name)
    return logger
