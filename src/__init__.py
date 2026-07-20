"""
Clinical Decision Support System (CDSS)
FAANG-Level Production Implementation
"""

__version__ = "1.0.0"
__author__ = "Healthcare AI Team"

from src.logger import get_logger
from src.config_manager import config_manager

__all__ = [
    "get_logger",
    "config_manager"
]