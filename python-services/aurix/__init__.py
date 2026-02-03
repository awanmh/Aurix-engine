"""
AURIX - Adaptive Crypto Trading Bot

Main package initialization.
"""

__version__ = "1.0.0"
__author__ = "AURIX Team"

from .config import load_config, AurixConfig
from .db import Database
from .redis_bus import RedisBus
