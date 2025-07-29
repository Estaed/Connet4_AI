"""
Core module for Connect4 RL Training System

Contains configuration management and other core utilities.
"""

from .config import Config, load_config, get_config

__all__ = ['Config', 'load_config', 'get_config']