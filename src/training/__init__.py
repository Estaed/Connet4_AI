"""
Training Module for Connect4 RL System - Hybrid Edition

This module contains all training-related components including:
- Hybrid vectorized trainer for high-performance training (CPU game logic + GPU neural networks)
- Training statistics tracking and analysis
- Interactive training interface with difficulty selection
- Three difficulty levels: Small (100), Medium (1000), Impossible (10000) environments
"""

from .hybrid_trainer import HybridTrainer
from .training_statistics import TrainingStatistics
from .training_interface import TrainingInterface

__all__ = [
    'HybridTrainer',
    'TrainingStatistics',
    'TrainingInterface'
]

# Version info for Hybrid implementation
__version__ = "2.0.0"
__architecture__ = "hybrid_vectorized"