"""
Connect4 RL Agents Package

This package contains all agent implementations for the Connect4 RL system,
including learning agents (PPO, A2C) and non-learning agents (Random, Minimax).
"""

from .base_agent import BaseAgent, create_agent
from .random_agent import RandomAgent, create_seeded_random_agent, create_random_agent_pair

__all__ = [
    'BaseAgent',
    'create_agent',
    'RandomAgent',
    'create_seeded_random_agent',
    'create_random_agent_pair',
]

# Version info
__version__ = '0.1.0'