"""
Pytest configuration and shared fixtures for Connect4 RL testing suite.

This module provides comprehensive fixtures for testing all components
of the Connect4 Reinforcement Learning system.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, patch

import pytest
import torch
import numpy as np

# Add src to Python path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.config import Config
from src.environments.connect4_game import Connect4Game
from src.environments.hybrid_vectorized_connect4 import HybridVectorizedConnect4
from src.agents.base_agent import BaseAgent
from src.agents.random_agent import RandomAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.networks import Connect4Network, Connect4DuelingNetwork
from src.utils.checkpointing import CheckpointManager
from src.utils.model_manager import ModelManager
from src.training.training_statistics import TrainingStatistics


# Device fixtures
@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing (CPU or CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Always return CPU device for CPU-specific tests."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def force_cpu():
    """Force all operations to use CPU during testing."""
    with patch('torch.cuda.is_available', return_value=False):
        yield


# Configuration fixtures
@pytest.fixture(scope="session")
def test_config_dict():
    """Basic test configuration dictionary."""
    return {
        'game': {
            'board_rows': 6,
            'board_cols': 7,
            'win_length': 4
        },
        'ppo': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_loss_coeff': 0.5,
            'entropy_coeff': 0.01,
            'max_grad_norm': 0.5
        },
        'network': {
            'conv_channels': [64, 128, 128],
            'conv_kernels': [4, 3, 3],
            'conv_padding': [2, 1, 1],
            'hidden_size': 128
        },
        'training': {
            'max_episodes': 1000,
            'update_frequency': 20,
            'checkpoint_interval': 100,
            'log_interval': 10
        },
        'device': {
            'training_device': 'cpu',
            'game_device': 'cpu'
        },
        'paths': {
            'models_dir': 'models',
            'logs_dir': 'logs',
            'checkpoints_dir': 'models/checkpoints'
        },
        'memory': {
            'buffer_size': 2048,
            'batch_size': 64
        },
        'logging': {
            'level': 'INFO',
            'log_to_file': False,
            'log_to_console': True
        },
        'seed': 42
    }


@pytest.fixture
def test_config(test_config_dict, tmp_path):
    """Create a temporary config file and Config object for testing."""
    import yaml
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config_dict, f)
    
    return Config(str(config_file))


@pytest.fixture
def minimal_config():
    """Minimal configuration for quick testing."""
    with patch('src.core.config.Config._load_config') as mock_load:
        mock_load.return_value = {
            'game': {'rows': 6, 'cols': 7, 'connect': 4},
            'ppo': {'learning_rate': 1e-3, 'batch_size': 8},
            'network': {'type': 'standard'},
            'training': {'max_episodes': 10},
            'device': {'training_device': 'cpu'}
        }
        return Config("dummy_path")


# Game fixtures
@pytest.fixture
def empty_game():
    """Create a fresh Connect4Game instance."""
    return Connect4Game()


@pytest.fixture
def game_with_moves():
    """Create a Connect4Game with some moves already made."""
    game = Connect4Game()
    # Player 1 moves: columns 0, 1, 2
    # Player 2 moves: columns 3, 4
    moves = [0, 3, 1, 4, 2]  # Alternating players
    for col in moves:
        game.drop_piece(col)
    return game


@pytest.fixture
def nearly_full_game():
    """Create a Connect4Game that's nearly full."""
    game = Connect4Game()
    # Fill most of the board leaving only top row partially empty
    for row in range(5):  # Fill bottom 5 rows
        for col in range(7):
            game.board[row, col] = 1 if (row + col) % 2 == 0 else -1
    game.current_player = 1
    return game


@pytest.fixture
def winning_game():
    """Create a Connect4Game with a winning position."""
    game = Connect4Game()
    # Create horizontal win for player 1 in bottom row
    for col in range(4):
        game.board[5, col] = 1
    game.current_player = 1
    return game


# Environment fixtures
@pytest.fixture
def small_vectorized_env(cpu_device):
    """Create a small vectorized environment for testing."""
    return HybridVectorizedConnect4(num_envs=4, device=cpu_device)


@pytest.fixture
def medium_vectorized_env(device):
    """Create a medium-sized vectorized environment."""
    return HybridVectorizedConnect4(num_envs=16, device=device)


# Agent fixtures
@pytest.fixture
def random_agent():
    """Create a RandomAgent for testing."""
    return RandomAgent()


@pytest.fixture
def ppo_agent(cpu_device, test_config):
    """Create a PPOAgent for testing."""
    return PPOAgent(device=cpu_device, config=test_config)


@pytest.fixture
def trained_ppo_agent(ppo_agent, small_vectorized_env):
    """Create a PPOAgent with some training experience."""
    agent = ppo_agent
    # Give it some minimal training experience
    env = small_vectorized_env
    
    for _ in range(5):  # Few training steps
        observations = env.get_observations_gpu()
        valid_moves = env.get_valid_moves_tensor()
        actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
        rewards, dones, _ = env.step_batch(actions.cpu().numpy())
        
        # Store experiences
        for i in range(len(observations)):
            if not dones[i]:
                agent.store_experience(
                    observations[i].cpu().numpy(),
                    actions[i].item(),
                    rewards[i],
                    observations[i].cpu().numpy(),  # Simplified
                    dones[i],
                    log_probs[i].item(),
                    values[i].item()
                )
    
    return agent


# Network fixtures
@pytest.fixture
def standard_network(cpu_device):
    """Create a standard Connect4Network for testing."""
    return Connect4Network().to(cpu_device)


@pytest.fixture
def dueling_network(cpu_device):
    """Create a dueling Connect4Network for testing."""
    return Connect4DuelingNetwork().to(cpu_device)


# Utility fixtures
@pytest.fixture
def temp_models_dir():
    """Create a temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir) / "models"
        models_dir.mkdir()
        yield models_dir


@pytest.fixture
def checkpoint_manager(temp_models_dir):
    """Create a CheckpointManager with temporary directory."""
    return CheckpointManager(str(temp_models_dir))


@pytest.fixture
def model_manager(temp_models_dir):
    """Create a ModelManager with temporary directory."""
    return ModelManager(str(temp_models_dir))


@pytest.fixture
def training_statistics():
    """Create a TrainingStatistics instance for testing."""
    return TrainingStatistics()


# Data fixtures
@pytest.fixture
def sample_board_states():
    """Generate sample board states for testing."""
    states = []
    
    # Empty board
    states.append(np.zeros((6, 7), dtype=np.int8))
    
    # Board with some pieces
    board1 = np.zeros((6, 7), dtype=np.int8)
    board1[5, 0] = 1
    board1[5, 1] = -1
    board1[4, 0] = 1
    states.append(board1)
    
    # Nearly full board
    board2 = np.random.choice([-1, 0, 1], size=(6, 7))
    board2[0, :] = 0  # Keep top row empty
    states.append(board2)
    
    return states


@pytest.fixture
def sample_experiences():
    """Generate sample experience tuples for testing."""
    experiences = []
    for i in range(10):
        obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
        action = np.random.randint(0, 7)
        reward = np.random.choice([-1, 0, 1, -0.01])
        next_obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
        done = np.random.choice([True, False], p=[0.1, 0.9])
        log_prob = np.random.normal(0, 1)
        value = np.random.normal(0, 1)
        
        experiences.append((obs, action, reward, next_obs, done, log_prob, value))
    
    return experiences


# Mock fixtures
@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability for testing GPU code paths."""
    with patch('torch.cuda.is_available') as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA unavailability for testing CPU fallback."""
    with patch('torch.cuda.is_available') as mock:
        mock.return_value = False
        yield mock


@pytest.fixture
def mock_tensorboard():
    """Mock TensorBoard SummaryWriter for testing logging."""
    with patch('torch.utils.tensorboard.SummaryWriter') as mock:
        yield mock


# Temporary file fixtures
@pytest.fixture
def temp_checkpoint_file(tmp_path):
    """Create a temporary checkpoint file path."""
    return tmp_path / "test_checkpoint.pth"


@pytest.fixture
def temp_config_file(tmp_path, test_config_dict):
    """Create a temporary configuration file."""
    import yaml
    
    config_file = tmp_path / "temp_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config_dict, f)
    
    return config_file


# Performance testing fixtures
@pytest.fixture
def benchmark_setup():
    """Setup for performance benchmarking tests."""
    return {
        'num_games': 100,
        'num_environments': [1, 4, 16],
        'timeout_seconds': 30
    }


# Integration testing fixtures
@pytest.fixture
def integration_setup(tmp_path):
    """Setup for integration tests with temporary directories."""
    setup = {
        'models_dir': tmp_path / "models",
        'logs_dir': tmp_path / "logs",
        'config_dir': tmp_path / "configs"
    }
    
    # Create directories
    for dir_path in setup.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return setup


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark slow tests
        if "slow" in item.name or "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark performance tests
        if "performance" in item.nodeid or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Automatically cleanup CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Seeds are reset before each test automatically