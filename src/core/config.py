"""
Connect4 RL Training System - Configuration Management

This module provides centralized configuration management for the Connect4 RL system.
It loads YAML configuration files, validates parameters, and handles device detection.
"""

import os
import yaml
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Centralized configuration management for Connect4 RL system.
    
    Features:
    - YAML-based configuration loading
    - Parameter validation
    - Automatic GPU/CPU device detection
    - Configuration value access with dot notation
    - Environment variable overrides
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration system.
        
        Args:
            config_path: Path to YAML configuration file. 
                        If None, loads default_config.yaml
        """
        # Set default config path if not provided
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "configs" / "default_config.yaml"
        
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        
        # Load configuration
        self._load_config()
        self._setup_device()
        self._create_directories()
        self._validate_config()
        
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            print(f"[OK] Configuration loaded from: {self.config_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
            
    def _setup_device(self) -> None:
        """Setup device configuration with automatic GPU/CPU detection."""
        training_device = self._config['device']['training_device']
        
        if training_device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"[OK] GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("[INFO] No GPU detected, using CPU for training")
        else:
            device = training_device
            
        # Validate device availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available, falling back to CPU")
            device = 'cpu'
            
        # Update configuration with resolved device
        self._config['device']['training_device'] = device
        
        # Game device is always CPU as specified in PRD
        self._config['device']['game_device'] = 'cpu'
        
    def _create_directories(self) -> None:
        """Create necessary directories for models, logs, etc."""
        project_root = Path(__file__).parent.parent.parent
        
        dirs_to_create = [
            project_root / self._config['paths']['models_dir'],
            project_root / self._config['paths']['logs_dir'], 
            project_root / self._config['paths']['checkpoints_dir'],
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        # Validate game configuration
        assert self._config['game']['board_rows'] > 0, "Board rows must be positive"
        assert self._config['game']['board_cols'] > 0, "Board columns must be positive" 
        assert self._config['game']['win_length'] > 0, "Win length must be positive"
        
        # Validate PPO parameters
        assert 0 < self._config['ppo']['learning_rate'] < 1, "Learning rate must be between 0 and 1"
        assert 0 < self._config['ppo']['gamma'] <= 1, "Gamma must be between 0 and 1"
        assert 0 < self._config['ppo']['clip_epsilon'] < 1, "Clip epsilon must be between 0 and 1"
        
        # Validate training parameters
        assert self._config['training']['max_episodes'] > 0, "Max episodes must be positive"
        assert self._config['training']['update_frequency'] > 0, "Update frequency must be positive"
        
        print("[OK] Configuration validation passed")
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'game.board_rows')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Example:
            >>> config.get('game.board_rows')
            6
            >>> config.get('ppo.learning_rate')
            0.0003
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
            
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        # Set final value
        config[keys[-1]] = value
        
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path
            
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self._config, f, default_flow_style=False, indent=2)
            
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
        
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to configuration."""
        return self.get(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of configuration."""
        self.set(key, value)
        
    def __str__(self) -> str:
        """String representation of configuration."""
        return yaml.safe_dump(self._config, default_flow_style=False, indent=2)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)


# Global configuration instance (lazy loaded)
_global_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Global Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration system...")
    
    config = load_config()
    
    # Test parameter access
    print(f"Board size: {config.get('game.board_rows')}x{config.get('game.board_cols')}")
    print(f"Training device: {config.get('device.training_device')}")
    print(f"Learning rate: {config.get('ppo.learning_rate')}")
    
    # Test device detection
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    print("[OK] Configuration system test completed!")