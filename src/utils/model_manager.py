"""
Model Manager for Connect4 RL System

Handles model selection, metadata management, and provides interface for:
- Browsing available trained models
- Model performance comparison
- Model loading for human vs AI gameplay
- Model metadata and statistics display

Features:
- Automatic model discovery and indexing
- Performance-based model ranking
- Easy model selection interface
- Model validation and compatibility checking
- Integration with render system for UI display
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import warnings

# Import logging system
try:
    from .logging_utils import LogLevel, TrainingLogger
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False
    LogLevel = None
    TrainingLogger = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available")
    TORCH_AVAILABLE = False
    torch = None

try:
    # Try relative imports first
    from .checkpointing import CheckpointManager
    from ..agents.ppo_agent import PPOAgent
except ImportError:
    # Fallback for when running as script or from different context
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from src.utils.checkpointing import CheckpointManager
        from src.agents.ppo_agent import PPOAgent
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        CheckpointManager = None
        PPOAgent = None


class ModelMetadata:
    """Container for model metadata and performance information."""
    
    def __init__(
        self,
        name: str,
        path: str,
        episode: int,
        timestamp: float,
        performance_summary: Dict[str, Any],
        file_size_mb: float,
        is_best: bool = False
    ):
        self.name = name
        self.path = path
        self.episode = episode
        self.timestamp = timestamp
        self.performance_summary = performance_summary
        self.file_size_mb = file_size_mb
        self.is_best = is_best
        
        # Extract key performance metrics
        self.win_rate = performance_summary.get('win_rate', 0.0)
        self.total_games = performance_summary.get('total_games', 0)
        self.avg_reward = performance_summary.get('avg_reward', 0.0)
        self.avg_game_length = performance_summary.get('avg_game_length', 0.0)
        
        # Format creation date
        self.creation_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(timestamp))
    
    def get_display_name(self) -> str:
        """Get a user-friendly display name for the model."""
        base_name = self.name.replace('.pt', '').replace('connect4_ppo_', '')
        if self.is_best:
            return f"🏆 {base_name} (Best)"
        return base_name
    
    def get_performance_grade(self) -> str:
        """Get a letter grade based on performance."""
        if self.win_rate >= 90:
            return "A+"
        elif self.win_rate >= 80:
            return "A"
        elif self.win_rate >= 70:
            return "B+"
        elif self.win_rate >= 60:
            return "B"
        elif self.win_rate >= 50:
            return "C+"
        elif self.win_rate >= 40:
            return "C"
        else:
            return "D"
    
    def get_skill_level(self) -> str:
        """Get a human-readable skill level."""
        if self.win_rate >= 85:
            return "Expert"
        elif self.win_rate >= 70:
            return "Advanced"
        elif self.win_rate >= 55:
            return "Intermediate"
        elif self.win_rate >= 40:
            return "Beginner"
        else:
            return "Learning"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'path': self.path,
            'episode': self.episode,
            'timestamp': self.timestamp,
            'performance_summary': self.performance_summary,
            'file_size_mb': self.file_size_mb,
            'is_best': self.is_best,
            'win_rate': self.win_rate,
            'total_games': self.total_games,
            'avg_reward': self.avg_reward,
            'creation_date': self.creation_date,
            'performance_grade': self.get_performance_grade(),
            'skill_level': self.get_skill_level()
        }


class ModelManager:
    """
    Comprehensive model management system for Connect4 RL.
    
    Provides functionality for:
    - Model discovery and indexing
    - Performance comparison and ranking
    - Model selection interface
    - Model loading and validation
    """
    
    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        cache_file: Optional[str] = "model_cache.json",
        auto_refresh: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory containing model checkpoints
            cache_file: File to cache model metadata (None to disable)
            auto_refresh: Whether to automatically refresh model list
            debug_mode: Whether to show detailed initialization logs
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_file = Path(models_dir) / cache_file if cache_file else None
        self.auto_refresh = auto_refresh
        self.debug_mode = debug_mode
        
        # Model storage
        self.models: Dict[str, ModelMetadata] = {}
        self.checkpoint_manager = None
        
        # Initialize checkpoint manager if available
        if CheckpointManager:
            try:
                self.checkpoint_manager = CheckpointManager(checkpoint_dir=models_dir)
            except Exception as e:
                print(f"Warning: Could not initialize checkpoint manager: {e}")
        
        # Load cached models and refresh
        self._load_cache()
        if auto_refresh:
            self.refresh_models()
        
        if self.debug_mode:
            print(f"[ModelManager] Initialized")
            print(f"  Models directory: {self.models_dir}")
            print(f"  Found {len(self.models)} models")
            print(f"  Cache file: {self.cache_file}")
    
    def refresh_models(self) -> None:
        """Refresh the list of available models."""
        if self.debug_mode:
            print("[ModelManager] Refreshing model list...")
        
        # Clear existing models
        old_count = len(self.models)
        self.models.clear()
        
        # Always use manual discovery for level-based folder structure
        # CheckpointManager only looks in base directory, but we have subdirectories
        self._manual_model_discovery()
        
        # Also try checkpoint manager as backup
        if self.checkpoint_manager:
            try:
                checkpoints = self.checkpoint_manager.list_checkpoints()
                for checkpoint in checkpoints:
                    # Only add if not already found by manual discovery
                    if checkpoint['name'] not in self.models:
                        metadata = ModelMetadata(
                            name=checkpoint['name'],
                            path=checkpoint['path'],
                            episode=checkpoint.get('episode', 0),
                            timestamp=checkpoint.get('timestamp', 0),
                            performance_summary=checkpoint.get('performance_summary', {}),
                            file_size_mb=checkpoint.get('file_size_mb', 0),
                            is_best=checkpoint.get('is_best', False)
                        )
                        self.models[checkpoint['name']] = metadata
            except Exception as e:
                print(f"Warning: Could not use checkpoint manager: {e}")
        
        new_count = len(self.models)
        if self.debug_mode:
            print(f"[ModelManager] Model refresh complete: {old_count} -> {new_count} models")
        
        # Save to cache
        self._save_cache()
    
    def _manual_model_discovery(self) -> None:
        """Manually discover models in the directory and subdirectories."""
        if not self.models_dir.exists():
            return
        
        # Search for models in main directory and all subdirectories
        for model_file in self.models_dir.glob('**/*.pt'):
            try:
                # Try to load basic metadata
                if TORCH_AVAILABLE:
                    checkpoint_data = torch.load(model_file, map_location='cpu', weights_only=False)
                    
                    # Include training level from subdirectory name
                    training_level = model_file.parent.name if model_file.parent != self.models_dir else "general"
                    display_name = f"[{training_level.upper()}] {model_file.name}"
                    
                    metadata = ModelMetadata(
                        name=display_name,
                        path=str(model_file),
                        episode=checkpoint_data.get('episode', 0),
                        timestamp=checkpoint_data.get('timestamp', model_file.stat().st_mtime),
                        performance_summary=checkpoint_data.get('performance_summary', {}),
                        file_size_mb=model_file.stat().st_size / (1024 * 1024),
                        is_best=checkpoint_data.get('is_best', False)
                    )
                    
                    self.models[display_name] = metadata
                else:
                    # Basic file info only
                    training_level = model_file.parent.name if model_file.parent != self.models_dir else "general"
                    display_name = f"[{training_level.upper()}] {model_file.name}"
                    
                    metadata = ModelMetadata(
                        name=display_name,
                        path=str(model_file),
                        episode=0,
                        timestamp=model_file.stat().st_mtime,
                        performance_summary={},
                        file_size_mb=model_file.stat().st_size / (1024 * 1024),
                        is_best=False
                    )
                    
                    self.models[display_name] = metadata
                    
            except Exception as e:
                print(f"Warning: Could not load metadata for {model_file}: {e}")
    
    def get_models(self, sort_by: str = 'performance') -> List[ModelMetadata]:
        """
        Get list of available models.
        
        Args:
            sort_by: Sort criteria ('performance', 'episode', 'date', 'name')
            
        Returns:
            Sorted list of model metadata
        """
        if self.auto_refresh:
            self.refresh_models()
        
        models_list = list(self.models.values())
        
        # Sort models
        if sort_by == 'performance':
            models_list.sort(key=lambda x: x.win_rate, reverse=True)
        elif sort_by == 'episode':
            models_list.sort(key=lambda x: x.episode, reverse=True)
        elif sort_by == 'date':
            models_list.sort(key=lambda x: x.timestamp, reverse=True)
        else:  # sort by name
            models_list.sort(key=lambda x: x.name)
        
        return models_list
    
    def get_best_models(self, limit: int = 5) -> List[ModelMetadata]:
        """Get the best performing models."""
        all_models = self.get_models(sort_by='performance')
        return all_models[:limit]
    
    def get_latest_models(self, limit: int = 5) -> List[ModelMetadata]:
        """Get the most recently created models."""
        all_models = self.get_models(sort_by='date')
        return all_models[:limit]
    
    def get_model_by_name(self, name: str) -> Optional[ModelMetadata]:
        """Get a specific model by name."""
        if self.auto_refresh:
            self.refresh_models()
        return self.models.get(name)
    
    def load_model_for_gameplay(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None
    ) -> Optional[Any]:  # PPOAgent
        """
        Load a model for human vs AI gameplay.
        
        Args:
            model_name: Name of the model to load
            device: Device to load the model on
            
        Returns:
            Loaded PPO agent or None if loading failed
        """
        if not TORCH_AVAILABLE or not PPOAgent:
            print("Error: PyTorch or PPOAgent not available")
            return None
        
        model_metadata = self.get_model_by_name(model_name)
        if not model_metadata:
            print(f"Error: Model '{model_name}' not found")
            return None
        
        if self.debug_mode:
            print(f"[ModelManager] Loading model for gameplay: {model_name}")
        
        try:
            # Determine device
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Create agent
            agent = PPOAgent(device=device)
            
            # Load checkpoint using checkpoint manager
            if self.checkpoint_manager:
                checkpoint_data = self.checkpoint_manager.load_checkpoint(
                    checkpoint_path=model_metadata.path,
                    agent=agent,
                    device=device,
                    load_optimizer=False  # Don't load optimizer for gameplay
                )
            else:
                # Manual loading
                checkpoint_data = torch.load(model_metadata.path, map_location=device)
                if hasattr(agent, 'network') and 'model_state_dict' in checkpoint_data:
                    agent.network.load_state_dict(checkpoint_data['model_state_dict'])
            
            if self.debug_mode:
                print(f"[ModelManager] Model loaded successfully")
                print(f"  Episode: {model_metadata.episode:,}")
                print(f"  Win Rate: {model_metadata.win_rate:.1f}%")
                print(f"  Skill Level: {model_metadata.get_skill_level()}")
            
            return agent
            
        except Exception as e:
            print(f"[ModelManager] Failed to load model: {e}")
            return None
    
    def validate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Validate a model file and return validation results.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        model_metadata = self.get_model_by_name(model_name)
        if not model_metadata:
            result['errors'].append(f"Model '{model_name}' not found")
            return result
        
        model_path = Path(model_metadata.path)
        
        # Check file exists
        if not model_path.exists():
            result['errors'].append("Model file does not exist")
            return result
        
        # Check file size
        if model_metadata.file_size_mb < 0.1:
            result['warnings'].append("Model file is very small (< 0.1 MB)")
        elif model_metadata.file_size_mb > 100:
            result['warnings'].append("Model file is very large (> 100 MB)")
        
        if not TORCH_AVAILABLE:
            result['warnings'].append("PyTorch not available - cannot validate model structure")
            result['valid'] = True  # Assume valid if we can't check
            return result
        
        try:
            # Try to load the model with weights_only=False for compatibility
            checkpoint_data = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Check required fields
            required_fields = ['model_state_dict', 'episode', 'timestamp']
            for field in required_fields:
                if field not in checkpoint_data:
                    result['warnings'].append(f"Missing optional field: {field}")
            
            # Check model state dict
            if 'model_state_dict' in checkpoint_data:
                state_dict = checkpoint_data['model_state_dict']
                if not isinstance(state_dict, dict):
                    result['errors'].append("Invalid model state dictionary")
                elif len(state_dict) == 0:
                    result['errors'].append("Empty model state dictionary")
                else:
                    result['info']['num_parameters'] = len(state_dict)
            
            # Check performance data
            perf_summary = checkpoint_data.get('performance_summary', {})
            if perf_summary:
                result['info']['performance'] = perf_summary
            else:
                result['warnings'].append("No performance data available")
            
            if len(result['errors']) == 0:
                result['valid'] = True
                result['info']['checkpoint_version'] = checkpoint_data.get('checkpoint_version', 'unknown')
                result['info']['pytorch_version'] = checkpoint_data.get('pytorch_version', 'unknown')
            
        except Exception as e:
            result['errors'].append(f"Failed to load model: {str(e)}")
        
        return result
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model file.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if deletion was successful
        """
        model_metadata = self.get_model_by_name(model_name)
        if not model_metadata:
            print(f"Error: Model '{model_name}' not found")
            return False
        
        try:
            model_path = Path(model_metadata.path)
            if model_path.exists():
                model_path.unlink()
                print(f"[ModelManager] Deleted model: {model_name}")
            
            # Remove from cache
            if model_name in self.models:
                del self.models[model_name]
            
            self._save_cache()
            return True
            
        except Exception as e:
            print(f"[ModelManager] Failed to delete model {model_name}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model manager statistics."""
        if self.auto_refresh:
            self.refresh_models()
        
        models_list = list(self.models.values())
        
        if not models_list:
            return {
                'total_models': 0,
                'best_models': 0,
                'avg_win_rate': 0,
                'total_size_mb': 0,
                'models_dir': str(self.models_dir),
                'performance_distribution': {'Expert': 0, 'Advanced': 0, 'Intermediate': 0, 'Beginner': 0, 'Learning': 0},
                'latest_model_episode': 0
            }
        
        total_size = sum(model.file_size_mb for model in models_list)
        avg_win_rate = sum(model.win_rate for model in models_list) / len(models_list)
        best_models = sum(1 for model in models_list if model.is_best)
        
        # Performance distribution
        performance_levels = {'Expert': 0, 'Advanced': 0, 'Intermediate': 0, 'Beginner': 0, 'Learning': 0}
        for model in models_list:
            level = model.get_skill_level()
            performance_levels[level] = performance_levels.get(level, 0) + 1
        
        return {
            'total_models': len(models_list),
            'best_models': best_models,
            'avg_win_rate': avg_win_rate,
            'total_size_mb': total_size,
            'models_dir': str(self.models_dir),
            'performance_distribution': performance_levels,
            'latest_model_episode': max((model.episode for model in models_list), default=0)
        }
    
    def _load_cache(self) -> None:
        """Load model cache from disk."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            for model_data in cache_data.get('models', []):
                metadata = ModelMetadata(
                    name=model_data['name'],
                    path=model_data['path'],
                    episode=model_data['episode'],
                    timestamp=model_data['timestamp'],
                    performance_summary=model_data['performance_summary'],
                    file_size_mb=model_data['file_size_mb'],
                    is_best=model_data.get('is_best', False)
                )
                self.models[model_data['name']] = metadata
            
            if self.debug_mode:
                print(f"[ModelManager] Loaded {len(self.models)} models from cache")
            
        except Exception as e:
            print(f"Warning: Could not load model cache: {e}")
    
    def _save_cache(self) -> None:
        """Save model cache to disk."""
        if not self.cache_file:
            return
        
        try:
            cache_data = {
                'timestamp': time.time(),
                'models': [model.to_dict() for model in self.models.values()]
            }
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
        except Exception as e:
            print(f"Warning: Could not save model cache: {e}")


# Factory functions
def create_model_manager(
    models_dir: str = "models",
    auto_refresh: bool = True,
    debug_mode: bool = False
) -> ModelManager:
    """
    Factory function to create a model manager.
    
    Args:
        models_dir: Directory containing model checkpoints
        auto_refresh: Whether to automatically refresh model list
        debug_mode: Whether to show detailed logs
        
    Returns:
        Configured ModelManager instance
    """
    return ModelManager(
        models_dir=models_dir,
        auto_refresh=auto_refresh,
        debug_mode=debug_mode
    )


if __name__ == "__main__":
    # Test model manager functionality
    print("Testing ModelManager...")
    
    # Create test model manager
    manager = ModelManager(models_dir="test_models", auto_refresh=True)
    
    # Get available models
    models = manager.get_models()
    print(f"Found {len(models)} models")
    
    for model in models:
        print(f"  - {model.get_display_name()}: {model.get_skill_level()} "
              f"({model.win_rate:.1f}% win rate)")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Statistics: {stats}")
    
    print("ModelManager test completed!")