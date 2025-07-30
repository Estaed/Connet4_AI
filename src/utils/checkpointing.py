"""
Checkpointing System for Connect4 RL Training

Implements robust save/load functionality for training state preservation:
- Complete training state saving (model, optimizer, metrics, config)
- Seamless training resumption from any checkpoint
- Automatic checkpoint management with versioning
- Model selection interface for human vs AI gameplay

Key Features:
- Auto-save every N episodes (configurable)
- Manual save triggers for experiment control
- Checkpoint integrity validation
- Multiple checkpoint versions for safety
- Optimized save/load performance for minimal training interruption
"""

import os
import time
import shutil
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import json
import torch

try:
    # Try relative imports first
    from ..agents.ppo_agent import PPOAgent
    from ..core.config import get_config, Config
except ImportError:
    # Fallback for when running as script or from different context
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from src.agents.ppo_agent import PPOAgent
        from src.core.config import get_config, Config
    except ImportError as e:
        print(f"Warning: Could not import PPO agent or config: {e}")
        PPOAgent = None
        get_config = None
        Config = None

# Add safe globals for PyTorch 2.6+ weights_only loading
if Config is not None:
    torch.serialization.add_safe_globals([Config])


class CheckpointManager:
    """
    Comprehensive training state preservation system.
    
    Manages saving and loading of complete training state including:
    - Neural network model weights
    - Optimizer state
    - Training metrics and progress
    - Configuration parameters
    - Random number generator states
    - Model metadata for selection interface
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "models",
        max_checkpoints: int = 5,
        auto_save_frequency: int = 1000,  # Every 1000 episodes
        enable_compression: bool = True,
        model_name_prefix: str = "connect4_ppo",
        difficulty_level: Optional[str] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            auto_save_frequency: Episodes between automatic saves
            enable_compression: Whether to compress checkpoint files
            model_name_prefix: Prefix for model filenames
            difficulty_level: Training difficulty level (small, medium, impossible, custom)
        """
        self.base_checkpoint_dir = Path(checkpoint_dir)
        self.difficulty_level = difficulty_level
        
        # Create difficulty-specific subdirectory if specified
        if difficulty_level:
            self.checkpoint_dir = self.base_checkpoint_dir / difficulty_level
        else:
            self.checkpoint_dir = self.base_checkpoint_dir
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.auto_save_frequency = auto_save_frequency
        self.enable_compression = enable_compression
        self.model_name_prefix = model_name_prefix
        
        # Track checkpoint history
        self.checkpoint_history: List[Dict[str, Any]] = []
        self.last_auto_save_episode = 0
        
        # Performance tracking
        self.save_times = []
        self.load_times = []
        
        # Load existing checkpoint history
        self._discover_existing_checkpoints()
        
        print(f"[CheckpointManager] Initialized")
        print(f"  Base directory: {self.base_checkpoint_dir}")
        if self.difficulty_level:
            print(f"  Difficulty folder: {self.difficulty_level}")
        print(f"  Full path: {self.checkpoint_dir}")
        print(f"  Auto-save frequency: {self.auto_save_frequency:,} episodes")
        print(f"  Max checkpoints: {self.max_checkpoints}")
        print(f"  Compression: {self.enable_compression}")
        print(f"  Found {len(self.checkpoint_history)} existing checkpoints")
    
    def save_checkpoint(
        self,
        agent,  # PPOAgent instance
        optimizer: Optional[torch.optim.Optimizer],
        episode: int,
        training_stats: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        checkpoint_name: Optional[str] = None,
        is_best: bool = False
    ) -> str:
        """
        Save complete training state to checkpoint.
        
        Args:
            agent: PPO agent to save
            optimizer: Optimizer state to save
            episode: Current episode number
            training_stats: Training statistics (win rates, game counts, etc.)
            training_metrics: PPO metrics (losses, rewards, etc.)
            additional_data: Additional data to save
            checkpoint_name: Custom checkpoint name (auto-generated if None)
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint
        """
        save_start_time = time.time()
        
        # Generate checkpoint name if not provided
        if checkpoint_name is None:
            timestamp = int(time.time())
            prefix = "best_" if is_best else ""
            checkpoint_name = f"{prefix}{self.model_name_prefix}_ep_{episode}_{timestamp}.pt"
        elif not checkpoint_name.endswith('.pt'):
            checkpoint_name += '.pt'
        
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        print(f"[CheckpointManager] Saving checkpoint: {checkpoint_name}")
        
        # Prepare checkpoint data
        checkpoint_data = {
            # Model and training state
            'model_state_dict': agent.network.state_dict() if hasattr(agent, 'network') else None,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            
            # Training progress
            'episode': episode,
            'training_stats': training_stats or {},
            'training_metrics': training_metrics or {},
            
            # Configuration
            'agent_type': type(agent).__name__,
            'device': str(getattr(agent, 'device', 'cpu')),
            
            # Agent-specific state
            'agent_config': getattr(agent, 'config', None),
            'update_count': getattr(agent, 'update_count', 0),
            
            # Random states for reproducibility
            'torch_rng_state': torch.get_rng_state(),
            'numpy_rng_state': None,  # Will add if numpy is used
            
            # Metadata
            'timestamp': time.time(),
            'pytorch_version': torch.__version__,
            'checkpoint_version': '2.0',
            'is_best': is_best,
            'model_name_prefix': self.model_name_prefix,
            
            # Model performance summary for selection interface
            'performance_summary': self._create_performance_summary(
                episode, training_stats, training_metrics
            )
        }
        
        # Add numpy random state if available
        try:
            import numpy as np
            checkpoint_data['numpy_rng_state'] = np.random.get_state()
        except ImportError:
            pass
        
        # Add CUDA random state if available
        if torch.cuda.is_available():
            checkpoint_data['cuda_rng_state'] = torch.cuda.get_rng_state()
        
        # Add additional data
        if additional_data:
            checkpoint_data['additional_data'] = additional_data
        
        # Save checkpoint
        try:
            if self.enable_compression:
                torch.save(checkpoint_data, checkpoint_path, _use_new_zipfile_serialization=False)
            else:
                torch.save(checkpoint_data, checkpoint_path)
            
            # Verify checkpoint was saved correctly
            if not checkpoint_path.exists():
                raise IOError(f"Checkpoint file was not created: {checkpoint_path}")
            
            file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            save_time = time.time() - save_start_time
            
            print(f"[CheckpointManager] Checkpoint saved successfully")
            print(f"  File: {checkpoint_name}")
            print(f"  Size: {file_size_mb:.1f} MB")
            print(f"  Save time: {save_time:.2f}s")
            
            # Update checkpoint history
            checkpoint_info = {
                'path': str(checkpoint_path),
                'name': checkpoint_name,
                'episode': episode,
                'timestamp': checkpoint_data['timestamp'],
                'file_size_mb': file_size_mb,
                'save_time': save_time,
                'is_best': is_best,
                'performance_summary': checkpoint_data['performance_summary']
            }
            
            self.checkpoint_history.append(checkpoint_info)
            self.save_times.append(save_time)
            
            # Clean up old checkpoints (but keep best models)
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            print(f"[CheckpointManager] Failed to save checkpoint: {e}")
            # Clean up partial file if it exists
            if checkpoint_path.exists():
                try:
                    checkpoint_path.unlink()
                except:
                    pass
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        agent = None,  # PPOAgent instance
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, torch.device]] = None,
        strict_loading: bool = True,
        load_optimizer: bool = True
    ) -> Dict[str, Any]:
        """
        Load training state from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            agent: PPO agent to load state into (optional)
            optimizer: Optimizer to load state into (optional)
            device: Device to map tensors to
            strict_loading: Whether to enforce strict state dict loading
            load_optimizer: Whether to load optimizer state
            
        Returns:
            Dictionary containing loaded checkpoint data
        """
        load_start_time = time.time()
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[CheckpointManager] Loading checkpoint: {checkpoint_path.name}")
        
        try:
            # Load checkpoint data with weights_only=False for backward compatibility
            if device is not None:
                device = torch.device(device) if isinstance(device, str) else device
                checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
            else:
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Validate checkpoint
            self._validate_checkpoint(checkpoint_data)
            
            # Load model state
            if agent is not None and 'model_state_dict' in checkpoint_data and checkpoint_data['model_state_dict']:
                try:
                    if hasattr(agent, 'network'):
                        agent.network.load_state_dict(
                            checkpoint_data['model_state_dict'], 
                            strict=strict_loading
                        )
                        print(f"[CheckpointManager] Model state loaded")
                    else:
                        print(f"[CheckpointManager] ⚠️  Agent has no 'network' attribute")
                except Exception as e:
                    if strict_loading:
                        raise
                    else:
                        warnings.warn(f"Could not load model state (non-strict): {e}")
            
            # Load optimizer state
            if (optimizer is not None and load_optimizer and 
                'optimizer_state_dict' in checkpoint_data and 
                checkpoint_data['optimizer_state_dict']):
                try:
                    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
                    print(f"[CheckpointManager] Optimizer state loaded")
                except Exception as e:
                    if strict_loading:
                        raise
                    else:
                        warnings.warn(f"Could not load optimizer state (non-strict): {e}")
            
            # Restore random states for reproducibility
            if 'torch_rng_state' in checkpoint_data:
                try:
                    torch.set_rng_state(checkpoint_data['torch_rng_state'])
                except Exception as e:
                    warnings.warn(f"Could not restore torch RNG state: {e}")
            
            if 'numpy_rng_state' in checkpoint_data and checkpoint_data['numpy_rng_state']:
                try:
                    import numpy as np
                    np.random.set_state(checkpoint_data['numpy_rng_state'])
                except Exception as e:
                    warnings.warn(f"Could not restore numpy RNG state: {e}")
            
            if 'cuda_rng_state' in checkpoint_data and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state(checkpoint_data['cuda_rng_state'])
                except Exception as e:
                    warnings.warn(f"Could not restore CUDA RNG state: {e}")
            
            # Restore agent state
            if agent is not None:
                if 'update_count' in checkpoint_data:
                    agent.update_count = checkpoint_data['update_count']
            
            load_time = time.time() - load_start_time
            self.load_times.append(load_time)
            
            print(f"[CheckpointManager] Checkpoint loaded successfully")
            print(f"  Episode: {checkpoint_data.get('episode', 'unknown'):,}")
            print(f"  Load time: {load_time:.2f}s")
            
            return checkpoint_data
            
        except Exception as e:
            print(f"[CheckpointManager] Failed to load checkpoint: {e}")
            raise
    
    def should_auto_save(self, episode: int) -> bool:
        """Check if automatic save should be triggered."""
        if self.auto_save_frequency <= 0:
            return False
        
        episodes_since_save = episode - self.last_auto_save_episode
        return episodes_since_save >= self.auto_save_frequency
    
    def trigger_auto_save(
        self,
        agent,
        optimizer: Optional[torch.optim.Optimizer],
        episode: int,
        training_stats: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Trigger automatic save if conditions are met.
        
        Returns:
            Path to saved checkpoint if save was triggered, None otherwise
        """
        if self.should_auto_save(episode):
            checkpoint_path = self.save_checkpoint(
                agent=agent,
                optimizer=optimizer,
                episode=episode,
                training_stats=training_stats,
                training_metrics=training_metrics,
                checkpoint_name=f"auto_save_ep_{episode}.pt"
            )
            
            self.last_auto_save_episode = episode
            return checkpoint_path
        
        return None
    
    def list_checkpoints(self, sort_by: str = 'episode') -> List[Dict[str, Any]]:
        """
        List all available checkpoints with metadata.
        
        Args:
            sort_by: Sort criteria ('episode', 'timestamp', 'name')
            
        Returns:
            Sorted list of checkpoint information
        """
        # Update with any checkpoints found in directory
        self._discover_existing_checkpoints()
        
        # Sort by specified criteria
        if sort_by == 'episode':
            key_func = lambda x: x.get('episode', 0)
        elif sort_by == 'timestamp':
            key_func = lambda x: x.get('timestamp', 0)
        else:  # sort by name
            key_func = lambda x: x.get('name', '')
        
        return sorted(self.checkpoint_history, key=key_func, reverse=True)
    
    def get_best_checkpoints(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the best performing checkpoints."""
        best_checkpoints = [cp for cp in self.checkpoint_history if cp.get('is_best', False)]
        return sorted(best_checkpoints, key=lambda x: x.get('episode', 0), reverse=True)[:limit]
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint by episode."""
        checkpoints = self.list_checkpoints(sort_by='episode')
        return checkpoints[0] if checkpoints else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get checkpoint manager statistics."""
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(self.checkpoint_history),
            'max_checkpoints': self.max_checkpoints,
            'auto_save_frequency': self.auto_save_frequency,
            'last_auto_save_episode': self.last_auto_save_episode,
            'avg_save_time': sum(self.save_times) / len(self.save_times) if self.save_times else 0,
            'avg_load_time': sum(self.load_times) / len(self.load_times) if self.load_times else 0,
            'total_saves': len(self.save_times),
            'total_loads': len(self.load_times),
            'best_checkpoints': len(self.get_best_checkpoints())
        }
    
    def _create_performance_summary(self, episode: int, training_stats: Optional[Dict], training_metrics: Optional[Dict]) -> Dict[str, Any]:
        """Create a performance summary for model selection interface."""
        summary = {
            'episode': episode,
            'training_complete': False,  # Can be updated when training finishes
        }
        
        if training_stats:
            total_games = training_stats.get('total_games', 0)
            if total_games > 0:
                summary.update({
                    'total_games': total_games,
                    'win_rate': training_stats.get('player1_wins', 0) / total_games * 100,
                    'avg_game_length': training_stats.get('avg_game_length', 0),
                })
        
        if training_metrics:
            summary.update({
                'avg_reward': training_metrics.get('avg_reward', 0),
                'policy_loss': training_metrics.get('policy_loss', 0),
                'value_loss': training_metrics.get('value_loss', 0),
            })
        
        return summary
    
    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Validate checkpoint data integrity."""
        required_keys = ['episode', 'timestamp', 'checkpoint_version']
        
        for key in required_keys:
            if key not in checkpoint_data:
                raise ValueError(f"Invalid checkpoint: missing required key '{key}'")
        
        # Check if checkpoint is too old (optional validation)
        checkpoint_time = checkpoint_data['timestamp']
        age_days = (time.time() - checkpoint_time) / (24 * 3600)
        
        if age_days > 30:  # Warn if checkpoint is over 30 days old
            warnings.warn(f"Loading old checkpoint ({age_days:.1f} days old)")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if self.max_checkpoints <= 0:
            return
        
        # Don't delete best models
        regular_checkpoints = [cp for cp in self.checkpoint_history if not cp.get('is_best', False)]
        
        # Sort regular checkpoints by timestamp (newest first)
        sorted_checkpoints = sorted(regular_checkpoints, key=lambda x: x['timestamp'], reverse=True)
        
        # Remove excess regular checkpoints
        if len(sorted_checkpoints) > self.max_checkpoints:
            checkpoints_to_remove = sorted_checkpoints[self.max_checkpoints:]
            
            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint_path = Path(checkpoint['path'])
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        print(f"[CheckpointManager] Removed old checkpoint: {checkpoint['name']}")
                    
                    # Remove from history
                    self.checkpoint_history.remove(checkpoint)
                except Exception as e:
                    warnings.warn(f"Could not remove old checkpoint {checkpoint['name']}: {e}")
    
    def _discover_existing_checkpoints(self) -> None:
        """Discover checkpoints in the directory that aren't in history."""
        if not self.checkpoint_dir.exists():
            return
        
        known_paths = {checkpoint['path'] for checkpoint in self.checkpoint_history}
        
        for checkpoint_file in self.checkpoint_dir.glob('*.pt'):
            if str(checkpoint_file) not in known_paths:
                try:
                    # Try to load metadata without loading full checkpoint
                    checkpoint_data = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
                    
                    checkpoint_info = {
                        'path': str(checkpoint_file),
                        'name': checkpoint_file.name,
                        'episode': checkpoint_data.get('episode', 0),
                        'timestamp': checkpoint_data.get('timestamp', checkpoint_file.stat().st_mtime),
                        'file_size_mb': checkpoint_file.stat().st_size / (1024 * 1024),
                        'save_time': None,
                        'is_best': checkpoint_data.get('is_best', False),
                        'performance_summary': checkpoint_data.get('performance_summary', {})
                    }
                    
                    self.checkpoint_history.append(checkpoint_info)
                    
                except Exception as e:
                    warnings.warn(f"Could not read checkpoint metadata from {checkpoint_file}: {e}")


# Factory function for easy checkpoint manager creation
def create_checkpoint_manager(
    checkpoint_dir: str = "models",
    max_checkpoints: int = 5,
    auto_save_frequency: int = 1000,
    model_name_prefix: str = "connect4_ppo",
    difficulty_level: Optional[str] = None
) -> CheckpointManager:
    """
    Factory function to create checkpoint manager.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        max_checkpoints: Maximum checkpoints to keep
        auto_save_frequency: Episodes between auto-saves
        model_name_prefix: Prefix for model filenames
        difficulty_level: Training difficulty level for folder organization
    
    Returns:
        Configured CheckpointManager
    """
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        auto_save_frequency=auto_save_frequency,
        model_name_prefix=model_name_prefix,
        difficulty_level=difficulty_level
    )


if __name__ == "__main__":
    # Test checkpoint manager functionality
    print("Testing CheckpointManager...")
    
    # Create test directory
    test_dir = Path("test_checkpoints")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create checkpoint manager
    manager = CheckpointManager(
        checkpoint_dir=test_dir,
        max_checkpoints=3,
        auto_save_frequency=100  # Low for testing
    )
    
    print(f"Manager created with {len(manager.checkpoint_history)} existing checkpoints")
    
    # Mock objects for testing (would normally be real PPO agent, etc.)
    class MockAgent:
        def __init__(self):
            self.network = torch.nn.Linear(10, 5)
            self.device = torch.device("cpu")
            self.update_count = 42
            self.config = {"learning_rate": 0.001}
    
    # Create mock objects
    agent = MockAgent()
    optimizer = torch.optim.Adam(agent.network.parameters())
    
    # Test saving
    training_stats = {
        'total_games': 500,
        'player1_wins': 300,
        'player2_wins': 150,
        'draws': 50,
        'avg_game_length': 25.5
    }
    
    training_metrics = {
        'policy_loss': 0.05,
        'value_loss': 0.03,
        'avg_reward': 0.15
    }
    
    checkpoint_path = manager.save_checkpoint(
        agent=agent,
        optimizer=optimizer,
        episode=1000,
        training_stats=training_stats,
        training_metrics=training_metrics,
        additional_data={'test_key': 'test_value'}
    )
    
    print(f"Saved checkpoint: {checkpoint_path}")
    
    # Test loading
    loaded_data = manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        agent=agent,
        optimizer=optimizer
    )
    
    print(f"Loaded checkpoint data keys: {list(loaded_data.keys())}")
    
    # Test auto-save functionality
    print(f"Should auto-save at episode 1100: {manager.should_auto_save(1100)}")
    
    auto_save_path = manager.trigger_auto_save(
        agent=agent,
        optimizer=optimizer,
        episode=1100,
        training_stats=training_stats,
        training_metrics=training_metrics
    )
    
    if auto_save_path:
        print(f"Auto-saved to: {auto_save_path}")
    
    # List checkpoints
    checkpoints = manager.list_checkpoints()
    print(f"Available checkpoints: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"  - {cp['name']}: Episode {cp['episode']:,}")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Manager statistics: {stats}")
    
    # Clean up test directory
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    print("CheckpointManager test completed!")