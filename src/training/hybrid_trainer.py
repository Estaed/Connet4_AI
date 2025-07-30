"""
Hybrid Vectorized Trainer for Connect4 RL System

This module implements high-performance parallel environment training using the hybrid
vectorized approach where game logic runs on CPU and neural networks on GPU.

Key Features:
- CPU game logic with GPU neural network acceleration
- Scalable from 100 to 10,000+ parallel environments
- Efficient batched operations for maximum throughput
- Memory-efficient design with minimal CPU-GPU transfers
- Three difficulty levels: Small (100), Medium (1000), Impossible (10000)
"""

import sys
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import torch

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.utils.render import (
    Colors,
    render_training_header,
    render_training_progress,
    clear_screen,
)

try:
    from src.environments.hybrid_vectorized_connect4 import HybridVectorizedConnect4
    from src.agents.ppo_agent import PPOAgent
    from .training_statistics import TrainingStatistics
    from src.utils.checkpointing import CheckpointManager
except ImportError as e:
    print(f"Error importing Connect4 components: {e}")
    print("Make sure you're running from the project root directory")
    print("and that all dependencies are installed")
    sys.exit(1)


class HybridTrainer:
    """
    High-performance trainer using hybrid vectorized environments.
    
    Features three difficulty levels based on environment count:
    - Small: 100 environments (development/testing)
    - Medium: 1000 environments (standard training)
    - Impossible: 10000 environments (maximum performance)
    """
    
    # Difficulty level configurations
    DIFFICULTY_CONFIGS = {
        'small': {
            'num_envs': 100,
            'batch_size': 64,
            'update_frequency': 20,
            'description': 'Small scale - Good for development and testing'
        },
        'medium': {
            'num_envs': 1000,
            'batch_size': 256,
            'update_frequency': 50,
            'description': 'Medium scale - Standard training setup'
        },
        'impossible': {
            'num_envs': 10000,
            'batch_size': 512,
            'update_frequency': 100,
            'description': 'Impossible scale - Maximum performance training'
        }
    }
    
    def __init__(self, 
                 difficulty: str = 'medium',
                 device: str = 'auto',
                 config: Optional[Dict] = None):
        """
        Initialize the hybrid trainer.
        
        Args:
            difficulty: Training difficulty level ('small', 'medium', 'impossible')
            device: Device for neural networks ('auto', 'cpu', 'cuda')
            config: Optional configuration overrides
        """
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"Difficulty must be one of {list(self.DIFFICULTY_CONFIGS.keys())}")
        
        self.difficulty = difficulty
        self.config = self.DIFFICULTY_CONFIGS[difficulty].copy()
        if config:
            self.config.update(config)
        
        # Device setup
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        print(f"Initializing Hybrid Trainer:")
        print(f"  Difficulty: {difficulty.upper()} ({self.config['description']})")
        print(f"  Environments: {self.config['num_envs']:,}")
        print(f"  Device: {self.device}")
        
        # Initialize components
        self.vec_env = HybridVectorizedConnect4(
            num_envs=self.config['num_envs'], 
            device=self.device
        )
        
        self.agent = PPOAgent(device=self.device)
        self.statistics = TrainingStatistics()
        self.checkpoint_manager = CheckpointManager()
        
        # Training state
        self.episode_count = 0
        self.total_timesteps = 0
        self.experience_buffer = []
        
        print(f"Hybrid trainer initialized successfully!")
    
    def collect_experiences(self, num_steps: int) -> List[Dict[str, Any]]:
        """
        Collect experiences from all environments for specified number of steps.
        
        Args:
            num_steps: Number of steps to collect from each environment
            
        Returns:
            List of experience dictionaries
        """
        experiences = []
        observations = self.vec_env.get_observations_gpu()  # Start with current observations
        
        for step in range(num_steps):
            # Get valid moves for action masking
            valid_moves_tensor = self.vec_env.get_valid_moves_tensor()
            
            # Get actions from agent (batched)
            with torch.no_grad():
                actions, log_probs, values = self.agent.get_actions_batch(
                    observations, valid_moves_tensor
                )
            
            # Step all environments
            next_observations, rewards, dones, info = self.vec_env.step_batch(actions)
            
            # Store experiences
            batch_experiences = []
            for env_id in range(self.config['num_envs']):
                experience = {
                    'obs': observations[env_id].cpu().numpy(),
                    'action': actions[env_id].cpu().item(),
                    'reward': rewards[env_id].cpu().item(),
                    'next_obs': next_observations[env_id].cpu().numpy(),
                    'done': dones[env_id].cpu().item(),
                    'log_prob': log_probs[env_id].cpu().item(),
                    'value': values[env_id].cpu().item(),
                    'env_id': env_id,
                    'timestep': self.total_timesteps + step
                }
                batch_experiences.append(experience)
            
            experiences.extend(batch_experiences)
            
            # Auto-reset finished environments
            reset_mask = self.vec_env.auto_reset_finished_games()
            
            # Update observations for next step
            observations = next_observations
            
            # Update statistics
            self.statistics.record_step_batch(info)
        
        self.total_timesteps += num_steps
        return experiences
    
    def train_episode_batch(self) -> Dict[str, float]:
        """
        Train on a batch of episodes using vectorized environments.
        
        Returns:
            Dictionary of training metrics
        """
        # Collect experiences
        experiences = self.collect_experiences(self.config['update_frequency'])
        
        # Train agent on collected experiences
        training_metrics = self.agent.train_on_experiences(experiences)
        
        # Update episode count (approximation based on done flags)
        num_episodes = len([exp for exp in experiences if exp['done']])
        self.episode_count += num_episodes
        
        # Record training metrics
        self.statistics.record_training_batch(training_metrics, num_episodes)
        
        return training_metrics
    
    def train(self, 
              max_episodes: int = 10000,
              checkpoint_interval: int = 1000,
              log_interval: int = 100,
              eval_interval: int = 500) -> Dict[str, Any]:
        """
        Main training loop using hybrid vectorized environments.
        
        Args:
            max_episodes: Maximum episodes to train
            checkpoint_interval: Save checkpoint every N episodes
            log_interval: Log progress every N episodes
            eval_interval: Evaluate agent every N episodes
            
        Returns:
            Training results dictionary
        """
        print(f"\nStarting training with {self.config['num_envs']:,} environments")
        print(f"Target episodes: {max_episodes:,}")
        
        training_start_time = time.time()
        
        # Reset environments
        self.vec_env.reset()
        
        # Training loop
        if TQDM_AVAILABLE:
            pbar = tqdm(total=max_episodes, desc="Training Progress")
        
        while self.episode_count < max_episodes:
            batch_start_time = time.time()
            
            # Train on episode batch
            training_metrics = self.train_episode_batch()
            
            batch_time = time.time() - batch_start_time
            
            # Update progress bar
            if TQDM_AVAILABLE:
                pbar.update(min(self.episode_count - pbar.n, max_episodes - pbar.n))
                pbar.set_postfix({
                    'Envs': f"{self.config['num_envs']:,}",
                    'Win%': f"{training_metrics.get('win_rate', 0):.1f}",
                    'Loss': f"{training_metrics.get('policy_loss', 0):.3f}",
                    'FPS': f"{self.config['num_envs'] * self.config['update_frequency'] / batch_time:.0f}"
                })
            
            # Logging
            if self.episode_count % log_interval == 0:
                self._log_progress(training_metrics, batch_time)
            
            # Checkpointing
            if self.episode_count % checkpoint_interval == 0:
                self._save_checkpoint()
            
            # Evaluation
            if self.episode_count % eval_interval == 0:
                eval_results = self._evaluate_agent()
                self.statistics.record_evaluation(eval_results)
        
        if TQDM_AVAILABLE:
            pbar.close()
        
        training_time = time.time() - training_start_time
        
        # Final results
        final_results = {
            'total_episodes': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'training_time': training_time,
            'episodes_per_second': self.episode_count / training_time,
            'difficulty': self.difficulty,
            'num_envs': self.config['num_envs'],
            'final_metrics': training_metrics,
            'statistics': self.statistics.get_summary()
        }
        
        print(f"\nTraining completed!")
        print(f"Total episodes: {self.episode_count:,}")
        print(f"Training time: {training_time:.1f}s")
        print(f"Episodes/second: {final_results['episodes_per_second']:.1f}")
        print(f"Environments/second: {self.config['num_envs'] * self.config['update_frequency'] / (training_time / (self.episode_count // self.config['update_frequency'])):.0f}")
        
        return final_results
    
    def _log_progress(self, metrics: Dict[str, float], batch_time: float):
        """Log training progress."""
        env_steps_per_sec = self.config['num_envs'] * self.config['update_frequency'] / batch_time
        
        print(f"\nEpisode {self.episode_count:,}/{self.config['num_envs']:,} envs")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Policy Loss: {metrics.get('policy_loss', 0):.4f}")
        print(f"  Value Loss: {metrics.get('value_loss', 0):.4f}")
        print(f"  Environment Steps/sec: {env_steps_per_sec:.0f}")
        print(f"  Memory Usage: {self.vec_env.get_statistics()['memory_allocated_mb']:.1f}MB")
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        checkpoint_data = {
            'agent_state': self.agent.get_state(),
            'episode_count': self.episode_count,
            'total_timesteps': self.total_timesteps,
            'statistics': self.statistics.get_state(),
            'config': self.config,
            'difficulty': self.difficulty
        }
        
        self.checkpoint_manager.save_checkpoint(
            self.agent,
            self.episode_count,
            checkpoint_data
        )
        print(f"Checkpoint saved at episode {self.episode_count}")
    
    def _evaluate_agent(self, num_eval_games: int = 100) -> Dict[str, float]:
        """Evaluate agent performance."""
        # Simple evaluation using existing environments
        eval_stats = self.vec_env.get_statistics()
        
        return {
            'games_finished': eval_stats['games_finished'],
            'avg_moves_per_game': eval_stats['avg_moves_per_game'],
            'total_moves': eval_stats['total_moves']
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'trainer_stats': {
                'difficulty': self.difficulty,
                'num_envs': self.config['num_envs'],
                'episode_count': self.episode_count,
                'total_timesteps': self.total_timesteps
            },
            'environment_stats': self.vec_env.get_statistics(),
            'training_stats': self.statistics.get_summary()
        }


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Hybrid Trainer')
    parser.add_argument('--difficulty', choices=['small', 'medium', 'impossible'], 
                       default='small', help='Training difficulty level')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of episodes to train')
    args = parser.parse_args()
    
    # Test the trainer
    trainer = HybridTrainer(difficulty=args.difficulty)
    
    print(f"\nTesting {args.difficulty} difficulty with {args.episodes} episodes")
    
    # Quick training test
    results = trainer.train(max_episodes=args.episodes, log_interval=50)
    
    print(f"\nTraining Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")