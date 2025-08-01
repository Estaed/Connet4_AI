"""
Training Statistics Management for Connect4 RL System

This module provides comprehensive statistics tracking for training sessions,
including win rates, PPO metrics, and performance monitoring.
"""

import time
from typing import Dict, Any, List


class TrainingStatistics:
    """Manages training statistics and metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.win_stats = {
            'total_games': 0,
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'avg_game_length': 0.0,
            'total_moves': 0
        }
        
        self.ppo_metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'total_loss': 0.0,
            'avg_reward': 0.0,
            'entropy': 0.0,
            'updates_count': 0
        }
        
        self.performance_stats = {
            'episodes_per_sec': 0.0,
            'games_per_sec': 0.0,
            'training_time': 0.0,
            'eta': 0.0,
            'start_time': time.time()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary for checkpointing."""
        return {
            'win_stats': self.win_stats.copy(),
            'ppo_metrics': self.ppo_metrics.copy(),
            'performance_stats': self.performance_stats.copy()
        }
    
    def update_game_result(self, winner: int, moves: int):
        """Update statistics after a game completion."""
        self.win_stats['total_games'] += 1
        self.win_stats['total_moves'] += moves
        
        if winner == 1:
            self.win_stats['player1_wins'] += 1
        elif winner == -1:
            self.win_stats['player2_wins'] += 1
        else:
            self.win_stats['draws'] += 1
        
        # Update average game length
        self.win_stats['avg_game_length'] = (
            self.win_stats['total_moves'] / self.win_stats['total_games']
        )
    
    def update_ppo_metrics(self, metrics: Dict[str, float]):
        """Update PPO training metrics."""
        for key, value in metrics.items():
            if key in self.ppo_metrics:
                # Use exponential moving average for smooth updates
                alpha = 0.1  # Smoothing factor
                self.ppo_metrics[key] = (
                    alpha * value + (1 - alpha) * self.ppo_metrics[key]
                )
        self.ppo_metrics['updates_count'] += 1
    
    def update_performance_stats(self, episode: int, total_episodes: int):
        """Update performance statistics."""
        current_time = time.time()
        elapsed_time = current_time - self.performance_stats['start_time']
        
        self.performance_stats['training_time'] = elapsed_time
        
        if elapsed_time > 0:
            self.performance_stats['episodes_per_sec'] = episode / elapsed_time
            self.performance_stats['games_per_sec'] = (
                self.win_stats['total_games'] / elapsed_time
            )
            
            # Estimate time remaining
            if episode > 0:
                time_per_episode = elapsed_time / episode
                remaining_episodes = total_episodes - episode
                self.performance_stats['eta'] = remaining_episodes * time_per_episode
    
    def record_step_batch(self, info_batch: List[Dict[str, Any]]) -> None:
        """Record information from a batch of environment steps."""
        if not info_batch:
            return
            
        # Process batch information - count games finished and moves
        for info in info_batch:
            if isinstance(info, dict):
                # Handle completed games
                if info.get('game_finished', False):
                    winner = info.get('winner', 0)
                    moves = info.get('moves', 0)
                    self.update_game_result(winner, moves)
    
    def record_training_batch(self, training_metrics: Dict[str, float], num_episodes: int):
        """Record metrics from a training batch."""
        # Update PPO metrics
        self.update_ppo_metrics(training_metrics)
        
        # Update performance stats if episodes completed
        if num_episodes > 0:
            self.update_performance_stats(self.win_stats['total_games'], 
                                        self.win_stats['total_games'] + num_episodes)
    
    def record_evaluation(self, eval_results: Dict[str, Any]):
        """Record evaluation results."""
        # Store evaluation results in performance stats
        if isinstance(eval_results, dict):
            for key, value in eval_results.items():
                eval_key = f"eval_{key}"
                self.performance_stats[eval_key] = value
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of all statistics."""
        return {
            'win_stats': self.win_stats.copy(),
            'ppo_metrics': self.ppo_metrics.copy(),
            'performance_stats': self.performance_stats.copy(),
            'win_rate': (self.win_stats['player1_wins'] / max(1, self.win_stats['total_games'])) * 100,
            'draw_rate': (self.win_stats['draws'] / max(1, self.win_stats['total_games'])) * 100
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing (same as to_dict but more explicit)."""
        return self.to_dict()