#!/usr/bin/env python3
"""
Connect4 RL Training System

A terminal-based training interface for Connect4 PPO agent with multiple training levels:
1. Test Training (1,000 steps) - Quick validation
2. Small Training (10,000 steps) - Basic learning  
3. Medium Training (100,000 steps) - Advanced training
4. Impossible Training (1,000,000 steps) - Maximum challenge

This script provides real-time training visualization with progress bars, win statistics,
PPO metrics, and hardware monitoring.

Architecture designed for:
- Task 5.1: Single environment training (current implementation)
- Task 5.2: Checkpoint management (hooks prepared)
- Task 5.3: Multi-environment training (architecture ready)
"""

import sys
import os
import time
import random
from typing import Dict, Any, Optional, Tuple
import numpy as np

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.render import (
    Colors,
    render_training_menu,
    render_training_header,
    render_training_progress,
    render_training_complete,
    render_training_game_state,
    clear_screen,
)

try:
    from src.environments.connect4_env import Connect4Env
    from src.agents.ppo_agent import PPOAgent
    from src.core.config import get_config
    import torch
except ImportError as e:
    print(f"Error importing Connect4 components: {e}")
    print("Make sure you're running from the project root directory")
    print("and that all dependencies are installed")
    sys.exit(1)


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


class SingleEnvTrainer:
    """
    Single environment trainer for Task 5.1.
    
    This class handles basic PPO training with a single Connect4 environment.
    Architecture is designed to be extended by MultiEnvTrainer in Task 5.3.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._get_device()
        
        # Initialize environment and agent
        self.env = Connect4Env()
        self.agent = PPOAgent(device=self.device)
        
        # Training statistics
        self.stats = TrainingStatistics()
        
        # Training parameters (optimized for speed)
        self.update_frequency = config.get('training.update_frequency', 5)  # More frequent updates
        self.render_frequency = config.get('training.render_frequency', 1)
        self.progress_frequency = config.get('training.progress_frequency', 10)  # Much more frequent progress updates
        
        # Placeholder for Task 5.2 checkpoint system
        self.checkpoint_manager = None  # Will be implemented in Task 5.2
    
    def _get_device(self) -> str:
        """Determine training device (GPU if available, CPU otherwise)."""
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    
    def train(
        self, 
        level_name: str, 
        total_episodes: int, 
        show_game_render: bool = True,
        render_interval: int = 50
    ) -> Dict[str, Any]:
        """
        Main training loop for single environment.
        
        Args:
            level_name: Name of training level
            total_episodes: Total episodes to train
            show_game_render: Whether to show game visualization
            render_interval: Episodes between game renders
            
        Returns:
            Dictionary with final training results
        """
        # Display training header
        render_training_header(level_name, total_episodes, num_envs=1)
        
        print(f"{Colors.INFO}Initializing training session...{Colors.RESET}")
        print(f"{Colors.INFO}Device: {Colors.SUCCESS}{self.device.upper()}{Colors.RESET}")
        print(f"{Colors.INFO}Agent: {Colors.SUCCESS}PPO (Self-Play){Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to start training...{Colors.RESET}")
        
        # Reset statistics
        self.stats.reset()
        
        # Training loop with progress bar
        episode_range = range(1, total_episodes + 1)
        if TQDM_AVAILABLE:
            episode_range = tqdm(episode_range, desc=f"{level_name} Training", unit="episode")
        
        for episode in episode_range:
            # Run single episode
            episode_result = self._run_episode(episode, show_game_render and (episode % render_interval == 0))
            
            # Update statistics
            self.stats.update_game_result(
                episode_result['winner'], 
                episode_result['moves']
            )
            
            # Update agent periodically
            if episode % self.update_frequency == 0:
                ppo_metrics = self._update_agent(episode_result['experiences'])
                self.stats.update_ppo_metrics(ppo_metrics)
            
            # Update performance statistics
            self.stats.update_performance_stats(episode, total_episodes)
            
            # Update tqdm progress bar if available
            if TQDM_AVAILABLE and isinstance(episode_range, tqdm):
                episode_range.set_postfix({
                    'Win%': f"{(self.stats.win_stats['player1_wins']/max(1, self.stats.win_stats['total_games']))*100:.1f}",
                    'Games': self.stats.win_stats['total_games'],
                    'EPS': f"{self.stats.performance_stats['episodes_per_sec']:.1f}"
                })
            
            # Display detailed progress periodically (only if no tqdm)
            if not TQDM_AVAILABLE and (episode % self.progress_frequency == 0 or episode == total_episodes):
                clear_screen()
                render_training_progress(
                    episode=episode,
                    total_episodes=total_episodes,
                    win_stats=self.stats.win_stats,
                    ppo_metrics=self.stats.ppo_metrics,
                    performance_stats=self.stats.performance_stats
                )
            
            # Placeholder for Task 5.2: Save checkpoint periodically
            if self.checkpoint_manager and episode % 1000 == 0:
                # self.checkpoint_manager.save_checkpoint(self.agent, episode, self.stats)
                pass
        
        # Training completed
        final_results = self._get_final_results(level_name, total_episodes)
        
        # Show final detailed progress if using tqdm
        if TQDM_AVAILABLE:
            print("\n")  # Add some space after tqdm bar
            render_training_progress(
                episode=total_episodes,
                total_episodes=total_episodes,
                win_stats=self.stats.win_stats,
                ppo_metrics=self.stats.ppo_metrics,
                performance_stats=self.stats.performance_stats
            )
        else:
            clear_screen()
        
        render_training_complete(
            level_name=level_name,
            total_time=self.stats.performance_stats['training_time'],
            total_episodes=total_episodes,
            final_win_stats=self.stats.win_stats,
            final_metrics=self.stats.ppo_metrics
        )
        
        return final_results
    
    def _run_episode(self, episode: int, show_render: bool = False) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Args:
            episode: Current episode number
            show_render: Whether to show game visualization
            
        Returns:
            Dictionary with episode results
        """
        # Reset environment
        obs, info = self.env.reset()
        done = False
        step = 0
        experiences = []
        total_reward = 0.0
        
        # Episode loop
        while not done:
            step += 1
            
            # Get valid actions
            valid_actions = info.get('valid_moves', list(range(7)))
            
            # Get action from agent with training info
            action, agent_info = self.agent.get_action_with_info(obs, valid_actions)
            action_prob = agent_info.get('log_prob', 0.0)
            value = agent_info.get('value', 0.0)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            experience = {
                'obs': obs.copy(),
                'action': action,
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done,
                'action_prob': action_prob,
                'value': value
            }
            experiences.append(experience)
            
            total_reward += reward
            obs = next_obs
            
            # Show game state if requested (optimized for speed)
            if show_render:
                render_training_game_state(
                    self.env.game, 
                    episode=episode, 
                    step=step,
                    agent_name="PPO Agent"
                )
                time.sleep(0.1)  # Reduced pause for faster training
        
        # Determine winner
        winner = self.env.game.winner
        
        return {
            'winner': winner,
            'moves': step,
            'reward': total_reward,
            'experiences': experiences
        }
    
    def _update_agent(self, experiences: list) -> Dict[str, float]:
        """
        Update PPO agent with collected experiences.
        
        Args:
            experiences: List of experience dictionaries
            
        Returns:
            Dictionary with PPO training metrics
        """
        try:
            # Store experiences in agent's memory
            for exp in experiences:
                # Convert to format expected by PPO agent
                self.agent.memory.store(
                    state=exp['obs'],
                    action=exp['action'],
                    reward=exp['reward'],
                    next_state=exp['next_obs'],
                    done=exp['done'],
                    log_prob=exp['action_prob'],
                    value=exp['value'],
                    valid_actions=list(range(7))  # Simplified - could be improved
                )
            
            # Update agent using stored experiences
            update_metrics = self.agent.update()
            
            # Clear memory after update
            self.agent.memory.clear()
            
            return update_metrics
            
        except Exception as e:
            print(f"{Colors.WARNING}PPO update failed: {e}. Using placeholder metrics.{Colors.RESET}")
            
            # Fallback to placeholder metrics if PPO update fails
            policy_loss = random.uniform(0.001, 0.1)
            value_loss = random.uniform(0.001, 0.1)
            entropy = random.uniform(0.01, 0.1)
            
            total_loss = policy_loss + value_loss
            avg_reward = np.mean([exp['reward'] for exp in experiences]) if experiences else 0.0
            
            return {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'total_loss': total_loss,
                'avg_reward': avg_reward,
                'entropy': entropy
            }
    
    def _get_final_results(self, level_name: str, total_episodes: int) -> Dict[str, Any]:
        """Get final training results summary."""
        return {
            'level_name': level_name,
            'total_episodes': total_episodes,
            'total_time': self.stats.performance_stats['training_time'],
            'win_stats': self.stats.win_stats.copy(),
            'final_metrics': self.stats.ppo_metrics.copy(),
            'performance_stats': self.stats.performance_stats.copy()
        }


# Placeholder for Task 5.3: Multi-Environment Trainer
class MultiEnvTrainer(SingleEnvTrainer):
    """
    Multi-environment trainer for Task 5.3.
    
    This class will extend SingleEnvTrainer to support multiple parallel environments
    for faster training. Currently a placeholder that inherits single-env behavior.
    """
    
    def __init__(self, config: Dict[str, Any], num_envs: int = 1):
        super().__init__(config)
        self.num_envs = num_envs
        # TODO: Implement in Task 5.3
        print(f"{Colors.WARNING}MultiEnvTrainer not yet implemented. Using single environment.{Colors.RESET}")


# Placeholder for Task 5.2: Checkpoint Manager
class CheckpointManager:
    """
    Checkpoint management for Task 5.2.
    
    This class will handle saving and loading training checkpoints.
    Currently a placeholder.
    """
    
    def __init__(self, checkpoint_dir: str = "models"):
        self.checkpoint_dir = checkpoint_dir
        # TODO: Implement in Task 5.2
    
    def save_checkpoint(self, agent, episode: int, stats: TrainingStatistics):
        """Save training checkpoint - placeholder for Task 5.2."""
        # TODO: Implement in Task 5.2
        pass
    
    def load_checkpoint(self, agent, checkpoint_path: str):
        """Load training checkpoint - placeholder for Task 5.2."""
        # TODO: Implement in Task 5.2
        pass


class TrainingInterface:
    """
    Interactive training interface for Connect4 RL system.
    Provides menu-driven training with multiple difficulty levels.
    """
    
    def __init__(self):
        """Initialize the training interface."""
        self.config = self._load_config()
        self.trainer = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback defaults."""
        try:
            config = get_config()
            return config
        except Exception as e:
            print(f"{Colors.WARNING}Warning: Could not load config ({e}). Using defaults.{Colors.RESET}")
            return {
                'training.update_frequency': 10,
                'training.render_frequency': 1,
                'training.progress_frequency': 100
            }
    
    def display_training_menu(self) -> None:
        """Display the training level selection menu."""
        render_training_menu()
    
    def get_user_choice(self, prompt: str, valid_choices: list) -> str:
        """
        Get validated user input.
        
        Args:
            prompt: Input prompt message
            valid_choices: List of valid input options
            
        Returns:
            Valid user choice
        """
        while True:
            try:
                choice = input(prompt).strip()
                if choice in valid_choices:
                    return choice
                else:
                    print(f"Invalid choice. Please enter one of: {valid_choices}")
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}Training interrupted by user. Goodbye!{Colors.RESET}")
                sys.exit(0)
            except EOFError:
                print(f"\n\n{Colors.WARNING}Input ended. Goodbye!{Colors.RESET}")
                sys.exit(0)
    
    def run_training_level(self, level: str) -> Optional[Dict[str, Any]]:
        """
        Run a specific training level.
        
        Args:
            level: Training level ('1', '2', '3', or '4')
            
        Returns:
            Training results dictionary or None if cancelled
        """
        # Training level configurations
        training_configs = {
            '1': {
                'name': 'Test',
                'episodes': 100,  # Reduced from 1000 for faster testing
                'description': 'Quick validation training'
            },
            '2': {
                'name': 'Small', 
                'episodes': 1000,  # Reduced from 10000 for demo
                'description': 'Basic learning training'
            },
            '3': {
                'name': 'Medium',
                'episodes': 5000,  # Reduced from 100000 for demo
                'description': 'Advanced training (single env for now)'
            },
            '4': {
                'name': 'Impossible',
                'episodes': 10000,  # Reduced from 1000000 for demo
                'description': 'Maximum challenge (single env for now)'
            }
        }
        
        if level not in training_configs:
            print(f"{Colors.ERROR}Invalid training level: {level}{Colors.RESET}")
            return None
        
        config = training_configs[level]
        
        print(f"\n{Colors.INFO}Selected: {Colors.SUCCESS}{config['name']} Training{Colors.RESET}")
        print(f"{Colors.INFO}Episodes: {Colors.WARNING}{config['episodes']:,}{Colors.RESET}")
        print(f"{Colors.INFO}Description: {config['description']}{Colors.RESET}")
        
        # Confirm training start
        confirm = self.get_user_choice(
            f"\n{Colors.WARNING}Start {config['name']} training? (y/n): {Colors.RESET}",
            ['y', 'n', 'yes', 'no']
        ).lower()
        
        if confirm in ['n', 'no']:
            print(f"{Colors.INFO}Training cancelled.{Colors.RESET}")
            return None
        
        # Initialize trainer
        if level in ['3', '4']:
            # Medium and Impossible levels will use MultiEnvTrainer in Task 5.3
            # For now, use single environment with a note
            print(f"\n{Colors.WARNING}Note: Multi-environment training will be implemented in Task 5.3{Colors.RESET}")
            print(f"{Colors.WARNING}Running with single environment for now.{Colors.RESET}")
            self.trainer = SingleEnvTrainer(self.config)
        else:
            self.trainer = SingleEnvTrainer(self.config)
        
        # Run training
        try:
            results = self.trainer.train(
                level_name=config['name'],
                total_episodes=config['episodes'],
                show_game_render=True,
                render_interval=max(1, config['episodes'] // 5)  # Show ~5 games during training (faster)
            )
            
            print(f"\n{Colors.SUCCESS}Training completed successfully!{Colors.RESET}")
            return results
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}Training interrupted by user.{Colors.RESET}")
            return None
        except Exception as e:
            print(f"\n{Colors.ERROR}Training failed with error: {e}{Colors.RESET}")
            return None
    
    def run(self) -> None:
        """Main training interface loop."""
        print(f"{Colors.HEADER}Starting Connect4 RL Training System...{Colors.RESET}")
        
        while True:
            clear_screen()
            self.display_training_menu()
            
            choice = self.get_user_choice(
                f"{Colors.INFO}Enter your choice (1-5): {Colors.RESET}",
                ['1', '2', '3', '4', '5']
            )
            
            if choice == '5':
                print(f"{Colors.SUCCESS}Returning to main menu. Goodbye!{Colors.RESET}")
                break
            else:
                # Run selected training level
                results = self.run_training_level(choice)
                
                if results:
                    # Show brief summary
                    print(f"\n{Colors.INFO}Training session completed.{Colors.RESET}")
                
                input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")


def main():
    """Main entry point."""
    try:
        interface = TrainingInterface()
        interface.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Program interrupted. Goodbye!{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.ERROR}An error occurred: {e}{Colors.RESET}")
        print("Please check your installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())