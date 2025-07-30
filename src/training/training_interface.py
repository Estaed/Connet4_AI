"""
Training Interface for Connect4 RL System - Hybrid Edition

This module provides the interactive menu-driven training interface with the new
hybrid vectorized environment approach, including:
- Three difficulty levels: Small (100), Medium (1000), Impossible (10000) environments
- Hybrid CPU/GPU training for optimal performance
- Model checkpoint management and resumption
- Training settings and configuration
- Real-time performance monitoring
"""

import sys
import os
import time
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from src.utils.render import (
    Colors,
    clear_screen,
)

try:
    from src.core.config import get_config
    from src.utils.checkpointing import CheckpointManager
    from .hybrid_trainer import HybridTrainer
    import torch
except ImportError as e:
    print(f"Error importing Connect4 components: {e}")
    print("Make sure you're running from the project root directory")
    print("and that all dependencies are installed")
    sys.exit(1)


class TrainingInterface:
    """
    Interactive training interface for Connect4 RL system using hybrid vectorized environments.
    Provides menu-driven training with three difficulty levels optimized for different use cases.
    """
    
    def __init__(self):
        """Initialize the training interface."""
        self.config = self._load_config()
        self.checkpoint_manager = CheckpointManager()
        
        # Display system information
        self._display_system_info()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with fallback defaults."""
        try:
            config = get_config()
            return config
        except Exception as e:
            print(f"{Colors.WARNING}Warning: Could not load config ({e}). Using defaults.{Colors.RESET}")
            return {
                'training.max_episodes': 10000,
                'training.checkpoint_interval': 1000,
                'training.log_interval': 100,
                'training.eval_interval': 500
            }
    
    def _display_system_info(self):
        """Display system capabilities for training."""
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.INFO}HYBRID TRAINING SYSTEM READY{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        # GPU information
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"{Colors.SUCCESS}GPU Detected: {gpu_name} ({gpu_memory:.1f}GB){Colors.RESET}")
        else:
            print(f"{Colors.WARNING}No GPU detected - will use CPU for neural networks{Colors.RESET}")
        
        # System specs
        import psutil
        ram_total = psutil.virtual_memory().total / 1024**3
        ram_available = psutil.virtual_memory().available / 1024**3
        print(f"{Colors.INFO}RAM: {ram_total:.1f}GB total, {ram_available:.1f}GB available{Colors.RESET}")
        
        print(f"{Colors.INFO}Training System: Hybrid (CPU game logic + GPU neural networks){Colors.RESET}")
    
    def display_training_menu(self) -> None:
        """Display the hybrid training difficulty selection menu."""
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.INFO}TRAINING DIFFICULTY SELECTION{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        print(f"\n{Colors.INFO}Choose your training difficulty level:{Colors.RESET}")
        
        # Small difficulty
        print(f"\n{Colors.SUCCESS}1. SMALL SCALE{Colors.RESET}")
        print(f"   - Environments: {Colors.WARNING}100{Colors.RESET}")
        print(f"   - Best for: Development, testing, debugging")
        print(f"   - Memory usage: ~80MB RAM")
        print(f"   - Training speed: Fast startup, moderate throughput")
        
        # Medium difficulty
        print(f"\n{Colors.WARNING}2. MEDIUM SCALE{Colors.RESET}")
        print(f"   - Environments: {Colors.WARNING}1,000{Colors.RESET}")
        print(f"   - Best for: Standard training, balanced performance")
        print(f"   - Memory usage: ~800MB RAM")
        print(f"   - Training speed: Good balance of speed and stability")
        
        # Impossible difficulty
        print(f"\n{Colors.ERROR}3. IMPOSSIBLE SCALE{Colors.RESET}")
        print(f"   - Environments: {Colors.ERROR}10,000{Colors.RESET}")
        print(f"   - Best for: Maximum performance, production training")
        print(f"   - Memory usage: ~8GB RAM")
        print(f"   - Training speed: Maximum throughput")
        
        # Additional options
        print(f"\n{Colors.INFO}4. RESUME TRAINING{Colors.RESET} - Continue from checkpoint")
        print(f"{Colors.INFO}5. BENCHMARK MODE{Colors.RESET} - Test performance")
        print(f"{Colors.INFO}6. BACK TO MAIN MENU{Colors.RESET}")
        
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
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
                    print(f"{Colors.ERROR}Invalid choice. Please enter one of: {valid_choices}{Colors.RESET}")
            except KeyboardInterrupt:
                print(f"\n\n{Colors.WARNING}Training interrupted by user. Goodbye!{Colors.RESET}")
                sys.exit(0)
            except EOFError:
                print(f"\n\n{Colors.WARNING}Input ended. Goodbye!{Colors.RESET}")
                sys.exit(0)
    
    def run_training_difficulty(self, difficulty: str) -> Optional[Dict[str, Any]]:
        """
        Run training with specified difficulty level.
        
        Args:
            difficulty: Difficulty level ('small', 'medium', 'impossible')
            
        Returns:
            Training results dictionary or None if cancelled
        """
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}INITIALIZING {difficulty.upper()} TRAINING{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        # Get training parameters
        max_episodes = self._get_episode_count(difficulty)
        if max_episodes is None:
            return None
        
        # Initialize trainer
        try:
            print(f"{Colors.INFO}Creating hybrid trainer...{Colors.RESET}")
            trainer = HybridTrainer(difficulty=difficulty)
            
            print(f"{Colors.SUCCESS}Trainer initialized successfully!{Colors.RESET}")
            
            # Display training information
            config = trainer.config
            print(f"\n{Colors.INFO}Training Configuration:{Colors.RESET}")
            print(f"  Environments: {config['num_envs']:,}")
            print(f"  Batch size: {config['batch_size']}")
            print(f"  Update frequency: {config['update_frequency']}")
            print(f"  Target episodes: {max_episodes:,}")
            print(f"  Device: {trainer.device}")
            
            # Confirm start
            confirm = self.get_user_choice(
                f"\n{Colors.WARNING}Start training? (y/n): {Colors.RESET}",
                ['y', 'n', 'yes', 'no']
            ).lower()
            
            if confirm in ['n', 'no']:
                print(f"{Colors.INFO}Training cancelled.{Colors.RESET}")
                return None
            
            # Start training
            print(f"\n{Colors.SUCCESS}Starting training...{Colors.RESET}")
            start_time = time.time()
            
            results = trainer.train(
                max_episodes=max_episodes,
                checkpoint_interval=self.config.get('training.checkpoint_interval', 1000),
                log_interval=self.config.get('training.log_interval', 100),
                eval_interval=self.config.get('training.eval_interval', 500)
            )
            
            training_time = time.time() - start_time
            
            # Display results
            self._display_training_results(results, training_time)
            
            return results
            
        except Exception as e:
            print(f"{Colors.ERROR}Training failed: {e}{Colors.RESET}")
            print(f"{Colors.INFO}Check system resources and try a smaller difficulty level.{Colors.RESET}")
            return None
    
    def _get_episode_count(self, difficulty: str) -> Optional[int]:
        """Get number of episodes to train based on difficulty."""
        default_episodes = {
            'small': 5000,
            'medium': 10000,
            'impossible': 50000
        }
        
        default = default_episodes.get(difficulty, 10000)
        
        print(f"\n{Colors.INFO}Episode Configuration:{Colors.RESET}")
        print(f"Default episodes for {difficulty}: {default:,}")
        
        choice = self.get_user_choice(
            f"{Colors.INFO}Use default or custom? (d/c): {Colors.RESET}",
            ['d', 'c', 'default', 'custom']
        ).lower()
        
        if choice in ['d', 'default']:
            return default
        
        # Custom episode count
        while True:
            try:
                episodes_str = input(f"{Colors.INFO}Enter number of episodes: {Colors.RESET}")
                episodes = int(episodes_str)
                if episodes > 0:
                    return episodes
                else:
                    print(f"{Colors.ERROR}Please enter a positive number.{Colors.RESET}")
            except ValueError:
                print(f"{Colors.ERROR}Please enter a valid number.{Colors.RESET}")
            except KeyboardInterrupt:
                print(f"\n{Colors.INFO}Cancelled.{Colors.RESET}")
                return None
    
    def _display_training_results(self, results: Dict[str, Any], training_time: float):
        """Display training completion results."""
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}TRAINING COMPLETED!{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        print(f"\n{Colors.INFO}Training Summary:{Colors.RESET}")
        print(f"  Total Episodes: {results['total_episodes']:,}")
        print(f"  Total Timesteps: {results['total_timesteps']:,}")
        print(f"  Training Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
        print(f"  Episodes/Second: {results['episodes_per_second']:.1f}")
        print(f"  Difficulty: {results['difficulty'].upper()}")
        print(f"  Environments: {results['num_envs']:,}")
        
        final_metrics = results.get('final_metrics', {})
        if final_metrics:
            print(f"\n{Colors.INFO}Final Performance:{Colors.RESET}")
            print(f"  Win Rate: {final_metrics.get('win_rate', 0):.1f}%")
            print(f"  Policy Loss: {final_metrics.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {final_metrics.get('value_loss', 0):.4f}")
        
        print(f"\n{Colors.SUCCESS}Model saved and ready for gameplay!{Colors.RESET}")
    
    def resume_training(self) -> Optional[Dict[str, Any]]:
        """Resume training from a checkpoint."""
        print(f"\n{Colors.INFO}Resume training functionality will be implemented.{Colors.RESET}")
        print(f"{Colors.INFO}For now, checkpoints are automatically saved during training.{Colors.RESET}")
        return None
    
    def benchmark_mode(self):
        """Run benchmark to test system performance."""
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}BENCHMARK MODE{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        print(f"{Colors.INFO}Testing all difficulty levels for 100 episodes each...{Colors.RESET}")
        
        for difficulty in ['small', 'medium', 'impossible']:
            print(f"\n{Colors.WARNING}Testing {difficulty.upper()}...{Colors.RESET}")
            
            try:
                trainer = HybridTrainer(difficulty=difficulty)
                
                start_time = time.time()
                results = trainer.train(max_episodes=100, log_interval=50)
                benchmark_time = time.time() - start_time
                
                print(f"  SUCCESS {difficulty.capitalize()}: {results['episodes_per_second']:.1f} eps/s")
                print(f"     Time: {benchmark_time:.1f}s, Memory: {trainer.vec_env.get_statistics()['memory_allocated_mb']:.1f}MB")
                
                del trainer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"  FAILED {difficulty.capitalize()}: Failed ({e})")
        
        print(f"\n{Colors.SUCCESS}Benchmark completed!{Colors.RESET}")
    
    def run(self) -> None:
        """Main training interface loop."""
        print(f"{Colors.SUCCESS}Welcome to Connect4 Hybrid Training System!{Colors.RESET}")
        
        while True:
            self.display_training_menu()
            
            choice = self.get_user_choice(
                f"{Colors.INFO}Enter your choice (1-6): {Colors.RESET}",
                ["1", "2", "3", "4", "5", "6"]
            )
            
            if choice == "1":
                self.run_training_difficulty('small')
            elif choice == "2":
                self.run_training_difficulty('medium')
            elif choice == "3":
                self.run_training_difficulty('impossible')
            elif choice == "4":
                self.resume_training()
            elif choice == "5":
                self.benchmark_mode()
            elif choice == "6":
                print(f"{Colors.SUCCESS}Returning to main menu...{Colors.RESET}")
                break
            
            # Pause before showing menu again
            if choice in ["1", "2", "3", "4", "5"]:
                input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")


# Example usage
if __name__ == "__main__":
    interface = TrainingInterface()
    interface.run()