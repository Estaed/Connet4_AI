"""
Logging and TensorBoard Integration for Connect4 RL Training

Provides comprehensive logging functionality with multiple output targets:
- Console logging with verbosity control (QUIET, NORMAL, DEBUG)
- TensorBoard integration for real-time training visualization
- Performance metrics tracking and visualization
- Training progress monitoring

Features:
- Real-time loss curves and metrics visualization
- Win rate progression over time
- Performance monitoring (FPS, memory usage)
- Hardware utilization tracking
- Model architecture visualization
- Configurable log levels and output formats
"""

import os
import time
import sys
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import warnings

# Try to import TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

# Try to import torch for GPU monitoring
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available for GPU monitoring")
    TORCH_AVAILABLE = False
    torch = None


class LogLevel(Enum):
    """Logging verbosity levels."""
    QUIET = 0      # Only critical messages
    NORMAL = 1     # Standard progress information
    DEBUG = 2      # Detailed debugging information


class TrainingLogger:
    """
    Comprehensive training logger with console and TensorBoard integration.
    
    Handles all logging needs for Connect4 RL training including:
    - Console output with configurable verbosity
    - TensorBoard logging for visualization
    - Performance metrics tracking
    - Training progress monitoring
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
        tensorboard_dir: Optional[Union[str, Path]] = None,
        log_level: LogLevel = LogLevel.NORMAL,
        enable_tensorboard: bool = True,
        experiment_name: Optional[str] = None,
        flush_secs: int = 30
    ):
        """
        Initialize the training logger.
        
        Args:
            log_dir: Base directory for all logs
            tensorboard_dir: Specific directory for TensorBoard logs (defaults to log_dir)
            log_level: Console logging verbosity level
            enable_tensorboard: Whether to enable TensorBoard logging
            experiment_name: Name for this training experiment
            flush_secs: Seconds between TensorBoard flushes
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up experiment name
        if experiment_name is None:
            timestamp = int(time.time())
            experiment_name = f"connect4_training_{timestamp}"
        self.experiment_name = experiment_name
        
        # Console logging setup
        self.log_level = log_level
        
        # TensorBoard setup
        self.enable_tensorboard = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.tensorboard_writer = None
        
        if self.enable_tensorboard:
            if tensorboard_dir is None:
                tensorboard_dir = self.log_dir / experiment_name
            else:
                tensorboard_dir = Path(tensorboard_dir) / experiment_name
            
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=str(tensorboard_dir),
                    flush_secs=flush_secs
                )
                self.info(f"TensorBoard logging enabled: {tensorboard_dir}")
                self.info("View with: tensorboard --logdir logs")
            except Exception as e:
                self.warning(f"Failed to initialize TensorBoard: {e}")
                self.enable_tensorboard = False
        
        # Performance tracking
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_count = 0
        
        self.info(f"Training logger initialized for experiment: {experiment_name}")
        self.info(f"Log level: {log_level.name}")
        self.info(f"TensorBoard: {'Enabled' if self.enable_tensorboard else 'Disabled'}")
    
    def set_level(self, level: LogLevel) -> None:
        """Set console logging level."""
        self.log_level = level
        self.debug(f"Log level changed to: {level.name}")
    
    def quiet(self, message: str, **kwargs) -> None:
        """Always print (for critical messages)."""
        print(f"[CRITICAL] {message}", **kwargs)
        sys.stdout.flush()
    
    def info(self, message: str, **kwargs) -> None:
        """Print in NORMAL and DEBUG modes."""
        if self.log_level.value >= LogLevel.NORMAL.value:
            print(f"[INFO] {message}", **kwargs)
            sys.stdout.flush()
    
    def debug(self, message: str, **kwargs) -> None:
        """Print only in DEBUG mode."""
        if self.log_level.value >= LogLevel.DEBUG.value:
            print(f"[DEBUG] {message}", **kwargs)
            sys.stdout.flush()
    
    def warning(self, message: str, **kwargs) -> None:
        """Print warnings (respects quiet mode)."""
        if self.log_level.value >= LogLevel.NORMAL.value:
            print(f"[WARNING] {message}", **kwargs)
            sys.stdout.flush()
    
    def error(self, message: str, **kwargs) -> None:
        """Always print errors."""
        print(f"[ERROR] {message}", **kwargs)
        sys.stdout.flush()
    
    def progress(self, message: str, **kwargs) -> None:
        """Progress updates (respects quiet mode)."""
        if self.log_level.value >= LogLevel.NORMAL.value:
            print(f"[PROGRESS] {message}", **kwargs)
            sys.stdout.flush()
    
    def log_training_metrics(
        self,
        episode: int,
        metrics: Dict[str, Any],
        prefix: str = "training"
    ) -> None:
        """
        Log training metrics to both console and TensorBoard.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metrics to log
            prefix: Prefix for TensorBoard metric names
        """
        self.step_count = episode
        
        # Console logging
        if self.log_level.value >= LogLevel.DEBUG.value:
            self.debug(f"Episode {episode} - Metrics: {metrics}")
        
        # TensorBoard logging
        if self.enable_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f"{prefix}/{key}", value, episode)
    
    def log_win_statistics(
        self,
        episode: int,
        win_stats: Dict[str, Any]
    ) -> None:
        """
        Log win rate statistics to TensorBoard.
        
        Args:
            episode: Current episode number
            win_stats: Dictionary with win statistics
        """
        if not (self.enable_tensorboard and self.tensorboard_writer):
            return
        
        total_games = win_stats.get('total_games', 0)
        if total_games > 0:
            # Calculate win rates
            p1_rate = (win_stats.get('player1_wins', 0) / total_games) * 100
            p2_rate = (win_stats.get('player2_wins', 0) / total_games) * 100
            draw_rate = (win_stats.get('draws', 0) / total_games) * 100
            
            # Log to TensorBoard
            self.tensorboard_writer.add_scalar('win_rates/player1', p1_rate, episode)
            self.tensorboard_writer.add_scalar('win_rates/player2', p2_rate, episode)
            self.tensorboard_writer.add_scalar('win_rates/draws', draw_rate, episode)
            self.tensorboard_writer.add_scalar('game_stats/total_games', total_games, episode)
            self.tensorboard_writer.add_scalar('game_stats/avg_game_length', 
                                             win_stats.get('avg_game_length', 0), episode)
    
    def log_performance_metrics(
        self,
        episode: int,
        performance_stats: Dict[str, Any]
    ) -> None:
        """
        Log performance metrics to TensorBoard.
        
        Args:
            episode: Current episode number
            performance_stats: Dictionary with performance statistics
        """
        if not (self.enable_tensorboard and self.tensorboard_writer):
            return
        
        # Log performance metrics
        for key, value in performance_stats.items():
            if isinstance(value, (int, float)) and key != 'start_time':
                self.tensorboard_writer.add_scalar(f'performance/{key}', value, episode)
    
    def log_hardware_stats(self, episode: int) -> None:
        """
        Log hardware utilization statistics.
        
        Args:
            episode: Current episode number
        """
        if not (self.enable_tensorboard and self.tensorboard_writer and TORCH_AVAILABLE):
            return
        
        try:
            # GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
                
                self.tensorboard_writer.add_scalar('hardware/gpu_memory_allocated', 
                                                 memory_allocated, episode)
                self.tensorboard_writer.add_scalar('hardware/gpu_memory_cached', 
                                                 memory_cached, episode)
            
            # CPU usage would require psutil, skip for now to avoid additional dependencies
        except Exception as e:
            self.debug(f"Failed to log hardware stats: {e}")
    
    def log_model_architecture(self, model: Any, input_shape: tuple = (1, 1, 6, 7)) -> None:
        """
        Log model architecture to TensorBoard.
        
        Args:
            model: PyTorch model to visualize
            input_shape: Input shape for the model
        """
        if not (self.enable_tensorboard and self.tensorboard_writer and TORCH_AVAILABLE):
            return
        
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Add model graph to TensorBoard
            self.tensorboard_writer.add_graph(model, dummy_input)
            self.info("Model architecture logged to TensorBoard")
        except Exception as e:
            self.warning(f"Failed to log model architecture: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Log hyperparameters and final metrics to TensorBoard.
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        if not (self.enable_tensorboard and self.tensorboard_writer):
            return
        
        try:
            # Filter out non-serializable values
            clean_hparams = {}
            for key, value in hparams.items():
                if isinstance(value, (int, float, str, bool)):
                    clean_hparams[key] = value
                else:
                    clean_hparams[key] = str(value)
            
            self.tensorboard_writer.add_hparams(clean_hparams, metrics)
            self.info("Hyperparameters logged to TensorBoard")
        except Exception as e:
            self.warning(f"Failed to log hyperparameters: {e}")
    
    def log_training_progress_summary(
        self,
        episode: int,
        total_episodes: int,
        win_stats: Dict[str, Any],
        ppo_metrics: Dict[str, Any],
        performance_stats: Dict[str, Any]
    ) -> None:
        """
        Log comprehensive training progress (console + TensorBoard).
        
        Args:
            episode: Current episode number
            total_episodes: Total episodes for training
            win_stats: Win statistics dictionary
            ppo_metrics: PPO training metrics
            performance_stats: Performance statistics
        """
        # Console progress
        progress_percent = (episode / total_episodes) * 100 if total_episodes > 0 else 0
        
        if episode % 100 == 0 or episode == total_episodes:  # Log every 100 episodes
            self.info(f"Training Progress - Episode {episode:,}/{total_episodes:,} ({progress_percent:.1f}%)")
            
            if win_stats.get('total_games', 0) > 0:
                total_games = win_stats['total_games']
                p1_rate = (win_stats.get('player1_wins', 0) / total_games) * 100
                self.info(f"  Win Rate: {p1_rate:.1f}% | Games: {total_games:,} | "
                         f"Avg Length: {win_stats.get('avg_game_length', 0):.1f}")
            
            if ppo_metrics:
                self.info(f"  PPO Loss: {ppo_metrics.get('total_loss', 0):.6f} | "
                         f"Reward: {ppo_metrics.get('avg_reward', 0):.3f}")
            
            if performance_stats:
                self.info(f"  Speed: {performance_stats.get('episodes_per_sec', 0):.1f} eps/sec | "
                         f"Time: {performance_stats.get('training_time', 0):.1f}s")
        
        # TensorBoard logging
        self.log_training_metrics(episode, ppo_metrics, "ppo")
        self.log_win_statistics(episode, win_stats)
        self.log_performance_metrics(episode, performance_stats)
        self.log_hardware_stats(episode)
    
    def log_training_completion(
        self,
        total_episodes: int,
        total_time: float,
        final_win_stats: Dict[str, Any],
        final_metrics: Dict[str, Any]
    ) -> None:
        """
        Log training completion summary.
        
        Args:
            total_episodes: Total episodes completed
            total_time: Total training time
            final_win_stats: Final win statistics
            final_metrics: Final training metrics
        """
        self.info("=" * 60)
        self.info("TRAINING COMPLETED!")
        self.info("=" * 60)
        self.info(f"Total Episodes: {total_episodes:,}")
        self.info(f"Total Time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
        self.info(f"Episodes/sec: {total_episodes/total_time:.2f}")
        
        if final_win_stats.get('total_games', 0) > 0:
            total_games = final_win_stats['total_games']
            p1_rate = (final_win_stats.get('player1_wins', 0) / total_games) * 100
            self.info(f"Final Win Rate: {p1_rate:.1f}%")
            self.info(f"Total Games: {total_games:,}")
        
        if final_metrics:
            self.info(f"Final Avg Reward: {final_metrics.get('avg_reward', 0):.3f}")
            self.info(f"Final Policy Loss: {final_metrics.get('policy_loss', 0):.6f}")
        
        # Log final metrics to TensorBoard
        if self.enable_tensorboard and self.tensorboard_writer:
            # Create a summary of final performance
            summary_metrics = {
                'final_win_rate': (final_win_stats.get('player1_wins', 0) / 
                                 max(1, final_win_stats.get('total_games', 1))) * 100,
                'total_training_time': total_time,
                'episodes_per_second': total_episodes / total_time,
                'total_episodes': total_episodes
            }
            summary_metrics.update(final_metrics)
            
            for key, value in summary_metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(f'final/{key}', value, total_episodes)
    
    def flush(self) -> None:
        """Flush all logs to disk."""
        if self.enable_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.flush()
    
    def close(self) -> None:
        """Close all loggers and clean up resources."""
        if self.enable_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
            self.info("TensorBoard writer closed")
        
        self.info(f"Training logger closed for experiment: {self.experiment_name}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global logger instance for backward compatibility
_global_logger: Optional[TrainingLogger] = None


def get_logger() -> TrainingLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = TrainingLogger()
    return _global_logger


def set_log_level(level: LogLevel) -> None:
    """Set global logging level."""
    logger = get_logger()
    logger.set_level(level)


def set_log_level_from_args(args) -> None:
    """Set logging level based on command line arguments."""
    if hasattr(args, 'quiet') and args.quiet:
        set_log_level(LogLevel.QUIET)
    elif hasattr(args, 'debug') and args.debug:
        set_log_level(LogLevel.DEBUG)
    elif hasattr(args, 'verbose') and args.verbose:
        set_log_level(LogLevel.DEBUG)  # verbose is alias for debug
    else:
        set_log_level(LogLevel.NORMAL)


def create_training_logger(
    experiment_name: str,
    log_dir: str = "logs",
    enable_tensorboard: bool = True,
    log_level: LogLevel = LogLevel.NORMAL
) -> TrainingLogger:
    """
    Factory function to create a training logger.
    
    Args:
        experiment_name: Name for this training experiment
        log_dir: Base directory for logs
        enable_tensorboard: Whether to enable TensorBoard
        log_level: Console logging verbosity
    
    Returns:
        Configured TrainingLogger instance
    """
    return TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        enable_tensorboard=enable_tensorboard,
        log_level=log_level
    )


if __name__ == "__main__":
    # Test logging functionality
    print("Testing TrainingLogger...")
    
    # Create test logger
    with create_training_logger("test_experiment", log_level=LogLevel.DEBUG) as logger:
        # Test basic logging
        logger.info("Testing info message")
        logger.debug("Testing debug message")
        logger.warning("Testing warning message")
        
        # Test training metrics
        test_metrics = {
            'policy_loss': 0.05,
            'value_loss': 0.03,
            'total_loss': 0.08,
            'avg_reward': 0.15,
            'entropy': 0.02
        }
        
        test_win_stats = {
            'total_games': 100,
            'player1_wins': 60,
            'player2_wins': 30,
            'draws': 10,
            'avg_game_length': 25.5
        }
        
        test_performance = {
            'episodes_per_sec': 15.2,
            'games_per_sec': 8.5,
            'training_time': 120.5
        }
        
        # Log metrics
        for episode in [100, 200, 300]:
            logger.log_training_progress_summary(
                episode=episode,
                total_episodes=1000,
                win_stats=test_win_stats,
                ppo_metrics=test_metrics,
                performance_stats=test_performance
            )
        
        # Test training completion
        logger.log_training_completion(
            total_episodes=1000,
            total_time=600.0,
            final_win_stats=test_win_stats,
            final_metrics=test_metrics
        )
        
        logger.info("Logger test completed")
    
    print("✅ TrainingLogger test completed!")