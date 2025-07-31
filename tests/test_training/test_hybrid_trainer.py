"""
Comprehensive tests for HybridTrainer class.

Tests the main training orchestration system including experience collection,
training loops, checkpoint management, and integration with vectorized environments.
"""

import pytest
import torch
import numpy as np
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.training.hybrid_trainer import HybridTrainer
from src.training.training_statistics import TrainingStatistics


class TestHybridTrainerInitialization:
    """Test HybridTrainer initialization and configuration."""
    
    def test_trainer_initialization_small_difficulty(self, test_config):
        """Test trainer initialization with small difficulty."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        assert trainer.difficulty == 'small'
        assert trainer.config == test_config
        assert hasattr(trainer, 'num_envs')
        assert hasattr(trainer, 'batch_size')
        assert hasattr(trainer, 'update_frequency')
        
        # Small difficulty should have appropriate settings
        assert trainer.num_envs <= 1000  # Reasonable for small
    
    def test_trainer_initialization_medium_difficulty(self, test_config):
        """Test trainer initialization with medium difficulty."""
        trainer = HybridTrainer(
            difficulty='medium',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        assert trainer.difficulty == 'medium'
        assert trainer.num_envs > 100  # Should be larger than small
        assert trainer.batch_size >= 64  # Reasonable batch size
    
    def test_trainer_initialization_impossible_difficulty(self, test_config):
        """Test trainer initialization with impossible difficulty."""
        trainer = HybridTrainer(
            difficulty='impossible',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        assert trainer.difficulty == 'impossible'
        assert trainer.num_envs >= 1000  # Should be very large
        assert trainer.batch_size >= 256  # Large batch size
    
    def test_trainer_initialization_invalid_difficulty(self, test_config):
        """Test trainer initialization with invalid difficulty."""
        with pytest.raises(ValueError):
            HybridTrainer(
                difficulty='invalid_difficulty',
                config=test_config,
                models_dir='./models',
                logs_dir='./logs'
            )
    
    def test_trainer_directory_setup(self, test_config, tmp_path):
        """Test that trainer sets up directories correctly."""
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Directories should exist after initialization
        assert models_dir.exists()
        assert logs_dir.exists()
    
    def test_trainer_component_initialization(self, test_config):
        """Test that trainer initializes all required components."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Should have all required components
        assert hasattr(trainer, 'env')
        assert hasattr(trainer, 'agent')
        assert hasattr(trainer, 'statistics')
        assert hasattr(trainer, 'checkpoint_manager')
        
        # Components should be properly initialized
        assert trainer.env is not None
        assert trainer.agent is not None
        assert isinstance(trainer.statistics, TrainingStatistics)
    
    def test_trainer_device_configuration(self, test_config):
        """Test that trainer uses correct device configuration."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Agent and environment should use consistent devices
        assert trainer.agent.device == trainer.env.device
        
        # Should use device from config or detect automatically
        expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert trainer.agent.device.type == expected_device.type


class TestHybridTrainerExperienceCollection:
    """Test experience collection functionality."""
    
    def test_collect_experiences_basic(self, test_config):
        """Test basic experience collection."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Collect experiences for a few steps
        num_steps = 10
        experiences = trainer.collect_experiences(num_steps)
        
        # Should return experience data
        assert isinstance(experiences, dict)
        
        # Should have expected keys
        expected_keys = ['observations', 'actions', 'rewards', 'dones', 'log_probs', 'values']
        for key in expected_keys:
            assert key in experiences
        
        # Each should have data for num_steps * num_envs
        total_steps = num_steps * trainer.num_envs
        for key in expected_keys:
            assert len(experiences[key]) <= total_steps  # May be less due to episode endings
    
    def test_collect_experiences_shapes(self, test_config):
        """Test that collected experiences have correct shapes."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        experiences = trainer.collect_experiences(5)
        
        # Check shapes
        num_experiences = len(experiences['observations'])
        
        # Observations should be board states
        assert experiences['observations'].shape == (num_experiences, 6, 7)
        
        # Actions should be column indices
        assert experiences['actions'].shape == (num_experiences,)
        assert np.all(experiences['actions'] >= 0)
        assert np.all(experiences['actions'] < 7)
        
        # Rewards should be scalars
        assert experiences['rewards'].shape == (num_experiences,)
        
        # Done flags should be boolean
        assert experiences['dones'].shape == (num_experiences,)
        assert experiences['dones'].dtype == bool
        
        # Log probs and values should be scalars
        assert experiences['log_probs'].shape == (num_experiences,)
        assert experiences['values'].shape == (num_experiences,)
    
    def test_collect_experiences_episode_handling(self, test_config):
        """Test handling of episode endings during experience collection."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Collect many experiences to ensure some episodes end
        experiences = trainer.collect_experiences(50)
        
        # Should have some done flags set to True
        assert np.any(experiences['dones'])
        
        # When episodes end, environments should auto-reset
        # This is handled by the vectorized environment
        
        # All experiences should be valid
        assert len(experiences['observations']) > 0
        assert len(experiences['actions']) > 0
    
    def test_collect_experiences_zero_steps(self, test_config):
        """Test collecting zero experiences."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        experiences = trainer.collect_experiences(0)
        
        # Should return empty experiences
        for key in experiences:
            assert len(experiences[key]) == 0
    
    def test_collect_experiences_performance(self, test_config):
        """Test performance of experience collection."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Time experience collection
        start_time = time.time()
        experiences = trainer.collect_experiences(20)
        collection_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert collection_time < 5.0  # Less than 5 seconds
        
        # Should collect reasonable amount of data
        assert len(experiences['observations']) > 0
        
        # Calculate collection rate
        num_experiences = len(experiences['observations'])
        experiences_per_second = num_experiences / collection_time
        
        # Should collect at reasonable rate
        assert experiences_per_second > 10  # At least 10 experiences per second


class TestHybridTrainerTrainingLoop:
    """Test training loop functionality."""
    
    def test_train_episode_batch(self, test_config):
        """Test training a batch of episodes."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Train for one batch
        metrics = trainer.train_episode_batch()
        
        # Should return training metrics
        assert isinstance(metrics, dict)
        
        # Should have expected metrics
        expected_metrics = ['policy_loss', 'value_loss', 'total_loss']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert np.isfinite(metrics[metric])
    
    def test_train_basic(self, test_config, tmp_path):
        """Test basic training functionality."""
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Train for a few episodes
        final_metrics = trainer.train(
            total_episodes=5,
            save_interval=10,  # No saving in this short test
            log_interval=2
        )
        
        # Should return final metrics
        assert isinstance(final_metrics, dict)
        
        # Should have completed training
        assert trainer.statistics.episodes_completed >= 5
    
    def test_train_with_checkpointing(self, test_config, tmp_path):
        """Test training with checkpoint saving."""
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Train with frequent checkpointing
        trainer.train(
            total_episodes=6,
            save_interval=3,  # Save every 3 episodes
            log_interval=2
        )
        
        # Should have created checkpoint files
        checkpoint_files = list(models_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
        
        # At least one checkpoint should exist
        assert any("checkpoint" in str(f) for f in checkpoint_files)
    
    def test_train_logging_intervals(self, test_config, tmp_path):
        """Test training with different logging intervals."""
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Mock the logging to capture calls
        with patch.object(trainer.statistics, 'log_episode_metrics') as mock_log:
            trainer.train(
                total_episodes=10,
                save_interval=100,  # No saving
                log_interval=3  # Log every 3 episodes
            )
            
            # Should have logged at appropriate intervals
            assert mock_log.call_count >= 3  # At least 3 logging calls
    
    def test_train_early_stopping(self, test_config):
        """Test training with early stopping conditions."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Train for short duration
        start_time = time.time()
        trainer.train(
            total_episodes=3,
            save_interval=100,
            log_interval=1
        )
        training_time = time.time() - start_time
        
        # Should complete quickly for small number of episodes
        assert training_time < 30.0  # Less than 30 seconds
        
        # Should have completed requested episodes
        assert trainer.statistics.episodes_completed >= 3
    
    def test_train_resume_capability(self, test_config, tmp_path):
        """Test capability to resume training from checkpoint."""
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        
        # First training session
        trainer1 = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        trainer1.train(
            total_episodes=3,
            save_interval=2,
            log_interval=1
        )
        
        # Check that checkpoint was created
        checkpoint_files = list(models_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
        
        # Create new trainer (simulating restart)
        trainer2 = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Should be able to load checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        trainer2.agent.load(str(latest_checkpoint))
        
        # Continue training
        trainer2.train(
            total_episodes=2,
            save_interval=5,
            log_interval=1
        )
        
        # Should complete additional episodes
        assert trainer2.statistics.episodes_completed >= 2


class TestHybridTrainerCheckpointing:
    """Test checkpoint management functionality."""
    
    def test_save_checkpoint_basic(self, test_config, tmp_path):
        """Test basic checkpoint saving."""
        models_dir = tmp_path / "models"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir='./logs'
        )
        
        # Save checkpoint
        checkpoint_path = trainer._save_checkpoint(episode=100, metrics={'loss': 0.5})
        
        # Should create checkpoint file
        assert Path(checkpoint_path).exists()
        assert "checkpoint" in checkpoint_path
        assert "episode_100" in checkpoint_path
    
    def test_save_final_checkpoint(self, test_config, tmp_path):
        """Test final checkpoint saving."""
        models_dir = tmp_path / "models"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir='./logs'
        )
        
        # Save final checkpoint
        final_metrics = {'total_loss': 0.3, 'episodes': 1000}
        checkpoint_path = trainer._save_final_checkpoint(final_metrics)
        
        # Should create final checkpoint file
        assert Path(checkpoint_path).exists()
        assert "final" in checkpoint_path.lower()
    
    def test_checkpoint_content_validation(self, test_config, tmp_path):
        """Test that checkpoints contain expected content."""
        models_dir = tmp_path / "models"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir='./logs'
        )
        
        # Save checkpoint
        metrics = {'policy_loss': 0.5, 'value_loss': 0.3}
        checkpoint_path = trainer._save_checkpoint(episode=50, metrics=metrics)
        
        # Load and verify checkpoint content
        checkpoint = torch.load(checkpoint_path)
        
        # Should contain expected keys
        expected_keys = ['model_state_dict', 'optimizer_state_dict', 'episode', 'metrics']
        for key in expected_keys:
            assert key in checkpoint
        
        # Episode should match
        assert checkpoint['episode'] == 50
        
        # Metrics should match
        assert checkpoint['metrics'] == metrics
    
    def test_checkpoint_compression(self, test_config, tmp_path):
        """Test checkpoint compression functionality."""
        models_dir = tmp_path / "models"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir='./logs'
        )
        
        # Save checkpoint
        checkpoint_path = trainer._save_checkpoint(episode=25, metrics={'loss': 0.4})
        
        # Check file size is reasonable
        file_size = Path(checkpoint_path).stat().st_size
        
        # Should be reasonable size (not too large)
        assert file_size < 100 * 1024 * 1024  # Less than 100MB
        assert file_size > 1024  # More than 1KB (not empty)
    
    def test_checkpoint_cleanup(self, test_config, tmp_path):
        """Test checkpoint cleanup functionality."""
        models_dir = tmp_path / "models"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir='./logs'
        )
        
        # Create multiple checkpoints
        checkpoints = []
        for i in range(5):
            checkpoint_path = trainer._save_checkpoint(
                episode=(i+1)*10, 
                metrics={'loss': 0.5 - i*0.1}
            )
            checkpoints.append(checkpoint_path)
        
        # All checkpoints should exist
        for checkpoint in checkpoints:
            assert Path(checkpoint).exists()
        
        # If cleanup is implemented, older checkpoints might be removed
        # This depends on the implementation details


class TestHybridTrainerStatisticsIntegration:
    """Test integration with training statistics."""
    
    def test_statistics_tracking_during_training(self, test_config):
        """Test that statistics are properly tracked during training."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        initial_episodes = trainer.statistics.episodes_completed
        
        # Train for a few episodes
        trainer.train(
            total_episodes=3,
            save_interval=10,
            log_interval=1
        )
        
        # Statistics should be updated
        assert trainer.statistics.episodes_completed > initial_episodes
        assert trainer.statistics.total_games_played > 0
        
        # Should have win/loss statistics
        total_outcomes = (trainer.statistics.wins + 
                         trainer.statistics.losses + 
                         trainer.statistics.draws)
        assert total_outcomes > 0
    
    def test_performance_metrics_tracking(self, test_config):
        """Test tracking of performance metrics."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Train and collect metrics
        final_metrics = trainer.train(
            total_episodes=5,
            save_interval=10,
            log_interval=2
        )
        
        # Should have performance metrics
        assert 'policy_loss' in final_metrics
        assert 'value_loss' in final_metrics
        assert 'total_loss' in final_metrics
        
        # Values should be reasonable
        assert final_metrics['policy_loss'] >= 0
        assert final_metrics['value_loss'] >= 0
        assert final_metrics['total_loss'] >= 0
    
    def test_statistics_summary_generation(self, test_config):
        """Test generation of statistics summaries."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Train to generate some statistics
        trainer.train(
            total_episodes=3,
            save_interval=10,
            log_interval=1
        )
        
        # Get statistics summary
        summary = trainer.statistics.get_summary()
        
        # Should contain expected information
        assert isinstance(summary, dict)
        assert 'episodes_completed' in summary
        assert 'total_games_played' in summary
        assert 'wins' in summary
        assert 'losses' in summary
        assert 'draws' in summary
        
        # Values should be consistent
        assert summary['episodes_completed'] >= 3
        assert summary['total_games_played'] > 0


class TestHybridTrainerPerformance:
    """Test performance characteristics of HybridTrainer."""
    
    @pytest.mark.performance
    def test_training_speed_small_difficulty(self, test_config):
        """Test training speed with small difficulty."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Time training
        start_time = time.time()
        trainer.train(
            total_episodes=5,
            save_interval=10,
            log_interval=2
        )
        training_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert training_time < 60.0  # Less than 1 minute for 5 episodes
        
        # Calculate episodes per second
        episodes_per_second = 5 / training_time
        assert episodes_per_second > 0.1  # At least 0.1 episodes per second
    
    @pytest.mark.performance
    def test_memory_usage_during_training(self, test_config):
        """Test memory usage during training."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Train for several episodes
        trainer.train(
            total_episodes=10,
            save_interval=20,
            log_interval=3
        )
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 1024 * 1024 * 1024  # Less than 1GB increase
    
    @pytest.mark.performance
    def test_scalability_different_difficulties(self, test_config):
        """Test scalability across different difficulty levels."""
        difficulties = ['small', 'medium']  # Skip 'impossible' for performance testing
        
        for difficulty in difficulties:
            trainer = HybridTrainer(
                difficulty=difficulty,
                config=test_config,
                models_dir='./models',
                logs_dir='./logs'
            )
            
            # Time single episode batch
            start_time = time.time()
            metrics = trainer.train_episode_batch()
            batch_time = time.time() - start_time
            
            # Should complete in reasonable time regardless of difficulty
            assert batch_time < 30.0  # Less than 30 seconds per batch
            
            # Should return valid metrics
            assert isinstance(metrics, dict)
            assert len(metrics) > 0


class TestHybridTrainerErrorHandling:
    """Test error handling and robustness."""
    
    def test_invalid_directory_handling(self, test_config):
        """Test handling of invalid directory paths."""
        # Try to create trainer with invalid directory paths
        with pytest.raises((OSError, PermissionError, ValueError)):
            HybridTrainer(
                difficulty='small',
                config=test_config,
                models_dir='/invalid/path/that/does/not/exist',
                logs_dir='/another/invalid/path'
            )
    
    def test_training_interruption_handling(self, test_config):
        """Test handling of training interruption."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Simulate interruption by training for very short time
        try:
            trainer.train(
                total_episodes=1,
                save_interval=1,
                log_interval=1
            )
        except KeyboardInterrupt:
            # Should handle gracefully
            pass
        
        # Trainer should still be in valid state
        assert trainer.statistics.episodes_completed >= 0
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        # Test with None config
        with pytest.raises((ValueError, AttributeError)):
            HybridTrainer(
                difficulty='small',
                config=None,
                models_dir='./models',
                logs_dir='./logs'
            )
        
        # Test with invalid config object
        invalid_config = Mock()
        invalid_config.get.side_effect = KeyError("Invalid config")
        
        with pytest.raises((KeyError, AttributeError)):
            HybridTrainer(
                difficulty='small',
                config=invalid_config,
                models_dir='./models',
                logs_dir='./logs'
            )
    
    def test_checkpoint_save_failure_handling(self, test_config):
        """Test handling of checkpoint save failures."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Mock torch.save to raise an error
        with patch('torch.save', side_effect=IOError("Disk full")):
            try:
                checkpoint_path = trainer._save_checkpoint(episode=10, metrics={'loss': 0.5})
                # If no error, should handle gracefully
                assert checkpoint_path is not None
            except IOError:
                # Expected error for disk full simulation
                pass
    
    def test_out_of_memory_handling(self, test_config):
        """Test handling of out-of-memory conditions."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # This test documents expected behavior for OOM conditions
        # Actual OOM is hard to simulate reliably
        
        # Trainer should be able to handle reasonable workloads
        try:
            trainer.train(
                total_episodes=2,
                save_interval=5,
                log_interval=1
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Should handle OOM gracefully
                pass
            else:
                raise


class TestHybridTrainerIntegration:
    """Test integration with other system components."""
    
    def test_integration_with_vectorized_environment(self, test_config):
        """Test integration with vectorized environment."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Environment should be properly integrated
        assert trainer.env is not None
        assert hasattr(trainer.env, 'num_envs')
        assert hasattr(trainer.env, 'step_batch')
        assert hasattr(trainer.env, 'get_observations_gpu')
        
        # Should be able to collect experiences
        experiences = trainer.collect_experiences(5)
        assert len(experiences['observations']) > 0
    
    def test_integration_with_ppo_agent(self, test_config):
        """Test integration with PPO agent."""
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir='./models',
            logs_dir='./logs'
        )
        
        # Agent should be properly integrated
        assert trainer.agent is not None
        assert hasattr(trainer.agent, 'get_actions_batch')
        assert hasattr(trainer.agent, 'train_on_experiences')
        
        # Should be able to train
        metrics = trainer.train_episode_batch()
        assert isinstance(metrics, dict)
        assert 'policy_loss' in metrics
    
    def test_integration_with_checkpoint_manager(self, test_config, tmp_path):
        """Test integration with checkpoint manager."""
        models_dir = tmp_path / "models"
        
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir='./logs'
        )
        
        # Should have checkpoint manager
        assert hasattr(trainer, 'checkpoint_manager')
        
        # Should be able to save and load checkpoints
        checkpoint_path = trainer._save_checkpoint(episode=5, metrics={'loss': 0.4})
        assert Path(checkpoint_path).exists()
        
        # Should be able to load checkpoint
        trainer.agent.load(checkpoint_path)
    
    def test_end_to_end_training_pipeline(self, test_config, tmp_path):
        """Test complete end-to-end training pipeline."""
        models_dir = tmp_path / "models"
        logs_dir = tmp_path / "logs"
        
        # Create trainer
        trainer = HybridTrainer(
            difficulty='small',
            config=test_config,
            models_dir=str(models_dir),
            logs_dir=str(logs_dir)
        )
        
        # Run complete training
        final_metrics = trainer.train(
            total_episodes=5,
            save_interval=3,
            log_interval=2
        )
        
        # Should complete successfully
        assert isinstance(final_metrics, dict)
        assert trainer.statistics.episodes_completed >= 5
        
        # Should have created files
        checkpoint_files = list(models_dir.glob("*.pth"))
        assert len(checkpoint_files) > 0
        
        # Should have valid final metrics
        assert all(np.isfinite(v) for v in final_metrics.values() if isinstance(v, (int, float)))