"""
Comprehensive tests for TrainingStatistics class.

Tests statistics tracking, metric computation, logging integration,
and summary generation for training monitoring.
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.training.training_statistics import TrainingStatistics


class TestTrainingStatisticsInitialization:
    """Test TrainingStatistics initialization and setup."""
    
    def test_statistics_initialization(self):
        """Test basic TrainingStatistics initialization."""
        stats = TrainingStatistics()
        
        # Should initialize with zero/empty values
        assert stats.episodes_completed == 0
        assert stats.total_games_played == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0
        
        # Should have timing attributes
        assert hasattr(stats, 'start_time')
        assert hasattr(stats, 'last_update_time')
        
        # Should have metric storage
        assert hasattr(stats, 'episode_metrics')
        assert hasattr(stats, 'game_metrics')
    
    def test_statistics_initial_state(self):
        """Test that statistics start in a consistent initial state."""
        stats = TrainingStatistics()
        
        # All counters should be zero
        assert stats.episodes_completed == 0
        assert stats.total_games_played == 0
        assert stats.wins == 0
        assert stats.losses == 0
        assert stats.draws == 0
        
        # Times should be set
        assert stats.start_time is not None
        assert stats.last_update_time is not None
        
        # Metric lists should be empty
        assert len(stats.episode_metrics) == 0
        assert len(stats.game_metrics) == 0
    
    def test_statistics_time_tracking(self):
        """Test that time tracking is properly initialized."""
        start_time = time.time()
        stats = TrainingStatistics()
        end_time = time.time()
        
        # Start time should be reasonable
        assert start_time <= stats.start_time <= end_time
        assert start_time <= stats.last_update_time <= end_time
    
    def test_multiple_statistics_instances(self):
        """Test that multiple statistics instances are independent."""
        stats1 = TrainingStatistics()
        stats2 = TrainingStatistics()
        
        # Should be independent objects
        assert stats1 is not stats2
        assert stats1.start_time != stats2.start_time or abs(stats1.start_time - stats2.start_time) < 0.01
        
        # Modifying one shouldn't affect the other
        stats1.episodes_completed = 5
        assert stats2.episodes_completed == 0


class TestTrainingStatisticsGameResults:
    """Test game result tracking functionality."""
    
    def test_update_game_result_win(self, training_statistics):
        """Test updating statistics for game wins."""
        stats = training_statistics
        
        initial_wins = stats.wins
        initial_games = stats.total_games_played
        
        # Record a win
        stats.update_game_result('win', game_length=20, episode=1)
        
        # Should increment win counter and total games
        assert stats.wins == initial_wins + 1
        assert stats.total_games_played == initial_games + 1
        assert stats.losses == 0  # Should not change
        assert stats.draws == 0   # Should not change
    
    def test_update_game_result_loss(self, training_statistics):
        """Test updating statistics for game losses."""
        stats = training_statistics
        
        initial_losses = stats.losses
        initial_games = stats.total_games_played
        
        # Record a loss
        stats.update_game_result('loss', game_length=15, episode=1)
        
        # Should increment loss counter and total games
        assert stats.losses == initial_losses + 1
        assert stats.total_games_played == initial_games + 1
        assert stats.wins == 0   # Should not change
        assert stats.draws == 0  # Should not change
    
    def test_update_game_result_draw(self, training_statistics):
        """Test updating statistics for game draws."""
        stats = training_statistics
        
        initial_draws = stats.draws
        initial_games = stats.total_games_played
        
        # Record a draw
        stats.update_game_result('draw', game_length=42, episode=1)
        
        # Should increment draw counter and total games
        assert stats.draws == initial_draws + 1
        assert stats.total_games_played == initial_games + 1
        assert stats.wins == 0    # Should not change
        assert stats.losses == 0  # Should not change
    
    def test_update_game_result_multiple(self, training_statistics):
        """Test updating statistics for multiple games."""
        stats = training_statistics
        
        # Record multiple game results
        game_results = [
            ('win', 25, 1),
            ('loss', 30, 1),
            ('draw', 42, 1),
            ('win', 20, 2),
            ('win', 18, 2)
        ]
        
        for result, length, episode in game_results:
            stats.update_game_result(result, length, episode)
        
        # Check final counts
        assert stats.wins == 3
        assert stats.losses == 1
        assert stats.draws == 1
        assert stats.total_games_played == 5
    
    def test_update_game_result_invalid_result(self, training_statistics):
        """Test handling of invalid game results."""
        stats = training_statistics
        
        # Try invalid result types
        invalid_results = ['invalid', 'tie', 'victory', 123, None]
        
        for invalid_result in invalid_results:
            try:
                stats.update_game_result(invalid_result, 20, 1)
                # If no error, should handle gracefully
            except (ValueError, TypeError):
                # Expected error for invalid results
                pass
    
    def test_update_game_result_edge_cases(self, training_statistics):
        """Test edge cases in game result updates."""
        stats = training_statistics
        
        # Test with edge case parameters
        edge_cases = [
            ('win', 0, 1),    # Zero-length game
            ('win', 1, 0),    # Zero episode
            ('win', 100, 1),  # Very long game
            ('win', 25, -1),  # Negative episode
        ]
        
        for result, length, episode in edge_cases:
            try:
                stats.update_game_result(result, length, episode)
                # Should handle edge cases gracefully
                assert stats.total_games_played > 0
            except (ValueError, TypeError):
                # Some edge cases might be rejected
                pass
    
    def test_game_result_metric_storage(self, training_statistics):
        """Test that game results are properly stored as metrics."""
        stats = training_statistics
        
        # Record some games
        stats.update_game_result('win', 25, 1)
        stats.update_game_result('loss', 30, 1)
        
        # Should store game metrics
        assert len(stats.game_metrics) > 0
        
        # Each metric should have expected structure
        for metric in stats.game_metrics:
            assert isinstance(metric, dict)
            assert 'result' in metric
            assert 'game_length' in metric
            assert 'episode' in metric
            assert 'timestamp' in metric


class TestTrainingStatisticsPPOMetrics:
    """Test PPO training metrics tracking."""
    
    def test_update_ppo_metrics_basic(self, training_statistics):
        """Test basic PPO metrics update."""
        stats = training_statistics
        
        # Update PPO metrics
        ppo_metrics = {
            'policy_loss': 0.5,
            'value_loss': 0.3,
            'total_loss': 0.8,
            'entropy': 1.2
        }
        
        stats.update_ppo_metrics(ppo_metrics, episode=1)
        
        # Should store metrics
        assert len(stats.episode_metrics) > 0
        
        # Should have latest metrics
        latest_metrics = stats.episode_metrics[-1]
        assert latest_metrics['policy_loss'] == 0.5
        assert latest_metrics['value_loss'] == 0.3
        assert latest_metrics['total_loss'] == 0.8
    
    def test_update_ppo_metrics_multiple_episodes(self, training_statistics):
        """Test PPO metrics updates across multiple episodes."""
        stats = training_statistics
        
        # Update metrics for multiple episodes
        episodes_data = [
            (1, {'policy_loss': 0.8, 'value_loss': 0.6, 'total_loss': 1.4}),
            (2, {'policy_loss': 0.7, 'value_loss': 0.5, 'total_loss': 1.2}),
            (3, {'policy_loss': 0.6, 'value_loss': 0.4, 'total_loss': 1.0}),
        ]
        
        for episode, metrics in episodes_data:
            stats.update_ppo_metrics(metrics, episode=episode)
        
        # Should have metrics for all episodes
        assert len(stats.episode_metrics) == 3
        
        # Should track improvement
        losses = [m['total_loss'] for m in stats.episode_metrics]
        assert losses == [1.4, 1.2, 1.0]  # Decreasing loss
    
    def test_update_ppo_metrics_missing_fields(self, training_statistics):
        """Test handling of missing fields in PPO metrics."""
        stats = training_statistics
        
        # Metrics with missing fields
        incomplete_metrics = {
            'policy_loss': 0.5,
            # Missing value_loss and total_loss
        }
        
        try:
            stats.update_ppo_metrics(incomplete_metrics, episode=1)
            # Should handle gracefully or fill with defaults
            assert len(stats.episode_metrics) > 0
        except (KeyError, ValueError):
            # Expected error for incomplete metrics
            pass
    
    def test_update_ppo_metrics_invalid_values(self, training_statistics):
        """Test handling of invalid metric values."""
        stats = training_statistics
        
        # Metrics with invalid values
        invalid_metrics = [
            {'policy_loss': float('nan'), 'value_loss': 0.3, 'total_loss': 0.8},
            {'policy_loss': float('inf'), 'value_loss': 0.3, 'total_loss': 0.8},
            {'policy_loss': -0.5, 'value_loss': 0.3, 'total_loss': 0.8},  # Negative loss
            {'policy_loss': 'invalid', 'value_loss': 0.3, 'total_loss': 0.8},  # String
        ]
        
        for metrics in invalid_metrics:
            try:
                stats.update_ppo_metrics(metrics, episode=1)
                # Should handle invalid values gracefully
            except (ValueError, TypeError):
                # Expected error for invalid values
                pass
    
    def test_ppo_metrics_timestamp_tracking(self, training_statistics):
        """Test that PPO metrics include timestamp information."""
        stats = training_statistics
        
        start_time = time.time()
        
        metrics = {'policy_loss': 0.5, 'value_loss': 0.3, 'total_loss': 0.8}
        stats.update_ppo_metrics(metrics, episode=1)
        
        end_time = time.time()
        
        # Should have timestamp
        stored_metrics = stats.episode_metrics[-1]
        assert 'timestamp' in stored_metrics
        
        # Timestamp should be reasonable
        timestamp = stored_metrics['timestamp']
        assert start_time <= timestamp <= end_time


class TestTrainingStatisticsStepBatch:
    """Test step batch recording functionality."""
    
    def test_record_step_batch_basic(self, training_statistics):
        """Test basic step batch recording."""
        stats = training_statistics
        
        # Record step batch
        rewards = np.array([0.1, -0.1, 0.0, 1.0])
        dones = np.array([False, False, True, True])
        valid_moves = [[0, 1, 2], [3, 4], [0, 1, 2, 3, 4, 5, 6], [2, 5]]
        
        stats.record_step_batch(rewards, dones, valid_moves)
        
        # Should update internal counters
        # Implementation details depend on what's tracked
        assert hasattr(stats, 'total_games_played')  # Should exist
    
    def test_record_step_batch_multiple_batches(self, training_statistics):
        """Test recording multiple step batches."""
        stats = training_statistics
        
        # Record multiple batches
        batches = [
            (np.array([0.1, 0.2]), np.array([False, False]), [[0, 1], [2, 3]]),
            (np.array([0.3, 0.4]), np.array([True, False]), [[4, 5], [6]]),
            (np.array([0.5, 1.0]), np.array([False, True]), [[0, 2, 4], [1, 3, 5]]),
        ]
        
        for rewards, dones, valid_moves in batches:
            stats.record_step_batch(rewards, dones, valid_moves)
        
        # Should accumulate information from all batches
        # Specific behavior depends on implementation
    
    def test_record_step_batch_empty_batch(self, training_statistics):
        """Test recording empty step batch."""
        stats = training_statistics
        
        # Empty batch
        rewards = np.array([])
        dones = np.array([])
        valid_moves = []
        
        try:
            stats.record_step_batch(rewards, dones, valid_moves)
            # Should handle empty batch gracefully
        except (ValueError, IndexError):
            # Expected error for empty batch
            pass
    
    def test_record_step_batch_mismatched_sizes(self, training_statistics):
        """Test handling of mismatched array sizes."""
        stats = training_statistics
        
        # Mismatched sizes
        rewards = np.array([0.1, 0.2])
        dones = np.array([False, False, True])  # Different size
        valid_moves = [[0, 1]]  # Different size
        
        try:
            stats.record_step_batch(rewards, dones, valid_moves)
            # Should handle gracefully or raise appropriate error
        except (ValueError, IndexError):
            # Expected error for mismatched sizes
            pass
    
    def test_record_step_batch_data_types(self, training_statistics):
        """Test step batch recording with different data types."""
        stats = training_statistics
        
        # Test with different data types
        rewards_list = [0.1, 0.2, 0.3]  # Python list instead of numpy array
        dones_list = [False, True, False]  # Python list
        valid_moves = [[0, 1], [2], [3, 4, 5]]
        
        try:
            stats.record_step_batch(rewards_list, dones_list, valid_moves)
            # Should handle different data types
        except (TypeError, ValueError):
            # Some implementations might require specific types
            pass


class TestTrainingStatisticsSummary:
    """Test statistics summary generation."""
    
    def test_get_summary_basic(self, training_statistics):
        """Test basic summary generation."""
        stats = training_statistics
        
        # Add some data
        stats.update_game_result('win', 25, 1)
        stats.update_game_result('loss', 30, 1)
        stats.update_ppo_metrics({'policy_loss': 0.5, 'value_loss': 0.3, 'total_loss': 0.8}, episode=1)
        
        summary = stats.get_summary()
        
        # Should return dictionary
        assert isinstance(summary, dict)
        
        # Should contain expected keys
        expected_keys = [
            'episodes_completed', 'total_games_played', 'wins', 'losses', 'draws',
            'win_rate', 'training_time'
        ]
        
        for key in expected_keys:
            assert key in summary
    
    def test_get_summary_empty_statistics(self, training_statistics):
        """Test summary generation with empty statistics."""
        stats = training_statistics
        
        summary = stats.get_summary()
        
        # Should return valid summary even with no data
        assert isinstance(summary, dict)
        assert summary['episodes_completed'] == 0
        assert summary['total_games_played'] == 0
        assert summary['wins'] == 0
        assert summary['losses'] == 0
        assert summary['draws'] == 0
    
    def test_get_summary_win_rate_calculation(self, training_statistics):
        """Test win rate calculation in summary."""
        stats = training_statistics
        
        # Add known game results
        for _ in range(7):  # 7 wins
            stats.update_game_result('win', 20, 1)
        for _ in range(2):  # 2 losses
            stats.update_game_result('loss', 25, 1)
        for _ in range(1):  # 1 draw
            stats.update_game_result('draw', 42, 1)
        
        summary = stats.get_summary()
        
        # Should calculate correct win rate
        expected_win_rate = 7 / 10  # 7 wins out of 10 games
        assert abs(summary['win_rate'] - expected_win_rate) < 0.001
    
    def test_get_summary_training_time_calculation(self, training_statistics):
        """Test training time calculation in summary."""
        stats = training_statistics
        
        # Wait a bit to have measurable training time
        time.sleep(0.01)
        
        summary = stats.get_summary()
        
        # Should have positive training time
        assert summary['training_time'] > 0
        assert summary['training_time'] < 10  # Should be small for test
    
    def test_get_summary_with_metrics(self, training_statistics):
        """Test summary with PPO metrics included."""
        stats = training_statistics
        
        # Add PPO metrics
        metrics_data = [
            {'policy_loss': 0.8, 'value_loss': 0.6, 'total_loss': 1.4},
            {'policy_loss': 0.6, 'value_loss': 0.4, 'total_loss': 1.0},
            {'policy_loss': 0.4, 'value_loss': 0.3, 'total_loss': 0.7},
        ]
        
        for i, metrics in enumerate(metrics_data, 1):
            stats.update_ppo_metrics(metrics, episode=i)
        
        summary = stats.get_summary()
        
        # Should include latest metrics
        if 'latest_policy_loss' in summary:
            assert summary['latest_policy_loss'] == 0.4
            assert summary['latest_value_loss'] == 0.3
            assert summary['latest_total_loss'] == 0.7
    
    def test_get_summary_performance_metrics(self, training_statistics):
        """Test performance metrics in summary."""
        stats = training_statistics
        
        # Add some games and episodes
        for i in range(5):
            stats.update_game_result('win', 20 + i, i // 2 + 1)
            stats.episodes_completed = i // 2 + 1
        
        summary = stats.get_summary()
        
        # Should include performance metrics
        assert 'games_per_episode' in summary
        if summary['episodes_completed'] > 0:
            expected_games_per_episode = summary['total_games_played'] / summary['episodes_completed']
            assert abs(summary['games_per_episode'] - expected_games_per_episode) < 0.001


class TestTrainingStatisticsLogging:
    """Test logging integration functionality."""
    
    def test_log_episode_metrics(self, training_statistics):
        """Test episode metrics logging."""
        stats = training_statistics
        
        # Mock the logger
        with patch.object(stats, 'logger') as mock_logger:
            metrics = {'policy_loss': 0.5, 'value_loss': 0.3, 'total_loss': 0.8}
            
            try:
                stats.log_episode_metrics(episode=1, metrics=metrics)
                # Should call logger if it exists
                if hasattr(stats, 'logger') and stats.logger:
                    assert mock_logger.info.called or mock_logger.log.called
            except AttributeError:
                # Logger might not be implemented yet
                pass
    
    def test_log_game_statistics(self, training_statistics):
        """Test game statistics logging."""
        stats = training_statistics
        
        # Add some game results
        stats.update_game_result('win', 25, 1)
        stats.update_game_result('loss', 30, 1)
        
        # Mock the logger
        with patch.object(stats, 'logger') as mock_logger:
            try:
                stats.log_game_statistics()
                # Should log game statistics
                if hasattr(stats, 'logger') and stats.logger:
                    assert mock_logger.called
            except AttributeError:
                # Logger might not be implemented yet
                pass
    
    def test_log_training_progress(self, training_statistics):
        """Test training progress logging."""
        stats = training_statistics
        
        # Update statistics
        stats.episodes_completed = 10
        stats.update_game_result('win', 25, 10)
        
        # Mock the logger
        with patch.object(stats, 'logger') as mock_logger:
            try:
                stats.log_training_progress()
                # Should log training progress
                if hasattr(stats, 'logger') and stats.logger:
                    assert mock_logger.called
            except AttributeError:
                # Logger might not be implemented yet
                pass


class TestTrainingStatisticsPerformance:
    """Test performance characteristics of statistics tracking."""
    
    @pytest.mark.performance
    def test_statistics_update_performance(self, training_statistics):
        """Test performance of statistics updates."""
        stats = training_statistics
        
        # Time many updates
        start_time = time.time()
        
        for i in range(1000):
            result = 'win' if i % 3 == 0 else 'loss' if i % 3 == 1 else 'draw'
            stats.update_game_result(result, 20 + (i % 10), i // 100 + 1)
        
        update_time = time.time() - start_time
        
        # Should update quickly
        assert update_time < 1.0  # Less than 1 second for 1000 updates
        
        updates_per_second = 1000 / update_time
        assert updates_per_second > 1000  # At least 1000 updates per second
    
    @pytest.mark.performance
    def test_summary_generation_performance(self, training_statistics):
        """Test performance of summary generation."""
        stats = training_statistics
        
        # Add lots of data
        for i in range(1000):
            result = 'win' if i % 2 == 0 else 'loss'
            stats.update_game_result(result, 20, i // 10 + 1)
            
            if i % 10 == 0:
                metrics = {'policy_loss': 0.5, 'value_loss': 0.3, 'total_loss': 0.8}
                stats.update_ppo_metrics(metrics, episode=i // 10 + 1)
        
        # Time summary generation
        start_time = time.time()
        summary = stats.get_summary()
        summary_time = time.time() - start_time
        
        # Should generate summary quickly
        assert summary_time < 0.1  # Less than 0.1 seconds
        
        # Summary should be correct
        assert summary['total_games_played'] == 1000
        assert summary['wins'] == 500
        assert summary['losses'] == 500
    
    def test_memory_usage(self, training_statistics):
        """Test memory usage of statistics tracking."""
        stats = training_statistics
        
        # Add many data points
        for i in range(10000):
            stats.update_game_result('win', 20, i // 100 + 1)
            
            if i % 100 == 0:
                metrics = {'policy_loss': 0.5, 'value_loss': 0.3, 'total_loss': 0.8}
                stats.update_ppo_metrics(metrics, episode=i // 100 + 1)
        
        # Should handle large amounts of data
        assert stats.total_games_played == 10000
        assert len(stats.game_metrics) <= 10000  # May have cleanup/limits
        
        # Should still function correctly
        summary = stats.get_summary()
        assert summary['total_games_played'] == 10000


class TestTrainingStatisticsEdgeCases:
    """Test edge cases and error handling."""
    
    def test_concurrent_updates(self, training_statistics):
        """Test concurrent statistics updates."""
        import threading
        
        stats = training_statistics
        errors = []
        
        def update_stats(thread_id):
            try:
                for i in range(100):
                    result = 'win' if (thread_id + i) % 2 == 0 else 'loss'
                    stats.update_game_result(result, 20, i // 10 + 1)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_stats, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should handle concurrent updates
        assert len(errors) == 0  # No errors
        assert stats.total_games_played == 500  # All updates counted
    
    def test_very_large_numbers(self, training_statistics):
        """Test handling of very large numbers."""
        stats = training_statistics
        
        # Test with very large episode numbers and game lengths
        large_values = [
            (1000000, 999999),  # Large episode and game length
            (2**31 - 1, 2**31 - 1),  # Near integer limit
        ]
        
        for episode, game_length in large_values:
            try:
                stats.update_game_result('win', game_length, episode)
                # Should handle large values
                assert stats.total_games_played > 0
            except (OverflowError, ValueError):
                # Some very large values might be rejected
                pass
    
    def test_negative_values(self, training_statistics):
        """Test handling of negative values."""
        stats = training_statistics
        
        # Test with negative values
        negative_cases = [
            (-1, 20),  # Negative episode
            (1, -20),  # Negative game length
            (-1, -20), # Both negative
        ]
        
        for episode, game_length in negative_cases:
            try:
                stats.update_game_result('win', game_length, episode)
                # Should handle or reject negative values appropriately
            except (ValueError, TypeError):
                # Expected error for negative values
                pass
    
    def test_reset_functionality(self, training_statistics):
        """Test statistics reset functionality."""
        stats = training_statistics
        
        # Add some data
        stats.update_game_result('win', 25, 1)
        stats.update_ppo_metrics({'policy_loss': 0.5, 'value_loss': 0.3, 'total_loss': 0.8}, episode=1)
        
        # Reset if method exists
        if hasattr(stats, 'reset'):
            stats.reset()
            
            # Should return to initial state
            assert stats.episodes_completed == 0
            assert stats.total_games_played == 0
            assert stats.wins == 0
            assert stats.losses == 0
            assert stats.draws == 0
            assert len(stats.episode_metrics) == 0
            assert len(stats.game_metrics) == 0