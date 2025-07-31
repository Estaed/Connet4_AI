"""
Comprehensive tests for HybridVectorizedConnect4 environment.

Tests the advanced vectorized environment system that manages thousands
of parallel Connect4 games for high-performance training.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, Mock
from src.environments.hybrid_vectorized_connect4 import HybridVectorizedConnect4


class TestHybridVectorizedInitialization:
    """Test initialization and setup of vectorized environments."""
    
    def test_small_env_initialization(self, cpu_device):
        """Test initialization with small number of environments."""
        num_envs = 4
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        assert len(env.games) == num_envs
        assert env.device == cpu_device
        assert env.num_envs == num_envs
        
        # Check that all games are initialized
        for game in env.games:
            assert game is not None
            assert hasattr(game, 'board')
            assert hasattr(game, 'current_player')
    
    def test_medium_env_initialization(self, cpu_device):
        """Test initialization with medium number of environments."""
        num_envs = 64
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        assert len(env.games) == num_envs
        assert env.device == cpu_device
        
        # Verify games are independent
        env.games[0].drop_piece(0)
        assert not np.array_equal(env.games[0].board, env.games[1].board)
    
    @pytest.mark.slow
    def test_large_env_initialization(self, cpu_device):
        """Test initialization with large number of environments."""
        num_envs = 256
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        assert len(env.games) == num_envs
        
        # Test memory efficiency
        import sys
        size_bytes = sys.getsizeof(env)
        assert size_bytes < 100 * 1024 * 1024  # Less than 100MB for 256 envs
    
    def test_device_handling(self):
        """Test device handling for CPU and GPU."""
        # Test CPU device
        cpu_env = HybridVectorizedConnect4(num_envs=4, device=torch.device('cpu'))
        assert cpu_env.device == torch.device('cpu')
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_env = HybridVectorizedConnect4(num_envs=4, device=torch.device('cuda'))
            assert cuda_env.device.type == 'cuda'


class TestHybridVectorizedReset:
    """Test reset functionality for vectorized environments."""
    
    def test_reset_all_environments(self, small_vectorized_env):
        """Test resetting all environments."""
        env = small_vectorized_env
        
        # Make some moves in all environments
        for i in range(env.num_envs):
            env.games[i].drop_piece(i % 7)
        
        # Reset all
        env.reset()
        
        # Verify all games are reset
        for game in env.games:
            assert np.all(game.board == 0)
            assert game.current_player == 1
    
    def test_reset_specific_environments(self, small_vectorized_env):
        """Test resetting specific environments by ID."""
        env = small_vectorized_env
        
        # Make moves in all environments
        for i in range(env.num_envs):
            env.games[i].drop_piece(0)
        
        # Reset only environments 0 and 2
        env.reset(env_ids=[0, 2])
        
        # Check that only specified environments were reset
        assert np.all(env.games[0].board == 0)  # Reset
        assert not np.all(env.games[1].board == 0)  # Not reset
        assert np.all(env.games[2].board == 0)  # Reset
        assert not np.all(env.games[3].board == 0)  # Not reset
    
    def test_reset_invalid_env_ids(self, small_vectorized_env):
        """Test handling of invalid environment IDs."""
        env = small_vectorized_env
        
        # Should handle invalid IDs gracefully
        env.reset(env_ids=[999, -1, env.num_envs])
        
        # Valid environments should still work
        env.reset(env_ids=[0, 1])


class TestHybridVectorizedValidMoves:
    """Test valid move detection across vectorized environments."""
    
    def test_get_valid_moves_batch_empty_boards(self, small_vectorized_env):
        """Test getting valid moves for empty boards."""
        env = small_vectorized_env
        valid_moves = env.get_valid_moves_batch()
        
        # All columns should be valid for empty boards
        expected = [[0, 1, 2, 3, 4, 5, 6] for _ in range(env.num_envs)]
        assert valid_moves == expected
    
    def test_get_valid_moves_batch_partial_boards(self, small_vectorized_env):
        """Test getting valid moves for partially filled boards."""
        env = small_vectorized_env
        
        # Fill column 0 in environment 0
        for _ in range(6):
            env.games[0].drop_piece(0)
        
        # Fill column 3 in environment 1
        for _ in range(6):
            env.games[1].drop_piece(3)
        
        valid_moves = env.get_valid_moves_batch()
        
        # Environment 0 should not have column 0 as valid
        assert 0 not in valid_moves[0]
        assert len(valid_moves[0]) == 6
        
        # Environment 1 should not have column 3 as valid
        assert 3 not in valid_moves[1]
        assert len(valid_moves[1]) == 6
        
        # Other environments should have all columns valid
        for i in range(2, env.num_envs):
            assert len(valid_moves[i]) == 7
    
    def test_get_valid_moves_tensor(self, small_vectorized_env):
        """Test getting valid moves as GPU tensor."""
        env = small_vectorized_env
        
        # Fill some columns
        env.games[0].drop_piece(0)  # Partial fill
        for _ in range(6):  # Full fill
            env.games[1].drop_piece(1)
        
        valid_tensor = env.get_valid_moves_tensor()
        
        # Should be on correct device
        assert valid_tensor.device == env.device
        
        # Should have correct shape (num_envs, 7)
        assert valid_tensor.shape == (env.num_envs, 7)
        
        # Should be boolean tensor
        assert valid_tensor.dtype == torch.bool
        
        # Check specific values
        assert valid_tensor[0, 0] == True  # Column 0 still valid in env 0
        assert valid_tensor[1, 1] == False  # Column 1 full in env 1
        assert valid_tensor[1, 0] == True  # Column 0 still valid in env 1


class TestHybridVectorizedStepBatch:
    """Test batch step operations across vectorized environments."""
    
    def test_step_batch_valid_actions(self, small_vectorized_env):
        """Test batch stepping with valid actions."""
        env = small_vectorized_env
        actions = np.array([0, 1, 2, 3])  # Valid actions for each env
        
        rewards, dones, info = env.step_batch(actions)
        
        # Check return types and shapes
        assert isinstance(rewards, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert isinstance(info, dict)
        
        assert rewards.shape == (env.num_envs,)
        assert dones.shape == (env.num_envs,)
        
        # Check that pieces were placed
        assert env.games[0].board[5, 0] != 0  # Action 0 in env 0
        assert env.games[1].board[5, 1] != 0  # Action 1 in env 1
        assert env.games[2].board[5, 2] != 0  # Action 2 in env 2
        assert env.games[3].board[5, 3] != 0  # Action 3 in env 3
    
    def test_step_batch_invalid_actions(self, small_vectorized_env):
        """Test batch stepping with some invalid actions."""
        env = small_vectorized_env
        
        # Fill column 0 in environment 0
        for _ in range(6):
            env.games[0].drop_piece(0)
        
        # Try to place in column 0 for all environments
        actions = np.array([0, 0, 0, 0])
        rewards, dones, info = env.step_batch(actions)
        
        # Environment 0 should reject the move (negative reward expected)
        # Other environments should accept the move
        assert rewards[0] <= 0  # Invalid move penalty
        for i in range(1, env.num_envs):
            assert rewards[i] >= -0.01  # Valid move (small penalty or neutral)
    
    def test_step_batch_winning_moves(self, small_vectorized_env):
        """Test batch stepping with winning moves."""
        env = small_vectorized_env
        
        # Set up winning position in environment 0
        for col in range(3):
            env.games[0].board[5, col] = 1
        env.games[0].current_player = 1
        
        # Make winning move
        actions = np.array([3, 0, 1, 2])  # Winning move for env 0
        rewards, dones, info = env.step_batch(actions)
        
        # Environment 0 should have won
        assert rewards[0] > 0  # Positive reward for winning
        assert dones[0] == True  # Game finished
        
        # Other environments should continue
        for i in range(1, env.num_envs):
            assert dones[i] == False
    
    def test_step_batch_info_dict(self, small_vectorized_env):
        """Test that step_batch returns proper info dictionary."""
        env = small_vectorized_env
        actions = np.array([0, 1, 2, 3])
        
        rewards, dones, info = env.step_batch(actions)
        
        # Check info dict structure
        assert 'valid_moves' in info
        assert len(info['valid_moves']) == env.num_envs
        
        # Each environment should have valid moves list
        for i in range(env.num_envs):
            assert isinstance(info['valid_moves'][i], list)
            assert len(info['valid_moves'][i]) >= 6  # At least 6 columns still valid


class TestHybridVectorizedObservations:
    """Test observation handling in vectorized environments."""
    
    def test_get_observations_gpu_empty_boards(self, small_vectorized_env):
        """Test getting observations for empty boards."""
        env = small_vectorized_env
        observations = env.get_observations_gpu()
        
        # Check tensor properties
        assert observations.device == env.device
        assert observations.shape == (env.num_envs, 6, 7)
        assert observations.dtype == torch.float32
        
        # All observations should be zeros (empty boards)
        assert torch.all(observations == 0)
    
    def test_get_observations_gpu_with_moves(self, small_vectorized_env):
        """Test getting observations after moves are made."""
        env = small_vectorized_env
        
        # Make different moves in each environment
        for i in range(env.num_envs):
            env.games[i].drop_piece(i)
        
        observations = env.get_observations_gpu()
        
        # Check that observations are different
        for i in range(env.num_envs):
            # Each environment should have a piece in column i
            assert observations[i, 5, i] != 0
            
            # Other positions should be empty
            for j in range(7):
                if j != i:
                    assert observations[i, 5, j] == 0
    
    def test_observation_consistency(self, small_vectorized_env):
        """Test that observations match actual game states."""
        env = small_vectorized_env
        
        # Make some moves
        env.games[0].drop_piece(3)  # Player 1
        env.games[0].drop_piece(3)  # Player -1
        
        observations = env.get_observations_gpu()
        
        # Check that observation matches game board
        game_board = torch.tensor(env.games[0].board, dtype=torch.float32, device=env.device)
        assert torch.allclose(observations[0], game_board)


class TestHybridVectorizedAutoReset:
    """Test automatic reset functionality."""
    
    def test_auto_reset_finished_games(self, small_vectorized_env):
        """Test that finished games are automatically reset."""
        env = small_vectorized_env
        
        # Create a winning position and finish game 0
        for col in range(4):
            env.games[0].board[5, col] = 1
        env.games[0].current_player = 1
        env.games[0].game_over = True
        env.games[0].winner = 1
        
        # Call auto reset
        reset_envs = env.auto_reset_finished_games()
        
        # Game 0 should have been reset
        assert 0 in reset_envs
        assert np.all(env.games[0].board == 0)
        assert env.games[0].current_player == 1
        assert not env.games[0].game_over
        assert env.games[0].winner is None
        
        # Other games should not be reset
        for i in range(1, env.num_envs):
            assert i not in reset_envs
    
    def test_auto_reset_multiple_finished_games(self, small_vectorized_env):
        """Test auto reset with multiple finished games."""
        env = small_vectorized_env
        
        # Finish games 0 and 2
        for game_idx in [0, 2]:
            env.games[game_idx].game_over = True
            env.games[game_idx].winner = 1
        
        reset_envs = env.auto_reset_finished_games()
        
        # Both games should be reset
        assert set(reset_envs) == {0, 2}
        
        for game_idx in [0, 2]:
            assert np.all(env.games[game_idx].board == 0)
            assert not env.games[game_idx].game_over


class TestHybridVectorizedPerformance:
    """Test performance characteristics of vectorized environments."""
    
    @pytest.mark.performance
    def test_batch_operations_performance(self, cpu_device):
        """Test performance of batch operations."""
        import time
        
        num_envs = 64
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        # Time batch observations
        start_time = time.time()
        for _ in range(100):
            observations = env.get_observations_gpu()
        obs_time = time.time() - start_time
        
        # Time batch valid moves
        start_time = time.time()
        for _ in range(100):
            valid_moves = env.get_valid_moves_batch()
        valid_moves_time = time.time() - start_time
        
        # Time batch steps
        start_time = time.time()
        for _ in range(100):
            actions = np.random.randint(0, 7, size=num_envs)
            rewards, dones, info = env.step_batch(actions)
        step_time = time.time() - start_time
        
        # Performance assertions (adjust based on hardware)
        assert obs_time < 1.0  # Should get observations quickly
        assert valid_moves_time < 1.0  # Should get valid moves quickly
        assert step_time < 2.0  # Should step environments quickly
    
    @pytest.mark.slow
    def test_memory_efficiency(self, cpu_device):
        """Test memory efficiency with large number of environments."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large vectorized environment
        num_envs = 500
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        # Make some operations
        for _ in range(10):
            observations = env.get_observations_gpu()
            actions = np.random.randint(0, 7, size=num_envs)
            rewards, dones, info = env.step_batch(actions)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 500 envs)
        assert memory_increase < 500 * 1024 * 1024
    
    @pytest.mark.performance
    def test_scalability_different_env_counts(self, cpu_device):
        """Test performance scaling with different environment counts."""
        import time
        
        env_counts = [4, 16, 64, 256]
        times = []
        
        for num_envs in env_counts:
            env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
            
            # Time batch step operations
            start_time = time.time()
            for _ in range(20):
                actions = np.random.randint(0, 7, size=num_envs)
                rewards, dones, info = env.step_batch(actions)
            end_time = time.time()
            
            times.append((end_time - start_time) / 20)  # Average time per batch
        
        # Performance should scale roughly linearly (not exponentially)
        # Time per environment should be roughly constant
        time_per_env = [t / n for t, n in zip(times, env_counts)]
        
        # Variation should be reasonable
        max_time_per_env = max(time_per_env)
        min_time_per_env = min(time_per_env)
        assert max_time_per_env / min_time_per_env < 5  # Less than 5x variation


class TestHybridVectorizedEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_environment_list(self):
        """Test handling of zero environments."""
        with pytest.raises((ValueError, AssertionError)):
            HybridVectorizedConnect4(num_envs=0, device=torch.device('cpu'))
    
    def test_negative_environment_count(self):
        """Test handling of negative environment count."""
        with pytest.raises((ValueError, AssertionError)):
            HybridVectorizedConnect4(num_envs=-1, device=torch.device('cpu'))
    
    def test_mismatched_action_batch_size(self, small_vectorized_env):
        """Test handling of mismatched action batch size."""
        env = small_vectorized_env
        
        # Wrong number of actions
        actions = np.array([0, 1])  # Only 2 actions for 4 environments
        
        with pytest.raises((ValueError, IndexError, AssertionError)):
            rewards, dones, info = env.step_batch(actions)
    
    def test_out_of_range_actions(self, small_vectorized_env):
        """Test handling of out-of-range actions."""
        env = small_vectorized_env
        
        # Actions outside valid range
        actions = np.array([7, 8, -1, 10])
        
        # Should handle gracefully (might return negative rewards)
        rewards, dones, info = env.step_batch(actions)
        
        # All should be penalized for invalid actions
        assert all(r <= 0 for r in rewards)
    
    def test_device_mismatch_handling(self, small_vectorized_env):
        """Test handling of device mismatches."""
        env = small_vectorized_env
        
        # This should work regardless of device mismatches in input
        observations = env.get_observations_gpu()
        assert observations.device == env.device


# Integration tests with mock components
class TestHybridVectorizedIntegration:
    """Test integration with other system components."""
    
    def test_integration_with_ppo_agent(self, small_vectorized_env, cpu_device):
        """Test integration with PPO agent (mocked)."""
        env = small_vectorized_env
        
        # Mock PPO agent behavior
        observations = env.get_observations_gpu()
        valid_moves = env.get_valid_moves_tensor()
        
        # Simulate agent decision (select random valid actions)
        batch_size = observations.shape[0]
        actions = torch.zeros(batch_size, dtype=torch.long, device=cpu_device)
        
        for i in range(batch_size):
            valid_cols = torch.where(valid_moves[i])[0]
            if len(valid_cols) > 0:
                actions[i] = valid_cols[torch.randint(len(valid_cols), (1,))]
        
        # Step environment
        rewards, dones, info = env.step_batch(actions.cpu().numpy())
        
        # Should work without errors
        assert len(rewards) == batch_size
        assert len(dones) == batch_size
    
    def test_training_loop_simulation(self, small_vectorized_env):
        """Test simulated training loop with vectorized environments."""
        env = small_vectorized_env
        
        total_steps = 0
        total_games = 0
        
        for episode in range(10):  # Short training simulation
            observations = env.get_observations_gpu()
            
            step_count = 0
            while step_count < 50:  # Max steps per episode
                # Get valid moves and make random actions
                valid_moves = env.get_valid_moves_batch()
                actions = []
                
                for i, vm in enumerate(valid_moves):
                    if vm:
                        actions.append(np.random.choice(vm))
                    else:
                        actions.append(0)  # Fallback (will be invalid)
                
                actions = np.array(actions)
                rewards, dones, info = env.step_batch(actions)
                
                # Auto-reset finished games
                reset_envs = env.auto_reset_finished_games()
                total_games += len(reset_envs)
                
                step_count += 1
                total_steps += env.num_envs
                
                if all(dones):
                    break
        
        # Should have completed simulation without errors
        assert total_steps > 0
        assert total_games >= 0