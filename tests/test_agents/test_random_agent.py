"""
Comprehensive tests for RandomAgent implementation.

Tests the baseline random agent that selects moves randomly
from valid actions for comparison and testing purposes.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock
from src.agents.random_agent import RandomAgent
from src.agents.base_agent import BaseAgent


class TestRandomAgentInitialization:
    """Test RandomAgent initialization and basic setup."""
    
    def test_inheritance(self):
        """Test that RandomAgent inherits from BaseAgent."""
        agent = RandomAgent()
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, RandomAgent)
    
    def test_default_initialization(self):
        """Test default initialization."""
        agent = RandomAgent()
        assert agent.device == 'cpu'  # Default device
    
    def test_device_initialization(self):
        """Test initialization with custom device."""
        agent_cpu = RandomAgent(device='cpu')
        assert agent_cpu.device == 'cpu'
        
        agent_cuda = RandomAgent(device='cuda')
        assert agent_cuda.device == 'cuda'
    
    def test_required_methods_implemented(self):
        """Test that required abstract methods are implemented."""
        agent = RandomAgent()
        
        # Should have all required methods
        assert hasattr(agent, 'get_action')
        assert hasattr(agent, 'update')
        assert callable(agent.get_action)
        assert callable(agent.update)


class TestRandomAgentActionSelection:
    """Test action selection functionality."""
    
    def test_get_action_with_valid_actions(self):
        """Test action selection with valid actions provided."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))  # Dummy observation
        valid_actions = [1, 3, 5]
        
        action = agent.get_action(observation, valid_actions)
        
        # Should return one of the valid actions
        assert action in valid_actions
        assert isinstance(action, (int, np.integer))
    
    def test_get_action_without_valid_actions(self):
        """Test action selection without valid actions provided."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        
        action = agent.get_action(observation)
        
        # Should return valid column index (0-6)
        assert 0 <= action <= 6
        assert isinstance(action, (int, np.integer))
    
    def test_get_action_single_valid_action(self):
        """Test action selection with only one valid action."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        valid_actions = [3]
        
        action = agent.get_action(observation, valid_actions)
        
        # Should return the only valid action
        assert action == 3
    
    def test_get_action_all_columns_valid(self):
        """Test action selection when all columns are valid."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Call multiple times to test randomness
        actions = []
        for _ in range(50):
            action = agent.get_action(observation, valid_actions)
            actions.append(action)
            assert action in valid_actions
        
        # Should use multiple different actions (randomness test)
        unique_actions = set(actions)
        assert len(unique_actions) > 1  # Should not always choose same action
    
    def test_get_action_empty_valid_actions(self):
        """Test handling of empty valid actions list."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        valid_actions = []
        
        # Should handle gracefully
        action = agent.get_action(observation, valid_actions)
        
        # Behavior depends on implementation - might return default or raise error
        if action is not None:
            assert isinstance(action, (int, np.integer))
            assert 0 <= action <= 6
    
    def test_get_action_none_valid_actions(self):
        """Test handling of None valid actions."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        
        action = agent.get_action(observation, valid_actions=None)
        
        # Should return valid column
        assert 0 <= action <= 6
        assert isinstance(action, (int, np.integer))


class TestRandomAgentRandomness:
    """Test randomness properties of RandomAgent."""
    
    def test_action_distribution(self):
        """Test that actions are roughly uniformly distributed."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Collect many actions
        actions = []
        num_trials = 1000
        
        for _ in range(num_trials):
            action = agent.get_action(observation, valid_actions)
            actions.append(action)
        
        # Count frequency of each action
        action_counts = {i: actions.count(i) for i in range(7)}
        
        # Should be roughly uniform distribution
        expected_count = num_trials / 7
        tolerance = expected_count * 0.3  # 30% tolerance
        
        for count in action_counts.values():
            assert abs(count - expected_count) < tolerance
    
    def test_action_independence(self):
        """Test that consecutive actions are independent."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Generate sequence of actions
        actions = []
        for _ in range(100):
            action = agent.get_action(observation, valid_actions)
            actions.append(action)
        
        # Test that there are no obvious patterns
        # (This is a basic test - more sophisticated randomness tests could be added)
        
        # Should not repeat the same action too many times in a row
        max_consecutive = 0
        current_consecutive = 1
        
        for i in range(1, len(actions)):
            if actions[i] == actions[i-1]:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
        
        max_consecutive = max(max_consecutive, current_consecutive)
        
        # Should not have extremely long runs (more than 10 in a row is very unlikely)
        assert max_consecutive < 10
    
    def test_different_agent_instances_independence(self):
        """Test that different agent instances produce different sequences."""
        observation = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Create two agents
        agent1 = RandomAgent()
        agent2 = RandomAgent()
        
        # Generate sequences from each
        sequence1 = [agent1.get_action(observation, valid_actions) for _ in range(20)]
        sequence2 = [agent2.get_action(observation, valid_actions) for _ in range(20)]
        
        # Sequences should not be identical (very unlikely with proper randomness)
        assert sequence1 != sequence2
    
    @patch('numpy.random.choice')
    def test_random_choice_usage(self, mock_choice):
        """Test that numpy.random.choice is used correctly."""
        mock_choice.return_value = 3
        
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        valid_actions = [1, 3, 5]
        
        action = agent.get_action(observation, valid_actions)
        
        # Should have called numpy.random.choice with valid_actions
        mock_choice.assert_called_once_with(valid_actions)
        assert action == 3
    
    @patch('numpy.random.randint')
    def test_random_int_fallback(self, mock_randint):
        """Test fallback behavior when no valid actions provided."""
        mock_randint.return_value = 4
        
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        
        # Implementation might use randint as fallback
        action = agent.get_action(observation, valid_actions=None)
        
        # Behavior depends on implementation
        assert isinstance(action, (int, np.integer))


class TestRandomAgentUpdate:
    """Test update functionality (should be no-op for RandomAgent)."""
    
    def test_update_no_op(self):
        """Test that update method exists but does nothing."""
        agent = RandomAgent()
        
        # Should not raise any errors
        agent.update([])
        agent.update(None)
        agent.update([1, 2, 3, 4, 5])
    
    def test_update_with_experiences(self, sample_experiences):
        """Test update with sample experiences."""
        agent = RandomAgent()
        
        # Should handle any type of experiences without error
        agent.update(sample_experiences)
        
        # Agent behavior should not change after update
        observation = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        action1 = agent.get_action(observation, valid_actions)
        agent.update(sample_experiences)
        action2 = agent.get_action(observation, valid_actions)
        
        # Both actions should be valid (behavior unchanged)
        assert action1 in valid_actions
        assert action2 in valid_actions


class TestRandomAgentSaveLoad:
    """Test save and load functionality."""
    
    def test_save_load_interface(self, tmp_path):
        """Test save and load interface."""
        agent = RandomAgent()
        save_path = tmp_path / "random_agent.pth"
        
        # Should not raise errors
        agent.save(str(save_path))
        agent.load(str(save_path))
    
    def test_save_load_behavior_unchanged(self, tmp_path):
        """Test that save/load doesn't change agent behavior."""
        agent = RandomAgent()
        save_path = tmp_path / "random_agent.pth"
        
        observation = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Test behavior before save
        actions_before = [agent.get_action(observation, valid_actions) for _ in range(10)]
        
        # Save and load
        agent.save(str(save_path))
        agent.load(str(save_path))
        
        # Test behavior after load
        actions_after = [agent.get_action(observation, valid_actions) for _ in range(10)]
        
        # Should still produce valid actions
        assert all(action in valid_actions for action in actions_before)
        assert all(action in valid_actions for action in actions_after)


class TestRandomAgentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_valid_actions_types(self):
        """Test handling of invalid valid_actions types."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        
        # Test various invalid types
        invalid_inputs = [
            "invalid",
            123,
            [1.5, 2.5, 3.5],  # Non-integer actions
            [-1, 0, 1],       # Negative actions
            [7, 8, 9],        # Out of range actions
        ]
        
        for invalid_input in invalid_inputs:
            try:
                action = agent.get_action(observation, invalid_input)
                # If no error, should return valid action
                assert isinstance(action, (int, np.integer))
                assert 0 <= action <= 6
            except (ValueError, TypeError, IndexError):
                # Appropriate error handling
                pass
    
    def test_invalid_observation_shapes(self):
        """Test handling of invalid observation shapes."""
        agent = RandomAgent()
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Test various invalid observation shapes
        invalid_observations = [
            np.zeros((5, 7)),    # Wrong number of rows
            np.zeros((6, 8)),    # Wrong number of columns
            np.zeros((6,)),      # Wrong dimensions
            np.zeros((6, 7, 1)), # Extra dimension
            None,                # None observation
            [],                  # Empty list
        ]
        
        for invalid_obs in invalid_observations:
            try:
                action = agent.get_action(invalid_obs, valid_actions)
                # RandomAgent might ignore observation, so could still work
                assert isinstance(action, (int, np.integer))
                assert action in valid_actions
            except (ValueError, TypeError, AttributeError):
                # Appropriate error handling
                pass
    
    def test_large_valid_actions_list(self):
        """Test behavior with unusually large valid actions lists."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        
        # Very long list of valid actions (with duplicates)
        large_valid_actions = list(range(7)) * 100  # 700 elements
        
        action = agent.get_action(observation, large_valid_actions)
        
        # Should still return valid action
        assert 0 <= action <= 6
    
    def test_out_of_range_valid_actions(self):
        """Test handling of out-of-range valid actions."""
        agent = RandomAgent()
        observation = np.zeros((6, 7))
        
        # Valid actions outside normal range
        out_of_range_actions = [10, 15, 100]
        
        try:
            action = agent.get_action(observation, out_of_range_actions)
            # If no error, behavior depends on implementation
            assert isinstance(action, (int, np.integer))
        except (ValueError, IndexError):
            # Appropriate error handling
            pass


class TestRandomAgentIntegration:
    """Test integration with other system components."""
    
    def test_integration_with_connect4_game(self, empty_game):
        """Test RandomAgent integration with Connect4Game."""
        agent = RandomAgent()
        game = empty_game
        
        # Play several moves
        for _ in range(10):
            observation = game.board.astype(np.int8)
            valid_actions = game.get_valid_moves()
            
            if valid_actions:
                action = agent.get_action(observation, valid_actions)
                assert action in valid_actions
                
                # Make the move
                result = game.drop_piece(action)
                assert result == True
                
                # Check for game end
                if game.check_win(action) or game.is_draw():
                    break
    
    def test_integration_with_vectorized_environment(self, small_vectorized_env):
        """Test RandomAgent with vectorized environments."""
        agent = RandomAgent()
        env = small_vectorized_env
        
        # Get batch observations
        observations = env.get_observations_gpu()
        valid_moves_batch = env.get_valid_moves_batch()
        
        # Agent should work with individual observations
        for i in range(env.num_envs):
            obs = observations[i].cpu().numpy()
            valid_actions = valid_moves_batch[i]
            
            action = agent.get_action(obs, valid_actions)
            assert action in valid_actions
    
    def test_performance_vs_other_agents(self, empty_game):
        """Test RandomAgent performance characteristics."""
        agent = RandomAgent()
        game = empty_game
        
        # Time action selection
        import time
        observation = game.board.astype(np.int8)
        valid_actions = game.get_valid_moves()
        
        start_time = time.time()
        
        # Many action selections
        for _ in range(1000):
            action = agent.get_action(observation, valid_actions)
        
        end_time = time.time()
        selection_time = end_time - start_time
        
        # Should be very fast (no computation required)
        assert selection_time < 0.1  # Less than 0.1 seconds for 1000 selections
        
        actions_per_second = 1000 / selection_time
        assert actions_per_second > 10000  # At least 10k actions per second
    
    def test_agent_factory_integration(self):
        """Test integration with agent factory."""
        from src.agents.base_agent import create_agent
        
        agent = create_agent('random')
        assert isinstance(agent, RandomAgent)
        
        # Should work with device specification
        agent_cpu = create_agent('random', device='cpu')
        assert agent_cpu.device == 'cpu'


class TestRandomAgentGameplay:
    """Test RandomAgent in actual gameplay scenarios."""
    
    def test_complete_game_simulation(self):
        """Test RandomAgent playing complete games."""
        from src.environments.connect4_game import Connect4Game
        
        # Simulate many games between two random agents
        games_completed = 0
        total_moves = 0
        
        for _ in range(10):  # Play 10 games
            game = Connect4Game()
            agent1 = RandomAgent()
            agent2 = RandomAgent()
            
            move_count = 0
            while not game.game_over and move_count < 42:
                current_agent = agent1 if game.current_player == 1 else agent2
                
                observation = game.board.astype(np.int8)
                valid_actions = game.get_valid_moves()
                
                if valid_actions:
                    action = current_agent.get_action(observation, valid_actions)
                    result = game.drop_piece(action)
                    assert result == True
                    
                    if game.check_win(action) or game.is_draw():
                        game.game_over = True
                
                move_count += 1
            
            games_completed += 1
            total_moves += move_count
        
        # Should complete all games
        assert games_completed == 10
        
        # Average game length should be reasonable
        avg_moves = total_moves / games_completed
        assert 7 <= avg_moves <= 42  # Between minimum and maximum possible
    
    def test_random_agent_vs_strategic_positions(self):
        """Test RandomAgent behavior in strategic positions."""
        agent = RandomAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Test position with obvious winning move
        observation[5, 0:3] = 1  # Three in a row, can win at column 3
        valid_actions = [3, 4, 5, 6]
        
        # RandomAgent should still choose randomly (no strategy)
        actions_chosen = []
        for _ in range(20):
            action = agent.get_action(observation, valid_actions)
            actions_chosen.append(action)
        
        # Should choose all available actions, not just the winning one
        unique_actions = set(actions_chosen)
        assert len(unique_actions) > 1  # Should not always choose same action
        
        # Should include the winning move sometimes, but not exclusively
        assert 3 in actions_chosen  # Should choose winning move sometimes
        assert any(a != 3 for a in actions_chosen)  # Should choose other moves too
    
    @pytest.mark.parametrize("game_state", ["early", "middle", "late"])
    def test_random_agent_consistency_across_game_states(self, game_state):
        """Test that RandomAgent behaves consistently across different game states."""
        agent = RandomAgent()
        
        # Create different game states
        if game_state == "early":
            observation = np.zeros((6, 7), dtype=np.int8)
            # Add just a few pieces
            observation[5, 3] = 1
            observation[5, 2] = -1
        elif game_state == "middle":
            observation = np.zeros((6, 7), dtype=np.int8)
            # Add more pieces in middle game
            for i in range(3):
                observation[5-i, 0] = 1
                observation[5-i, 6] = -1
                observation[5-i, 3] = (-1) ** i
        else:  # late game
            observation = np.random.choice([-1, 0, 1], size=(6, 7))
            observation[0:2, :] = 0  # Keep some top rows empty
        
        valid_actions = [i for i in range(7) if observation[0, i] == 0]
        
        if valid_actions:
            # Should work consistently regardless of game state
            actions = []
            for _ in range(10):
                action = agent.get_action(observation, valid_actions)
                actions.append(action)
                assert action in valid_actions
            
            # Should still show randomness
            if len(valid_actions) > 1:
                assert len(set(actions)) > 1 or len(actions) < 5  # Allow some repetition in small samples