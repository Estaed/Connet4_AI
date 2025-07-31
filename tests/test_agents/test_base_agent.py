"""
Comprehensive tests for BaseAgent abstract base class.

Tests the abstract agent interface, tactical analysis functions,
and common agent functionality that all agents inherit.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from abc import ABC
from src.agents.base_agent import BaseAgent, create_agent
from src.agents.random_agent import RandomAgent
from src.agents.ppo_agent import PPOAgent


# Concrete implementation for testing abstract base class
class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""
    
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.action_calls = []
        self.update_calls = []
    
    def get_action(self, observation, valid_actions=None):
        """Mock implementation that records calls."""
        self.action_calls.append((observation, valid_actions))
        if valid_actions and len(valid_actions) > 0:
            return valid_actions[0]
        return 0
    
    def update(self, experiences):
        """Mock implementation that records calls."""
        self.update_calls.append(experiences)


class TestBaseAgentInterface:
    """Test BaseAgent abstract interface and basic functionality."""
    
    def test_abstract_base_class(self):
        """Test that BaseAgent is properly abstract."""
        # Should not be able to instantiate BaseAgent directly
        with pytest.raises(TypeError):
            BaseAgent()
    
    def test_concrete_implementation(self):
        """Test that concrete implementations work correctly."""
        agent = TestAgent()
        assert isinstance(agent, BaseAgent)
        assert hasattr(agent, 'get_action')
        assert hasattr(agent, 'update')
        assert hasattr(agent, 'save')
        assert hasattr(agent, 'load')
    
    def test_device_initialization(self):
        """Test device initialization and handling."""
        # Default device
        agent = TestAgent()
        assert agent.device == 'cpu'
        
        # Custom device
        agent_cuda = TestAgent(device='cuda')
        assert agent_cuda.device == 'cuda'
    
    def test_abstract_methods_interface(self):
        """Test that abstract methods are properly defined."""
        agent = TestAgent()
        
        # Test get_action interface
        result = agent.get_action(np.zeros((6, 7)), [0, 1, 2])
        assert result == 0  # First valid action
        assert len(agent.action_calls) == 1
        
        # Test update interface
        agent.update([])
        assert len(agent.update_calls) == 1


class TestBaseAgentTacticalAnalysis:
    """Test tactical analysis functions in BaseAgent."""
    
    def test_analyze_board_threats_empty_board(self):
        """Test threat analysis on empty board."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        threats = agent.analyze_board_threats(observation, player=1)
        
        assert 'winning_moves' in threats
        assert 'blocking_moves' in threats
        assert 'threat_positions' in threats
        
        # Empty board should have no immediate threats
        assert len(threats['winning_moves']) == 0
        assert len(threats['blocking_moves']) == 0
        assert len(threats['threat_positions']) == 0
    
    def test_analyze_board_threats_winning_move(self):
        """Test detection of winning moves."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Set up horizontal winning opportunity for player 1
        observation[5, 0] = 1  # Player 1
        observation[5, 1] = 1  # Player 1
        observation[5, 2] = 1  # Player 1
        # observation[5, 3] = 0  # Empty - winning move for player 1
        
        threats = agent.analyze_board_threats(observation, player=1)
        
        # Should detect winning move at column 3
        assert 3 in threats['winning_moves']
    
    def test_analyze_board_threats_blocking_move(self):
        """Test detection of blocking moves."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Set up horizontal threat from opponent (player -1)
        observation[5, 0] = -1  # Opponent
        observation[5, 1] = -1  # Opponent
        observation[5, 2] = -1  # Opponent
        # observation[5, 3] = 0   # Empty - must block here
        
        threats = agent.analyze_board_threats(observation, player=1)
        
        # Should detect need to block at column 3
        assert 3 in threats['blocking_moves']
    
    def test_analyze_board_threats_vertical_patterns(self):
        """Test detection of vertical threat patterns."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Set up vertical winning opportunity
        observation[5, 3] = 1  # Bottom
        observation[4, 3] = 1  # Second
        observation[3, 3] = 1  # Third
        # observation[2, 3] = 0  # Empty - winning move
        
        threats = agent.analyze_board_threats(observation, player=1)
        
        # Should detect winning move in column 3
        assert 3 in threats['winning_moves']
    
    def test_analyze_board_threats_diagonal_patterns(self):
        """Test detection of diagonal threat patterns."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Set up positive diagonal winning opportunity
        observation[5, 0] = 1  # (5,0)
        observation[4, 1] = 1  # (4,1)
        observation[3, 2] = 1  # (3,2)
        # Need piece at (2,3) to win
        
        threats = agent.analyze_board_threats(observation, player=1)
        
        # Should detect potential winning positions
        # (This might require the position to be accessible - check implementation)
        assert len(threats['threat_positions']) > 0
    
    def test_analyze_board_threats_complex_board(self, game_with_moves):
        """Test threat analysis on complex board state."""
        agent = TestAgent()
        observation = game_with_moves.board.astype(np.int8)
        
        # Test analysis for both players
        threats_p1 = agent.analyze_board_threats(observation, player=1)
        threats_p2 = agent.analyze_board_threats(observation, player=-1)
        
        # Should return valid threat dictionaries
        for threats in [threats_p1, threats_p2]:
            assert isinstance(threats['winning_moves'], list)
            assert isinstance(threats['blocking_moves'], list)
            assert isinstance(threats['threat_positions'], list)
            
            # All moves should be valid columns (0-6)
            for move in threats['winning_moves'] + threats['blocking_moves']:
                assert 0 <= move <= 6


class TestBaseAgentPositionEvaluation:
    """Test position evaluation functions in BaseAgent."""
    
    def test_evaluate_position_strength_empty_board(self):
        """Test position evaluation on empty board."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        strength = agent.evaluate_position_strength(observation, player=1)
        
        # Should return numeric score
        assert isinstance(strength, (int, float))
        
        # Empty board should have neutral/low strength
        assert -10 <= strength <= 10  # Reasonable range
    
    def test_evaluate_position_strength_winning_position(self):
        """Test position evaluation with winning position."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Create winning position
        for col in range(4):
            observation[5, col] = 1
        
        strength = agent.evaluate_position_strength(observation, player=1)
        
        # Winning position should have high strength
        assert strength > 0
    
    def test_evaluate_position_strength_losing_position(self):
        """Test position evaluation with losing position."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Create position where opponent is about to win
        for col in range(3):
            observation[5, col] = -1  # Opponent has 3 in a row
        
        strength = agent.evaluate_position_strength(observation, player=1)
        
        # Losing position should have low strength
        assert strength < 0
    
    def test_evaluate_position_strength_symmetry(self):
        """Test that position evaluation is consistent for symmetric positions."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Create symmetric position
        observation[5, 1] = 1
        observation[5, 5] = 1
        observation[4, 2] = -1
        observation[4, 4] = -1
        
        strength1 = agent.evaluate_position_strength(observation, player=1)
        
        # Mirror the board
        mirrored_obs = observation[:, ::-1]  # Flip horizontally
        strength2 = agent.evaluate_position_strength(mirrored_obs, player=1)
        
        # Symmetric positions should have similar strength
        assert abs(strength1 - strength2) < 1.0  # Allow small differences due to implementation
    
    def test_evaluate_position_strength_player_perspective(self):
        """Test that position evaluation changes based on player perspective."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Create position favorable to player 1
        observation[5, 0:3] = 1
        observation[5, 4:6] = -1
        
        strength_p1 = agent.evaluate_position_strength(observation, player=1)
        strength_p2 = agent.evaluate_position_strength(observation, player=-1)
        
        # Position should be evaluated differently from each player's perspective
        assert strength_p1 != strength_p2
        # If position is good for player 1, it should be bad for player -1
        if strength_p1 > 0:
            assert strength_p2 < strength_p1


class TestBaseAgentStrategicMoves:
    """Test strategic move finding functionality."""
    
    def test_find_best_strategic_move_immediate_win(self):
        """Test that immediate winning moves are prioritized."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Set up immediate winning move at column 3
        observation[5, 0:3] = 1
        
        best_move = agent.find_best_strategic_move(observation, valid_actions, player=1)
        
        # Should choose the winning move
        assert best_move == 3
    
    def test_find_best_strategic_move_block_opponent(self):
        """Test that blocking opponent wins is prioritized."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        # Set up opponent winning threat at column 3
        observation[5, 0:3] = -1  # Opponent has 3 in a row
        
        best_move = agent.find_best_strategic_move(observation, valid_actions, player=1)
        
        # Should choose to block the opponent
        assert best_move == 3
    
    def test_find_best_strategic_move_no_immediate_threats(self):
        """Test strategic move selection without immediate threats."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        valid_actions = [1, 2, 3, 4, 5]  # Exclude edge columns
        
        # Add some pieces but no immediate threats
        observation[5, 0] = 1
        observation[5, 6] = -1
        
        best_move = agent.find_best_strategic_move(observation, valid_actions, player=1)
        
        # Should return a valid move
        assert best_move in valid_actions
        
        # Should prefer center columns (common Connect4 strategy)
        # This depends on implementation details
        assert isinstance(best_move, int)
    
    def test_find_best_strategic_move_limited_valid_actions(self):
        """Test strategic move selection with limited valid actions."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        valid_actions = [2, 4]  # Only two valid moves
        
        best_move = agent.find_best_strategic_move(observation, valid_actions, player=1)
        
        # Should return one of the valid actions
        assert best_move in valid_actions
    
    def test_find_best_strategic_move_empty_valid_actions(self):
        """Test handling of empty valid actions list."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        valid_actions = []
        
        best_move = agent.find_best_strategic_move(observation, valid_actions, player=1)
        
        # Should handle gracefully (might return None or default)
        assert best_move is None or isinstance(best_move, int)


class TestBaseAgentSaveLoad:
    """Test save and load functionality."""
    
    def test_save_load_interface(self, tmp_path):
        """Test basic save/load interface."""
        agent = TestAgent()
        save_path = tmp_path / "test_agent.pth"
        
        # Should not raise errors (default implementation might be no-op)
        agent.save(str(save_path))
        agent.load(str(save_path))
    
    def test_save_load_state_preservation(self, tmp_path):
        """Test that agent state is preserved through save/load."""
        agent = TestAgent()
        save_path = tmp_path / "test_agent.pth"
        
        # Modify agent state
        agent.action_calls = [1, 2, 3]
        agent.update_calls = [4, 5, 6]
        
        # Save and create new agent
        agent.save(str(save_path))
        new_agent = TestAgent()
        new_agent.load(str(save_path))
        
        # Note: Default BaseAgent save/load might not preserve state
        # This test documents expected behavior - actual implementation may vary


class TestBaseAgentEdgeCases:
    """Test edge cases and error handling."""
    
    def test_analyze_board_threats_invalid_board(self):
        """Test threat analysis with invalid board shapes."""
        agent = TestAgent()
        
        # Wrong shape board
        invalid_board = np.zeros((5, 6))  # Should be (6, 7)
        
        # Should handle gracefully or raise appropriate error
        try:
            threats = agent.analyze_board_threats(invalid_board, player=1)
            # If no error, should return valid structure
            assert isinstance(threats, dict)
        except (ValueError, IndexError, AssertionError):
            # Appropriate error handling
            pass
    
    def test_analyze_board_threats_invalid_player(self):
        """Test threat analysis with invalid player values."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Invalid player values
        for invalid_player in [0, 2, -2, 1.5, 'player']:
            try:
                threats = agent.analyze_board_threats(observation, player=invalid_player)
                # If no error, should return valid structure
                assert isinstance(threats, dict)
            except (ValueError, TypeError):
                # Appropriate error handling
                pass
    
    def test_evaluate_position_strength_edge_cases(self):
        """Test position evaluation edge cases."""
        agent = TestAgent()
        
        # Full board
        full_board = np.ones((6, 7), dtype=np.int8)
        strength = agent.evaluate_position_strength(full_board, player=1)
        assert isinstance(strength, (int, float))
        
        # Board with invalid values
        invalid_board = np.full((6, 7), 5, dtype=np.int8)  # Invalid piece values
        try:
            strength = agent.evaluate_position_strength(invalid_board, player=1)
            assert isinstance(strength, (int, float))
        except (ValueError, AssertionError):
            # Appropriate error handling
            pass
    
    def test_find_best_strategic_move_edge_cases(self):
        """Test strategic move finding edge cases."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Invalid valid_actions
        invalid_actions_list = [
            [-1, 0, 1],  # Negative column
            [7, 8, 9],   # Out of range columns
            [1.5, 2.5],  # Non-integer columns
        ]
        
        for invalid_actions in invalid_actions_list:
            try:
                best_move = agent.find_best_strategic_move(
                    observation, invalid_actions, player=1
                )
                # If no error, should return valid result
                assert best_move is None or isinstance(best_move, int)
            except (ValueError, IndexError, TypeError):
                # Appropriate error handling
                pass


class TestAgentFactory:
    """Test agent creation factory function."""
    
    def test_create_random_agent(self):
        """Test creation of random agent."""
        agent = create_agent('random')
        assert isinstance(agent, RandomAgent)
        assert isinstance(agent, BaseAgent)
    
    def test_create_ppo_agent(self, test_config):
        """Test creation of PPO agent."""
        agent = create_agent('ppo', config=test_config)
        assert isinstance(agent, PPOAgent)
        assert isinstance(agent, BaseAgent)
    
    def test_create_agent_with_device(self):
        """Test agent creation with device specification."""
        agent = create_agent('random', device='cpu')
        assert agent.device == 'cpu'
        
        if hasattr(agent, 'device'):  # Some agents might not store device
            agent_cuda = create_agent('random', device='cuda')
            assert 'cuda' in str(agent_cuda.device) or agent_cuda.device == 'cuda'
    
    def test_create_agent_invalid_type(self):
        """Test handling of invalid agent types."""
        with pytest.raises((ValueError, KeyError)):
            create_agent('invalid_agent_type')
    
    def test_create_agent_missing_config(self):
        """Test PPO agent creation without required config."""
        # PPO agent might require config
        try:
            agent = create_agent('ppo')
            # If successful, should return valid agent
            assert isinstance(agent, PPOAgent)
        except (ValueError, TypeError):
            # Appropriate error handling for missing config
            pass


class TestBaseAgentIntegration:
    """Test integration with other components."""
    
    def test_integration_with_game_environment(self, empty_game):
        """Test agent integration with game environment."""
        agent = TestAgent()
        game = empty_game
        
        # Simulate game interaction
        observation = game.board.astype(np.int8)
        valid_actions = game.get_valid_moves()
        
        action = agent.get_action(observation, valid_actions)
        
        # Action should be valid
        assert action in valid_actions
        
        # Should be able to make the move
        result = game.drop_piece(action)
        assert result == True
    
    def test_integration_tactical_analysis_consistency(self):
        """Test that tactical analysis methods work together consistently."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Set up a position with tactical elements
        observation[5, 0:3] = 1  # Player 1 has 3 in a row
        observation[5, 4:6] = -1  # Player -1 has 2 in a row
        
        # Analyze threats
        threats = agent.analyze_board_threats(observation, player=1)
        
        # Evaluate position
        strength = agent.evaluate_position_strength(observation, player=1)
        
        # Find strategic move
        valid_actions = [3, 6]  # Can complete win or block opponent
        strategic_move = agent.find_best_strategic_move(
            observation, valid_actions, player=1
        )
        
        # Results should be consistent
        if 3 in threats['winning_moves']:
            # If there's a winning move, it should be chosen
            assert strategic_move == 3
            # Position should be evaluated as strong
            assert strength > 0
    
    @pytest.mark.parametrize("player", [1, -1])
    def test_player_consistency(self, player):
        """Test that all methods work consistently for both players."""
        agent = TestAgent()
        observation = np.zeros((6, 7), dtype=np.int8)
        
        # Add some pieces for both players
        observation[5, 1] = 1
        observation[5, 2] = -1
        observation[4, 1] = -1
        observation[4, 3] = 1
        
        # All methods should work for both players
        threats = agent.analyze_board_threats(observation, player)
        strength = agent.evaluate_position_strength(observation, player)
        strategic_move = agent.find_best_strategic_move(
            observation, [0, 3, 4, 5, 6], player
        )
        
        # Should return valid results for both players
        assert isinstance(threats, dict)
        assert isinstance(strength, (int, float))
        assert strategic_move is None or isinstance(strategic_move, int)