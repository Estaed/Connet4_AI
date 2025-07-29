"""
Connect4 Gymnasium Environment Wrapper

This module provides a Gymnasium-compatible wrapper for the Connect4 game,
enabling integration with reinforcement learning frameworks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .connect4_game import Connect4Game


class Connect4Env(gym.Env):
    """
    Gymnasium environment wrapper for Connect4 game.
    
    This wrapper provides a standard RL interface for the Connect4 game,
    with action masking and proper reward structure for AI training.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Connect4 environment.
        
        Args:
            render_mode: Mode for rendering ('human' or 'ansi')
        """
        super().__init__()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(7)  # 7 columns
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(6, 7), dtype=np.int8
        )
        
        # Initialize game
        self.game = Connect4Game()
        self.render_mode = render_mode
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Column index (0-6) to drop piece
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")
        
        if action < 0 or action >= 7:
            raise ValueError(f"Action must be between 0 and 6, got {action}")
        
        # Check if action is valid
        if not self.game.is_valid_move(action):
            # Invalid move penalty
            return (
                self.game.board.copy(),
                -1.0,  # Heavy penalty for invalid move
                True,  # Game terminates
                False,
                {
                    'valid_moves': self.game.get_valid_moves(),
                    'action_mask': self._get_action_mask(),
                    'invalid_move': True
                }
            )
        
        # Make the move
        success = self.game.drop_piece(action)
        assert success, "Move should be valid at this point"
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get current state
        observation = self.game.board.copy()
        terminated = self.game.game_over
        truncated = False  # No time limits in basic Connect4
        
        info = {
            'valid_moves': self.game.get_valid_moves(),
            'action_mask': self._get_action_mask(),
            'current_player': self.game.current_player,
            'move_count': self.game.move_count,
            'winner': self.game.winner,
            'invalid_move': False
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed (not used in deterministic Connect4)
            options: Additional options (not used)
            
        Returns:
            Tuple of (initial_observation, info)
        """
        # Handle seeding (though Connect4 is deterministic)
        super().reset(seed=seed)
        
        # Reset the game
        observation = self.game.reset()
        
        info = {
            'valid_moves': self.game.get_valid_moves(),
            'action_mask': self._get_action_mask(),
            'current_player': self.game.current_player,
            'move_count': self.game.move_count,
            'winner': None
        }
        
        return observation, info
    
    def render(self, mode: Optional[str] = None) -> Optional[str]:
        """
        Render the environment.
        
        Args:
            mode: Render mode override
            
        Returns:
            String representation if mode is 'ansi', None for 'human'
        """
        render_mode = mode if mode is not None else self.render_mode
        
        if render_mode == 'ansi':
            return self.game.render(mode='return', show_stats=False)
        elif render_mode == 'human':
            self.game.render(mode='human', show_stats=True)
            return None
        else:
            # Default to string representation
            return str(self.game)
    
    def close(self):
        """Clean up the environment."""
        pass
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on current game state.
        
        Reward structure:
        - +2.0 for winning
        - -0.1 for draw  
        - -1.0 for losing
        - -0.01 for each move (to encourage shorter games)
        
        Returns:
            Reward value
        """
        if self.game.game_over:
            if self.game.winner == 1:  # Player 1 (current perspective)
                return 2.0
            elif self.game.winner == -1:  # Player 2
                return -1.0
            else:  # Draw (winner == 0)
                return -0.1
        else:
            # Small negative reward for each move to encourage efficiency
            return -0.01
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Get action mask for valid moves.
        
        Returns:
            Boolean array where True indicates valid actions
        """
        mask = np.zeros(7, dtype=bool)
        valid_moves = self.game.get_valid_moves()
        for move in valid_moves:
            mask[move] = True
        return mask