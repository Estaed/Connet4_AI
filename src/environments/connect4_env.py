"""
Connect4 Gymnasium Environment Wrapper

This module provides a Gymnasium-compatible wrapper for the Connect4 game,
enabling integration with reinforcement learning frameworks.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Any, Tuple, Optional

from .connect4_game import Connect4Game


class Connect4Env(gym.Env):
    """
    Gymnasium environment wrapper for Connect4 game.
    
    This wrapper provides a standard RL interface for the Connect4 game,
    with action masking and proper reward structure for AI training.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, render_mode: Optional[str] = None, random_start: bool = True):
        """
        Initialize the Connect4 environment.
        
        Args:
            render_mode: Mode for rendering ('human' or 'ansi')
            random_start: Whether to randomly select starting player
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
        self.random_start = random_start
        
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
        
        # Store previous board state for reward calculation
        previous_board = self.game.board.copy()
        previous_player = self.game.current_player
        
        # Make the move
        success = self.game.drop_piece(action)
        assert success, "Move should be valid at this point"
        
        # Calculate strategic reward
        reward = self._calculate_strategic_reward(previous_board, action, previous_player)
        
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
            seed: Random seed for starting player selection
            options: Additional options (not used)
            
        Returns:
            Tuple of (initial_observation, info)
        """
        # Handle seeding
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        
        # Reset the game
        observation = self.game.reset()
        
        # Randomly select starting player if enabled
        if self.random_start:
            starting_player = random.choice([1, -1])
            self.game.current_player = starting_player
        
        info = {
            'valid_moves': self.game.get_valid_moves(),
            'action_mask': self._get_action_mask(),
            'current_player': self.game.current_player,
            'move_count': self.game.move_count,
            'winner': None,
            'starting_player': self.game.current_player
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
    
    def _calculate_strategic_reward(self, previous_board: np.ndarray, action: int, player: int) -> float:
        """
        Calculate strategic reward based on game state changes and tactical analysis.
        
        Reward structure:
        Game Outcome (Terminal Rewards):
        - Win: +3.0 (The ultimate goal)
        - Loss: -3.0 (The ultimate failure)
        - Draw: +0.1 (Slightly positive neutral outcome)
        
        Immediate Threats & Defenses (High-Value Intermediate Rewards):
        - Making a winning move: +1.0 (immediate win reward)
        - Blocking opponent's winning move: +0.5
        - Setting up "three-in-a-row": +0.2
        
        Positional & Strategic Advantages (Low-Value Intermediate Rewards):
        - Setting up "two-in-a-row": +0.05
        - Center column control: +0.01
        
        Penalties (Negative Intermediate Rewards):
        - Allowing opponent immediate win: -0.7
        
        Args:
            previous_board: Board state before the move
            action: Column where piece was placed
            player: Player who made the move
            
        Returns:
            Strategic reward value
        """
        # Check for game ending first
        if self.game.game_over:
            if self.game.winner == player:
                return 3.0  # Win
            elif self.game.winner == -player:
                return -3.0  # Loss
            else:
                return 0.1  # Draw (slightly positive)
        
        reward = 0.0
        
        # Get the row where the piece was placed
        placed_row = None
        for row in range(self.game.board_rows):
            if previous_board[row, action] == 0 and self.game.board[row, action] == player:
                placed_row = row
                break
        
        if placed_row is None:
            return reward  # Shouldn't happen, but safety check
        
        # 1. Center Column Control (+0.01)
        if action == 3:  # Center column (0-indexed)
            reward += 0.01
        
        # 2. Check if this move blocked an opponent's winning move (+0.5)
        if self._would_win_if_played(previous_board, action, -player):
            reward += 0.5
        
        # 3. Check if this move allows opponent to win next turn (-0.7)
        if self._allows_opponent_win(self.game.board, player):
            reward -= 0.7
        
        # 4. Check for setting up three-in-a-row (+0.2)
        if self._creates_three_in_row(self.game.board, placed_row, action, player):
            reward += 0.2
        
        # 5. Check for setting up two-in-a-row (+0.05)
        elif self._creates_two_in_row(self.game.board, placed_row, action, player):
            reward += 0.05
        
        return reward
    
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
    
    def _would_win_if_played(self, board: np.ndarray, col: int, player: int) -> bool:
        """
        Check if playing in a column would result in a win for the player.
        
        Args:
            board: Board state to check
            col: Column to test
            player: Player to test for
            
        Returns:
            True if playing in this column would win
        """
        # Find where the piece would land
        row = None
        for r in range(self.game.board_rows):
            if board[r, col] == 0:
                row = r
                break
        
        if row is None:
            return False  # Column is full
        
        # Temporarily place the piece
        test_board = board.copy()
        test_board[row, col] = player
        
        # Check for win in all directions
        return self._check_win_at_position(test_board, row, col, player)
    
    def _check_win_at_position(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """
        Check if there's a win starting from a specific position.
        
        Args:
            board: Board state
            row: Row position
            col: Column position
            player: Player to check for
            
        Returns:
            True if there's a winning line through this position
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the placed piece
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                   board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                   board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False
    
    def _allows_opponent_win(self, board: np.ndarray, current_player: int) -> bool:
        """
        Check if the current board state allows opponent to win on their next move.
        
        Args:
            board: Current board state
            current_player: Player who just moved
            
        Returns:
            True if opponent can win on next move
        """
        opponent = -current_player
        
        for col in range(self.game.board_cols):
            if self._would_win_if_played(board, col, opponent):
                return True
        
        return False
    
    def _creates_three_in_row(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """
        Check if the placed piece creates a three-in-a-row threat.
        
        Args:
            board: Current board state
            row: Row of placed piece
            col: Column of placed piece
            player: Player who placed the piece
            
        Returns:
            True if this creates a three-in-a-row with open ends
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the placed piece
            open_ends = 0
            
            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                   board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Check if positive end is open and playable
            if (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                board[r, c] == 0 and (r == self.game.board_rows - 1 or board[r + 1, c] != 0)):
                open_ends += 1
            
            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                   board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            # Check if negative end is open and playable
            if (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                board[r, c] == 0 and (r == self.game.board_rows - 1 or board[r + 1, c] != 0)):
                open_ends += 1
            
            # Three in a row with at least one open end is a threat
            if count == 3 and open_ends > 0:
                return True
        
        return False
    
    def _creates_two_in_row(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """
        Check if the placed piece creates a two-in-a-row building block.
        
        Args:
            board: Current board state
            row: Row of placed piece
            col: Column of placed piece
            player: Player who placed the piece
            
        Returns:
            True if this creates a two-in-a-row with open ends
        """
        directions = [
            (0, 1),   # Horizontal
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the placed piece
            open_ends = 0
            
            # Check positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                   board[r, c] == player):
                count += 1
                r, c = r + dr, c + dc
            
            # Check if positive end is open and has space for expansion
            if (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                board[r, c] == 0):
                open_ends += 1
            
            # Check negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                   board[r, c] == player):
                count += 1
                r, c = r - dr, c - dc
            
            # Check if negative end is open and has space for expansion
            if (0 <= r < self.game.board_rows and 0 <= c < self.game.board_cols and 
                board[r, c] == 0):
                open_ends += 1
            
            # Two in a row with both ends open (room to grow to 4)
            if count == 2 and open_ends >= 2:
                return True
        
        return False