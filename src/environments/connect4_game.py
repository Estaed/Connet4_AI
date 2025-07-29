"""
Connect4 Game Logic - Pure Python Implementation

This module contains the core Connect4 game logic without any ML dependencies.
Designed for AI training with simple, efficient game mechanics.

Features:
- 6x7 board representation using NumPy
- Win detection (horizontal, vertical, diagonal)
- Action validation and piece dropping with gravity
- Terminal rendering with performance statistics
- CPU-only execution for consistency
"""

import numpy as np
import time
import sys
import os
from typing import List, Tuple, Optional, Dict, Any

# Add utils directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.render import render_connect4_game


class Connect4Game:
    """
    Pure Python Connect4 game implementation optimized for AI training.
    
    Board representation:
    - 0: Empty cell
    - 1: Player 1 (X)
    - -1: Player 2 (O)
    
    Coordinate system:
    - board[row][col] where row 0 is TOP, row 5 is BOTTOM
    - Pieces drop from top to bottom (gravity effect)
    """
    
    def __init__(self):
        """Initialize a new Connect4 game."""
        self.board_rows = 6
        self.board_cols = 7
        self.win_length = 4
        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8)
        self.current_player = 1  # Player 1 starts
        self.move_count = 0
        self.game_over = False
        self.winner = None
        
        # Performance tracking for AI training observation
        self.start_time = time.time()
        self.last_render_time = time.time()
        self.render_count = 0
        
    def reset(self) -> np.ndarray:
        """
        Reset the game to initial state.
        
        Returns:
            Initial board state as numpy array
        """
        self.board = np.zeros((self.board_rows, self.board_cols), dtype=np.int8)
        self.current_player = 1
        self.move_count = 0
        self.game_over = False
        self.winner = None
        self.start_time = time.time()
        return self.board.copy()
        
    def get_valid_moves(self) -> List[int]:
        """
        Get list of valid column indices where pieces can be dropped.
        
        Returns:
            List of valid column indices (0-6)
        """
        valid_moves = []
        for col in range(self.board_cols):
            if self.board[0][col] == 0:  # Top row is empty
                valid_moves.append(col)
        return valid_moves
        
    def is_valid_move(self, col: int) -> bool:
        """
        Check if a move in the given column is valid.
        
        Args:
            col: Column index (0-6)
            
        Returns:
            True if move is valid, False otherwise
        """
        if col < 0 or col >= self.board_cols:
            return False
        return self.board[0][col] == 0
        
    def drop_piece(self, col: int) -> bool:
        """
        Drop a piece in the specified column.
        
        Args:
            col: Column index (0-6)
            
        Returns:
            True if piece was successfully dropped, False if invalid move
        """
        if not self.is_valid_move(col) or self.game_over:
            return False
            
        # Find the lowest empty row in the column (gravity effect)
        for row in range(self.board_rows - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.move_count += 1
                break
                
        # Check for win or draw
        if self.check_win(col):
            self.game_over = True
            self.winner = self.current_player
        elif self.is_draw():
            self.game_over = True
            self.winner = 0  # Draw
        else:
            # Switch players
            self.current_player = -self.current_player
            
        return True
        
    def check_win(self, last_col: int) -> bool:
        """
        Check if the current player has won the game.
        Optimized to only check around the last dropped piece.
        
        Args:
            last_col: Column where the last piece was dropped
            
        Returns:
            True if current player won, False otherwise
        """
        # Find the row where the last piece was dropped
        last_row = -1
        for row in range(self.board_rows):
            if self.board[row][last_col] == self.current_player:
                last_row = row
                break
                
        if last_row == -1:
            return False
            
        # Check all four directions: horizontal, vertical, diagonal
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical  
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the piece we just dropped
            
            # Check positive direction
            r, c = last_row + dr, last_col + dc
            while (0 <= r < self.board_rows and 0 <= c < self.board_cols and 
                   self.board[r][c] == self.current_player):
                count += 1
                r, c = r + dr, c + dc
                
            # Check negative direction
            r, c = last_row - dr, last_col - dc
            while (0 <= r < self.board_rows and 0 <= c < self.board_cols and 
                   self.board[r][c] == self.current_player):
                count += 1
                r, c = r - dr, c - dc
                
            if count >= self.win_length:
                return True
                
        return False
        
    def is_draw(self) -> bool:
        """
        Check if the game is a draw (board is full with no winner).
        
        Returns:
            True if game is a draw, False otherwise
        """
        return len(self.get_valid_moves()) == 0
        
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get current game state information.
        
        Returns:
            Dictionary containing game state information
        """
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'valid_moves': self.get_valid_moves(),
            'game_over': self.game_over,
            'winner': self.winner,
            'move_count': self.move_count,
            'game_time': time.time() - self.start_time
        }
        
    def render(self, mode: str = 'human', show_stats: bool = True) -> Optional[str]:
        """
        Render the game board and statistics using centralized rendering.
        
        Args:
            mode: Rendering mode ('human' for terminal output)
            show_stats: Whether to show performance statistics
            
        Returns:
            String representation if mode != 'human', None otherwise
        """
        return render_connect4_game(self, mode, show_stats)
            
    def __str__(self) -> str:
        """String representation of the game board."""
        return self.render(mode='return', show_stats=False)
        
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Connect4Game(current_player={self.current_player}, moves={self.move_count}, game_over={self.game_over})"