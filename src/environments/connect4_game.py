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
from typing import List, Tuple, Optional, Dict, Any


# ANSI Color Codes for terminal rendering
class Colors:
    """ANSI color codes for terminal styling."""
    
    # Player colors
    PLAYER1 = '\033[91m'  # Red for Player 1 (X) - Human player
    PLAYER2 = '\033[94m'  # Blue for Player 2 (O) - AI/Second player
    EMPTY = '\033[90m'    # Gray for empty spaces
    
    # UI colors
    HEADER = '\033[95m'   # Magenta for headers
    SUCCESS = '\033[92m'  # Green for success/good stats
    WARNING = '\033[93m'  # Yellow for warnings
    ERROR = '\033[91m'    # Red for errors
    INFO = '\033[96m'     # Cyan for info
    
    # Special
    BOLD = '\033[1m'      # Bold text
    UNDERLINE = '\033[4m' # Underlined text
    RESET = '\033[0m'     # Reset to default


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
        Render the game board and statistics.
        
        Args:
            mode: Rendering mode ('human' for terminal output)
            show_stats: Whether to show performance statistics
            
        Returns:
            String representation if mode != 'human', None otherwise
        """
        self.render_count += 1
        current_time = time.time()
        
        # Calculate FPS
        time_since_last = current_time - self.last_render_time
        fps = 1.0 / time_since_last if time_since_last > 0 else 0.0
        self.last_render_time = current_time
        
        # Build output string
        output_lines = []
        
        # Column numbers with color (1-7 instead of 0-6)
        header = " " + "".join(f" {i+1}  " for i in range(self.board_cols))
        output_lines.append(f"{Colors.HEADER}{header}{Colors.RESET}")
        output_lines.append(f"{Colors.HEADER}{'='*30}{Colors.RESET}")
        
        # Board display with colors
        for row in range(self.board_rows):
            row_str = f"{Colors.HEADER}|{Colors.RESET}"
            for col in range(self.board_cols):
                cell = self.board[row][col]
                
                if cell == 1:  # Player 1 (Human - Red X)
                    symbol = f"{Colors.PLAYER1}X{Colors.RESET}"
                elif cell == -1:  # Player 2 (AI - Blue O)
                    symbol = f"{Colors.PLAYER2}O{Colors.RESET}"
                else:  # Empty cell
                    symbol = f"{Colors.EMPTY}.{Colors.RESET}"
                
                row_str += f" {symbol} {Colors.HEADER}|{Colors.RESET}"
            output_lines.append(row_str)
            
        output_lines.append(f"{Colors.HEADER}{'='*23}{Colors.RESET}")
        
        # Current player info with colors
        if self.current_player == 1:
            player_symbol = f"{Colors.PLAYER1}X{Colors.RESET}"
            player_display = f"{Colors.PLAYER1}Player 1{Colors.RESET}"
        else:
            player_symbol = f"{Colors.PLAYER2}O{Colors.RESET}"
            player_display = f"{Colors.PLAYER2}Player 2{Colors.RESET}"
            
        output_lines.append(f"Current: {player_symbol} {player_display}")
        
        if show_stats:
            # Performance metrics section with colors
            output_lines.append("")
            output_lines.append(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
            output_lines.append(f"{Colors.HEADER}{Colors.BOLD}CONNECT4 RL TRAINING - REAL-TIME STATS{Colors.RESET}")
            output_lines.append(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
            output_lines.append("")
            
            # Performance metrics with colors
            output_lines.append(f"{Colors.INFO}{Colors.BOLD}[PERFORMANCE METRICS]{Colors.RESET}")
            game_time = current_time - self.start_time
            output_lines.append(f"Games/sec:        {Colors.WARNING}0{Colors.RESET}")
            output_lines.append(f"Total Games:      {Colors.INFO}1{Colors.RESET}")
            output_lines.append(f"Episode:          {Colors.INFO}0{Colors.RESET}")
            output_lines.append(f"Training Time:    {Colors.INFO}{game_time:08.2f}{Colors.RESET}")
            output_lines.append("")
            
            # Win statistics with player colors
            output_lines.append(f"{Colors.INFO}{Colors.BOLD}[WIN STATISTICS]{Colors.RESET}")
            output_lines.append(f"Player 1 (X):     {Colors.PLAYER1}0.0%{Colors.RESET} (0 wins)")
            output_lines.append(f"Player 2 (O):     {Colors.PLAYER2}0.0%{Colors.RESET} (0 wins)")
            output_lines.append(f"Draws:            {Colors.WARNING}0.0%{Colors.RESET} (0 draws)")
            output_lines.append(f"Avg Game Len:     {Colors.INFO}{self.move_count:.1f}{Colors.RESET} moves")
            output_lines.append("")
            
            # GPU statistics (placeholder for future AI training)
            output_lines.append(f"{Colors.INFO}{Colors.BOLD}[GPU STATISTICS]{Colors.RESET}")
            output_lines.append(f"GPU Usage:        {Colors.WARNING}Not added yet{Colors.RESET}")
            output_lines.append(f"GPU Memory:       {Colors.WARNING}Not added yet{Colors.RESET}")
            output_lines.append(f"GPU Device:       {Colors.WARNING}Not added yet{Colors.RESET}")
            output_lines.append("")
            
            output_lines.append(f"{Colors.HEADER}{'='*60}{Colors.RESET}")
            output_lines.append(f"{Colors.INFO}Updates: 60 FPS | Renderer FPS: {fps:.1f}{Colors.RESET}")
            output_lines.append(f"{Colors.HEADER}{'='*60}{Colors.RESET}")
            
        # Game over message with colors
        if self.game_over:
            output_lines.append("")
            if self.winner == 0:
                output_lines.append(f"{Colors.WARNING}{Colors.BOLD}GAME OVER - DRAW!{Colors.RESET}")
            else:
                if self.winner == 1:
                    winner_symbol = f"{Colors.PLAYER1}X{Colors.RESET}"
                    winner_display = f"{Colors.PLAYER1}{Colors.BOLD}Player 1{Colors.RESET}"
                else:
                    winner_symbol = f"{Colors.PLAYER2}O{Colors.RESET}" 
                    winner_display = f"{Colors.PLAYER2}{Colors.BOLD}Player 2{Colors.RESET}"
                output_lines.append(f"{Colors.SUCCESS}{Colors.BOLD}GAME OVER - {winner_display} ({winner_symbol}) WINS!{Colors.RESET}")
        else:
            output_lines.append("")
            if self.current_player == 1:
                prompt_player = f"{Colors.PLAYER1}Player X{Colors.RESET}"
            else:
                prompt_player = f"{Colors.PLAYER2}Player O{Colors.RESET}"
            output_lines.append(f"{prompt_player}, enter column (1-{self.board_cols}):")
            
        output_str = "\n".join(output_lines)
        
        if mode == 'human':
            # Clear screen and print (for terminal play)
            print("\n" * 50)  # Simple screen clear
            print(output_str)
            return None
        else:
            return output_str
            
    def __str__(self) -> str:
        """String representation of the game board."""
        return self.render(mode='return', show_stats=False)
        
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Connect4Game(current_player={self.current_player}, moves={self.move_count}, game_over={self.game_over})"