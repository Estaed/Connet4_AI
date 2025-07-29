"""
Base Agent Interface for Connect4 RL System

This module provides the abstract base class for all Connect4 agents,
defining the standard interface and common functionality.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import pickle
import logging

from ..core.config import get_config


class BaseAgent(ABC):
    """
    Abstract base class for all Connect4 agents.
    
    This class defines the standard interface that all agents must implement,
    providing common functionality for device management, serialization,
    and Connect4-specific utilities.
    
    Supports various agent types:
    - Learning agents (PPO, A2C, etc.)
    - Non-learning agents (Random, Minimax, Monte Carlo Tree Search)
    - Human agents (for interactive play)
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 name: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize base agent.
        
        Args:
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            name: Human-readable name for the agent
            config_path: Path to configuration file (uses default if None)
        """
        self.config = get_config()
        
        # Device setup
        if device is None:
            device = self.config.get('device.training_device', 'cpu')
        self.device = self._validate_device(device)
        
        # Agent metadata
        self.name = name or self.__class__.__name__
        self.total_games = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        # Game state tracking
        self.last_observation = None
        self.last_action = None
        
        # Set up logging
        self.logger = logging.getLogger(f"Agent.{self.name}")
        
    def _validate_device(self, device: str) -> str:
        """
        Validate and return appropriate device.
        
        Args:
            device: Requested device
            
        Returns:
            Validated device string
        """
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, using CPU")
            return 'cpu'
        return device
    
    @abstractmethod
    def get_action(self, 
                   observation: np.ndarray, 
                   valid_actions: Optional[List[int]] = None,
                   **kwargs) -> int:
        """
        Get action for given observation.
        
        Args:
            observation: Board state as 6x7 numpy array
            valid_actions: List of valid column indices (0-6)
            **kwargs: Additional agent-specific parameters
            
        Returns:
            Column index (0-6) for piece placement
            
        Raises:
            ValueError: If action is invalid or agent cannot decide
        """
        pass
    
    @abstractmethod 
    def update(self, 
               experiences: Optional[Dict[str, Any]] = None,
               **kwargs) -> Dict[str, float]:
        """
        Update agent based on experiences.
        
        For learning agents, this performs training updates.
        For non-learning agents, this may do nothing or update statistics.
        
        Args:
            experiences: Training experiences (trajectories, rewards, etc.)
            **kwargs: Additional update parameters
            
        Returns:
            Dictionary of training metrics (loss, accuracy, etc.)
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save agent state to file.
        
        Args:
            path: File path to save agent state
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Basic agent state
        agent_state = {
            'name': self.name,
            'device': self.device,
            'total_games': self.total_games,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'class_name': self.__class__.__name__,
        }
        
        # Let subclasses add their specific state
        agent_state.update(self._get_save_state())
        
        with open(path, 'wb') as f:
            pickle.dump(agent_state, f)
            
        self.logger.info(f"Agent saved to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load agent state from file.
        
        Args:
            path: File path to load agent state from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Agent state file not found: {path}")
            
        with open(path, 'rb') as f:
            agent_state = pickle.load(f)
            
        # Restore basic state
        self.name = agent_state.get('name', self.name)
        self.total_games = agent_state.get('total_games', 0)
        self.wins = agent_state.get('wins', 0)
        self.losses = agent_state.get('losses', 0)
        self.draws = agent_state.get('draws', 0)
        
        # Let subclasses restore their specific state
        self._set_load_state(agent_state)
        
        self.logger.info(f"Agent loaded from {path}")
    
    def _get_save_state(self) -> Dict[str, Any]:
        """
        Get agent-specific state for saving.
        Override in subclasses to add model weights, etc.
        
        Returns:
            Dictionary of agent-specific state
        """
        return {}
    
    def _set_load_state(self, state: Dict[str, Any]) -> None:
        """
        Set agent-specific state from loaded data.
        Override in subclasses to restore model weights, etc.
        
        Args:
            state: Loaded state dictionary
        """
        pass
    
    def reset_episode(self) -> None:
        """
        Reset agent state for new episode.
        Called at the start of each game.
        """
        self.last_observation = None
        self.last_action = None
        
    def end_episode(self, result: str) -> None:
        """
        Handle end of episode.
        
        Args:
            result: Game result ('win', 'loss', 'draw')
        """
        self.total_games += 1
        if result == 'win':
            self.wins += 1
        elif result == 'loss':
            self.losses += 1
        elif result == 'draw':
            self.draws += 1
            
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Get agent performance statistics.
        
        Returns:
            Dictionary of performance metrics
        """
        if self.total_games == 0:
            return {
                'total_games': 0,
                'wins': 0,
                'losses': 0,
                'draws': 0,
                'win_rate': 0.0,
                'loss_rate': 0.0,
                'draw_rate': 0.0
            }
            
        return {
            'total_games': self.total_games,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / self.total_games,
            'loss_rate': self.losses / self.total_games,
            'draw_rate': self.draws / self.total_games
        }
    
    def is_learning_agent(self) -> bool:
        """
        Check if this is a learning agent.
        
        Returns:
            True if agent can learn from experiences
        """
        # Override in subclasses - default assumes non-learning
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get agent information and metadata.
        
        Returns:
            Dictionary with agent information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'device': self.device,
            'is_learning': self.is_learning_agent(),
            'statistics': self.get_statistics()
        }
    
    def __str__(self) -> str:
        """String representation of agent."""
        stats = self.get_statistics()
        return f"{self.name} (Games: {stats['total_games']}, Win Rate: {stats['win_rate']:.2%})"
    
    def __repr__(self) -> str:
        """Developer representation of agent."""
        return f"{self.__class__.__name__}(name='{self.name}', device='{self.device}')"
    
    # Strategic Analysis Methods for Connect4
    def analyze_board_threats(self, observation: np.ndarray, player: int) -> Dict[str, List[int]]:
        """
        Analyze board for immediate threats and opportunities.
        
        Args:
            observation: Current board state (6x7 numpy array)
            player: Player perspective (1 or -1)
            
        Returns:
            Dictionary containing:
            - 'winning_moves': Columns that result in immediate win
            - 'blocking_moves': Columns that block opponent's win
            - 'threat_moves': Columns that create 3-in-a-row threats
            - 'building_moves': Columns that create 2-in-a-row setups
        """
        analysis = {
            'winning_moves': [],
            'blocking_moves': [],
            'threat_moves': [],
            'building_moves': []
        }
        
        opponent = -player
        
        for col in range(7):
            if not self._is_valid_move(observation, col):
                continue
                
            # Find where piece would land
            row = self._get_drop_row(observation, col)
            if row is None:
                continue
            
            # Test the move
            test_board = observation.copy()
            test_board[row, col] = player
            
            # Check for winning move
            if self._check_win_at_position(test_board, row, col, player):
                analysis['winning_moves'].append(col)
            
            # Check for blocking opponent's win
            elif self._would_opponent_win_if_not_blocked(observation, col, opponent):
                analysis['blocking_moves'].append(col)
            
            # Check for creating three-in-a-row threat
            elif self._creates_three_in_row(test_board, row, col, player):
                analysis['threat_moves'].append(col)
            
            # Check for creating two-in-a-row building block
            elif self._creates_two_in_row(test_board, row, col, player):
                analysis['building_moves'].append(col)
        
        return analysis
    
    def evaluate_position_strength(self, observation: np.ndarray, player: int) -> float:
        """
        Evaluate the overall strength of a position for a player.
        
        Args:
            observation: Current board state
            player: Player to evaluate for
            
        Returns:
            Position strength score (higher = better for player)
        """
        score = 0.0
        
        # Count potential winning lines
        for row in range(6):
            for col in range(7):
                if observation[row, col] == 0:  # Empty cell
                    # Check how many winning lines pass through this cell
                    lines_through_cell = self._count_potential_lines(observation, row, col, player)
                    score += lines_through_cell * 0.1
        
        # Evaluate center control (center columns more valuable)
        center_weights = [1, 2, 3, 4, 3, 2, 1]
        for col in range(7):
            for row in range(6):
                if observation[row, col] == player:
                    score += center_weights[col] * 0.05
                elif observation[row, col] == -player:
                    score -= center_weights[col] * 0.05
        
        return score
    
    def find_best_strategic_move(self, observation: np.ndarray, valid_actions: List[int], player: int) -> int:
        """
        Find the best strategic move based on tactical analysis.
        
        Args:
            observation: Current board state
            valid_actions: List of valid column moves
            player: Player making the move
            
        Returns:
            Best column to play
        """
        analysis = self.analyze_board_threats(observation, player)
        
        # Priority 1: Take winning move
        if analysis['winning_moves']:
            return analysis['winning_moves'][0]
        
        # Priority 2: Block opponent's winning move
        if analysis['blocking_moves']:
            return analysis['blocking_moves'][0]
        
        # Priority 3: Create three-in-a-row threat
        if analysis['threat_moves']:
            # Choose the threat move that gives best position
            best_col = analysis['threat_moves'][0]
            best_score = -float('inf')
            
            for col in analysis['threat_moves']:
                test_board = observation.copy()
                row = self._get_drop_row(observation, col)
                test_board[row, col] = player
                score = self.evaluate_position_strength(test_board, player)
                
                if score > best_score:
                    best_score = score
                    best_col = col
            
            return best_col
        
        # Priority 4: Create two-in-a-row building block
        if analysis['building_moves']:
            return analysis['building_moves'][0]
        
        # Priority 5: Play center column if available
        if 3 in valid_actions:
            return 3
        
        # Priority 6: Evaluate all moves and pick best positional move
        best_col = valid_actions[0]
        best_score = -float('inf')
        
        for col in valid_actions:
            test_board = observation.copy()
            row = self._get_drop_row(observation, col)
            test_board[row, col] = player
            score = self.evaluate_position_strength(test_board, player)
            
            if score > best_score:
                best_score = score
                best_col = col
        
        return best_col
    
    # Helper methods for strategic analysis
    def _is_valid_move(self, board: np.ndarray, col: int) -> bool:
        """Check if a move is valid."""
        return 0 <= col < 7 and board[0, col] == 0
    
    def _get_drop_row(self, board: np.ndarray, col: int) -> Optional[int]:
        """Get the row where a piece would land in a column."""
        for row in range(6):  # Top to bottom
            if board[row, col] == 0:
                return row
        return None
    
    def _check_win_at_position(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if there's a win starting from a specific position."""
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
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 4:
                return True
        
        return False
    
    def _would_opponent_win_if_not_blocked(self, board: np.ndarray, col: int, opponent: int) -> bool:
        """Check if opponent would win by playing in this column."""
        row = self._get_drop_row(board, col)
        if row is None:
            return False
        
        test_board = board.copy()
        test_board[row, col] = opponent
        return self._check_win_at_position(test_board, row, col, opponent)
    
    def _creates_three_in_row(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if move creates a three-in-a-row threat."""
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the placed piece
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count == 3:
                # Check if there's room to extend to 4
                end1_valid = (0 <= row + 3*dr < 6 and 0 <= col + 3*dc < 7 and 
                             board[row + 3*dr, col + 3*dc] == 0)
                end2_valid = (0 <= row - 3*dr < 6 and 0 <= col - 3*dc < 7 and 
                             board[row - 3*dr, col - 3*dc] == 0)
                
                if end1_valid or end2_valid:
                    return True
        
        return False
    
    def _creates_two_in_row(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """Check if move creates a two-in-a-row building block."""
        directions = [
            (0, 1),   # Horizontal
            (1, 1),   # Diagonal /
            (1, -1),  # Diagonal \
        ]
        
        for dr, dc in directions:
            count = 1  # Count the placed piece
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r + dr, c + dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count == 2:
                # Check if there's room to extend in both directions
                space_positive = 0
                r, c = row + dr, col + dc
                while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == 0:
                    space_positive += 1
                    r, c = r + dr, c + dc
                
                space_negative = 0
                r, c = row - dr, col - dc
                while 0 <= r < 6 and 0 <= c < 7 and board[r, c] == 0:
                    space_negative += 1
                    r, c = r - dr, c - dc
                
                # Need at least 2 more spaces to potentially make 4-in-a-row
                if space_positive + space_negative >= 2:
                    return True
        
        return False
    
    def _count_potential_lines(self, board: np.ndarray, row: int, col: int, player: int) -> int:
        """Count how many potential 4-in-a-row lines pass through a position."""
        count = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # Check if a 4-length line through this position could be formed
            line_positions = []
            for i in range(-3, 4):  # -3, -2, -1, 0, 1, 2, 3
                r, c = row + i*dr, col + i*dc
                if 0 <= r < 6 and 0 <= c < 7:
                    line_positions.append((r, c))
            
            # Check all possible 4-length segments in this line
            for start_idx in range(len(line_positions) - 3):
                segment = line_positions[start_idx:start_idx + 4]
                
                # Check if this segment could potentially be won by player
                can_win = True
                for r, c in segment:
                    if board[r, c] == -player:  # Opponent piece blocks this line
                        can_win = False
                        break
                
                if can_win:
                    count += 1
        
        return count


# Utility functions for agent management
def create_agent(agent_type: str, **kwargs) -> BaseAgent:
    """
    Factory function to create agents by type name.
    
    Args:
        agent_type: Type of agent to create ('random', 'ppo', etc.)
        **kwargs: Arguments passed to agent constructor
        
    Returns:
        Instantiated agent
        
    Raises:
        ValueError: If agent type is not recognized
    """
    # Import here to avoid circular imports
    from .random_agent import RandomAgent
    
    # This will be populated as more agents are implemented
    agent_registry = {
        'random': RandomAgent,  # Task 3.2 - COMPLETED
        # 'ppo': PPOAgent,       # Task 4.1
    }
    
    if agent_type.lower() not in agent_registry:
        available_types = list(agent_registry.keys())
        raise ValueError(f"Unknown agent type '{agent_type}'. Available: {available_types}")
    
    agent_class = agent_registry[agent_type.lower()]
    return agent_class(**kwargs)