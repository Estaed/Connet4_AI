"""
Comprehensive tests for Connect4Game core game logic.

Tests all game mechanics, win detection, edge cases, and board management
functionality of the pure Python Connect4 game implementation.
"""

import pytest
import numpy as np
from src.environments.connect4_game import Connect4Game


class TestConnect4GameInitialization:
    """Test game initialization and basic setup."""
    
    def test_initial_state(self, empty_game):
        """Test that game initializes with correct initial state."""
        assert empty_game.board.shape == (6, 7)
        assert np.all(empty_game.board == 0)
        assert empty_game.current_player == 1
        assert not empty_game.game_over
        assert empty_game.winner is None
        
    def test_board_dtype(self, empty_game):
        """Test that board uses correct data type."""
        assert empty_game.board.dtype == np.int8
        
    def test_reset_functionality(self, game_with_moves):
        """Test that reset returns game to initial state."""
        # Verify game has moves
        assert np.any(game_with_moves.board != 0)
        
        # Reset and verify
        game_with_moves.reset()
        assert np.all(game_with_moves.board == 0)
        assert game_with_moves.current_player == 1
        assert not game_with_moves.game_over
        assert game_with_moves.winner is None


class TestConnect4GameMoveMechanics:
    """Test piece dropping and move mechanics."""
    
    def test_valid_move_detection(self, empty_game):
        """Test valid move detection in various states."""
        # All columns valid on empty board
        assert empty_game.get_valid_moves() == [0, 1, 2, 3, 4, 5, 6]
        
        # Test individual column validity
        for col in range(7):
            assert empty_game.is_valid_move(col)
    
    def test_invalid_move_detection(self, empty_game):
        """Test invalid move detection."""
        # Fill column 0 completely
        for _ in range(6):
            empty_game.drop_piece(0)
        
        # Column 0 should now be invalid
        assert not empty_game.is_valid_move(0)
        assert 0 not in empty_game.get_valid_moves()
        
        # Other columns should still be valid
        for col in range(1, 7):
            assert empty_game.is_valid_move(col)
    
    def test_drop_piece_mechanics(self, empty_game):
        """Test piece dropping mechanics and gravity."""
        # Drop piece in column 3
        result = empty_game.drop_piece(3)
        assert result == True
        assert empty_game.board[5, 3] == 1  # Bottom row
        assert empty_game.current_player == -1  # Player switched
        
        # Drop another piece in same column
        result = empty_game.drop_piece(3)
        assert result == True
        assert empty_game.board[4, 3] == -1  # One row up
        assert empty_game.current_player == 1  # Player switched back
    
    def test_drop_piece_full_column(self, empty_game):
        """Test dropping piece in full column fails."""
        # Fill column 0
        for _ in range(6):
            empty_game.drop_piece(0)
        
        # Attempting to drop in full column should fail
        result = empty_game.drop_piece(0)
        assert result == False
        # Player should not switch on failed move
        assert empty_game.current_player == 1
    
    def test_drop_piece_invalid_column(self, empty_game):
        """Test dropping piece in invalid column indices."""
        # Test negative column
        result = empty_game.drop_piece(-1)
        assert result == False
        
        # Test column too high
        result = empty_game.drop_piece(7)
        assert result == False
        
        # Player should not switch on invalid moves
        assert empty_game.current_player == 1
    
    def test_alternating_players(self, empty_game):
        """Test that players alternate correctly."""
        players = []
        for col in range(5):
            players.append(empty_game.current_player)
            empty_game.drop_piece(col)
        
        # Should alternate between 1 and -1
        assert players == [1, -1, 1, -1, 1]


class TestConnect4GameWinDetection:
    """Test win detection for all winning patterns."""
    
    def test_horizontal_win_detection(self, empty_game):
        """Test horizontal win detection."""
        # Create horizontal win for player 1 in bottom row
        for col in range(4):
            empty_game.board[5, col] = 1
            empty_game.current_player = 1
        
        # Test win detection
        assert empty_game.check_win(3) == True  # Last piece at column 3
        
        # Test in middle of board
        empty_game.reset()
        for col in range(2, 6):
            empty_game.board[3, col] = -1
            empty_game.current_player = -1
        
        assert empty_game.check_win(5) == True
    
    def test_vertical_win_detection(self, empty_game):
        """Test vertical win detection."""
        # Create vertical win for player 1 in column 0
        for row in range(4, 6):  # Bottom 4 rows of column 0
            empty_game.board[row, 0] = 1
        for row in range(2, 4):  # Next 2 rows
            empty_game.board[row, 0] = 1
            
        empty_game.current_player = 1
        assert empty_game.check_win(0) == True
    
    def test_diagonal_win_detection_positive_slope(self, empty_game):
        """Test positive slope diagonal win detection."""
        # Create positive slope diagonal win
        positions = [(5, 0), (4, 1), (3, 2), (2, 3)]
        for row, col in positions:
            empty_game.board[row, col] = 1
        
        empty_game.current_player = 1
        assert empty_game.check_win(3) == True  # Last piece at (2,3)
    
    def test_diagonal_win_detection_negative_slope(self, empty_game):
        """Test negative slope diagonal win detection."""
        # Create negative slope diagonal win
        positions = [(2, 0), (3, 1), (4, 2), (5, 3)]
        for row, col in positions:
            empty_game.board[row, col] = -1
        
        empty_game.current_player = -1
        assert empty_game.check_win(3) == True  # Last piece at (5,3)
    
    def test_no_win_detection(self, empty_game):
        """Test that no win is detected when there isn't one."""
        # Place some pieces that don't form a win
        positions = [(5, 0), (5, 2), (4, 1), (3, 3)]
        for row, col in positions:
            empty_game.board[row, col] = 1
        
        empty_game.current_player = 1
        for col in [0, 2, 1, 3]:
            assert empty_game.check_win(col) == False
    
    def test_win_detection_edge_cases(self, empty_game):
        """Test win detection at board edges."""
        # Horizontal win at right edge
        for col in range(3, 7):
            empty_game.board[5, col] = 1
        empty_game.current_player = 1
        assert empty_game.check_win(6) == True
        
        # Vertical win at top
        empty_game.reset()
        for row in range(0, 4):
            empty_game.board[row, 6] = -1
        empty_game.current_player = -1
        assert empty_game.check_win(6) == True
    
    def test_mixed_players_no_win(self, empty_game):
        """Test that mixed player pieces don't create wins."""
        # Horizontal with mixed players
        empty_game.board[5, 0] = 1
        empty_game.board[5, 1] = -1
        empty_game.board[5, 2] = 1
        empty_game.board[5, 3] = 1
        
        empty_game.current_player = 1
        assert empty_game.check_win(3) == False


class TestConnect4GameDrawDetection:
    """Test draw detection functionality."""
    
    def test_draw_detection_full_board(self, empty_game):
        """Test draw detection when board is completely full."""
        # Fill entire board with alternating pattern (no wins)
        for row in range(6):
            for col in range(7):
                empty_game.board[row, col] = 1 if (row + col) % 2 == 0 else -1
        
        assert empty_game.is_draw() == True
    
    def test_no_draw_detection_partial_board(self, game_with_moves):
        """Test that draw is not detected on partial board."""
        assert game_with_moves.is_draw() == False
    
    def test_no_draw_detection_empty_board(self, empty_game):
        """Test that draw is not detected on empty board."""
        assert empty_game.is_draw() == False


class TestConnect4GameStateManagement:
    """Test game state management and information retrieval."""
    
    def test_get_game_state(self, game_with_moves):
        """Test game state information retrieval."""
        state = game_with_moves.get_game_state()
        
        assert 'board' in state
        assert 'current_player' in state
        assert 'game_over' in state
        assert 'winner' in state
        assert 'valid_moves' in state
        
        # Verify types
        assert isinstance(state['board'], np.ndarray)
        assert isinstance(state['current_player'], int)
        assert isinstance(state['game_over'], bool)
        assert isinstance(state['valid_moves'], list)
    
    def test_game_state_consistency(self, empty_game):
        """Test that game state remains consistent across operations."""
        initial_state = empty_game.get_game_state()
        
        # Make some moves
        empty_game.drop_piece(3)
        empty_game.drop_piece(2)
        
        new_state = empty_game.get_game_state()
        
        # Board should have changed
        assert not np.array_equal(initial_state['board'], new_state['board'])
        # Current player should have changed
        assert initial_state['current_player'] != new_state['current_player']


class TestConnect4GameRenderAndDisplay:
    """Test game rendering and display functionality."""
    
    def test_render_empty_board(self, empty_game, capsys):
        """Test rendering of empty board."""
        empty_game.render()
        captured = capsys.readouterr()
        
        # Should contain board representation
        assert '|' in captured.out
        assert '0 1 2 3 4 5 6' in captured.out
    
    def test_render_with_pieces(self, game_with_moves, capsys):
        """Test rendering of board with pieces."""
        game_with_moves.render()
        captured = capsys.readouterr()
        
        # Should contain piece representations
        assert 'X' in captured.out or 'O' in captured.out


class TestConnect4GameEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_board_boundary_conditions(self, empty_game):
        """Test various boundary conditions."""
        # Test all corner positions
        corners = [(0, 0), (0, 6), (5, 0), (5, 6)]
        for row, col in corners:
            empty_game.board[row, col] = 1
            # Should not cause errors
            empty_game.check_win(col)
    
    def test_game_continuation_after_win(self, winning_game):
        """Test that game behaves correctly after a win."""
        # Game should detect win
        assert winning_game.check_win(3) == True
        
        # Further moves should be rejected or handled gracefully
        # (This depends on implementation - document expected behavior)
    
    def test_large_number_of_moves(self, empty_game):
        """Test game behavior with many moves."""
        move_count = 0
        
        # Fill board systematically
        for col in range(7):
            for _ in range(6):
                if empty_game.is_valid_move(col):
                    result = empty_game.drop_piece(col)
                    assert result == True
                    move_count += 1
        
        # Should have made 42 moves (full board)
        assert move_count == 42
        assert empty_game.is_draw() == True
    
    def test_player_values(self, empty_game):
        """Test that player values are consistently 1 and -1."""
        players_seen = set()
        
        # Make several moves and track player values
        for col in range(7):
            players_seen.add(empty_game.current_player)
            empty_game.drop_piece(col)
        
        # Should only see values 1 and -1
        assert players_seen == {1, -1}


class TestConnect4GamePerformance:
    """Test performance characteristics of game operations."""
    
    def test_win_detection_performance(self, empty_game):
        """Test that win detection is reasonably fast."""
        import time
        
        # Create a winning position
        for col in range(4):
            empty_game.board[5, col] = 1
        empty_game.current_player = 1
        
        # Time win detection
        start_time = time.time()
        for _ in range(1000):
            empty_game.check_win(3)
        end_time = time.time()
        
        # Should complete quickly (less than 0.1 seconds for 1000 checks)
        assert (end_time - start_time) < 0.1
    
    @pytest.mark.slow
    def test_many_game_simulations(self, empty_game):
        """Test playing many complete games for performance."""
        import time
        
        games_played = 0
        start_time = time.time()
        
        # Play random games for a short time
        while time.time() - start_time < 1.0:  # 1 second
            empty_game.reset()
            
            # Play a random game
            moves = 0
            while not empty_game.game_over and moves < 42:
                valid_moves = empty_game.get_valid_moves()
                if valid_moves:
                    import random
                    col = random.choice(valid_moves)
                    empty_game.drop_piece(col)
                    
                    # Check for game end
                    if empty_game.check_win(col) or empty_game.is_draw():
                        break
                    
                moves += 1
            
            games_played += 1
        
        # Should be able to play many games per second
        games_per_second = games_played / 1.0
        assert games_per_second > 10  # At least 10 games per second


# Parametrized tests for comprehensive coverage
class TestConnect4GameParametrized:
    """Parametrized tests for comprehensive coverage."""
    
    @pytest.mark.parametrize("col", range(7))
    def test_drop_piece_all_columns(self, empty_game, col):
        """Test dropping pieces in each column."""
        result = empty_game.drop_piece(col)
        assert result == True
        assert empty_game.board[5, col] != 0
    
    @pytest.mark.parametrize("player", [1, -1])
    def test_win_detection_both_players(self, empty_game, player):
        """Test win detection for both players."""
        # Create horizontal win
        for col in range(4):
            empty_game.board[5, col] = player
        
        empty_game.current_player = player
        assert empty_game.check_win(3) == True
    
    @pytest.mark.parametrize("row", range(6))
    @pytest.mark.parametrize("col", range(7))
    def test_valid_board_positions(self, empty_game, row, col):
        """Test that all board positions can hold pieces."""
        empty_game.board[row, col] = 1
        # Should not raise any exceptions
        assert empty_game.board[row, col] == 1