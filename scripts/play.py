#!/usr/bin/env python3
"""
Connect4 Interactive Game Interface

A terminal-based game interface for Connect4 with multiple game modes:
1. Random vs Random (for testing)
2. Human vs Human (interactive play)  
3. Human vs AI (main feature - future implementation)

This script provides a user-friendly interface for playing Connect4 games
and testing the game logic before AI training.
"""

import sys
import os
import time
from typing import Optional


# ANSI Color Codes for terminal styling
class Colors:
    """ANSI color codes for terminal styling."""
    
    # Player colors
    PLAYER1 = '\033[91m'  # Red for Player 1 (X)
    PLAYER2 = '\033[94m'  # Blue for Player 2 (O)
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

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from environments.connect4_game import Connect4Game
except ImportError as e:
    print(f"Error importing Connect4Game: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class GameInterface:
    """
    Interactive game interface for Connect4.
    Handles user input, game flow, and display.
    """
    
    def __init__(self):
        """Initialize the game interface."""
        self.game = Connect4Game()
        self.stats = {
            'games_played': 0,
            'player1_wins': 0,
            'player2_wins': 0,
            'draws': 0,
            'total_moves': 0
        }
        
    def display_main_menu(self) -> None:
        """Display the main menu with game mode options."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}*** CONNECT4 RL TRAINING SYSTEM ***{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"\n{Colors.INFO}Select Game Mode:{Colors.RESET}")
        print(f"{Colors.SUCCESS}1. Human vs Human{Colors.RESET} (Interactive - Player 1 goes first)")
        print(f"{Colors.WARNING}2. Random vs Random{Colors.RESET} (Testing)")
        print(f"{Colors.INFO}3. Human vs AI{Colors.RESET} (Coming Soon - Human is Player 1)")
        print(f"{Colors.HEADER}4. View Statistics{Colors.RESET}")
        print(f"{Colors.ERROR}5. Exit{Colors.RESET}")
        print(f"\n{Colors.HEADER}{'-' * 50}{Colors.RESET}")
        
    def get_user_choice(self, prompt: str, valid_choices: list) -> str:
        """
        Get validated user input.
        
        Args:
            prompt: Input prompt message
            valid_choices: List of valid input options
            
        Returns:
            Valid user choice
        """
        while True:
            try:
                choice = input(prompt).strip()
                if choice in valid_choices:
                    return choice
                else:
                    print(f"Invalid choice. Please enter one of: {valid_choices}")
            except KeyboardInterrupt:
                print(f"\n\nGame interrupted by user. Goodbye!")
                sys.exit(0)
            except EOFError:
                print(f"\n\nInput ended. Goodbye!")
                sys.exit(0)
                
    def get_column_input(self, player_symbol: str) -> int:
        """
        Get column input from human player.
        
        Args:
            player_symbol: Current player symbol (X or O)
            
        Returns:
            Valid column number (0-6)
        """
        valid_moves = self.game.get_valid_moves()
        valid_moves_str = [str(col) for col in valid_moves]
        
        prompt = f"Player {player_symbol}, enter column (1-7): "
        
        while True:
            choice = input(prompt).strip()
            
            if choice == 'q' or choice == 'quit':
                print("Game ended by player.")
                return -1
                
            try:
                col_input = int(choice)
                if col_input < 1 or col_input > 7:
                    print("Please enter a number between 1-7, or 'q' to quit.")
                    continue
                    
                col = col_input - 1  # Convert from 1-7 to 0-6
                if col in valid_moves:
                    return col
                else:
                    # Display column numbers in 1-7 format for user
                    valid_display = [str(c + 1) for c in valid_moves]
                    print(f"Column {col_input} is full or invalid. Valid columns: {valid_display}")
            except ValueError:
                print("Please enter a number between 1-7, or 'q' to quit.")
                
    def play_human_vs_human(self) -> None:
        """Play Human vs Human mode."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}>>> HUMAN vs HUMAN MODE <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.INFO}Instructions:{Colors.RESET}")
        print(f"- {Colors.INFO}Players take turns dropping pieces{Colors.RESET}")
        print(f"- {Colors.PLAYER1}Player 1 (RED X){Colors.RESET}: Goes first - Human player")  
        print(f"- {Colors.PLAYER2}Player 2 (BLUE O){Colors.RESET}: Goes second - Second player")
        print(f"- {Colors.WARNING}Enter column number (1-7) to drop piece{Colors.RESET}")
        print(f"- {Colors.ERROR}Type 'q' to quit game{Colors.RESET}")
        print(f"- {Colors.SUCCESS}Connect 4 pieces in a row to win!{Colors.RESET}")
        print(f"- {Colors.INFO}Board is colorized: {Colors.PLAYER1}RED{Colors.RESET} for Player 1, {Colors.PLAYER2}BLUE{Colors.RESET} for Player 2")
        print(f"\n{Colors.WARNING}Press Enter to start...{Colors.RESET}")
        input()
        
        # Initialize new game
        self.game.reset()
        game_start_time = time.time()
        
        # Game loop
        while not self.game.game_over:
            # Render current state
            self.game.render(show_stats=True)
            
            # Get current player info
            player_symbol = "X" if self.game.current_player == 1 else "O"
            
            # Get player input
            col = self.get_column_input(player_symbol)
            
            if col == -1:  # Player quit
                print(f"{Colors.WARNING}Game abandoned.{Colors.RESET}")
                input(f"{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")
                return
                
            # Make move
            if not self.game.drop_piece(col):
                print(f"{Colors.ERROR}Invalid move! Please try again.{Colors.RESET}")
                input(f"{Colors.INFO}Press Enter to continue...{Colors.RESET}")
                continue
                
        # Game over - show final state
        self.game.render(show_stats=True)
        
        # Update statistics
        game_time = time.time() - game_start_time
        self.stats['games_played'] += 1
        self.stats['total_moves'] += self.game.move_count
        
        if self.game.winner == 1:
            self.stats['player1_wins'] += 1
        elif self.game.winner == -1:
            self.stats['player2_wins'] += 1
        else:
            self.stats['draws'] += 1
            
        # Show game summary
        print(f"\n{Colors.INFO}{Colors.BOLD}Game Summary:{Colors.RESET}")
        print(f"- {Colors.INFO}Duration: {game_time:.1f} seconds{Colors.RESET}")
        print(f"- {Colors.INFO}Total moves: {self.game.move_count}{Colors.RESET}")
        print(f"- {Colors.INFO}Moves per second: {self.game.move_count / game_time:.1f}{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
        
    def play_random_vs_random(self) -> None:
        """Play Random vs Random mode (for testing)."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.WARNING}{Colors.BOLD}>>> RANDOM vs RANDOM MODE <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.INFO}This mode will be implemented when random agents are available.{Colors.RESET}")
        print(f"{Colors.WARNING}Currently in development...{Colors.RESET}")
        input(f"{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")
        
    def play_human_vs_ai(self) -> None:
        """Play Human vs AI mode (future implementation)."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.INFO}{Colors.BOLD}>>> HUMAN vs AI MODE <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.INFO}This mode will be available after PPO agent implementation.{Colors.RESET}")  
        print(f"{Colors.WARNING}Coming in Phase 4 of the project...{Colors.RESET}")
        input(f"{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")
        
    def view_statistics(self) -> None:
        """Display game statistics."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.HEADER}{Colors.BOLD}>>> GAME STATISTICS <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        
        if self.stats['games_played'] == 0:
            print(f"{Colors.WARNING}No games played yet.{Colors.RESET}")
        else:
            total_games = self.stats['games_played']
            avg_moves = self.stats['total_moves'] / total_games
            
            print(f"{Colors.INFO}Total Games Played: {Colors.SUCCESS}{total_games}{Colors.RESET}")
            print(f"{Colors.PLAYER1}Player 1 (X) Wins:{Colors.RESET}  {self.stats['player1_wins']} ({self.stats['player1_wins']/total_games*100:.1f}%)")
            print(f"{Colors.PLAYER2}Player 2 (O) Wins:{Colors.RESET}  {self.stats['player2_wins']} ({self.stats['player2_wins']/total_games*100:.1f}%)")
            print(f"{Colors.WARNING}Draws:{Colors.RESET}              {self.stats['draws']} ({self.stats['draws']/total_games*100:.1f}%)")
            print(f"{Colors.INFO}Average Game Length: {avg_moves:.1f} moves{Colors.RESET}")
            
        input(f"\n{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")
        
    def run(self) -> None:
        """Main game loop."""
        print("Starting Connect4 Interactive Game Interface...")
        
        while True:
            self.display_main_menu()
            
            choice = self.get_user_choice(
                f"{Colors.INFO}Enter your choice (1-5): {Colors.RESET}",
                ['1', '2', '3', '4', '5']
            )
            
            if choice == '1':
                self.play_human_vs_human()
            elif choice == '2':
                self.play_random_vs_random()
            elif choice == '3':
                self.play_human_vs_ai()
            elif choice == '4':
                self.view_statistics()
            elif choice == '5':
                print(f"{Colors.SUCCESS}Thanks for playing Connect4! Goodbye!{Colors.RESET}")
                break


def main():
    """Main entry point."""
    try:
        interface = GameInterface()
        interface.run()
    except KeyboardInterrupt:
        print(f"\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())