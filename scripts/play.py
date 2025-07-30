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
import random

# Add utils directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.render import (
    Colors,
    render_main_menu,
    render_game_mode_header,
    render_game_instructions,
    render_game_summary,
    render_statistics,
    render_development_message,
)

try:
    from src.environments.connect4_game import Connect4Game
    from src.agents.random_agent import RandomAgent
except ImportError as e:
    print(f"Error importing Connect4 components: {e}")
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
            "games_played": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "total_moves": 0,
        }

    def display_main_menu(self) -> None:
        """Display the main menu with game mode options."""
        render_main_menu()

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
                print("\n\nGame interrupted by user. Goodbye!")
                sys.exit(0)
            except EOFError:
                print("\n\nInput ended. Goodbye!")
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

        prompt = f"Player {player_symbol}, enter column (1-7): "

        while True:
            choice = input(prompt).strip()

            if choice == "q" or choice == "quit":
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
                    print(
                        f"Column {col_input} is full or invalid. "
                        f"Valid columns: {valid_display}"
                    )
            except ValueError:
                print("Please enter a number between 1-7, or 'q' to quit.")

    def play_human_vs_human(self) -> None:
        """Play Human vs Human mode."""
        render_game_mode_header("HUMAN vs HUMAN MODE")
        render_game_instructions()

        # Randomly decide who starts first
        starting_player = random.choice([1, -1])
        if starting_player == 1:
            print(
                f"{Colors.HEADER}🎲 Random selection: Player X (Human) starts first!{Colors.RESET}"
            )
        else:
            print(
                f"{Colors.HEADER}🎲 Random selection: Player O (Human) starts first!{Colors.RESET}"
            )

        # Brief pause to show the selection
        time.sleep(0.5)

        input()

        # Initialize new game with random starting player
        self.game.reset()
        self.game.current_player = starting_player
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
                input(
                    f"{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}"
                )
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
        self.stats["games_played"] += 1
        self.stats["total_moves"] += self.game.move_count

        if self.game.winner == 1:
            self.stats["player1_wins"] += 1
        elif self.game.winner == -1:
            self.stats["player2_wins"] += 1
        else:
            self.stats["draws"] += 1

        # Show game summary
        render_game_summary(game_time, self.game.move_count)

        input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")

    def play_random_vs_random(self) -> None:
        """Play Random vs Random mode (for testing)."""
        render_game_mode_header("RANDOM vs RANDOM MODE", Colors.WARNING)

        print(f"{Colors.INFO}Instructions:{Colors.RESET}")
        print(
            f"- {Colors.INFO}Two random agents will play "
            f"against each other{Colors.RESET}"
        )
        print(f"- {Colors.PLAYER1}Random Agent 1 (RED X){Colors.RESET}: Goes first")
        print(f"- {Colors.PLAYER2}Random Agent 2 (BLUE O){Colors.RESET}: Goes second")
        print(
            f"- {Colors.WARNING}Watch the agents make random valid moves{Colors.RESET}"
        )
        print(f"- {Colors.INFO}Game will auto-play until completion{Colors.RESET}")

        # Get user preferences
        num_games = self.get_user_choice(
            f"\n{Colors.INFO}How many games to play? (1-10): {Colors.RESET}",
            [str(i) for i in range(1, 11)],
        )
        num_games = int(num_games)

        auto_advance = self.get_user_choice(
            f"{Colors.INFO}Auto-advance between moves? (y/n): {Colors.RESET}",
            ["y", "n", "yes", "no"],
        ).lower() in ["y", "yes"]

        delay_time = 1.0 if auto_advance else 0.0

        print(f"\n{Colors.WARNING}Starting {num_games} game(s)...{Colors.RESET}")
        input(f"{Colors.INFO}Press Enter to begin...{Colors.RESET}")

        # Create random agents with different seeds
        agent1 = RandomAgent(seed=42, name="RandomAgent_1")
        agent2 = RandomAgent(seed=123, name="RandomAgent_2")

        game_results = []

        for game_num in range(num_games):
            print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}GAME {game_num + 1} of {num_games}{Colors.RESET}")
            print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")

            # Initialize new game with random starting player
            self.game.reset()
            starting_player = random.choice([1, -1])
            self.game.current_player = starting_player

            # Announce which agent starts first
            if starting_player == 1:
                print(
                    f"{Colors.SUCCESS}🎲 Random selection: RandomAgent_1 (X) starts first!{Colors.RESET}"
                )
            else:
                print(
                    f"{Colors.WARNING}🎲 Random selection: RandomAgent_2 (O) starts first!{Colors.RESET}"
                )

            # Brief pause to show the selection
            time.sleep(0.5)

            # Reset agents for new episode
            agent1.reset_episode()
            agent2.reset_episode()

            current_agent = agent1 if starting_player == 1 else agent2
            other_agent = agent2 if starting_player == 1 else agent1

            game_start_time = time.time()
            move_count = 0

            # Game loop
            while not self.game.game_over:
                # Show current board state
                self.game.render(show_stats=True)

                # Get valid moves
                valid_moves = self.game.get_valid_moves()

                # Get action from current agent
                action = current_agent.get_action(self.game.board, valid_moves)

                # Display agent's choice
                agent_symbol = "X" if self.game.current_player == 1 else "O"
                agent_name = current_agent.name
                print(
                    f"\n{Colors.INFO}🤖 {agent_name} ({agent_symbol}) "
                    f"chooses column {action + 1}{Colors.RESET}"
                )

                # Make the move
                if not self.game.drop_piece(action):
                    print(
                        f"{Colors.ERROR}Invalid move by {agent_name}! "
                        f"This shouldn't happen.{Colors.RESET}"
                    )
                    break

                move_count += 1

                # Delay or wait for user input
                if auto_advance:
                    time.sleep(delay_time)
                else:
                    input(f"{Colors.WARNING}Press Enter for next move...{Colors.RESET}")

                # Switch agents
                current_agent, other_agent = other_agent, current_agent

            # Game completed - show final state
            self.game.render(show_stats=True)

            game_time = time.time() - game_start_time

            # Record results
            if self.game.winner == 1:
                winner_name = "RandomAgent_1 (X)"
                agent1.end_episode("win")
                agent2.end_episode("loss")
                self.stats["player1_wins"] += 1
            elif self.game.winner == -1:
                winner_name = "RandomAgent_2 (O)"
                agent1.end_episode("loss")
                agent2.end_episode("win")
                self.stats["player2_wins"] += 1
            else:
                winner_name = "Draw"
                agent1.end_episode("draw")
                agent2.end_episode("draw")
                self.stats["draws"] += 1

            game_results.append(
                {
                    "game": game_num + 1,
                    "winner": winner_name,
                    "moves": move_count,
                    "time": game_time,
                }
            )

            self.stats["games_played"] += 1
            self.stats["total_moves"] += move_count

            print(
                f"\n{Colors.SUCCESS}🏆 Game {game_num + 1} Result: "
                f"{winner_name}{Colors.RESET}"
            )
            print(
                f"{Colors.INFO}Duration: {game_time:.1f}s, "
                f"Moves: {move_count}{Colors.RESET}"
            )

            if game_num < num_games - 1:  # Not the last game
                if not auto_advance:
                    input(f"{Colors.WARNING}Press Enter for next game...{Colors.RESET}")
                else:
                    time.sleep(1.0)

        # Show session summary
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}SESSION SUMMARY - {num_games} GAMES{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")

        agent1_wins = sum(1 for r in game_results if r["winner"] == "RandomAgent_1 (X)")
        agent2_wins = sum(1 for r in game_results if r["winner"] == "RandomAgent_2 (O)")
        draws = sum(1 for r in game_results if r["winner"] == "Draw")

        print(f"{Colors.PLAYER1}RandomAgent_1 (X) Wins: {agent1_wins}{Colors.RESET}")
        print(f"{Colors.PLAYER2}RandomAgent_2 (O) Wins: {agent2_wins}{Colors.RESET}")
        print(f"{Colors.WARNING}Draws: {draws}{Colors.RESET}")

        avg_moves = sum(r["moves"] for r in game_results) / len(game_results)
        avg_time = sum(r["time"] for r in game_results) / len(game_results)

        print(f"{Colors.INFO}Average game length: {avg_moves:.1f} moves{Colors.RESET}")
        print(f"{Colors.INFO}Average game time: {avg_time:.1f} seconds{Colors.RESET}")

        # Show agent statistics
        print(f"\n{Colors.INFO}Final Agent Statistics:{Colors.RESET}")
        print(f"{Colors.PLAYER1}Agent 1: {agent1}{Colors.RESET}")
        print(f"{Colors.PLAYER2}Agent 2: {agent2}{Colors.RESET}")

        input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")

    def play_human_vs_ai(self) -> None:
        """Play Human vs AI mode (future implementation)."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.INFO}{Colors.BOLD}>>> HUMAN vs AI MODE <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        render_development_message("PPO agents", "Coming in Phase 4 of the project...")
        input(f"{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")

    def view_statistics(self) -> None:
        """Display game statistics."""
        render_statistics(self.stats)
        input(f"\n{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")

    def start_training(self) -> None:
        """Launch the training interface."""
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.INFO}{Colors.BOLD}>>> LAUNCHING TRAINING SYSTEM <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        
        try:
            # Import training interface
            from train import TrainingInterface
            
            print(f"{Colors.INFO}Initializing PPO training system...{Colors.RESET}")
            training_interface = TrainingInterface()
            
            print(f"{Colors.SUCCESS}Training system ready!{Colors.RESET}")
            input(f"{Colors.WARNING}Press Enter to continue to training menu...{Colors.RESET}")
            
            # Launch training interface
            training_interface.run()
            
        except ImportError as e:
            print(f"{Colors.ERROR}Error: Could not import training system: {e}{Colors.RESET}")
            print(f"{Colors.INFO}Make sure train.py is in the scripts directory.{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.ERROR}Error launching training: {e}{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")

    def run(self) -> None:
        """Main game loop."""
        print("Starting Connect4 Interactive Game Interface...")

        while True:
            self.display_main_menu()

            choice = self.get_user_choice(
                f"{Colors.INFO}Enter your choice (1-6): {Colors.RESET}",
                ["1", "2", "3", "4", "5", "6"],
            )

            if choice == "1":
                self.play_human_vs_human()
            elif choice == "2":
                self.play_random_vs_random()
            elif choice == "3":
                self.play_human_vs_ai()
            elif choice == "4":
                self.start_training()
            elif choice == "5":
                self.view_statistics()
            elif choice == "6":
                print(
                    f"{Colors.SUCCESS}Thanks for playing Connect4! "
                    f"Goodbye!{Colors.RESET}"
                )
                break


def main():
    """Main entry point."""
    try:
        interface = GameInterface()
        interface.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your installation and try again.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
