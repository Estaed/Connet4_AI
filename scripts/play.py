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

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(project_root, "src"))

from src.utils.render import (
    Colors,
    render_main_menu,
    render_game_mode_header,
    render_game_instructions,
    render_game_summary,
    render_statistics,
    render_development_message,
    render_model_selection_menu,
    render_model_details,
    render_human_vs_ai_setup,
    render_model_browser_menu,
    render_model_selection_error,
    render_model_loading_progress,
    render_model_loaded_success,
)

try:
    from src.environments.connect4_game import Connect4Game
    from src.agents.random_agent import RandomAgent
    from src.utils.model_manager import ModelManager
    from src.utils.checkpointing import CheckpointManager
except ImportError as e:
    print(f"Error importing Connect4 components: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class GameInterface:
    """
    Interactive game interface for Connect4.
    Handles user input, game flow, and display.
    """

    def __init__(self, debug_mode: bool = False):
        """Initialize the game interface.
        
        Args:
            debug_mode: Whether to show detailed initialization logs
        """
        self.debug_mode = debug_mode
        self.game = Connect4Game()
        self.stats = {
            "games_played": 0,
            "player1_wins": 0,
            "player2_wins": 0,
            "draws": 0,
            "total_moves": 0,
        }
        
        # Initialize model management system
        try:
            self.model_manager = ModelManager(
                models_dir="models", 
                auto_refresh=True,
                debug_mode=debug_mode
            )
            if debug_mode:
                print(f"[GameInterface] Model manager initialized with {len(self.model_manager.get_models())} models")
        except Exception as e:
            print(f"Warning: Could not initialize model manager: {e}")
            self.model_manager = None

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
        """Play Human vs AI mode with model selection."""
        if not self.model_manager:
            render_model_selection_error("Model manager not available")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        # Get available models
        models = self.model_manager.get_models(sort_by='performance')
        
        if not models:
            render_model_selection_error("No trained models available")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        # Display model selection menu
        render_model_selection_menu(models, "Select AI Opponent")
        
        # Get user's model choice
        valid_choices = [str(i) for i in range(1, len(models) + 1)] + ['b', 'back']
        choice = self.get_user_choice(
            f"\n{Colors.INFO}Select model (1-{len(models)}) or 'b' for back: {Colors.RESET}",
            valid_choices
        )
        
        if choice.lower() in ['b', 'back']:
            return
        
        # Get selected model
        model_index = int(choice) - 1
        selected_model = models[model_index]
        
        # Show model details
        render_model_details(selected_model)
        
        # Confirm selection
        confirm = self.get_user_choice(
            f"\n{Colors.WARNING}Play against this AI? (y/n): {Colors.RESET}",
            ['y', 'n', 'yes', 'no']
        ).lower()
        
        if confirm in ['n', 'no']:
            print(f"{Colors.INFO}Model selection cancelled.{Colors.RESET}")
            input(f"{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        # Load the selected model
        render_model_loading_progress(selected_model.name)
        
        try:
            ai_agent = self.model_manager.load_model_for_gameplay(selected_model.name)
            
            if not ai_agent:
                render_model_selection_error(f"Failed to load model: {selected_model.name}")
                input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
                return
            
            # Show successful loading
            model_info = {
                'episode': selected_model.episode,
                'win_rate': selected_model.win_rate,
                'skill_level': selected_model.get_skill_level()
            }
            render_model_loaded_success(selected_model.name, model_info)
            
            # Show game setup
            render_human_vs_ai_setup(selected_model.to_dict())
            
            input(f"\n{Colors.WARNING}Press Enter to start the game...{Colors.RESET}")
            
            # Play the game
            self._play_human_vs_ai_game(ai_agent, selected_model)
            
        except Exception as e:
            render_model_selection_error(f"Error loading model: {str(e)}")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
    
    def _play_human_vs_ai_game(self, ai_agent, model_metadata) -> None:
        """Play a human vs AI game."""
        # Initialize game with human starting first
        self.game.reset()
        self.game.current_player = 1  # Human starts (Player X)
        game_start_time = time.time()
        
        print(f"\n{Colors.SUCCESS}🎮 Starting Human vs AI Game!{Colors.RESET}")
        print(f"{Colors.INFO}AI Opponent: {model_metadata.get_display_name()} ({model_metadata.get_skill_level()}){Colors.RESET}")
        
        # Game loop
        while not self.game.game_over:
            # Render current state (show column numbers only for human turns)
            show_columns = (self.game.current_player == 1)
            self.game.render(show_stats=True, show_column_numbers=show_columns)
            
            if self.game.current_player == 1:  # Human turn
                player_symbol = "X"
                print(f"\n{Colors.PLAYER1}Your turn (Player {player_symbol})!{Colors.RESET}")
                
                # Get human input
                col = self.get_column_input(player_symbol)
                
                if col == -1:  # Player quit
                    print(f"{Colors.WARNING}Game ended by human player.{Colors.RESET}")
                    input(f"{Colors.INFO}Press Enter to return to main menu...{Colors.RESET}")
                    return
                
                # Make human move
                if not self.game.drop_piece(col):
                    print(f"{Colors.ERROR}Invalid move! Please try again.{Colors.RESET}")
                    input(f"{Colors.INFO}Press Enter to continue...{Colors.RESET}")
                    continue
                
                print(f"{Colors.SUCCESS}You placed your piece in column {col + 1}!{Colors.RESET}")
                
            else:  # AI turn
                player_symbol = "O"
                print(f"\n{Colors.PLAYER2}AI turn (Player {player_symbol})...{Colors.RESET}")
                
                # Get AI action
                try:
                    # Convert board to format expected by agent
                    # The AI expects the board from its perspective (it's player -1)
                    ai_observation = self.game.board.copy()
                    valid_actions = self.game.get_valid_moves()
                    
                    # Get AI action
                    ai_action = ai_agent.get_action(ai_observation, valid_actions)
                    
                    print(f"{Colors.INFO}🤖 AI is thinking...{Colors.RESET}")
                    time.sleep(0.5)  # Brief pause for dramatic effect
                    
                    # Make AI move
                    if ai_action in valid_actions and self.game.drop_piece(ai_action):
                        print(f"{Colors.SUCCESS}AI placed its piece in column {ai_action + 1}!{Colors.RESET}")
                    else:
                        print(f"{Colors.ERROR}AI made invalid move! Using random fallback.{Colors.RESET}")
                        # Fallback to random valid move
                        fallback_action = random.choice(valid_actions)
                        self.game.drop_piece(fallback_action)
                        print(f"{Colors.INFO}AI placed piece in column {fallback_action + 1} (random).{Colors.RESET}")
                    
                except Exception as e:
                    print(f"{Colors.ERROR}AI error: {e}. Using random move.{Colors.RESET}")
                    valid_actions = self.game.get_valid_moves()
                    fallback_action = random.choice(valid_actions)
                    self.game.drop_piece(fallback_action)
                    print(f"{Colors.INFO}AI placed piece in column {fallback_action + 1} (random).{Colors.RESET}")
                
            # Brief pause between moves
            time.sleep(0.3)
        
        # Game over - show final state with column numbers for reference
        self.game.render(show_stats=True, show_column_numbers=True)
        
        # Update statistics
        game_time = time.time() - game_start_time
        self.stats["games_played"] += 1
        self.stats["total_moves"] += self.game.move_count
        
        # Determine result and update stats
        if self.game.winner == 1:
            self.stats["player1_wins"] += 1
            result_msg = f"{Colors.SUCCESS}🏆 Congratulations! You beat the AI!{Colors.RESET}"
        elif self.game.winner == -1:
            self.stats["player2_wins"] += 1
            result_msg = f"{Colors.ERROR}🤖 AI wins! Better luck next time!{Colors.RESET}"
        else:
            self.stats["draws"] += 1
            result_msg = f"{Colors.WARNING}🤝 It's a draw! Great game!{Colors.RESET}"
        
        print(f"\n{result_msg}")
        
        # Show game summary
        render_game_summary(game_time, self.game.move_count)
        
        # Ask if player wants to play again with same AI
        play_again = self.get_user_choice(
            f"\n{Colors.INFO}Play again with same AI? (y/n): {Colors.RESET}",
            ['y', 'n', 'yes', 'no']
        ).lower()
        
        if play_again in ['y', 'yes']:
            self._play_human_vs_ai_game(ai_agent, model_metadata)
        else:
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")

    def play_ai_vs_ai(self) -> None:
        """Play AI vs AI mode with model selection."""
        if not self.model_manager:
            render_model_selection_error("Model manager not available")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        # Get available models
        models = self.model_manager.get_models(sort_by='performance')
        
        if len(models) < 2:
            render_model_selection_error("Need at least 2 trained models for AI vs AI")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        render_game_mode_header("AI vs AI MODE", Colors.PLAYER2)
        
        print(f"{Colors.INFO}Instructions:{Colors.RESET}")
        print(f"- {Colors.INFO}Select two AI models to compete against each other{Colors.RESET}")
        print(f"- {Colors.PLAYER1}AI Model 1 (RED X){Colors.RESET}: First player")
        print(f"- {Colors.PLAYER2}AI Model 2 (BLUE O){Colors.RESET}: Second player")
        print(f"- {Colors.WARNING}Watch different AI strategies compete!{Colors.RESET}")
        print(f"- {Colors.INFO}Great way to evaluate training progress{Colors.RESET}")
        
        # Select first AI model
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}STEP 1: Select First AI Model (Player X){Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        
        render_model_selection_menu(models, "Select AI Model 1 (Player X)")
        
        valid_choices = [str(i) for i in range(1, len(models) + 1)] + ['b', 'back']
        choice1 = self.get_user_choice(
            f"\n{Colors.INFO}Select model 1 (1-{len(models)}) or 'b' for back: {Colors.RESET}",
            valid_choices
        )
        
        if choice1.lower() in ['b', 'back']:
            return
        
        model1_index = int(choice1) - 1
        ai_model_1 = models[model1_index]
        
        print(f"\n{Colors.SUCCESS}✅ AI Model 1 Selected: {ai_model_1.get_display_name()}{Colors.RESET}")
        
        # Select second AI model
        print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        print(f"{Colors.BOLD}STEP 2: Select Second AI Model (Player O){Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
        
        # Show available models (excluding the first one selected)
        available_models = [m for i, m in enumerate(models) if i != model1_index]
        render_model_selection_menu(available_models, "Select AI Model 2 (Player O)")
        
        valid_choices = [str(i) for i in range(1, len(available_models) + 1)] + ['b', 'back']
        choice2 = self.get_user_choice(
            f"\n{Colors.INFO}Select model 2 (1-{len(available_models)}) or 'b' for back: {Colors.RESET}",
            valid_choices
        )
        
        if choice2.lower() in ['b', 'back']:
            return
        
        model2_index = int(choice2) - 1
        ai_model_2 = available_models[model2_index]
        
        print(f"\n{Colors.SUCCESS}✅ AI Model 2 Selected: {ai_model_2.get_display_name()}{Colors.RESET}")
        
        # Show matchup summary
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}AI vs AI MATCHUP{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.PLAYER1}Player X (RED): {ai_model_1.get_display_name()}{Colors.RESET}")
        print(f"  Episode: {ai_model_1.episode:,}, Win Rate: {ai_model_1.win_rate:.1f}%")
        print(f"  Skill Level: {ai_model_1.get_skill_level()}")
        print()
        print(f"{Colors.PLAYER2}Player O (BLUE): {ai_model_2.get_display_name()}{Colors.RESET}")
        print(f"  Episode: {ai_model_2.episode:,}, Win Rate: {ai_model_2.win_rate:.1f}%")
        print(f"  Skill Level: {ai_model_2.get_skill_level()}")
        
        # Get game settings
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
        
        # Load the AI models
        print(f"\n{Colors.INFO}Loading AI models...{Colors.RESET}")
        
        try:
            agent1 = self.model_manager.load_model_for_gameplay(ai_model_1.name)
            agent2 = self.model_manager.load_model_for_gameplay(ai_model_2.name)
            
            if not agent1 or not agent2:
                render_model_selection_error("Failed to load one or both AI models")
                input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
                return
                
            print(f"{Colors.SUCCESS}✅ Both AI models loaded successfully!{Colors.RESET}")
            
        except Exception as e:
            render_model_selection_error(f"Error loading AI models: {str(e)}")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        print(f"\n{Colors.WARNING}Starting {num_games} AI vs AI game(s)...{Colors.RESET}")
        input(f"{Colors.INFO}Press Enter to begin the battle...{Colors.RESET}")
        
        # Play the games
        self._play_ai_vs_ai_games(agent1, agent2, ai_model_1, ai_model_2, num_games, auto_advance, delay_time)

    def _play_ai_vs_ai_games(self, agent1, agent2, model1_meta, model2_meta, num_games, auto_advance, delay_time):
        """Play multiple AI vs AI games."""
        game_results = []
        
        for game_num in range(num_games):
            print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
            print(f"{Colors.BOLD}AI vs AI GAME {game_num + 1} of {num_games}{Colors.RESET}")
            print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
            
            # Initialize new game with random starting player
            self.game.reset()
            starting_player = random.choice([1, -1])
            self.game.current_player = starting_player
            
            # Announce which AI starts first
            if starting_player == 1:
                print(f"{Colors.SUCCESS}🎲 Random selection: {model1_meta.get_display_name()} (X) starts first!{Colors.RESET}")
                current_agent = agent1
                other_agent = agent2
                current_model = model1_meta
                other_model = model2_meta
            else:
                print(f"{Colors.WARNING}🎲 Random selection: {model2_meta.get_display_name()} (O) starts first!{Colors.RESET}")
                current_agent = agent2
                other_agent = agent1
                current_model = model2_meta
                other_model = model1_meta
            
            # Brief pause to show the selection
            time.sleep(0.5)
            
            game_start_time = time.time()
            move_count = 0
            
            # Game loop
            while not self.game.game_over:
                # Show current board state
                self.game.render(show_stats=True)
                
                # Get valid moves
                valid_moves = self.game.get_valid_moves()
                
                # Get AI action
                try:
                    # Convert board to format expected by agent
                    ai_observation = self.game.board.copy()
                    action = current_agent.get_action(ai_observation, valid_moves)
                    
                    # Display AI's choice
                    agent_symbol = "X" if self.game.current_player == 1 else "O"
                    agent_name = current_model.get_display_name()
                    
                    print(f"\n{Colors.INFO}🤖 {agent_name} ({agent_symbol}) is thinking...{Colors.RESET}")
                    
                    if auto_advance:
                        time.sleep(0.3)  # Brief thinking time
                    
                    print(f"{Colors.SUCCESS}🤖 {agent_name} ({agent_symbol}) chooses column {action + 1}{Colors.RESET}")
                    
                    # Make the move
                    if action in valid_moves and self.game.drop_piece(action):
                        move_count += 1
                    else:
                        print(f"{Colors.ERROR}Invalid move by {agent_name}! Using random fallback.{Colors.RESET}")
                        # Fallback to random valid move
                        fallback_action = random.choice(valid_moves)
                        self.game.drop_piece(fallback_action)
                        move_count += 1
                        print(f"{Colors.INFO}Used column {fallback_action + 1} instead.{Colors.RESET}")
                    
                except Exception as e:
                    print(f"{Colors.ERROR}AI error for {current_model.get_display_name()}: {e}. Using random move.{Colors.RESET}")
                    valid_actions = self.game.get_valid_moves()
                    fallback_action = random.choice(valid_actions)
                    self.game.drop_piece(fallback_action)
                    move_count += 1
                    print(f"{Colors.INFO}Used column {fallback_action + 1} (random).{Colors.RESET}")
                
                # Delay or wait for user input
                if auto_advance:
                    time.sleep(delay_time)
                else:
                    input(f"{Colors.WARNING}Press Enter for next move...{Colors.RESET}")
                
                # Switch agents
                current_agent, other_agent = other_agent, current_agent
                current_model, other_model = other_model, current_model
            
            # Game completed - show final state
            self.game.render(show_stats=True)
            
            game_time = time.time() - game_start_time
            
            # Record results
            if self.game.winner == 1:
                winner_name = f"{model1_meta.get_display_name()} (X)"
                self.stats["player1_wins"] += 1
            elif self.game.winner == -1:
                winner_name = f"{model2_meta.get_display_name()} (O)"
                self.stats["player2_wins"] += 1
            else:
                winner_name = "Draw"
                self.stats["draws"] += 1
            
            game_results.append({
                "game": game_num + 1,
                "winner": winner_name,
                "moves": move_count,
                "time": game_time,
            })
            
            self.stats["games_played"] += 1
            self.stats["total_moves"] += move_count
            
            print(f"\n{Colors.SUCCESS}🏆 Game {game_num + 1} Result: {winner_name}{Colors.RESET}")
            print(f"{Colors.INFO}Duration: {game_time:.1f}s, Moves: {move_count}{Colors.RESET}")
            
            if game_num < num_games - 1:  # Not the last game
                if not auto_advance:
                    input(f"{Colors.WARNING}Press Enter for next game...{Colors.RESET}")
                else:
                    time.sleep(1.0)
        
        # Show session summary
        print(f"\n{Colors.HEADER}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.BOLD}AI vs AI SESSION SUMMARY - {num_games} GAMES{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 70}{Colors.RESET}")
        
        model1_wins = sum(1 for r in game_results if model1_meta.get_display_name() in r["winner"])
        model2_wins = sum(1 for r in game_results if model2_meta.get_display_name() in r["winner"])
        draws = sum(1 for r in game_results if r["winner"] == "Draw")
        
        print(f"\n{Colors.PLAYER1}{model1_meta.get_display_name()} (X) Wins: {model1_wins}{Colors.RESET}")
        print(f"  Win Rate: {(model1_wins/num_games)*100:.1f}%")
        print(f"  Training: Episode {model1_meta.episode:,}, {model1_meta.get_skill_level()}")
        
        print(f"\n{Colors.PLAYER2}{model2_meta.get_display_name()} (O) Wins: {model2_wins}{Colors.RESET}")
        print(f"  Win Rate: {(model2_wins/num_games)*100:.1f}%")
        print(f"  Training: Episode {model2_meta.episode:,}, {model2_meta.get_skill_level()}")
        
        print(f"\n{Colors.WARNING}Draws: {draws} ({(draws/num_games)*100:.1f}%){Colors.RESET}")
        
        avg_moves = sum(r["moves"] for r in game_results) / len(game_results)
        avg_time = sum(r["time"] for r in game_results) / len(game_results)
        
        print(f"\n{Colors.INFO}Average game length: {avg_moves:.1f} moves{Colors.RESET}")
        print(f"{Colors.INFO}Average game time: {avg_time:.1f} seconds{Colors.RESET}")
        
        # Determine overall winner
        if model1_wins > model2_wins:
            print(f"\n{Colors.SUCCESS}🏆 OVERALL WINNER: {model1_meta.get_display_name()}!{Colors.RESET}")
        elif model2_wins > model1_wins:
            print(f"\n{Colors.SUCCESS}🏆 OVERALL WINNER: {model2_meta.get_display_name()}!{Colors.RESET}")
        else:
            print(f"\n{Colors.WARNING}🤝 SERIES TIED! Both AIs performed equally well!{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")

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
            from src.training.training_interface import TrainingInterface
            
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
    
    def manage_models(self) -> None:
        """Model management interface."""
        if not self.model_manager:
            render_model_selection_error("Model manager not available")
            input(f"\n{Colors.WARNING}Press Enter to return to main menu...{Colors.RESET}")
            return
        
        while True:
            render_model_browser_menu()
            
            choice = self.get_user_choice(
                f"{Colors.INFO}Enter your choice (1-7): {Colors.RESET}",
                ["1", "2", "3", "4", "5", "6", "7"]
            )
            
            if choice == "1":
                self._browse_all_models()
            elif choice == "2":
                self._show_model_details()
            elif choice == "3":
                self._validate_models()
            elif choice == "4":
                self._delete_models()
            elif choice == "5":
                self._show_model_statistics()
            elif choice == "6":
                self._export_model()
            elif choice == "7":
                break
    
    def _browse_all_models(self) -> None:
        """Browse all available models."""
        models = self.model_manager.get_models(sort_by='performance')
        
        if not models:
            render_model_selection_error("No models found")
            input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            return
        
        render_model_selection_menu(models, "All Available Models")
        
        # Option to view details
        print(f"\n{Colors.INFO}Options:{Colors.RESET}")
        print(f"{Colors.SUCCESS}d <number>{Colors.RESET} - View details (e.g., 'd 1')")
        print(f"{Colors.WARNING}b{Colors.RESET} - Back to model management")
        
        choice = input(f"\n{Colors.INFO}Enter choice: {Colors.RESET}").strip().lower()
        
        if choice.startswith('d '):
            try:
                model_num = int(choice.split()[1])
                if 1 <= model_num <= len(models):
                    render_model_details(models[model_num - 1])
                    input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
                else:
                    print(f"{Colors.ERROR}Invalid model number{Colors.RESET}")
                    input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            except (ValueError, IndexError):
                print(f"{Colors.ERROR}Invalid format. Use 'd <number>'{Colors.RESET}")
                input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
    
    def _show_model_details(self) -> None:
        """Show detailed information for a specific model."""
        models = self.model_manager.get_models()
        
        if not models:
            render_model_selection_error("No models found")
            input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            return
        
        render_model_selection_menu(models, "Select Model for Details")
        
        valid_choices = [str(i) for i in range(1, len(models) + 1)] + ['b', 'back']
        choice = self.get_user_choice(
            f"\n{Colors.INFO}Select model (1-{len(models)}) or 'b' for back: {Colors.RESET}",
            valid_choices
        )
        
        if choice.lower() in ['b', 'back']:
            return
        
        model_index = int(choice) - 1
        selected_model = models[model_index]
        
        render_model_details(selected_model)
        
        # Show validation info
        validation_result = self.model_manager.validate_model(selected_model.name)
        
        print(f"\n{Colors.INFO}{Colors.BOLD}[VALIDATION RESULTS]{Colors.RESET}")
        if validation_result['valid']:
            print(f"Status: {Colors.SUCCESS}Valid{Colors.RESET}")
        else:
            print(f"Status: {Colors.ERROR}Invalid{Colors.RESET}")
        
        if validation_result['errors']:
            print(f"Errors: {Colors.ERROR}{', '.join(validation_result['errors'])}{Colors.RESET}")
        
        if validation_result['warnings']:
            print(f"Warnings: {Colors.WARNING}{', '.join(validation_result['warnings'])}{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
    
    def _validate_models(self) -> None:
        """Validate all models."""
        models = self.model_manager.get_models()
        
        if not models:
            render_model_selection_error("No models found")
            input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            return
        
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.INFO}>>> MODEL VALIDATION RESULTS <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        valid_count = 0
        invalid_count = 0
        
        for i, model in enumerate(models, 1):
            print(f"\n{Colors.INFO}Validating {i}/{len(models)}: {model.name}{Colors.RESET}")
            
            validation_result = self.model_manager.validate_model(model.name)
            
            if validation_result['valid']:
                print(f"  Status: {Colors.SUCCESS}✅ Valid{Colors.RESET}")
                valid_count += 1
            else:
                print(f"  Status: {Colors.ERROR}❌ Invalid{Colors.RESET}")
                if validation_result['errors']:
                    print(f"  Errors: {Colors.ERROR}{', '.join(validation_result['errors'])}{Colors.RESET}")
                invalid_count += 1
            
            if validation_result['warnings']:
                print(f"  Warnings: {Colors.WARNING}{', '.join(validation_result['warnings'])}{Colors.RESET}")
        
        print(f"\n{Colors.INFO}{Colors.BOLD}[VALIDATION SUMMARY]{Colors.RESET}")
        print(f"Total models: {Colors.INFO}{len(models)}{Colors.RESET}")
        print(f"Valid models: {Colors.SUCCESS}{valid_count}{Colors.RESET}")
        print(f"Invalid models: {Colors.ERROR}{invalid_count}{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
    
    def _delete_models(self) -> None:
        """Delete models interface."""
        models = self.model_manager.get_models()
        
        if not models:
            render_model_selection_error("No models found")
            input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            return
        
        print(f"\n{Colors.ERROR}{Colors.BOLD}⚠️  MODEL DELETION ⚠️{Colors.RESET}")
        print(f"{Colors.WARNING}This will permanently delete the selected model file.{Colors.RESET}")
        
        render_model_selection_menu(models, "Select Model to Delete")
        
        valid_choices = [str(i) for i in range(1, len(models) + 1)] + ['b', 'back']
        choice = self.get_user_choice(
            f"\n{Colors.INFO}Select model (1-{len(models)}) or 'b' for back: {Colors.RESET}",
            valid_choices
        )
        
        if choice.lower() in ['b', 'back']:
            return
        
        model_index = int(choice) - 1
        selected_model = models[model_index]
        
        # Show model details before deletion
        render_model_details(selected_model)
        
        # Double confirmation
        confirm1 = self.get_user_choice(
            f"\n{Colors.ERROR}Are you sure you want to delete this model? (yes/no): {Colors.RESET}",
            ['yes', 'no']
        ).lower()
        
        if confirm1 != 'yes':
            print(f"{Colors.INFO}Deletion cancelled.{Colors.RESET}")
            input(f"{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            return
        
        confirm2 = self.get_user_choice(
            f"{Colors.ERROR}Final confirmation - Delete {selected_model.name}? (DELETE/cancel): {Colors.RESET}",
            ['DELETE', 'cancel']
        )
        
        if confirm2 != 'DELETE':
            print(f"{Colors.INFO}Deletion cancelled.{Colors.RESET}")
            input(f"{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
            return
        
        # Delete the model
        if self.model_manager.delete_model(selected_model.name):
            print(f"{Colors.SUCCESS}✅ Model deleted successfully: {selected_model.name}{Colors.RESET}")
        else:
            print(f"{Colors.ERROR}❌ Failed to delete model: {selected_model.name}{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
    
    def _show_model_statistics(self) -> None:
        """Show overall model statistics."""
        stats = self.model_manager.get_statistics()
        
        print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.INFO}>>> MODEL STATISTICS <<<{Colors.RESET}")
        print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
        
        print(f"\n{Colors.INFO}{Colors.BOLD}[OVERVIEW]{Colors.RESET}")
        print(f"Total models:     {Colors.WARNING}{stats['total_models']}{Colors.RESET}")
        print(f"Best models:      {Colors.SUCCESS}{stats['best_models']}{Colors.RESET}")
        print(f"Average win rate: {Colors.WARNING}{stats['avg_win_rate']:.1f}%{Colors.RESET}")
        print(f"Total storage:    {Colors.INFO}{stats['total_size_mb']:.1f} MB{Colors.RESET}")
        print(f"Models directory: {Colors.INFO}{stats['models_dir']}{Colors.RESET}")
        
        # Performance distribution
        perf_dist = stats.get('performance_distribution', {})
        if perf_dist and any(perf_dist.values()):
            print(f"\n{Colors.INFO}{Colors.BOLD}[SKILL LEVEL DISTRIBUTION]{Colors.RESET}")
            for level, count in perf_dist.items():
                if count > 0:
                    if level in ['Expert', 'Advanced']:
                        color = Colors.SUCCESS
                    elif level == 'Intermediate':
                        color = Colors.WARNING
                    else:
                        color = Colors.ERROR
                    print(f"{level:12} {color}{count:3} models{Colors.RESET}")
        
        # Latest model info
        if stats['latest_model_episode'] > 0:
            print(f"\n{Colors.INFO}{Colors.BOLD}[LATEST MODEL]{Colors.RESET}")
            print(f"Latest episode:   {Colors.INFO}{stats['latest_model_episode']:,}{Colors.RESET}")
        
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")
    
    def _export_model(self) -> None:
        """Export model (placeholder for future implementation)."""
        print(f"\n{Colors.INFO}Model export functionality will be implemented in a future update.{Colors.RESET}")
        print(f"{Colors.INFO}For now, you can manually copy .pt files from the models/ directory.{Colors.RESET}")
        input(f"\n{Colors.WARNING}Press Enter to continue...{Colors.RESET}")

    def run(self) -> None:
        """Main game loop."""
        if not self.debug_mode:
            print("Starting Connect4 Interactive Game Interface...")
        else:
            print("Starting Connect4 Interactive Game Interface...")
            print("[DEBUG MODE] Detailed logging enabled")

        while True:
            self.display_main_menu()

            choice = self.get_user_choice(
                f"{Colors.INFO}Enter your choice (1-8): {Colors.RESET}",
                ["1", "2", "3", "4", "5", "6", "7", "8"],
            )

            if choice == "1":
                self.play_human_vs_human()
            elif choice == "2":
                self.play_random_vs_random()
            elif choice == "3":
                self.play_human_vs_ai()
            elif choice == "4":
                self.play_ai_vs_ai()
            elif choice == "5":
                self.start_training()
            elif choice == "6":
                self.manage_models()
            elif choice == "7":
                self.view_statistics()
            elif choice == "8":
                print(
                    f"{Colors.SUCCESS}Thanks for playing Connect4! "
                    f"Goodbye!{Colors.RESET}"
                )
                break


def main():
    """Main entry point."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Connect4 Interactive Game Interface')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode with detailed logging')
    parser.add_argument('--debug-mode', action='store_true', 
                       help='Enable debug mode with detailed logging (alias for --debug)')
    
    args = parser.parse_args()
    debug_mode = args.debug or args.debug_mode
    
    try:
        interface = GameInterface(debug_mode=debug_mode)
        interface.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please check that all dependencies are installed: pip install -r requirements.txt")
        return 1
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure you're running from the project root directory.")
        return 1
    except PermissionError as e:
        print(f"Permission error: {e}")
        print("Please check file permissions or run with appropriate privileges.")
        return 1
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        print("Please check your installation and try again.")
        if debug_mode:
            import traceback
            print("\nFull traceback (debug mode):")
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
