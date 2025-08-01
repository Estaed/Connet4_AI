"""
Terminal Rendering System for Connect4 RL Training

This module centralizes all terminal rendering functionality including:
- Game board rendering with colors and statistics
- Menu and UI rendering
- Statistics and progress displays
- User interface components

All rendering functions use consistent ANSI color codes and formatting.
"""

from typing import Any


class Colors:
    """ANSI color codes for terminal styling."""

    # Player colors
    PLAYER1 = "\033[91m"  # Red for Player 1 (X) - Human player
    PLAYER2 = "\033[94m"  # Blue for Player 2 (O) - AI/Second player
    EMPTY = "\033[90m"  # Gray for empty spaces

    # UI colors
    HEADER = "\033[95m"  # Magenta for headers
    SUCCESS = "\033[92m"  # Green for success/good stats
    WARNING = "\033[93m"  # Yellow for warnings
    ERROR = "\033[91m"  # Red for errors
    INFO = "\033[96m"  # Cyan for info

    # Special
    BOLD = "\033[1m"  # Bold text
    UNDERLINE = "\033[4m"  # Underlined text
    RESET = "\033[0m"  # Reset to default


def clear_screen() -> None:
    """Clear the terminal screen for better rendering."""
    print("\n" * 50)  # Simple screen clear


def format_cell_symbol(cell_value: int) -> str:
    """
    Format a board cell with appropriate color and symbol.

    Args:
        cell_value: Cell value (1 for Player 1, -1 for Player 2, 0 for empty)

    Returns:
        Formatted string with color and symbol
    """
    if cell_value == 1:  # Player 1 (Human - Red X)
        return f"{Colors.PLAYER1}X{Colors.RESET}"
    elif cell_value == -1:  # Player 2 (AI - Blue O)
        return f"{Colors.PLAYER2}O{Colors.RESET}"
    else:  # Empty cell
        return f"{Colors.EMPTY}.{Colors.RESET}"


def format_player_display(player: int) -> tuple[str, str]:
    """
    Format player display with appropriate colors.

    Args:
        player: Player number (1 or -1)

    Returns:
        Tuple of (symbol, display_name) with colors
    """
    if player == 1:
        symbol = f"{Colors.PLAYER1}X{Colors.RESET}"
        display = f"{Colors.PLAYER1}Player 1{Colors.RESET}"
    else:
        symbol = f"{Colors.PLAYER2}O{Colors.RESET}"
        display = f"{Colors.PLAYER2}Player 2{Colors.RESET}"

    return symbol, display


def format_header(text: str, width: int = 60) -> str:
    """
    Format a header with consistent styling.

    Args:
        text: Header text
        width: Total width of header

    Returns:
        Formatted header string
    """
    return f"{Colors.HEADER}{Colors.BOLD}{text.center(width)}{Colors.RESET}"


def format_separator(width: int = 60, char: str = "=") -> str:
    """
    Format a separator line with consistent styling.

    Args:
        width: Width of separator
        char: Character to use for separator

    Returns:
        Formatted separator string
    """
    return f"{Colors.HEADER}{char * width}{Colors.RESET}"


def format_stats_section(title: str) -> str:
    """
    Format a statistics section header.

    Args:
        title: Section title

    Returns:
        Formatted section header
    """
    return f"{Colors.INFO}{Colors.BOLD}[{title}]{Colors.RESET}"


def format_metric(label: str, value: str, color: str = Colors.INFO) -> str:
    """
    Format a metric display line.

    Args:
        label: Metric label
        value: Metric value
        color: Color for the value

    Returns:
        Formatted metric string
    """
    return f"{label:<20} {color}{value}{Colors.RESET}"


def render_connect4_game(game_obj: Any, mode: str = "human", show_stats: bool = True, show_column_numbers: bool = True) -> None:
    """
    Render the Connect4 game board and statistics.

    Args:
        game_obj: Connect4Game instance
        mode: Rendering mode ('human' for terminal output)
        show_stats: Whether to show performance statistics

    Returns:
        String representation if mode != 'human', None otherwise
    """
    import time

    game_obj.render_count += 1
    current_time = time.time()

    # Calculate FPS
    time_since_last = current_time - game_obj.last_render_time
    fps = 1.0 / time_since_last if time_since_last > 0 else 0.0
    game_obj.last_render_time = current_time

    # Build output string
    output_lines = []

    # Column numbers with color (1-7 instead of 0-6) - only if requested
    if show_column_numbers:
        header = " " + "".join(f" {i+1}  " for i in range(game_obj.board_cols))
        output_lines.append(f"{Colors.HEADER}{header}{Colors.RESET}")
    output_lines.append(f"{Colors.HEADER}{'='*30}{Colors.RESET}")

    # Board display with colors
    for row in range(game_obj.board_rows):
        row_str = f"{Colors.HEADER}|{Colors.RESET}"
        for col in range(game_obj.board_cols):
            cell = game_obj.board[row][col]
            symbol = format_cell_symbol(cell)
            row_str += f" {symbol} {Colors.HEADER}|{Colors.RESET}"
        output_lines.append(row_str)

    output_lines.append(f"{Colors.HEADER}{'='*30}{Colors.RESET}")

    # Current player info with colors
    player_symbol, player_display = format_player_display(game_obj.current_player)
    output_lines.append(f"Current: {player_symbol} {player_display}")

    if show_stats:
        # Performance metrics section with colors
        output_lines.append("")
        output_lines.append(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
        output_lines.append(
            f"{Colors.HEADER}{Colors.BOLD}CONNECT4 RL TRAINING - REAL-TIME STATS{Colors.RESET}"
        )
        output_lines.append(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
        output_lines.append("")

        # Performance metrics with colors
        output_lines.append(
            f"{Colors.INFO}{Colors.BOLD}[PERFORMANCE METRICS]{Colors.RESET}"
        )
        game_time = current_time - game_obj.start_time
        output_lines.append(f"Games/sec:        {Colors.WARNING}0{Colors.RESET}")
        output_lines.append(f"Total Games:      {Colors.INFO}1{Colors.RESET}")
        output_lines.append(f"Episode:          {Colors.INFO}0{Colors.RESET}")
        output_lines.append(
            f"Training Time:    {Colors.INFO}{game_time:08.2f}{Colors.RESET}"
        )
        output_lines.append("")

        # Win statistics with player colors
        output_lines.append(f"{Colors.INFO}{Colors.BOLD}[WIN STATISTICS]{Colors.RESET}")
        output_lines.append(
            f"Player 1 (X):     {Colors.PLAYER1}0.0%{Colors.RESET} (0 wins)"
        )
        output_lines.append(
            f"Player 2 (O):     {Colors.PLAYER2}0.0%{Colors.RESET} (0 wins)"
        )
        output_lines.append(
            f"Draws:            {Colors.WARNING}0.0%{Colors.RESET} (0 draws)"
        )
        output_lines.append(
            f"Avg Game Len:     {Colors.INFO}{game_obj.move_count:.1f}{Colors.RESET} moves"
        )
        output_lines.append("")

        # Device/Hardware statistics
        output_lines.append(
            f"{Colors.INFO}{Colors.BOLD}[DEVICE STATISTICS]{Colors.RESET}"
        )

        # Import here to avoid circular imports
        try:
            import torch
            from src.core.config import get_config

            config = get_config()

            # Game device (always CPU as per PRD)
            game_device = config.get("device.game_device", "cpu").upper()
            output_lines.append(
                f"Game Logic:       {Colors.INFO}{game_device}{Colors.RESET}"
            )

            # Training device (CPU/GPU based on availability)
            training_device = config.get("device.training_device", "cpu").upper()
            if training_device == "CUDA":
                gpu_name = (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else "Unknown"
                )
                output_lines.append(
                    f"Training Device:  {Colors.SUCCESS}{training_device}{Colors.RESET}"
                )
                output_lines.append(
                    f"GPU Name:         {Colors.SUCCESS}{gpu_name}{Colors.RESET}"
                )

                # Show GPU memory if available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
                    output_lines.append(
                        f"GPU Memory:       {Colors.INFO}{memory_allocated:.2f}GB / {memory_cached:.2f}GB{Colors.RESET}"
                    )
                else:
                    output_lines.append(
                        f"GPU Memory:       {Colors.WARNING}N/A{Colors.RESET}"
                    )
            else:
                output_lines.append(
                    f"Training Device:  {Colors.INFO}{training_device}{Colors.RESET}"
                )
                output_lines.append(
                    f"GPU Available:    {Colors.WARNING}{'Yes' if torch.cuda.is_available() else 'No'}{Colors.RESET}"
                )

        except ImportError:
            # Fallback if config system not available
            output_lines.append(f"Game Logic:       {Colors.INFO}CPU{Colors.RESET}")
            output_lines.append(
                f"Training Device:  {Colors.WARNING}Unknown{Colors.RESET}"
            )
            output_lines.append(
                f"Config Status:    {Colors.ERROR}Not loaded{Colors.RESET}"
            )
        output_lines.append("")

        output_lines.append(f"{Colors.HEADER}{'='*60}{Colors.RESET}")
        output_lines.append(
            f"{Colors.INFO}Updates: 60 FPS | Renderer FPS: {fps:.1f}{Colors.RESET}"
        )
        output_lines.append(f"{Colors.HEADER}{'='*60}{Colors.RESET}")

    # Game over message with colors
    if game_obj.game_over:
        output_lines.append("")
        if game_obj.winner == 0:
            output_lines.append(
                f"{Colors.WARNING}{Colors.BOLD}GAME OVER - DRAW!{Colors.RESET}"
            )
        else:
            winner_symbol, winner_display = format_player_display(game_obj.winner)
            winner_display = winner_display.replace("Player", f"{Colors.BOLD}Player")
            output_lines.append(
                f"{Colors.SUCCESS}{Colors.BOLD}GAME OVER - {winner_display} ({winner_symbol}) WINS!{Colors.RESET}"
            )
    else:
        output_lines.append("")
        if game_obj.current_player == 1:
            prompt_player = f"{Colors.PLAYER1}Player X{Colors.RESET}"
        else:
            prompt_player = f"{Colors.PLAYER2}Player O{Colors.RESET}"
        output_lines.append(f"{prompt_player}, enter column (1-{game_obj.board_cols}):")

    output_str = "\n".join(output_lines)

    if mode == "human":
        # Clear screen and print (for terminal play)
        clear_screen()
        print(output_str)
        return None
    else:
        return output_str


def render_main_menu() -> None:
    """Display the main menu with game mode options."""
    print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    print(
        f"{Colors.BOLD}{Colors.HEADER}*** CONNECT4 RL TRAINING SYSTEM ***{Colors.RESET}"
    )
    print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    print(f"\n{Colors.INFO}Select Mode:{Colors.RESET}")
    print(
        f"{Colors.SUCCESS}1. Human vs Human{Colors.RESET} (Interactive - Player 1 goes first)"
    )
    print(
        f"{Colors.WARNING}2. Random vs Random{Colors.RESET} (Agent Testing - Available!)"
    )
    print(
        f"{Colors.INFO}3. Human vs AI{Colors.RESET} (Select trained model - Available!)"
    )
    print(
        f"{Colors.PLAYER2}4. AI vs AI{Colors.RESET} (Select two models to compete - Available!)"
    )
    print(f"{Colors.HEADER}{Colors.BOLD}5. Start Training{Colors.RESET} (PPO Agent Training - Available!)")
    print(f"{Colors.INFO}6. Model Management{Colors.RESET} (Browse, validate, manage models)")
    print(f"{Colors.HEADER}7. View Statistics{Colors.RESET}")
    print(f"{Colors.ERROR}8. Exit{Colors.RESET}")
    print(f"\n{Colors.HEADER}{'-' * 50}{Colors.RESET}")


def render_game_mode_header(mode_name: str, mode_color: str = Colors.SUCCESS) -> None:
    """Render a game mode header."""
    print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    print(f"{mode_color}{Colors.BOLD}>>> {mode_name} <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")


def render_game_instructions() -> None:
    """Render Human vs Human game instructions."""
    print(f"{Colors.INFO}Instructions:{Colors.RESET}")
    print(f"- {Colors.INFO}Players take turns dropping pieces{Colors.RESET}")
    print(
        f"- {Colors.PLAYER1}Player 1 (RED X){Colors.RESET}: Goes first - Human player"
    )
    print(
        f"- {Colors.PLAYER2}Player 2 (BLUE O){Colors.RESET}: Goes second - Second player"
    )
    print(f"- {Colors.WARNING}Enter column number (1-7) to drop piece{Colors.RESET}")
    print(f"- {Colors.ERROR}Type 'q' to quit game{Colors.RESET}")
    print(f"- {Colors.SUCCESS}Connect 4 pieces in a row to win!{Colors.RESET}")
    print(
        f"- {Colors.INFO}Board is colorized: {Colors.PLAYER1}RED{Colors.RESET} for Player 1, {Colors.PLAYER2}BLUE{Colors.RESET} for Player 2"
    )
    print(f"\n{Colors.WARNING}Press Enter to start...{Colors.RESET}")


def render_game_summary(game_time: float, move_count: int) -> None:
    """Render game summary statistics."""
    print(f"\n{Colors.INFO}{Colors.BOLD}Game Summary:{Colors.RESET}")
    print(f"- {Colors.INFO}Duration: {game_time:.1f} seconds{Colors.RESET}")
    print(f"- {Colors.INFO}Total moves: {move_count}{Colors.RESET}")
    print(
        f"- {Colors.INFO}Moves per second: {move_count / game_time:.1f}{Colors.RESET}"
    )


def render_statistics(stats: dict) -> None:
    """Display game statistics."""
    print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}>>> GAME STATISTICS <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")

    if stats["games_played"] == 0:
        print(f"{Colors.WARNING}No games played yet.{Colors.RESET}")
    else:
        total_games = stats["games_played"]
        avg_moves = stats["total_moves"] / total_games

        print(
            f"{Colors.INFO}Total Games Played: {Colors.SUCCESS}{total_games}{Colors.RESET}"
        )
        print(
            f"{Colors.PLAYER1}Player 1 (X) Wins:{Colors.RESET}  {stats['player1_wins']} ({stats['player1_wins']/total_games*100:.1f}%)"
        )
        print(
            f"{Colors.PLAYER2}Player 2 (O) Wins:{Colors.RESET}  {stats['player2_wins']} ({stats['player2_wins']/total_games*100:.1f}%)"
        )
        print(
            f"{Colors.WARNING}Draws:{Colors.RESET}              {stats['draws']} ({stats['draws']/total_games*100:.1f}%)"
        )
        print(f"{Colors.INFO}Average Game Length: {avg_moves:.1f} moves{Colors.RESET}")


def render_development_message(
    feature_name: str, phase_info: str = "Currently in development..."
) -> None:
    """Render a development/coming soon message."""
    print(
        f"{Colors.INFO}This mode will be implemented when {feature_name} are available.{Colors.RESET}"
    )
    print(f"{Colors.WARNING}{phase_info}{Colors.RESET}")


def render_training_menu() -> None:
    """Display the training level selection menu."""
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}*** CONNECT4 RL TRAINING SYSTEM ***{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"\n{Colors.INFO}Select Training Level:{Colors.RESET}")
    print(f"{Colors.SUCCESS}1. Test Training{Colors.RESET}       - 1,000 steps (Quick validation)")
    print(f"{Colors.WARNING}2. Small Training{Colors.RESET}      - 10,000 steps (Basic learning)")
    print(f"{Colors.INFO}3. Medium Training{Colors.RESET}     - 100,000 steps (Advanced training)")
    print(f"{Colors.ERROR}4. Impossible Training{Colors.RESET} - 1,000,000 steps (Maximum challenge)")
    print(f"{Colors.BOLD}5. Custom Training{Colors.RESET}     - Choose your own steps & environments")
    print(f"{Colors.HEADER}6. Training Settings{Colors.RESET}   - Configure checkpoint & TensorBoard options")
    print(f"{Colors.HEADER}7. Back to Main Menu{Colors.RESET}")
    print(f"\n{Colors.HEADER}{'-' * 60}{Colors.RESET}")


def render_training_header(level_name: str, total_steps: int, num_envs: int = 1) -> None:
    """Render training session header."""
    print(f"\n{Colors.HEADER}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.SUCCESS}>>> {level_name.upper()} TRAINING SESSION <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.INFO}Training Steps: {Colors.WARNING}{total_steps:,}{Colors.RESET}")
    print(f"{Colors.INFO}Environments: {Colors.WARNING}{num_envs}{Colors.RESET}")
    print(f"{Colors.INFO}Training Mode: {Colors.SUCCESS}Self-Play PPO{Colors.RESET}")
    print(f"{Colors.HEADER}{'-' * 70}{Colors.RESET}")


def render_training_progress(
    episode: int,
    total_episodes: int,
    win_stats: dict,
    ppo_metrics: dict,
    performance_stats: dict
) -> None:
    """
    Render real-time training progress and statistics.
    
    Args:
        episode: Current episode number
        total_episodes: Total episodes for this training session
        win_stats: Dictionary with win statistics (player1_wins, player2_wins, draws)
        ppo_metrics: Dictionary with PPO training metrics (loss, reward, etc.)
        performance_stats: Dictionary with performance metrics (fps, games_per_sec, etc.)
    """
    import time
    
    # Progress calculation
    progress_percent = (episode / total_episodes) * 100 if total_episodes > 0 else 0
    
    print(f"\n{Colors.HEADER}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}CONNECT4 RL TRAINING - LIVE PROGRESS{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 80}{Colors.RESET}")
    
    # Training Progress Section
    print(f"{Colors.INFO}{Colors.BOLD}[TRAINING PROGRESS]{Colors.RESET}")
    print(f"Episode:          {Colors.WARNING}{episode:,}{Colors.RESET} / {Colors.INFO}{total_episodes:,}{Colors.RESET}")
    print(f"Progress:         {Colors.SUCCESS}{progress_percent:.1f}%{Colors.RESET}")
    
    # Create simple progress bar (ASCII only for Windows compatibility)
    bar_width = 40
    filled_width = int(bar_width * progress_percent / 100)
    bar = "#" * filled_width + "-" * (bar_width - filled_width)
    print(f"Progress Bar:     {Colors.SUCCESS}[{bar}]{Colors.RESET}")
    print()
    
    # Win Statistics Section
    print(f"{Colors.INFO}{Colors.BOLD}[WIN STATISTICS]{Colors.RESET}")
    total_games = win_stats.get('total_games', 0)
    if total_games > 0:
        p1_wins = win_stats.get('player1_wins', 0)
        p2_wins = win_stats.get('player2_wins', 0)
        draws = win_stats.get('draws', 0)
        
        p1_rate = (p1_wins / total_games) * 100
        p2_rate = (p2_wins / total_games) * 100
        draw_rate = (draws / total_games) * 100
        
        print(f"Total Games:      {Colors.INFO}{total_games:,}{Colors.RESET}")
        print(f"Player 1 (X):     {Colors.PLAYER1}{p1_rate:.1f}%{Colors.RESET} ({p1_wins:,} wins)")
        print(f"Player 2 (O):     {Colors.PLAYER2}{p2_rate:.1f}%{Colors.RESET} ({p2_wins:,} wins)")
        print(f"Draws:            {Colors.WARNING}{draw_rate:.1f}%{Colors.RESET} ({draws:,} draws)")
        print(f"Avg Game Length:  {Colors.INFO}{win_stats.get('avg_game_length', 0):.1f}{Colors.RESET} moves")
    else:
        print(f"Total Games:      {Colors.WARNING}0{Colors.RESET} (Starting...)")
    print()
    
    # PPO Metrics Section
    print(f"{Colors.INFO}{Colors.BOLD}[PPO TRAINING METRICS]{Colors.RESET}")
    print(f"Policy Loss:      {Colors.WARNING}{ppo_metrics.get('policy_loss', 0.0):.6f}{Colors.RESET}")
    print(f"Value Loss:       {Colors.WARNING}{ppo_metrics.get('value_loss', 0.0):.6f}{Colors.RESET}")
    print(f"Total Loss:       {Colors.ERROR}{ppo_metrics.get('total_loss', 0.0):.6f}{Colors.RESET}")
    print(f"Avg Reward:       {Colors.SUCCESS}{ppo_metrics.get('avg_reward', 0.0):.3f}{Colors.RESET}")
    print(f"Entropy:          {Colors.INFO}{ppo_metrics.get('entropy', 0.0):.6f}{Colors.RESET}")
    print()
    
    # Performance Statistics Section
    print(f"{Colors.INFO}{Colors.BOLD}[PERFORMANCE METRICS]{Colors.RESET}")
    print(f"Episodes/sec:     {Colors.WARNING}{performance_stats.get('episodes_per_sec', 0.0):.2f}{Colors.RESET}")
    print(f"Games/sec:        {Colors.WARNING}{performance_stats.get('games_per_sec', 0.0):.2f}{Colors.RESET}")
    print(f"Training Time:    {Colors.INFO}{performance_stats.get('training_time', 0.0):.1f}{Colors.RESET} seconds")
    print(f"Est. Remaining:   {Colors.INFO}{performance_stats.get('eta', 0.0):.1f}{Colors.RESET} seconds")
    print()
    
    # Device Information Section  
    print(f"{Colors.INFO}{Colors.BOLD}[DEVICE STATISTICS]{Colors.RESET}")
    print(f"Game Logic:       {Colors.INFO}CPU{Colors.RESET}")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
            print(f"Training Device:  {Colors.SUCCESS}GPU{Colors.RESET}")
            print(f"GPU Name:         {Colors.SUCCESS}{device_name}{Colors.RESET}")
            print(f"GPU Memory:       {Colors.INFO}{memory_allocated:.2f}GB / {memory_cached:.2f}GB{Colors.RESET}")
        else:
            print(f"Training Device:  {Colors.WARNING}CPU{Colors.RESET}")
            print(f"GPU Available:    {Colors.ERROR}No{Colors.RESET}")
    except ImportError:
        print(f"Training Device:  {Colors.WARNING}CPU{Colors.RESET}")
        print(f"PyTorch:          {Colors.ERROR}Not available{Colors.RESET}")
    
    print(f"{Colors.HEADER}{'=' * 80}{Colors.RESET}")


def render_training_complete(
    level_name: str,
    total_time: float,
    total_episodes: int,
    final_win_stats: dict,
    final_metrics: dict
) -> None:
    """Render training completion summary."""
    print(f"\n{Colors.SUCCESS}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.SUCCESS}{Colors.BOLD}>>> {level_name.upper()} TRAINING COMPLETED! <<<{Colors.RESET}")
    print(f"{Colors.SUCCESS}{'=' * 70}{Colors.RESET}")
    
    print(f"\n{Colors.INFO}{Colors.BOLD}Final Training Summary:{Colors.RESET}")
    print(f"Total Time:       {Colors.INFO}{total_time:.1f}{Colors.RESET} seconds")
    print(f"Total Episodes:   {Colors.INFO}{total_episodes:,}{Colors.RESET}")
    print(f"Episodes/sec:     {Colors.WARNING}{total_episodes/total_time:.2f}{Colors.RESET}")
    
    # Final win statistics
    total_games = final_win_stats.get('total_games', 0)
    if total_games > 0:
        print(f"\n{Colors.INFO}{Colors.BOLD}Final Win Statistics:{Colors.RESET}")
        p1_wins = final_win_stats.get('player1_wins', 0)
        p2_wins = final_win_stats.get('player2_wins', 0)
        draws = final_win_stats.get('draws', 0)
        
        print(f"Player 1 (X):     {Colors.PLAYER1}{(p1_wins/total_games)*100:.1f}%{Colors.RESET} ({p1_wins:,} wins)")
        print(f"Player 2 (O):     {Colors.PLAYER2}{(p2_wins/total_games)*100:.1f}%{Colors.RESET} ({p2_wins:,} wins)")
        print(f"Draws:            {Colors.WARNING}{(draws/total_games)*100:.1f}%{Colors.RESET} ({draws:,} draws)")
    
    # Final PPO metrics
    if final_metrics:
        print(f"\n{Colors.INFO}{Colors.BOLD}Final PPO Metrics:{Colors.RESET}")
        print(f"Final Policy Loss: {Colors.WARNING}{final_metrics.get('policy_loss', 0.0):.6f}{Colors.RESET}")
        print(f"Final Value Loss:  {Colors.WARNING}{final_metrics.get('value_loss', 0.0):.6f}{Colors.RESET}")
        print(f"Final Avg Reward:  {Colors.SUCCESS}{final_metrics.get('avg_reward', 0.0):.3f}{Colors.RESET}")
    
    print(f"\n{Colors.SUCCESS}Training session saved and completed successfully!{Colors.RESET}")
    print(f"{Colors.SUCCESS}{'=' * 70}{Colors.RESET}")


def render_training_game_state(game_obj, episode: int, step: int, agent_name: str = "PPO Agent"):
    """
    Render the current game state during training with training-specific information.
    
    Args:
        game_obj: Connect4Game instance
        episode: Current episode number
        step: Current step in episode
        agent_name: Name of the training agent
    """
    # Use existing game rendering with additional training context
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}TRAINING SESSION - Episode {episode}, Step {step}{Colors.RESET}")
    print(f"{Colors.INFO}Agent: {agent_name} (Self-Play Mode){Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    # Use existing Connect4 game rendering
    render_connect4_game(game_obj, mode="human", show_stats=False)
    
    print(f"{Colors.INFO}Training in progress... {agent_name} is learning to play Connect4{Colors.RESET}")


# ============================================================================
# MODEL SELECTION AND MANAGEMENT UI
# ============================================================================

def render_model_selection_menu(models: list, title: str = "Select AI Model") -> None:
    """
    Display model selection menu with performance information.
    
    Args:
        models: List of ModelMetadata objects
        title: Menu title
    """
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}>>> {title.upper()} <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    if not models:
        print(f"{Colors.WARNING}No trained models available.{Colors.RESET}")
        print(f"{Colors.INFO}Train some models first using the training system.{Colors.RESET}")
        return
    
    print(f"\n{Colors.INFO}Available Models:{Colors.RESET}")
    print(f"{Colors.HEADER}{'ID':<3} {'Name':<25} {'Level':<12} {'Win Rate':<8} {'Games':<8} {'Date':<16}{Colors.RESET}")
    print(f"{Colors.HEADER}{'-' * 76}{Colors.RESET}")
    
    for i, model in enumerate(models, 1):
        # Format model display
        display_name = model.get_display_name()[:24]  # Truncate if too long
        skill_level = model.get_skill_level()
        win_rate = f"{model.win_rate:.1f}%"
        total_games = f"{model.total_games:,}" if model.total_games > 0 else "N/A"
        
        # Color coding based on performance
        if model.win_rate >= 70:
            name_color = Colors.SUCCESS
        elif model.win_rate >= 50:
            name_color = Colors.WARNING
        else:
            name_color = Colors.ERROR
        
        print(f"{Colors.INFO}{i:<3}{Colors.RESET} "
              f"{name_color}{display_name:<25}{Colors.RESET} "
              f"{Colors.INFO}{skill_level:<12}{Colors.RESET} "
              f"{name_color}{win_rate:<8}{Colors.RESET} "
              f"{Colors.INFO}{total_games:<8}{Colors.RESET} "
              f"{Colors.INFO}{model.creation_date}{Colors.RESET}")
    
    print(f"\n{Colors.INFO}Legend:{Colors.RESET}")
    print(f"  * = Best model  |  {Colors.SUCCESS}Green{Colors.RESET} = Strong (70%+)  |  "
          f"{Colors.WARNING}Yellow{Colors.RESET} = Average (50-70%)  |  {Colors.ERROR}Red{Colors.RESET} = Weak (<50%)")
    print(f"\n{Colors.HEADER}{'-' * 60}{Colors.RESET}")


def render_model_details(model: Any, show_validation: bool = True) -> None:
    """
    Display detailed information about a specific model.
    
    Args:
        model: ModelMetadata object
        show_validation: Whether to show validation information
    """
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.INFO}>>> MODEL DETAILS <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    # Model identification
    print(f"{Colors.INFO}{Colors.BOLD}[MODEL INFORMATION]{Colors.RESET}")
    print(f"Name:             {Colors.SUCCESS}{model.get_display_name()}{Colors.RESET}")
    print(f"File Path:        {Colors.INFO}{model.path}{Colors.RESET}")
    print(f"Creation Date:    {Colors.INFO}{model.creation_date}{Colors.RESET}")
    print(f"File Size:        {Colors.INFO}{model.file_size_mb:.1f} MB{Colors.RESET}")
    print(f"Training Episode: {Colors.INFO}{model.episode:,}{Colors.RESET}")
    print()
    
    # Performance metrics
    print(f"{Colors.INFO}{Colors.BOLD}[PERFORMANCE METRICS]{Colors.RESET}")
    print(f"Skill Level:      {Colors.SUCCESS}{model.get_skill_level()}{Colors.RESET}")
    print(f"Performance Grade: {Colors.WARNING}{model.get_performance_grade()}{Colors.RESET}")
    print(f"Win Rate:         {Colors.SUCCESS if model.win_rate >= 60 else Colors.WARNING}{model.win_rate:.1f}%{Colors.RESET}")
    print(f"Total Games:      {Colors.INFO}{model.total_games:,}{Colors.RESET}")
    
    if model.avg_game_length > 0:
        print(f"Avg Game Length:  {Colors.INFO}{model.avg_game_length:.1f} moves{Colors.RESET}")
    
    if model.avg_reward != 0:
        print(f"Avg Reward:       {Colors.INFO}{model.avg_reward:.3f}{Colors.RESET}")
    
    print()
    
    # Model recommendation
    print(f"{Colors.INFO}{Colors.BOLD}[RECOMMENDATION]{Colors.RESET}")
    if model.win_rate >= 80:
        print(f"{Colors.SUCCESS}[*] Excellent choice! This model is very strong.{Colors.RESET}")
    elif model.win_rate >= 60:
        print(f"{Colors.SUCCESS}[+] Good choice! This model plays well.{Colors.RESET}")
    elif model.win_rate >= 40:
        print(f"{Colors.WARNING}[!] Fair choice. This model is still learning.{Colors.RESET}")
    else:
        print(f"{Colors.ERROR}[-] This model is weak. Consider training more or choosing another.{Colors.RESET}")
    
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")


def render_human_vs_ai_setup(selected_model: dict) -> None:
    """
    Display human vs AI game setup information.
    
    Args:
        selected_model: Dictionary with selected model information
    """
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.SUCCESS}>>> HUMAN vs AI SETUP <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.INFO}{Colors.BOLD}[AI OPPONENT DETAILS]{Colors.RESET}")
    print(f"Model Name:      {Colors.SUCCESS}{selected_model.get('name', 'Unknown')}{Colors.RESET}")
    print(f"Skill Level:     {Colors.SUCCESS}{selected_model.get('skill_level', 'Unknown')}{Colors.RESET}")
    print(f"Win Rate:        {Colors.WARNING}{selected_model.get('win_rate', 0):.1f}%{Colors.RESET}")
    print(f"Training Games:  {Colors.INFO}{selected_model.get('total_games', 0):,}{Colors.RESET}")
    
    print(f"\n{Colors.INFO}{Colors.BOLD}[GAME RULES]{Colors.RESET}")
    print(f"• {Colors.PLAYER1}Human Player (RED X){Colors.RESET}: You go first")
    print(f"• {Colors.PLAYER2}AI Player (BLUE O){Colors.RESET}: AI plays second")
    print(f"• Enter column number (1-7) to drop your piece")
    print(f"• Type 'q' to quit the game at any time")
    print(f"• Connect 4 pieces in a row to win!")
    
    # Difficulty assessment
    win_rate = selected_model.get('win_rate', 0)
    if win_rate >= 80:
        difficulty = f"{Colors.ERROR}Very Hard{Colors.RESET} - This AI is very strong!"
    elif win_rate >= 65:
        difficulty = f"{Colors.WARNING}Hard{Colors.RESET} - Challenging opponent"
    elif win_rate >= 50:
        difficulty = f"{Colors.SUCCESS}Medium{Colors.RESET} - Good for practice"
    else:
        difficulty = f"{Colors.INFO}Easy{Colors.RESET} - Still learning"
    
    print(f"\n{Colors.INFO}Difficulty Level: {difficulty}{Colors.RESET}")
    
    print(f"\n{Colors.HEADER}{'-' * 60}{Colors.RESET}")


def render_model_browser_menu() -> None:
    """Display the model browser and management menu."""
    print(f"\n{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.HEADER}*** MODEL MANAGEMENT CENTER ***{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"\n{Colors.INFO}Model Management Options:{Colors.RESET}")
    print(f"{Colors.SUCCESS}1. Browse All Models{Colors.RESET}      - View and select from available models")
    print(f"{Colors.SUCCESS}2. Model Details{Colors.RESET}          - View detailed model information")
    print(f"{Colors.SUCCESS}3. Validate Models{Colors.RESET}        - Check model integrity and performance")
    print(f"{Colors.WARNING}4. Delete Models{Colors.RESET}          - Remove unwanted models")
    print(f"{Colors.INFO}5. Model Statistics{Colors.RESET}       - View overall model statistics")
    print(f"{Colors.INFO}6. Export Model{Colors.RESET}           - Export model for sharing")
    print(f"{Colors.HEADER}7. Back to Main Menu{Colors.RESET}")
    print(f"\n{Colors.HEADER}{'-' * 60}{Colors.RESET}")


def render_model_selection_error(error_message: str) -> None:
    """
    Display an error message for model selection issues.
    
    Args:
        error_message: The error message to display
    """
    print(f"\n{Colors.ERROR}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.ERROR}{Colors.BOLD}>>> MODEL SELECTION ERROR <<<{Colors.RESET}")
    print(f"{Colors.ERROR}{'=' * 50}{Colors.RESET}")
    print(f"\n{Colors.ERROR}Error: {error_message}{Colors.RESET}")
    
    if "not available" in error_message.lower():
        print(f"\n{Colors.INFO}Troubleshooting:{Colors.RESET}")
        print(f"• Check that the models directory exists")
        print(f"• Verify model files are properly saved")
        print(f"• Try training a model first")
    elif "no trained models" in error_message.lower() or "no models" in error_message.lower():
        print(f"\n{Colors.INFO}Solution:{Colors.RESET}")
        print(f"• Train models using the training system (option 4 in main menu)")
        print(f"• Models will be saved in the 'models/' directory")
        print(f"• Once trained, they will appear in the model selection")
    elif "failed to load" in error_message.lower():
        print(f"\n{Colors.INFO}Troubleshooting:{Colors.RESET}")
        print(f"• Check that the model file exists and is not corrupted")
        print(f"• Verify the model was saved correctly during training")
        print(f"• Try selecting a different model")
    
    print(f"\n{Colors.ERROR}{'-' * 50}{Colors.RESET}")


def render_model_loading_progress(model_name: str) -> None:
    """
    Display loading progress for a model.
    
    Args:
        model_name: Name of the model being loaded
    """
    print(f"\n{Colors.INFO}Loading model: {Colors.SUCCESS}{model_name}{Colors.RESET}")
    print(f"{Colors.INFO}Please wait...{Colors.RESET}")


def render_model_loaded_success(model_name: str, model_info: dict = None) -> None:
    """
    Display success message when a model is loaded.
    
    Args:
        model_name: Name of the loaded model
        model_info: Optional dictionary with model information
    """
    print(f"\n{Colors.SUCCESS}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.SUCCESS}{Colors.BOLD}>>> MODEL LOADED SUCCESSFULLY <<<{Colors.RESET}")
    print(f"{Colors.SUCCESS}{'=' * 50}{Colors.RESET}")
    
    print(f"\n{Colors.INFO}Model: {Colors.SUCCESS}{model_name}{Colors.RESET}")
    
    if model_info:
        if 'skill_level' in model_info:
            print(f"Skill Level: {Colors.SUCCESS}{model_info['skill_level']}{Colors.RESET}")
        if 'win_rate' in model_info:
            print(f"Win Rate: {Colors.WARNING}{model_info['win_rate']:.1f}%{Colors.RESET}")
        if 'total_games' in model_info:
            print(f"Training Games: {Colors.INFO}{model_info['total_games']:,}{Colors.RESET}")
    
    print(f"\n{Colors.SUCCESS}Ready to play!{Colors.RESET}")
    print(f"{Colors.SUCCESS}{'-' * 50}{Colors.RESET}")