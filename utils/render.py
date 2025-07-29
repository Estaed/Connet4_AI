"""
Terminal Rendering System for Connect4 RL Training

This module centralizes all terminal rendering functionality including:
- Game board rendering with colors and statistics
- Menu and UI rendering
- Statistics and progress displays
- User interface components

All rendering functions use consistent ANSI color codes and formatting.
"""


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


def format_separator(width: int = 60, char: str = '=') -> str:
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


def render_connect4_game(game_obj, mode: str = 'human', show_stats: bool = True):
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
    
    # Column numbers with color (1-7 instead of 0-6)
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
        output_lines.append(f"{Colors.HEADER}{Colors.BOLD}CONNECT4 RL TRAINING - REAL-TIME STATS{Colors.RESET}")
        output_lines.append(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
        output_lines.append("")
        
        # Performance metrics with colors
        output_lines.append(f"{Colors.INFO}{Colors.BOLD}[PERFORMANCE METRICS]{Colors.RESET}")
        game_time = current_time - game_obj.start_time
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
        output_lines.append(f"Avg Game Len:     {Colors.INFO}{game_obj.move_count:.1f}{Colors.RESET} moves")
        output_lines.append("")
        
        # Device/Hardware statistics  
        output_lines.append(f"{Colors.INFO}{Colors.BOLD}[DEVICE STATISTICS]{Colors.RESET}")
        
        # Import here to avoid circular imports
        try:
            import torch
            from src.core.config import get_config
            config = get_config()
            
            # Game device (always CPU as per PRD)
            game_device = config.get('device.game_device', 'cpu').upper()
            output_lines.append(f"Game Logic:       {Colors.INFO}{game_device}{Colors.RESET}")
            
            # Training device (CPU/GPU based on availability)
            training_device = config.get('device.training_device', 'cpu').upper()
            if training_device == 'CUDA':
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
                output_lines.append(f"Training Device:  {Colors.SUCCESS}{training_device}{Colors.RESET}")
                output_lines.append(f"GPU Name:         {Colors.SUCCESS}{gpu_name}{Colors.RESET}")
                
                # Show GPU memory if available
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB  
                    output_lines.append(f"GPU Memory:       {Colors.INFO}{memory_allocated:.2f}GB / {memory_cached:.2f}GB{Colors.RESET}")
                else:
                    output_lines.append(f"GPU Memory:       {Colors.WARNING}N/A{Colors.RESET}")
            else:
                output_lines.append(f"Training Device:  {Colors.INFO}{training_device}{Colors.RESET}")
                output_lines.append(f"GPU Available:    {Colors.WARNING}{'Yes' if torch.cuda.is_available() else 'No'}{Colors.RESET}")
                
        except ImportError:
            # Fallback if config system not available
            output_lines.append(f"Game Logic:       {Colors.INFO}CPU{Colors.RESET}")
            output_lines.append(f"Training Device:  {Colors.WARNING}Unknown{Colors.RESET}")
            output_lines.append(f"Config Status:    {Colors.ERROR}Not loaded{Colors.RESET}")
        output_lines.append("")
        
        output_lines.append(f"{Colors.HEADER}{'='*60}{Colors.RESET}")
        output_lines.append(f"{Colors.INFO}Updates: 60 FPS | Renderer FPS: {fps:.1f}{Colors.RESET}")
        output_lines.append(f"{Colors.HEADER}{'='*60}{Colors.RESET}")
        
    # Game over message with colors
    if game_obj.game_over:
        output_lines.append("")
        if game_obj.winner == 0:
            output_lines.append(f"{Colors.WARNING}{Colors.BOLD}GAME OVER - DRAW!{Colors.RESET}")
        else:
            winner_symbol, winner_display = format_player_display(game_obj.winner)
            winner_display = winner_display.replace("Player", f"{Colors.BOLD}Player")
            output_lines.append(f"{Colors.SUCCESS}{Colors.BOLD}GAME OVER - {winner_display} ({winner_symbol}) WINS!{Colors.RESET}")
    else:
        output_lines.append("")
        if game_obj.current_player == 1:
            prompt_player = f"{Colors.PLAYER1}Player X{Colors.RESET}"
        else:
            prompt_player = f"{Colors.PLAYER2}Player O{Colors.RESET}"
        output_lines.append(f"{prompt_player}, enter column (1-{game_obj.board_cols}):")
        
    output_str = "\n".join(output_lines)
    
    if mode == 'human':
        # Clear screen and print (for terminal play)
        clear_screen()
        print(output_str)
        return None
    else:
        return output_str


def render_main_menu() -> None:
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


def render_game_mode_header(mode_name: str, mode_color: str = Colors.SUCCESS) -> None:
    """Render a game mode header."""
    print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    print(f"{mode_color}{Colors.BOLD}>>> {mode_name} <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")


def render_game_instructions() -> None:
    """Render Human vs Human game instructions."""  
    print(f"{Colors.INFO}Instructions:{Colors.RESET}")
    print(f"- {Colors.INFO}Players take turns dropping pieces{Colors.RESET}")
    print(f"- {Colors.PLAYER1}Player 1 (RED X){Colors.RESET}: Goes first - Human player")  
    print(f"- {Colors.PLAYER2}Player 2 (BLUE O){Colors.RESET}: Goes second - Second player")
    print(f"- {Colors.WARNING}Enter column number (1-7) to drop piece{Colors.RESET}")
    print(f"- {Colors.ERROR}Type 'q' to quit game{Colors.RESET}")
    print(f"- {Colors.SUCCESS}Connect 4 pieces in a row to win!{Colors.RESET}")
    print(f"- {Colors.INFO}Board is colorized: {Colors.PLAYER1}RED{Colors.RESET} for Player 1, {Colors.PLAYER2}BLUE{Colors.RESET} for Player 2")
    print(f"\n{Colors.WARNING}Press Enter to start...{Colors.RESET}")


def render_game_summary(game_time: float, move_count: int) -> None:
    """Render game summary statistics."""
    print(f"\n{Colors.INFO}{Colors.BOLD}Game Summary:{Colors.RESET}")
    print(f"- {Colors.INFO}Duration: {game_time:.1f} seconds{Colors.RESET}")
    print(f"- {Colors.INFO}Total moves: {move_count}{Colors.RESET}")
    print(f"- {Colors.INFO}Moves per second: {move_count / game_time:.1f}{Colors.RESET}")


def render_statistics(stats: dict) -> None:
    """Display game statistics."""
    print(f"\n{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.HEADER}{Colors.BOLD}>>> GAME STATISTICS <<<{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 50}{Colors.RESET}")
    
    if stats['games_played'] == 0:
        print(f"{Colors.WARNING}No games played yet.{Colors.RESET}")
    else:
        total_games = stats['games_played']
        avg_moves = stats['total_moves'] / total_games
        
        print(f"{Colors.INFO}Total Games Played: {Colors.SUCCESS}{total_games}{Colors.RESET}")
        print(f"{Colors.PLAYER1}Player 1 (X) Wins:{Colors.RESET}  {stats['player1_wins']} ({stats['player1_wins']/total_games*100:.1f}%)")
        print(f"{Colors.PLAYER2}Player 2 (O) Wins:{Colors.RESET}  {stats['player2_wins']} ({stats['player2_wins']/total_games*100:.1f}%)")
        print(f"{Colors.WARNING}Draws:{Colors.RESET}              {stats['draws']} ({stats['draws']/total_games*100:.1f}%)")
        print(f"{Colors.INFO}Average Game Length: {avg_moves:.1f} moves{Colors.RESET}")


def render_development_message(feature_name: str, phase_info: str = "Currently in development...") -> None:
    """Render a development/coming soon message."""
    print(f"{Colors.INFO}This mode will be implemented when {feature_name} are available.{Colors.RESET}")
    print(f"{Colors.WARNING}{phase_info}{Colors.RESET}")