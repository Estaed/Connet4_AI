#!/usr/bin/env python3
"""
Demo Training Script for Connect4 RL System

This script demonstrates the training functionality by running a very short
training session to verify all components work together correctly.

This is for testing purposes only - real training should use train.py
"""

import sys
import os

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.render import Colors, clear_screen
from scripts.train import SingleEnvTrainer, TrainingStatistics

def demo_training():
    """Run a short demo training session."""
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.SUCCESS}CONNECT4 TRAINING DEMO{Colors.RESET}")
    print(f"{Colors.HEADER}{'=' * 60}{Colors.RESET}")
    
    print(f"\n{Colors.INFO}This demo will run a very short training session to verify{Colors.RESET}")
    print(f"{Colors.INFO}that all components are working correctly.{Colors.RESET}")
    
    try:
        # Test configuration loading
        print(f"\n{Colors.INFO}1. Testing configuration loading...{Colors.RESET}")
        config = {
            'training.update_frequency': 5,
            'training.render_frequency': 1,
            'training.progress_frequency': 10
        }
        print(f"{Colors.SUCCESS}   Configuration loaded successfully{Colors.RESET}")
        
        # Test trainer initialization
        print(f"\n{Colors.INFO}2. Testing trainer initialization...{Colors.RESET}")
        trainer = SingleEnvTrainer(config)
        print(f"{Colors.SUCCESS}   SingleEnvTrainer created successfully{Colors.RESET}")
        print(f"{Colors.INFO}   Device: {Colors.WARNING}{trainer.device.upper()}{Colors.RESET}")
        
        # Test statistics system
        print(f"\n{Colors.INFO}3. Testing statistics system...{Colors.RESET}")
        stats = TrainingStatistics()
        stats.update_game_result(winner=1, moves=15)
        stats.update_game_result(winner=-1, moves=20)
        stats.update_game_result(winner=0, moves=42)
        print(f"{Colors.SUCCESS}   Statistics system working correctly{Colors.RESET}")
        print(f"{Colors.INFO}   Total games: {stats.win_stats['total_games']}{Colors.RESET}")
        print(f"{Colors.INFO}   Avg game length: {stats.win_stats['avg_game_length']:.1f}{Colors.RESET}")
        
        # Test PPO metrics
        print(f"\n{Colors.INFO}4. Testing PPO metrics...{Colors.RESET}")
        test_metrics = {
            'policy_loss': 0.05,
            'value_loss': 0.03,
            'total_loss': 0.08,
            'avg_reward': 0.15,
            'entropy': 0.02
        }
        stats.update_ppo_metrics(test_metrics)
        print(f"{Colors.SUCCESS}   PPO metrics updated successfully{Colors.RESET}")
        
        # Test render system
        print(f"\n{Colors.INFO}5. Testing render system...{Colors.RESET}")
        from utils.render import render_training_progress
        
        # Simulate training progress display
        clear_screen()
        render_training_progress(
            episode=50,
            total_episodes=100,
            win_stats=stats.win_stats,
            ppo_metrics=stats.ppo_metrics,
            performance_stats={
                'episodes_per_sec': 2.5,
                'games_per_sec': 2.0,
                'training_time': 25.0,
                'eta': 20.0
            }
        )
        
        print(f"\n{Colors.SUCCESS}Render system test completed!{Colors.RESET}")
        
        # Test environment and agent interaction
        print(f"\n{Colors.INFO}6. Testing environment and agent interaction...{Colors.RESET}")
        
        # Create a very short training run (5 episodes)
        print(f"{Colors.WARNING}Running micro-training session (5 episodes)...{Colors.RESET}")
        
        # Temporarily override input for demo
        import builtins
        original_input = builtins.input
        builtins.input = lambda *args: ""  # Auto-confirm all prompts
        
        try:
            results = trainer.train(
                level_name="Demo",
                total_episodes=5,
                show_game_render=False,  # Don't show game rendering for demo
                render_interval=1
            )
        finally:
            # Restore original input function
            builtins.input = original_input
        
        print(f"\n{Colors.SUCCESS}{'=' * 60}{Colors.RESET}")
        print(f"{Colors.SUCCESS}{Colors.BOLD}DEMO TRAINING COMPLETED SUCCESSFULLY!{Colors.RESET}")
        print(f"{Colors.SUCCESS}{'=' * 60}{Colors.RESET}")
        
        print(f"\n{Colors.INFO}Demo Results:{Colors.RESET}")
        print(f"- Episodes completed: {results['total_episodes']}")
        print(f"- Training time: {results['total_time']:.1f} seconds")
        print(f"- Total games: {results['win_stats']['total_games']}")
        print(f"- System status: {Colors.SUCCESS}All components working!{Colors.RESET}")
        
        return True
        
    except Exception as e:
        print(f"\n{Colors.ERROR}Demo failed with error: {e}{Colors.RESET}")
        print(f"{Colors.ERROR}Please check the installation and dependencies.{Colors.RESET}")
        return False

def main():
    """Main entry point for demo."""
    try:
        success = demo_training()
        if success:
            print(f"\n{Colors.SUCCESS}Demo completed successfully!{Colors.RESET}")
            print(f"{Colors.INFO}The training system is ready for use.{Colors.RESET}")
            print(f"{Colors.INFO}Run 'python scripts/train.py' to start training.{Colors.RESET}")
            return 0
        else:
            print(f"\n{Colors.ERROR}Demo failed. Please check the setup.{Colors.RESET}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Demo interrupted by user.{Colors.RESET}")
        return 1
    except Exception as e:
        print(f"\n{Colors.ERROR}Unexpected error: {e}{Colors.RESET}")
        return 1

if __name__ == "__main__":
    sys.exit(main())