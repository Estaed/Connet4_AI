"""
Performance benchmarks and tests for Connect4 environments.

Tests performance characteristics, memory usage, and scalability
of both single game and vectorized environment systems.
"""

import pytest
import time
import psutil
import os
import torch
import numpy as np
from typing import List, Tuple
from src.environments.connect4_game import Connect4Game
from src.environments.hybrid_vectorized_connect4 import HybridVectorizedConnect4


class TestConnect4GamePerformance:
    """Performance benchmarks for single Connect4 game operations."""
    
    @pytest.mark.performance
    def test_game_initialization_speed(self):
        """Test speed of game initialization."""
        start_time = time.time()
        
        games = []
        for _ in range(1000):
            game = Connect4Game()
            games.append(game)
        
        end_time = time.time()
        init_time = end_time - start_time
        
        # Should initialize 1000 games quickly
        assert init_time < 0.1  # Less than 0.1 seconds
        
        # Verify games are properly initialized
        assert len(games) == 1000
        assert all(np.all(game.board == 0) for game in games)
    
    @pytest.mark.performance
    def test_move_execution_speed(self, empty_game):
        """Test speed of move execution."""
        game = empty_game
        
        start_time = time.time()
        
        # Execute many moves
        for _ in range(10000):
            col = np.random.randint(0, 7)
            if game.is_valid_move(col):
                game.drop_piece(col)
            
            # Reset periodically to avoid filling board
            if np.random.random() < 0.1:
                game.reset()
        
        end_time = time.time()
        move_time = end_time - start_time
        
        # Should execute moves quickly
        assert move_time < 1.0  # Less than 1 second for 10000 moves
        moves_per_second = 10000 / move_time
        assert moves_per_second > 5000  # At least 5000 moves per second
    
    @pytest.mark.performance
    def test_win_detection_speed(self, empty_game):
        """Test speed of win detection algorithms."""
        game = empty_game
        
        # Set up various board states for testing
        test_cases = []
        
        # Empty board
        test_cases.append((game.board.copy(), 3))
        
        # Horizontal win
        game.board[5, 0:4] = 1
        test_cases.append((game.board.copy(), 3))
        
        # Vertical win
        game.reset()
        game.board[2:6, 3] = 1
        test_cases.append((game.board.copy(), 3))
        
        # Diagonal win
        game.reset()
        for i in range(4):
            game.board[5-i, i] = 1
        test_cases.append((game.board.copy(), 3))
        
        # Time win detection
        start_time = time.time()
        
        iterations = 10000
        for _ in range(iterations):
            for board, col in test_cases:
                game.board = board.copy()
                game.current_player = 1
                game.check_win(col)
        
        end_time = time.time()
        detection_time = end_time - start_time
        
        # Should detect wins quickly
        total_checks = iterations * len(test_cases)
        checks_per_second = total_checks / detection_time
        assert checks_per_second > 50000  # At least 50k checks per second
    
    @pytest.mark.performance
    def test_full_game_simulation_speed(self):
        """Test speed of complete game simulations."""
        start_time = time.time()
        
        games_completed = 0
        target_time = 1.0  # 1 second
        
        while time.time() - start_time < target_time:
            game = Connect4Game()
            
            # Play random game until completion
            moves = 0
            while not game.game_over and moves < 42:
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    col = np.random.choice(valid_moves)
                    game.drop_piece(col)
                    
                    if game.check_win(col) or game.is_draw():
                        game.game_over = True
                
                moves += 1
            
            games_completed += 1
        
        games_per_second = games_completed / target_time
        
        # Should simulate many games per second
        assert games_per_second > 50  # At least 50 games per second
    
    @pytest.mark.performance
    def test_memory_usage_single_game(self):
        """Test memory usage of single games."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many games
        games = []
        for _ in range(1000):
            game = Connect4Game()
            games.append(game)
        
        memory_after_creation = process.memory_info().rss
        
        # Play games to test memory during operations
        for game in games[:100]:  # Test subset
            for _ in range(20):  # 20 moves per game
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    col = np.random.choice(valid_moves)
                    game.drop_piece(col)
        
        final_memory = process.memory_info().rss
        
        # Memory usage should be reasonable
        creation_memory = memory_after_creation - initial_memory
        operation_memory = final_memory - memory_after_creation
        
        # Each game should use less than 1KB
        memory_per_game = creation_memory / 1000
        assert memory_per_game < 1024  # Less than 1KB per game
        
        # Operations shouldn't cause significant memory growth
        assert operation_memory < creation_memory * 0.5  # Less than 50% growth


class TestHybridVectorizedPerformance:
    """Performance benchmarks for vectorized environments."""
    
    @pytest.mark.performance
    def test_vectorized_initialization_speed(self, cpu_device):
        """Test speed of vectorized environment initialization."""
        env_counts = [10, 50, 100, 500]
        
        for num_envs in env_counts:
            start_time = time.time()
            env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
            end_time = time.time()
            
            init_time = end_time - start_time
            time_per_env = init_time / num_envs
            
            # Initialization should be efficient
            assert time_per_env < 0.001  # Less than 1ms per environment
    
    @pytest.mark.performance
    def test_batch_operations_speed(self, cpu_device):
        """Test speed of batch operations."""
        num_envs = 100
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        # Test observation batching speed
        start_time = time.time()
        for _ in range(1000):
            observations = env.get_observations_gpu()
        obs_time = time.time() - start_time
        
        # Test valid moves batching speed
        start_time = time.time()
        for _ in range(1000):
            valid_moves = env.get_valid_moves_batch()
        valid_moves_time = time.time() - start_time
        
        # Test step batching speed
        start_time = time.time()
        for _ in range(1000):
            actions = np.random.randint(0, 7, size=num_envs)
            rewards, dones, info = env.step_batch(actions)
        step_time = time.time() - start_time
        
        # Performance assertions
        obs_per_second = 1000 / obs_time
        valid_moves_per_second = 1000 / valid_moves_time
        steps_per_second = 1000 / step_time
        
        assert obs_per_second > 500  # At least 500 obs/sec
        assert valid_moves_per_second > 1000  # At least 1000 valid_moves/sec
        assert steps_per_second > 200  # At least 200 steps/sec
    
    @pytest.mark.performance
    def test_scalability_performance(self, cpu_device):
        """Test how performance scales with environment count."""
        env_counts = [10, 50, 100, 200]
        results = {}
        
        for num_envs in env_counts:
            env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
            
            # Time batch step operations
            start_time = time.time()
            iterations = 100
            
            for _ in range(iterations):
                actions = np.random.randint(0, 7, size=num_envs)
                rewards, dones, info = env.step_batch(actions)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            steps_per_second = (iterations * num_envs) / total_time
            time_per_env = total_time / (iterations * num_envs)
            
            results[num_envs] = {
                'steps_per_second': steps_per_second,
                'time_per_env': time_per_env,
                'total_time': total_time
            }
        
        # Check scalability
        # Steps per second should increase with more environments
        assert results[200]['steps_per_second'] > results[10]['steps_per_second']
        
        # Time per environment should remain roughly constant
        time_ratios = []
        base_time = results[10]['time_per_env']
        for num_envs in env_counts[1:]:
            ratio = results[num_envs]['time_per_env'] / base_time
            time_ratios.append(ratio)
        
        # Scaling should be reasonable (not exponential)
        assert all(ratio < 3.0 for ratio in time_ratios)  # Less than 3x slowdown
    
    @pytest.mark.performance
    def test_memory_efficiency_vectorized(self, cpu_device):
        """Test memory efficiency of vectorized environments."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        env_counts = [50, 100, 200, 500]
        memory_usage = {}
        
        for num_envs in env_counts:
            # Create environment
            env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
            
            # Perform operations to test memory usage
            for _ in range(50):
                observations = env.get_observations_gpu()
                actions = np.random.randint(0, 7, size=num_envs)
                rewards, dones, info = env.step_batch(actions)
                
                # Reset some environments periodically
                if np.any(dones):
                    reset_envs = np.where(dones)[0].tolist()
                    env.reset(env_ids=reset_envs)
            
            current_memory = process.memory_info().rss
            memory_usage[num_envs] = current_memory - initial_memory
        
        # Check memory scaling
        for num_envs in env_counts:
            memory_per_env = memory_usage[num_envs] / num_envs
            # Each environment should use reasonable memory
            assert memory_per_env < 10 * 1024  # Less than 10KB per environment
        
        # Memory should scale roughly linearly
        memory_ratio = memory_usage[500] / memory_usage[50]
        env_ratio = 500 / 50
        
        # Memory scaling should be reasonable (not much worse than linear)
        assert memory_ratio < env_ratio * 2  # Less than 2x worse than linear
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_large_scale_performance(self, cpu_device):
        """Test performance with large number of environments."""
        large_env_counts = [500, 1000]
        
        for num_envs in large_env_counts:
            start_time = time.time()
            
            # Create large environment
            env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
            
            creation_time = time.time() - start_time
            
            # Test operations
            operation_start = time.time()
            
            total_steps = 0
            for episode in range(10):  # Limited episodes for large scale
                step_count = 0
                while step_count < 20:  # Limited steps per episode
                    actions = np.random.randint(0, 7, size=num_envs)
                    rewards, dones, info = env.step_batch(actions)
                    
                    # Auto-reset finished games
                    env.auto_reset_finished_games()
                    
                    step_count += 1
                    total_steps += num_envs
            
            operation_time = time.time() - operation_start
            
            # Performance assertions for large scale
            assert creation_time < 5.0  # Less than 5 seconds to create
            
            steps_per_second = total_steps / operation_time
            assert steps_per_second > 1000  # At least 1000 env steps per second
    
    @pytest.mark.performance
    def test_gpu_vs_cpu_performance(self):
        """Test performance comparison between GPU and CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for GPU performance testing")
        
        num_envs = 100
        operations = 100
        
        # Test CPU performance
        cpu_env = HybridVectorizedConnect4(num_envs=num_envs, device=torch.device('cpu'))
        
        start_time = time.time()
        for _ in range(operations):
            observations = cpu_env.get_observations_gpu()
            actions = np.random.randint(0, 7, size=num_envs)
            rewards, dones, info = cpu_env.step_batch(actions)
        cpu_time = time.time() - start_time
        
        # Test GPU performance
        gpu_env = HybridVectorizedConnect4(num_envs=num_envs, device=torch.device('cuda'))
        
        start_time = time.time()
        for _ in range(operations):
            observations = gpu_env.get_observations_gpu()
            actions = np.random.randint(0, 7, size=num_envs)
            rewards, dones, info = gpu_env.step_batch(actions)
        gpu_time = time.time() - start_time
        
        # Both should complete in reasonable time
        assert cpu_time < 10.0  # Less than 10 seconds on CPU
        assert gpu_time < 10.0  # Less than 10 seconds on GPU
        
        # GPU might be faster for large batches, but both should work
        # Note: For small operations, CPU might actually be faster due to GPU overhead


class TestEnvironmentStressTests:
    """Stress tests for environment robustness."""
    
    @pytest.mark.slow
    def test_long_running_stability(self, small_vectorized_env):
        """Test stability over long-running operations."""
        env = small_vectorized_env
        
        total_steps = 0
        total_games = 0
        start_time = time.time()
        target_duration = 5.0  # 5 seconds
        
        while time.time() - start_time < target_duration:
            # Random actions
            valid_moves = env.get_valid_moves_batch()
            actions = []
            
            for vm_list in valid_moves:
                if vm_list:
                    actions.append(np.random.choice(vm_list))
                else:
                    actions.append(0)  # Will be invalid, but should be handled
            
            actions = np.array(actions)
            rewards, dones, info = env.step_batch(actions)
            
            # Reset finished games
            reset_envs = env.auto_reset_finished_games()
            total_games += len(reset_envs)
            total_steps += env.num_envs
            
            # Verify environment integrity periodically
            if total_steps % 100 == 0:
                observations = env.get_observations_gpu()
                assert observations.shape == (env.num_envs, 6, 7)
                assert torch.all(torch.abs(observations) <= 1)  # Valid values
        
        # Should have processed many steps without errors
        assert total_steps > 1000
        assert total_games > 0
    
    @pytest.mark.performance
    def test_memory_leak_detection(self, cpu_device):
        """Test for memory leaks in long-running operations."""
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        initial_memory = process.memory_info().rss
        
        # Run operations that might cause leaks
        for cycle in range(5):
            env = HybridVectorizedConnect4(num_envs=100, device=cpu_device)
            
            # Perform many operations
            for _ in range(100):
                observations = env.get_observations_gpu()
                actions = np.random.randint(0, 7, size=100)
                rewards, dones, info = env.step_batch(actions)
                env.auto_reset_finished_games()
            
            # Delete environment
            del env
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Check memory after each cycle
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # Memory growth should be bounded
            assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth
    
    @pytest.mark.performance
    def test_concurrent_environment_performance(self, cpu_device):
        """Test performance with multiple concurrent environments."""
        import threading
        import queue
        
        num_threads = 4
        num_envs_per_thread = 25
        results_queue = queue.Queue()
        
        def worker_thread(thread_id):
            env = HybridVectorizedConnect4(num_envs=num_envs_per_thread, device=cpu_device)
            
            start_time = time.time()
            steps_completed = 0
            
            for _ in range(50):  # 50 operations per thread
                actions = np.random.randint(0, 7, size=num_envs_per_thread)
                rewards, dones, info = env.step_batch(actions)
                env.auto_reset_finished_games()
                steps_completed += num_envs_per_thread
            
            end_time = time.time()
            results_queue.put({
                'thread_id': thread_id,
                'time': end_time - start_time,
                'steps': steps_completed
            })
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            thread.start()
            threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Collect results
        total_steps = 0
        max_thread_time = 0
        
        for _ in range(num_threads):
            result = results_queue.get()
            total_steps += result['steps']
            max_thread_time = max(max_thread_time, result['time'])
        
        # Performance assertions
        steps_per_second = total_steps / total_time
        assert steps_per_second > 500  # At least 500 steps/sec total
        assert max_thread_time < 5.0  # Each thread completes in reasonable time


class TestEnvironmentBenchmarkSuite:
    """Comprehensive benchmark suite for performance comparison."""
    
    @pytest.mark.performance
    def test_benchmark_suite(self, benchmark_setup, cpu_device):
        """Run comprehensive benchmark suite."""
        results = {}
        
        # Single game benchmarks
        single_game_start = time.time()
        game = Connect4Game()
        
        games_completed = 0
        while time.time() - single_game_start < 1.0:
            game.reset()
            moves = 0
            while not game.game_over and moves < 42:
                valid_moves = game.get_valid_moves()
                if valid_moves:
                    col = np.random.choice(valid_moves)
                    game.drop_piece(col)
                    if game.check_win(col) or game.is_draw():
                        game.game_over = True
                moves += 1
            games_completed += 1
        
        results['single_game_per_sec'] = games_completed
        
        # Vectorized environment benchmarks
        for num_envs in [10, 50, 100]:
            env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
            
            start_time = time.time()
            total_env_steps = 0
            
            while time.time() - start_time < 1.0:
                actions = np.random.randint(0, 7, size=num_envs)
                rewards, dones, info = env.step_batch(actions)
                env.auto_reset_finished_games()
                total_env_steps += num_envs
            
            results[f'vectorized_{num_envs}_steps_per_sec'] = total_env_steps
        
        # Print benchmark results for analysis
        print(f"\nBenchmark Results:")
        for key, value in results.items():
            print(f"{key}: {value:.2f}")
        
        # Assertions for minimum performance
        assert results['single_game_per_sec'] > 10
        assert results['vectorized_100_steps_per_sec'] > 1000
    
    def test_performance_regression_detection(self, cpu_device):
        """Test to detect performance regressions."""
        # This test can be extended to compare against historical benchmarks
        # For now, it ensures basic performance thresholds are met
        
        num_envs = 50
        env = HybridVectorizedConnect4(num_envs=num_envs, device=cpu_device)
        
        # Baseline performance test
        start_time = time.time()
        operations = 100
        
        for _ in range(operations):
            observations = env.get_observations_gpu()
            actions = np.random.randint(0, 7, size=num_envs)
            rewards, dones, info = env.step_batch(actions)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        ops_per_second = operations / total_time
        
        # Performance regression threshold
        # Adjust these values based on expected performance on target hardware
        assert ops_per_second > 50  # At least 50 operations per second
        
        # Memory usage check
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss
        
        # Memory usage should be reasonable
        # This is a basic check - more sophisticated memory profiling could be added
        assert current_memory < 1024 * 1024 * 1024  # Less than 1GB total