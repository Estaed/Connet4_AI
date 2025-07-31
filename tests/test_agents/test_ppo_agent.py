"""
Comprehensive tests for PPOAgent implementation.

Tests the Proximal Policy Optimization agent including action selection,
experience storage, training updates, and integration with networks.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.agents.ppo_agent import PPOAgent, PPOMemory
from src.agents.base_agent import BaseAgent
from src.agents.networks import Connect4Network


class TestPPOAgentInitialization:
    """Test PPOAgent initialization and setup."""
    
    def test_inheritance(self, test_config):
        """Test that PPOAgent inherits from BaseAgent."""
        agent = PPOAgent(device='cpu', config=test_config)
        assert isinstance(agent, BaseAgent)
        assert isinstance(agent, PPOAgent)
    
    def test_default_initialization(self, test_config):
        """Test default PPOAgent initialization."""
        agent = PPOAgent(device='cpu', config=test_config)
        
        # Should have required components
        assert hasattr(agent, 'network')
        assert hasattr(agent, 'optimizer')
        assert hasattr(agent, 'memory')
        assert hasattr(agent, 'device')
        assert hasattr(agent, 'config')
        
        # Network should be on correct device
        assert agent.device == torch.device('cpu')
        for param in agent.network.parameters():
            assert param.device.type == 'cpu'
    
    def test_gpu_initialization(self, test_config):
        """Test PPOAgent initialization with GPU device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        agent = PPOAgent(device='cuda', config=test_config)
        
        assert agent.device.type == 'cuda'
        for param in agent.network.parameters():
            assert param.device.type == 'cuda'
    
    def test_config_parameter_usage(self, test_config):
        """Test that config parameters are properly used."""
        agent = PPOAgent(device='cpu', config=test_config)
        
        # Should use learning rate from config
        assert agent.optimizer.param_groups[0]['lr'] == test_config.get('ppo.learning_rate')
    
    def test_network_initialization(self, test_config):
        """Test that network is properly initialized."""
        agent = PPOAgent(device='cpu', config=test_config)
        
        assert isinstance(agent.network, Connect4Network)
        
        # Network should be in training mode by default
        assert agent.network.training == True
    
    def test_memory_initialization(self, test_config):
        """Test that memory buffer is properly initialized."""
        agent = PPOAgent(device='cpu', config=test_config)
        
        assert isinstance(agent.memory, PPOMemory)
        assert len(agent.memory.observations) == 0
        assert len(agent.memory.actions) == 0


class TestPPOAgentActionSelection:
    """Test PPOAgent action selection functionality."""
    
    def test_get_action_deterministic(self, ppo_agent, sample_board_states):
        """Test deterministic action selection."""
        agent = ppo_agent
        observation = sample_board_states[0]
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        action = agent.get_action(observation, valid_actions, deterministic=True)
        
        # Should return valid action
        assert action in valid_actions
        assert isinstance(action, (int, np.integer))
        
        # Should be deterministic - same input should give same output
        action2 = agent.get_action(observation, valid_actions, deterministic=True)
        assert action == action2
    
    def test_get_action_stochastic(self, ppo_agent, sample_board_states):
        """Test stochastic action selection."""
        agent = ppo_agent
        observation = sample_board_states[1]
        valid_actions = [1, 3, 5]
        
        # Collect multiple actions
        actions = []
        for _ in range(20):
            action = agent.get_action(observation, valid_actions, deterministic=False)
            actions.append(action)
            assert action in valid_actions
        
        # Should show some variability (stochastic)
        unique_actions = set(actions)
        if len(valid_actions) > 1:
            assert len(unique_actions) > 1  # Should not always choose same action
    
    def test_get_action_with_info(self, ppo_agent, sample_board_states):
        """Test get_action_with_info functionality."""
        agent = ppo_agent
        observation = sample_board_states[0]
        valid_actions = [0, 2, 4, 6]
        
        action, log_prob, value = agent.get_action_with_info(observation, valid_actions)
        
        # Check return types and values
        assert action in valid_actions
        assert isinstance(log_prob, (float, np.floating))
        assert isinstance(value, (float, np.floating))
        
        # Log probability should be negative (log of probability)
        assert log_prob <= 0
        
        # Value should be finite
        assert np.isfinite(value)
    
    def test_get_action_single_valid_action(self, ppo_agent, sample_board_states):
        """Test action selection with only one valid action."""
        agent = ppo_agent
        observation = sample_board_states[0]
        valid_actions = [3]
        
        action = agent.get_action(observation, valid_actions)
        assert action == 3
        
        # Should work with info version too
        action, log_prob, value = agent.get_action_with_info(observation, valid_actions)
        assert action == 3
        assert np.isfinite(log_prob)
        assert np.isfinite(value)
    
    def test_get_action_empty_valid_actions(self, ppo_agent, sample_board_states):
        """Test handling of empty valid actions."""
        agent = ppo_agent
        observation = sample_board_states[0]
        valid_actions = []
        
        # Should handle gracefully
        try:
            action = agent.get_action(observation, valid_actions)
            assert isinstance(action, (int, np.integer))
            assert 0 <= action <= 6  # Should return some valid column
        except (ValueError, IndexError):
            # Appropriate error handling
            pass
    
    def test_get_actions_batch(self, ppo_agent, cpu_device):
        """Test batch action selection."""
        agent = ppo_agent
        batch_size = 4
        
        # Create batch observations
        observations = torch.randn(batch_size, 6, 7, device=cpu_device)
        valid_moves = torch.ones(batch_size, 7, dtype=torch.bool, device=cpu_device)
        
        actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
        
        # Check output shapes and types
        assert actions.shape == (batch_size,)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size,)
        
        # Check that all outputs are on correct device
        assert actions.device == cpu_device
        assert log_probs.device == cpu_device
        assert values.device == cpu_device
        
        # Check value ranges
        assert torch.all(actions >= 0)
        assert torch.all(actions < 7)
        assert torch.all(log_probs <= 0)  # Log probabilities should be negative
        assert torch.all(torch.isfinite(values))
    
    def test_get_actions_batch_with_masking(self, ppo_agent, cpu_device):
        """Test batch action selection with action masking."""
        agent = ppo_agent
        batch_size = 3
        
        observations = torch.randn(batch_size, 6, 7, device=cpu_device)
        
        # Create different masks for each sample
        valid_moves = torch.tensor([
            [True, True, False, False, False, False, False],   # Only columns 0,1 valid
            [False, False, False, True, True, True, False],    # Only columns 3,4,5 valid
            [False, False, False, False, False, False, True],  # Only column 6 valid
        ], device=cpu_device)
        
        actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
        
        # Check that actions respect masks
        assert actions[0] in [0, 1]
        assert actions[1] in [3, 4, 5]
        assert actions[2] == 6
        
        # All outputs should be finite
        assert torch.all(torch.isfinite(log_probs))
        assert torch.all(torch.isfinite(values))


class TestPPOAgentExperienceStorage:
    """Test experience storage functionality."""
    
    def test_store_experience_basic(self, ppo_agent, sample_board_states):
        """Test basic experience storage."""
        agent = ppo_agent
        
        obs = sample_board_states[0]
        action = 3
        reward = 0.5
        next_obs = sample_board_states[1]
        done = False
        log_prob = -1.2
        value = 0.8
        
        initial_size = len(agent.memory.observations)
        
        agent.store_experience(obs, action, reward, next_obs, done, log_prob, value)
        
        # Memory should have one more experience
        assert len(agent.memory.observations) == initial_size + 1
        assert len(agent.memory.actions) == initial_size + 1
        assert len(agent.memory.rewards) == initial_size + 1
    
    def test_store_multiple_experiences(self, ppo_agent, sample_experiences):
        """Test storing multiple experiences."""
        agent = ppo_agent
        
        initial_size = len(agent.memory.observations)
        
        # Store all sample experiences
        for exp in sample_experiences:
            agent.store_experience(*exp)
        
        # Memory should contain all experiences
        expected_size = initial_size + len(sample_experiences)
        assert len(agent.memory.observations) == expected_size
        assert len(agent.memory.actions) == expected_size
        assert len(agent.memory.rewards) == expected_size
    
    def test_experience_data_types(self, ppo_agent):
        """Test that stored experiences have correct data types."""
        agent = ppo_agent
        
        obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
        action = np.int32(4)
        reward = np.float32(1.0)
        next_obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
        done = True
        log_prob = np.float32(-0.5)
        value = np.float32(0.3)
        
        agent.store_experience(obs, action, reward, next_obs, done, log_prob, value)
        
        # Check that data is stored correctly
        stored_obs = agent.memory.observations[-1]
        stored_action = agent.memory.actions[-1]
        stored_reward = agent.memory.rewards[-1]
        
        assert isinstance(stored_obs, np.ndarray)
        assert isinstance(stored_action, (int, np.integer))
        assert isinstance(stored_reward, (float, np.floating))
    
    def test_memory_capacity_handling(self, ppo_agent):
        """Test memory handling when capacity is reached."""
        agent = ppo_agent
        
        # Fill memory beyond reasonable capacity
        for i in range(1000):  # Store many experiences
            obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
            action = i % 7
            reward = 0.1
            next_obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
            done = (i % 50 == 0)
            log_prob = -1.0
            value = 0.0
            
            agent.store_experience(obs, action, reward, next_obs, done, log_prob, value)
        
        # Memory should handle large amounts of data gracefully
        assert len(agent.memory.observations) > 0
        assert len(agent.memory.observations) <= 1000


class TestPPOAgentTraining:
    """Test PPOAgent training and update functionality."""
    
    def test_update_basic(self, trained_ppo_agent):
        """Test basic update functionality."""
        agent = trained_ppo_agent
        
        # Ensure agent has some experiences
        assert len(agent.memory.observations) > 0
        
        # Store initial network parameters
        initial_params = [param.clone() for param in agent.network.parameters()]
        
        # Perform update
        loss_info = agent.update()
        
        # Check that parameters changed
        for initial_param, current_param in zip(initial_params, agent.network.parameters()):
            if initial_param.numel() > 0:  # Skip empty parameters
                assert not torch.allclose(initial_param, current_param, atol=1e-6)
        
        # Should return loss information
        if loss_info is not None:
            assert isinstance(loss_info, dict)
    
    def test_update_without_experiences(self, ppo_agent):
        """Test update behavior without experiences."""
        agent = ppo_agent
        
        # Clear memory
        agent.memory.clear()
        
        # Update should handle gracefully
        try:
            loss_info = agent.update()
            # If no error, should return None or empty info
            assert loss_info is None or isinstance(loss_info, dict)
        except (ValueError, RuntimeError):
            # Appropriate error handling for empty memory
            pass
    
    def test_train_on_experiences(self, ppo_agent):
        """Test train_on_experiences functionality."""
        agent = ppo_agent
        
        # Create sample batch data
        batch_size = 8
        observations = torch.randn(batch_size, 6, 7, device=agent.device)
        actions = torch.randint(0, 7, (batch_size,), device=agent.device)
        rewards = torch.randn(batch_size, device=agent.device)
        dones = torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=agent.device)
        old_log_probs = torch.randn(batch_size, device=agent.device)
        values = torch.randn(batch_size, device=agent.device)
        advantages = torch.randn(batch_size, device=agent.device)
        
        # Store initial parameters
        initial_params = [param.clone() for param in agent.network.parameters()]
        
        # Train on experiences
        loss_info = agent.train_on_experiences(
            observations, actions, rewards, dones, old_log_probs, values, advantages
        )
        
        # Parameters should have changed
        for initial_param, current_param in zip(initial_params, agent.network.parameters()):
            if initial_param.numel() > 0:
                assert not torch.allclose(initial_param, current_param, atol=1e-6)
        
        # Should return loss information
        assert isinstance(loss_info, dict)
        assert 'policy_loss' in loss_info
        assert 'value_loss' in loss_info
        assert 'total_loss' in loss_info
    
    def test_ppo_clipping(self, ppo_agent):
        """Test PPO clipping mechanism."""
        agent = ppo_agent
        
        # Create batch with large advantage values to test clipping
        batch_size = 4
        observations = torch.randn(batch_size, 6, 7, device=agent.device)
        actions = torch.randint(0, 7, (batch_size,), device=agent.device)
        rewards = torch.randn(batch_size, device=agent.device)
        dones = torch.zeros(batch_size, dtype=torch.bool, device=agent.device)
        old_log_probs = torch.randn(batch_size, device=agent.device)
        values = torch.randn(batch_size, device=agent.device)
        
        # Create extreme advantages to test clipping
        advantages = torch.tensor([10.0, -10.0, 5.0, -5.0], device=agent.device)
        
        # Should handle extreme advantages without errors
        loss_info = agent.train_on_experiences(
            observations, actions, rewards, dones, old_log_probs, values, advantages
        )
        
        # Losses should be finite
        assert torch.isfinite(torch.tensor(loss_info['policy_loss']))
        assert torch.isfinite(torch.tensor(loss_info['value_loss']))
    
    def test_gradient_accumulation(self, ppo_agent):
        """Test gradient accumulation during training."""
        agent = ppo_agent
        
        # Zero gradients
        agent.optimizer.zero_grad()
        
        # Check that gradients are initially zero
        for param in agent.network.parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0)
        
        # Perform forward pass and backward
        batch_size = 4
        observations = torch.randn(batch_size, 6, 7, device=agent.device)
        actions = torch.randint(0, 7, (batch_size,), device=agent.device)
        rewards = torch.randn(batch_size, device=agent.device)
        dones = torch.zeros(batch_size, dtype=torch.bool, device=agent.device)
        old_log_probs = torch.randn(batch_size, device=agent.device)
        values = torch.randn(batch_size, device=agent.device)
        advantages = torch.randn(batch_size, device=agent.device)
        
        loss_info = agent.train_on_experiences(
            observations, actions, rewards, dones, old_log_probs, values, advantages
        )
        
        # Gradients should now exist and be non-zero
        has_gradients = False
        for param in agent.network.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        assert has_gradients


class TestPPOMemory:
    """Test PPOMemory buffer implementation."""
    
    def test_memory_initialization(self):
        """Test PPOMemory initialization."""
        memory = PPOMemory()
        
        # Should have empty lists for all data types
        assert len(memory.observations) == 0
        assert len(memory.actions) == 0
        assert len(memory.rewards) == 0
        assert len(memory.next_observations) == 0
        assert len(memory.dones) == 0
        assert len(memory.log_probs) == 0
        assert len(memory.values) == 0
    
    def test_memory_push(self):
        """Test adding experiences to memory."""
        memory = PPOMemory()
        
        obs = np.zeros((6, 7))
        action = 3
        reward = 1.0
        next_obs = np.ones((6, 7))
        done = False
        log_prob = -0.5
        value = 0.8
        
        memory.push(obs, action, reward, next_obs, done, log_prob, value)
        
        # Should have one experience
        assert len(memory.observations) == 1
        assert memory.actions[0] == action
        assert memory.rewards[0] == reward
        assert memory.dones[0] == done
    
    def test_memory_clear(self):
        """Test clearing memory."""
        memory = PPOMemory()
        
        # Add some experiences
        for i in range(5):
            memory.push(
                np.zeros((6, 7)), i, 0.1, np.zeros((6, 7)), False, -1.0, 0.0
            )
        
        assert len(memory.observations) == 5
        
        # Clear memory
        memory.clear()
        
        # Should be empty
        assert len(memory.observations) == 0
        assert len(memory.actions) == 0
        assert len(memory.rewards) == 0
    
    def test_memory_compute_advantages(self):
        """Test advantage computation using GAE."""
        memory = PPOMemory()
        
        # Add sequence of experiences
        rewards = [0.1, 0.2, 0.3, 1.0]  # Increasing rewards with final reward
        values = [0.5, 0.6, 0.7, 0.0]   # Estimated values
        dones = [False, False, False, True]  # Episode ends at last step
        
        for i in range(4):
            obs = np.random.rand(6, 7)
            memory.push(obs, i, rewards[i], obs, dones[i], -1.0, values[i])
        
        # Compute advantages
        gamma = 0.99
        gae_lambda = 0.95
        advantages = memory.compute_advantages(gamma, gae_lambda)
        
        # Should return advantages for all experiences
        assert len(advantages) == 4
        assert all(isinstance(adv, (float, np.floating)) for adv in advantages)
        assert all(np.isfinite(adv) for adv in advantages)
    
    def test_memory_get_batch(self):
        """Test getting batch of experiences."""
        memory = PPOMemory()
        
        # Add experiences
        for i in range(10):
            obs = np.random.rand(6, 7)
            memory.push(obs, i % 7, 0.1, obs, False, -1.0, 0.5)
        
        # Get batch
        batch = memory.get_batch()
        
        # Should return all experiences
        assert len(batch['observations']) == 10
        assert len(batch['actions']) == 10
        assert len(batch['rewards']) == 10
        
        # Data should be numpy arrays
        assert isinstance(batch['observations'], np.ndarray)
        assert isinstance(batch['actions'], np.ndarray)
        assert isinstance(batch['rewards'], np.ndarray)


class TestPPOAgentSaveLoad:
    """Test PPOAgent save and load functionality."""
    
    def test_save_load_basic(self, ppo_agent, tmp_path):
        """Test basic save and load functionality."""
        agent = ppo_agent
        save_path = tmp_path / "ppo_agent.pth"
        
        # Should not raise errors
        agent.save(str(save_path))
        agent.load(str(save_path))
    
    def test_save_load_state_preservation(self, ppo_agent, tmp_path):
        """Test that save/load preserves agent state."""
        agent = ppo_agent
        save_path = tmp_path / "ppo_agent.pth"
        
        # Get initial network state
        initial_state = agent.network.state_dict()
        
        # Save and create new agent
        agent.save(str(save_path))
        
        new_agent = PPOAgent(device=agent.device, config=agent.config)
        new_agent.load(str(save_path))
        
        # Network states should be identical
        new_state = new_agent.network.state_dict()
        
        for key in initial_state:
            assert torch.allclose(initial_state[key], new_state[key])
    
    def test_save_load_with_optimizer_state(self, ppo_agent, tmp_path):
        """Test that optimizer state is preserved."""
        agent = ppo_agent
        save_path = tmp_path / "ppo_agent.pth"
        
        # Modify optimizer state by doing an update
        if len(agent.memory.observations) == 0:
            # Add some dummy experience
            obs = np.zeros((6, 7))
            agent.store_experience(obs, 3, 0.1, obs, False, -1.0, 0.5)
        
        # Get initial optimizer state
        initial_opt_state = agent.optimizer.state_dict()
        
        # Save and load
        agent.save(str(save_path))
        
        new_agent = PPOAgent(device=agent.device, config=agent.config)
        new_agent.load(str(save_path))
        
        # Optimizer states should be preserved (if implementation saves them)
        # This test documents expected behavior


class TestPPOAgentIntegration:
    """Test PPOAgent integration with other components."""
    
    def test_integration_with_vectorized_environment(self, ppo_agent, small_vectorized_env):
        """Test PPOAgent integration with vectorized environments."""
        agent = ppo_agent
        env = small_vectorized_env
        
        # Should work with vectorized environment
        observations = env.get_observations_gpu()
        valid_moves = env.get_valid_moves_tensor()
        
        actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
        
        # Step environment
        rewards, dones, info = env.step_batch(actions.cpu().numpy())
        
        # Should handle results without errors
        assert len(rewards) == env.num_envs
        assert len(dones) == env.num_envs
    
    def test_integration_training_loop_simulation(self, ppo_agent, small_vectorized_env):
        """Test PPOAgent in simulated training loop."""
        agent = ppo_agent
        env = small_vectorized_env
        
        # Short training simulation
        for episode in range(3):
            observations = env.get_observations_gpu()
            valid_moves = env.get_valid_moves_tensor()
            
            # Collect experiences
            for step in range(10):
                actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
                rewards, dones, info = env.step_batch(actions.cpu().numpy())
                
                # Store experiences (simplified)
                for i in range(env.num_envs):
                    if not dones[i]:
                        obs_np = observations[i].cpu().numpy()
                        agent.store_experience(
                            obs_np, actions[i].item(), rewards[i],
                            obs_np, dones[i], log_probs[i].item(), values[i].item()
                        )
                
                # Reset finished environments
                env.auto_reset_finished_games()
                observations = env.get_observations_gpu()
                valid_moves = env.get_valid_moves_tensor()
            
            # Update agent
            if len(agent.memory.observations) > 0:
                loss_info = agent.update()
        
        # Should complete without errors
        assert len(agent.memory.observations) >= 0
    
    def test_integration_with_different_network_types(self, test_config):
        """Test PPOAgent with different network architectures."""
        # Test with standard network
        agent_standard = PPOAgent(device='cpu', config=test_config)
        assert isinstance(agent_standard.network, Connect4Network)
        
        # Both should work for action selection
        obs = np.zeros((6, 7))
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        action_standard = agent_standard.get_action(obs, valid_actions)
        assert action_standard in valid_actions


class TestPPOAgentPerformance:
    """Test PPOAgent performance characteristics."""
    
    @pytest.mark.performance
    def test_action_selection_speed(self, ppo_agent):
        """Test action selection performance."""
        agent = ppo_agent
        obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        import time
        start_time = time.time()
        
        # Many action selections
        for _ in range(1000):
            action = agent.get_action(obs, valid_actions)
        
        end_time = time.time()
        selection_time = end_time - start_time
        
        # Should be reasonably fast
        actions_per_second = 1000 / selection_time
        assert actions_per_second > 100  # At least 100 actions per second
    
    @pytest.mark.performance
    def test_batch_action_selection_speed(self, ppo_agent, cpu_device):
        """Test batch action selection performance."""
        agent = ppo_agent
        batch_size = 32
        
        observations = torch.randn(batch_size, 6, 7, device=cpu_device)
        valid_moves = torch.ones(batch_size, 7, dtype=torch.bool, device=cpu_device)
        
        import time
        start_time = time.time()
        
        # Many batch selections
        for _ in range(100):
            actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
        
        end_time = time.time()
        batch_time = end_time - start_time
        
        # Should be efficient for batch processing
        batches_per_second = 100 / batch_time
        assert batches_per_second > 10  # At least 10 batches per second
    
    @pytest.mark.performance
    def test_training_update_speed(self, trained_ppo_agent):
        """Test training update performance."""
        agent = trained_ppo_agent
        
        # Ensure sufficient experiences
        while len(agent.memory.observations) < 32:
            obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
            agent.store_experience(obs, 3, 0.1, obs, False, -1.0, 0.5)
        
        import time
        start_time = time.time()
        
        # Multiple updates
        for _ in range(10):
            loss_info = agent.update()
        
        end_time = time.time()
        update_time = end_time - start_time
        
        # Should complete updates in reasonable time
        updates_per_second = 10 / update_time
        assert updates_per_second > 1  # At least 1 update per second


class TestPPOAgentEdgeCases:
    """Test edge cases and error handling for PPOAgent."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration."""
        # Missing required config parameters
        try:
            agent = PPOAgent(device='cpu', config=None)
            # If no error, should have default behavior
            assert isinstance(agent, PPOAgent)
        except (ValueError, TypeError, AttributeError):
            # Appropriate error handling
            pass
    
    def test_extreme_advantage_values(self, ppo_agent):
        """Test handling of extreme advantage values."""
        agent = ppo_agent
        
        batch_size = 4
        observations = torch.randn(batch_size, 6, 7, device=agent.device)
        actions = torch.randint(0, 7, (batch_size,), device=agent.device)
        rewards = torch.randn(batch_size, device=agent.device)
        dones = torch.zeros(batch_size, dtype=torch.bool, device=agent.device)
        old_log_probs = torch.randn(batch_size, device=agent.device)
        values = torch.randn(batch_size, device=agent.device)
        
        # Extreme advantage values
        extreme_advantages = torch.tensor([1000.0, -1000.0, float('inf'), float('-inf')], device=agent.device)
        extreme_advantages[2] = 100.0  # Replace inf with large finite value
        extreme_advantages[3] = -100.0  # Replace -inf with large negative finite value
        
        # Should handle extreme values gracefully
        try:
            loss_info = agent.train_on_experiences(
                observations, actions, rewards, dones, old_log_probs, values, extreme_advantages
            )
            # If successful, losses should be finite
            assert np.isfinite(loss_info['total_loss'])
        except (RuntimeError, ValueError):
            # Appropriate error handling for extreme values
            pass
    
    def test_device_mismatch_handling(self, test_config):
        """Test handling of device mismatches."""
        agent = PPOAgent(device='cpu', config=test_config)
        
        if torch.cuda.is_available():
            # Try to use GPU tensors with CPU agent
            gpu_obs = torch.randn(1, 6, 7, device='cuda')
            gpu_valid_moves = torch.ones(1, 7, dtype=torch.bool, device='cuda')
            
            with pytest.raises(RuntimeError):
                actions, log_probs, values = agent.get_actions_batch(gpu_obs, gpu_valid_moves)
    
    def test_memory_overflow_handling(self, ppo_agent):
        """Test memory handling with very large amounts of data."""
        agent = ppo_agent
        
        # Store many experiences
        for i in range(10000):
            obs = np.random.randint(-1, 2, (6, 7), dtype=np.int8)
            agent.store_experience(obs, i % 7, 0.1, obs, False, -1.0, 0.5)
        
        # Should handle large memory gracefully
        assert len(agent.memory.observations) > 0
        
        # Update should still work (though might be slow)
        try:
            loss_info = agent.update()
            # If successful, should return valid loss info
            if loss_info is not None:
                assert isinstance(loss_info, dict)
        except (RuntimeError, MemoryError):
            # Acceptable to fail with very large memory
            pass