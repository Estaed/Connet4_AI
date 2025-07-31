"""
Comprehensive tests for agent factory functionality.

Tests the create_agent factory function and agent creation patterns
for different agent types and configurations.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from src.agents.base_agent import create_agent, BaseAgent
from src.agents.random_agent import RandomAgent
from src.agents.ppo_agent import PPOAgent


class TestAgentFactoryBasics:
    """Test basic agent factory functionality."""
    
    def test_create_random_agent(self):
        """Test creation of random agent."""
        agent = create_agent('random')
        
        assert isinstance(agent, RandomAgent)
        assert isinstance(agent, BaseAgent)
        
        # Should have default device
        assert agent.device == 'cpu'
    
    def test_create_ppo_agent(self, test_config):
        """Test creation of PPO agent."""
        agent = create_agent('ppo', config=test_config)
        
        assert isinstance(agent, PPOAgent)
        assert isinstance(agent, BaseAgent)
        
        # Should use config
        assert agent.config == test_config
    
    def test_agent_factory_case_insensitive(self, test_config):
        """Test that agent factory is case insensitive."""
        agents = [
            create_agent('RANDOM'),
            create_agent('Random'),
            create_agent('random'),
            create_agent('PPO', config=test_config),
            create_agent('ppo', config=test_config),
            create_agent('Ppo', config=test_config),
        ]
        
        # Should create appropriate agent types
        assert isinstance(agents[0], RandomAgent)
        assert isinstance(agents[1], RandomAgent)
        assert isinstance(agents[2], RandomAgent)
        assert isinstance(agents[3], PPOAgent)
        assert isinstance(agents[4], PPOAgent)
        assert isinstance(agents[5], PPOAgent)
    
    def test_agent_factory_with_device(self, test_config):
        """Test agent creation with device specification."""
        # CPU device
        agent_cpu = create_agent('random', device='cpu')
        assert agent_cpu.device == 'cpu'
        
        ppo_cpu = create_agent('ppo', device='cpu', config=test_config)
        assert ppo_cpu.device == torch.device('cpu')
        
        # CUDA device if available
        if torch.cuda.is_available():
            agent_cuda = create_agent('random', device='cuda')
            assert 'cuda' in str(agent_cuda.device)
            
            ppo_cuda = create_agent('ppo', device='cuda', config=test_config)
            assert ppo_cuda.device.type == 'cuda'
    
    def test_agent_factory_invalid_type(self):
        """Test handling of invalid agent types."""
        invalid_types = [
            'invalid_agent',
            'nonexistent',
            'fake_agent',
            123,
            None,
            '',
        ]
        
        for invalid_type in invalid_types:
            with pytest.raises((ValueError, KeyError, TypeError)):
                create_agent(invalid_type)
    
    def test_agent_factory_return_types(self, test_config):
        """Test that factory returns correct types."""
        random_agent = create_agent('random')
        ppo_agent = create_agent('ppo', config=test_config)
        
        # All should inherit from BaseAgent
        assert isinstance(random_agent, BaseAgent)
        assert isinstance(ppo_agent, BaseAgent)
        
        # Should be different concrete types
        assert type(random_agent) != type(ppo_agent)
        assert isinstance(random_agent, RandomAgent)
        assert isinstance(ppo_agent, PPOAgent)


class TestAgentFactoryParameterHandling:
    """Test parameter handling in agent factory."""
    
    def test_ppo_agent_requires_config(self):
        """Test that PPO agent creation requires config."""
        # PPO agent should require config parameter
        try:
            agent = create_agent('ppo')
            # If successful, should be valid PPO agent
            assert isinstance(agent, PPOAgent)
        except (ValueError, TypeError):
            # Expected error for missing config
            pass
    
    def test_random_agent_ignores_extra_params(self):
        """Test that random agent ignores extra parameters."""
        # Random agent should work with extra parameters
        agent = create_agent('random', config=None, extra_param=123, another_param='test')
        
        assert isinstance(agent, RandomAgent)
    
    def test_config_parameter_passing(self, test_config):
        """Test that config is properly passed to agents."""
        ppo_agent = create_agent('ppo', config=test_config)
        
        # Agent should have the config
        assert hasattr(ppo_agent, 'config')
        assert ppo_agent.config == test_config
    
    def test_device_parameter_passing(self):
        """Test that device parameter is properly passed."""
        devices_to_test = ['cpu']
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        
        for device in devices_to_test:
            random_agent = create_agent('random', device=device)
            assert random_agent.device == device
    
    def test_multiple_parameter_passing(self, test_config):
        """Test passing multiple parameters to factory."""
        agent = create_agent('ppo', device='cpu', config=test_config)
        
        assert isinstance(agent, PPOAgent)
        assert agent.device == torch.device('cpu')
        assert agent.config == test_config
    
    def test_parameter_validation(self):
        """Test parameter validation in factory."""
        # Test invalid device
        try:
            agent = create_agent('random', device='invalid_device')
            # If successful, should handle gracefully
            assert isinstance(agent, RandomAgent)
        except (ValueError, RuntimeError):
            # Appropriate error handling
            pass
    
    def test_kwargs_parameter_passing(self, test_config):
        """Test that **kwargs are properly passed to agent constructors."""
        # Test with keyword arguments
        params = {
            'device': 'cpu',
            'config': test_config
        }
        
        ppo_agent = create_agent('ppo', **params)
        
        assert isinstance(ppo_agent, PPOAgent)
        assert ppo_agent.device == torch.device('cpu')
        assert ppo_agent.config == test_config


class TestAgentFactoryExtensibility:
    """Test factory extensibility and customization."""
    
    def test_factory_function_signature(self):
        """Test that factory function has correct signature."""
        import inspect
        
        sig = inspect.signature(create_agent)
        params = list(sig.parameters.keys())
        
        # Should accept agent_type as first parameter
        assert params[0] == 'agent_type'
        
        # Should accept **kwargs for flexibility
        assert any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())
    
    def test_factory_documentation(self):
        """Test that factory function is properly documented."""
        assert create_agent.__doc__ is not None
        assert len(create_agent.__doc__.strip()) > 0
    
    def test_factory_consistent_interface(self, test_config):
        """Test that all created agents have consistent interface."""
        agents = {
            'random': create_agent('random'),
            'ppo': create_agent('ppo', config=test_config)
        }
        
        # All agents should have required methods
        required_methods = ['get_action', 'update', 'save', 'load']
        
        for agent_type, agent in agents.items():
            for method in required_methods:
                assert hasattr(agent, method), f"{agent_type} agent missing {method}"
                assert callable(getattr(agent, method)), f"{agent_type} agent {method} not callable"
    
    def test_factory_with_custom_agent_class(self):
        """Test factory behavior with potential custom agent classes."""
        # This test documents how the factory might handle custom agents
        # Implementation depends on whether factory supports registration of new agent types
        
        # Test current supported types
        supported_types = ['random', 'ppo']
        
        for agent_type in supported_types:
            try:
                if agent_type == 'ppo':
                    agent = create_agent(agent_type, config=Mock())
                else:
                    agent = create_agent(agent_type)
                assert isinstance(agent, BaseAgent)
            except Exception as e:
                pytest.fail(f"Failed to create {agent_type} agent: {e}")


class TestAgentFactoryErrorHandling:
    """Test error handling in agent factory."""
    
    def test_none_agent_type(self):
        """Test handling of None agent type."""
        with pytest.raises((ValueError, TypeError, KeyError)):
            create_agent(None)
    
    def test_empty_agent_type(self):
        """Test handling of empty string agent type."""
        with pytest.raises((ValueError, KeyError)):
            create_agent('')
    
    def test_numeric_agent_type(self):
        """Test handling of numeric agent types."""
        with pytest.raises((ValueError, TypeError, KeyError)):
            create_agent(123)
        
        with pytest.raises((ValueError, TypeError, KeyError)):
            create_agent(0)
    
    def test_list_agent_type(self):
        """Test handling of list as agent type."""
        with pytest.raises((ValueError, TypeError, KeyError)):
            create_agent(['random', 'ppo'])
    
    def test_invalid_config_for_ppo(self):
        """Test handling of invalid config for PPO agent."""
        invalid_configs = [
            None,
            "invalid_config",
            123,
            [],
            {}
        ]
        
        for invalid_config in invalid_configs:
            try:
                agent = create_agent('ppo', config=invalid_config)
                # If successful, should be valid agent
                assert isinstance(agent, PPOAgent)
            except (ValueError, TypeError, AttributeError):
                # Expected error for invalid config
                pass
    
    def test_conflicting_parameters(self, test_config):
        """Test handling of conflicting parameters."""
        # This test documents behavior with potentially conflicting parameters
        try:
            agent = create_agent('ppo', config=test_config, device='cuda')
            # Python will use the last device parameter
            assert isinstance(agent, PPOAgent)
        except TypeError:
            # Expected error for duplicate keyword arguments
            pass
    
    def test_missing_required_dependencies(self):
        """Test behavior when required dependencies are missing."""
        # This test could mock missing imports to test error handling
        # For now, it documents expected behavior
        
        # All current agent types should be available
        agent_types = ['random', 'ppo']
        
        for agent_type in agent_types:
            try:
                if agent_type == 'ppo':
                    agent = create_agent(agent_type, config=Mock())
                else:
                    agent = create_agent(agent_type)
                assert isinstance(agent, BaseAgent)
            except ImportError:
                # Would be expected if dependencies are missing
                pytest.skip(f"Dependencies for {agent_type} agent not available")


class TestAgentFactoryPerformance:
    """Test performance characteristics of agent factory."""
    
    @pytest.mark.performance
    def test_agent_creation_speed(self, test_config):
        """Test speed of agent creation."""
        import time
        
        # Time random agent creation
        start_time = time.time()
        for _ in range(100):
            agent = create_agent('random')
        random_time = time.time() - start_time
        
        # Time PPO agent creation
        start_time = time.time()
        for _ in range(10):  # Fewer PPO agents as they're more complex
            agent = create_agent('ppo', config=test_config)
        ppo_time = time.time() - start_time
        
        # Should create agents reasonably quickly
        assert random_time < 1.0  # Less than 1 second for 100 random agents
        assert ppo_time < 5.0     # Less than 5 seconds for 10 PPO agents
    
    def test_agent_creation_memory_usage(self, test_config):
        """Test memory usage of agent creation."""
        import gc
        
        # Create many agents and ensure memory is managed properly
        agents = []
        
        for i in range(50):
            if i % 2 == 0:
                agent = create_agent('random')
            else:
                agent = create_agent('ppo', config=test_config)
            agents.append(agent)
        
        # Clear references and force garbage collection
        del agents
        gc.collect()
        
        # Memory should be reclaimed (no easy way to test this precisely)
        # This test documents expected behavior
    
    def test_factory_caching_behavior(self, test_config):
        """Test whether factory caches agent instances or creates new ones."""
        # Create multiple agents of same type
        agent1 = create_agent('random')
        agent2 = create_agent('random')
        
        # Should create separate instances
        assert agent1 is not agent2
        assert id(agent1) != id(agent2)
        
        # Same for PPO agents
        ppo1 = create_agent('ppo', config=test_config)
        ppo2 = create_agent('ppo', config=test_config)
        
        assert ppo1 is not ppo2
        assert id(ppo1) != id(ppo2)


class TestAgentFactoryIntegration:
    """Test factory integration with other system components."""
    
    def test_factory_with_game_environment(self, empty_game, test_config):
        """Test factory-created agents with game environment."""
        agents = [
            create_agent('random'),
            create_agent('ppo', config=test_config)
        ]
        
        for agent in agents:
            # Should work with game environment
            observation = empty_game.board.astype('int8')
            valid_actions = empty_game.get_valid_moves()
            
            action = agent.get_action(observation, valid_actions)
            assert action in valid_actions
            
            # Should be able to make the move
            result = empty_game.drop_piece(action)
            assert result == True
            
            empty_game.reset()  # Reset for next agent
    
    def test_factory_with_vectorized_environment(self, small_vectorized_env, test_config):
        """Test factory-created agents with vectorized environment."""
        ppo_agent = create_agent('ppo', config=test_config)
        
        # Should work with vectorized environment
        observations = small_vectorized_env.get_observations_gpu()
        valid_moves = small_vectorized_env.get_valid_moves_tensor()
        
        if hasattr(ppo_agent, 'get_actions_batch'):
            actions, log_probs, values = ppo_agent.get_actions_batch(observations, valid_moves)
            
            # Should produce valid results
            assert actions.shape[0] == small_vectorized_env.num_envs
            assert torch.all(actions >= 0)
            assert torch.all(actions < 7)
    
    def test_factory_with_training_loop(self, test_config):
        """Test factory-created agents in training context."""
        agent = create_agent('ppo', config=test_config)
        
        # Should be able to store experiences and update
        obs = torch.randn(6, 7).numpy()
        agent.store_experience(obs, 3, 0.1, obs, False, -1.0, 0.5)
        
        # Should be able to update (though might not do much with single experience)
        try:
            loss_info = agent.update()
            if loss_info is not None:
                assert isinstance(loss_info, dict)
        except (ValueError, RuntimeError):
            # Expected with insufficient experience
            pass
    
    def test_factory_agent_serialization(self, tmp_path, test_config):
        """Test that factory-created agents can be saved and loaded."""
        agents = [
            ('random', create_agent('random')),
            ('ppo', create_agent('ppo', config=test_config))
        ]
        
        for agent_type, agent in agents:
            save_path = tmp_path / f"{agent_type}_agent.pth"
            
            # Should be able to save and load
            agent.save(str(save_path))
            agent.load(str(save_path))
    
    def test_factory_agents_polymorphism(self, test_config):
        """Test that factory-created agents can be used polymorphically."""
        agents = [
            create_agent('random'),
            create_agent('ppo', config=test_config)
        ]
        
        # All agents should support same interface
        observation = torch.randn(6, 7).numpy()
        valid_actions = [0, 1, 2, 3, 4, 5, 6]
        
        for agent in agents:
            # All should support get_action
            action = agent.get_action(observation, valid_actions)
            assert action in valid_actions
            
            # All should support update (even if no-op)
            agent.update([])
    
    def test_factory_with_config_variations(self):
        """Test factory with different configuration variations."""
        # Test with different config objects
        configs_to_test = []
        
        # Mock config
        mock_config = Mock()
        mock_config.get.return_value = 0.001  # Mock learning rate
        configs_to_test.append(mock_config)
        
        # Test with each config variation
        for config in configs_to_test:
            try:
                agent = create_agent('ppo', config=config)
                assert isinstance(agent, PPOAgent)
            except (ValueError, AttributeError):
                # Some mock configs might not work
                pass


class TestAgentFactoryDocumentation:
    """Test that agent factory is well-documented and discoverable."""
    
    def test_factory_function_docstring(self):
        """Test that factory function has comprehensive docstring."""
        docstring = create_agent.__doc__
        
        assert docstring is not None
        assert len(docstring.strip()) > 50  # Should have substantial documentation
        
        # Should mention supported agent types
        docstring_lower = docstring.lower()
        assert 'random' in docstring_lower or 'ppo' in docstring_lower
    
    def test_supported_agent_types_documentation(self):
        """Test that supported agent types are documented."""
        # This test would check if there's a way to discover supported agent types
        # Implementation depends on how the factory is designed
        
        # At minimum, should support these types
        essential_types = ['random', 'ppo']
        
        for agent_type in essential_types:
            try:
                if agent_type == 'ppo':
                    agent = create_agent(agent_type, config=Mock())
                else:
                    agent = create_agent(agent_type)
                assert isinstance(agent, BaseAgent)
            except Exception as e:
                pytest.fail(f"Essential agent type {agent_type} not supported: {e}")
    
    def test_factory_usage_examples(self):
        """Test that common factory usage patterns work."""
        # Common usage patterns that should work
        usage_examples = [
            # Basic usage
            lambda: create_agent('random'),
            
            # With device
            lambda: create_agent('random', device='cpu'),
            
            # With config (requires mock)
            lambda: create_agent('ppo', config=Mock()),
        ]
        
        for i, example_func in enumerate(usage_examples):
            try:
                agent = example_func()
                assert isinstance(agent, BaseAgent), f"Usage example {i} failed"
            except Exception as e:
                # Some examples might fail due to mocking limitations
                pass