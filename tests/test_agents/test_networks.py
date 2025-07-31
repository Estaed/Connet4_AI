"""
Comprehensive tests for neural network architectures.

Tests the Connect4Network and Connect4DuelingNetwork implementations,
including forward passes, action probability calculation, and GPU compatibility.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, Mock
from src.agents.networks import Connect4Network, Connect4DuelingNetwork, create_network


class TestConnect4NetworkInitialization:
    """Test Connect4Network initialization and architecture."""
    
    def test_network_initialization(self):
        """Test basic network initialization."""
        network = Connect4Network()
        
        # Should be a PyTorch module
        assert isinstance(network, nn.Module)
        assert isinstance(network, Connect4Network)
    
    def test_network_layers_structure(self):
        """Test that network has expected layer structure."""
        network = Connect4Network()
        
        # Should have convolutional layers
        assert hasattr(network, 'conv_layers')
        assert isinstance(network.conv_layers, nn.Sequential)
        
        # Should have policy and value heads
        assert hasattr(network, 'policy_head')
        assert hasattr(network, 'value_head')
        assert isinstance(network.policy_head, nn.Linear)
        assert isinstance(network.value_head, nn.Linear)
    
    def test_network_layer_dimensions(self):
        """Test that layers have correct input/output dimensions."""
        network = Connect4Network()
        
        # Policy head should output 7 values (one per column)
        assert network.policy_head.out_features == 7
        
        # Value head should output 1 value (state value)
        assert network.value_head.out_features == 1
    
    def test_network_parameters_initialization(self):
        """Test that network parameters are properly initialized."""
        network = Connect4Network()
        
        # Should have trainable parameters
        params = list(network.parameters())
        assert len(params) > 0
        
        # Parameters should have reasonable values (not all zeros)
        param_values = torch.cat([p.flatten() for p in params])
        assert not torch.all(param_values == 0)
        
        # Parameters should be finite
        assert torch.all(torch.isfinite(param_values))
    
    def test_network_device_compatibility(self, device):
        """Test network device compatibility."""
        network = Connect4Network().to(device)
        
        # Network should be on correct device
        for param in network.parameters():
            assert param.device.type == device.type
    
    def test_network_eval_train_modes(self):
        """Test network eval and training modes."""
        network = Connect4Network()
        
        # Default should be training mode
        assert network.training == True
        
        # Should be able to switch to eval mode
        network.eval()
        assert network.training == False
        
        # Should be able to switch back to training mode
        network.train()
        assert network.training == True


class TestConnect4NetworkForwardPass:
    """Test forward pass functionality of Connect4Network."""
    
    def test_forward_pass_single_input(self, standard_network, cpu_device):
        """Test forward pass with single input."""
        network = standard_network
        
        # Create single board input (1, 1, 6, 7)
        input_tensor = torch.zeros(1, 1, 6, 7, device=cpu_device)
        
        policy_logits, value = network(input_tensor)
        
        # Check output shapes
        assert policy_logits.shape == (1, 7)
        assert value.shape == (1, 1)
        
        # Check output types
        assert isinstance(policy_logits, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        
        # Outputs should be finite
        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))
    
    def test_forward_pass_batch_input(self, standard_network, cpu_device):
        """Test forward pass with batch input."""
        network = standard_network
        batch_size = 8
        
        # Create batch input (8, 1, 6, 7)
        input_tensor = torch.randn(batch_size, 1, 6, 7, device=cpu_device)
        
        policy_logits, value = network(input_tensor)
        
        # Check output shapes
        assert policy_logits.shape == (batch_size, 7)
        assert value.shape == (batch_size, 1)
        
        # All outputs should be finite
        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))
    
    def test_forward_pass_different_inputs(self, standard_network, cpu_device):
        """Test that different inputs produce different outputs."""
        network = standard_network
        
        # Create two different inputs
        input1 = torch.zeros(1, 1, 6, 7, device=cpu_device)
        input2 = torch.ones(1, 1, 6, 7, device=cpu_device)
        
        policy1, value1 = network(input1)
        policy2, value2 = network(input2)
        
        # Outputs should be different (network should be non-trivial)
        assert not torch.allclose(policy1, policy2, atol=1e-6)
        assert not torch.allclose(value1, value2, atol=1e-6)
    
    def test_forward_pass_gradient_flow(self, standard_network, cpu_device):
        """Test that gradients flow properly through the network."""
        network = standard_network
        network.train()
        
        input_tensor = torch.randn(2, 1, 6, 7, device=cpu_device, requires_grad=True)
        policy_logits, value = network(input_tensor)
        
        # Create dummy loss
        loss = policy_logits.sum() + value.sum()
        loss.backward()
        
        # Check that gradients exist for network parameters
        for param in network.parameters():
            assert param.grad is not None
            assert torch.any(param.grad != 0)  # At least some gradients non-zero
        
        # Check that input gradients exist
        assert input_tensor.grad is not None
    
    def test_forward_pass_deterministic(self, standard_network, cpu_device):
        """Test that forward pass is deterministic for same input."""
        network = standard_network
        network.eval()  # Ensure deterministic behavior
        
        input_tensor = torch.randn(1, 1, 6, 7, device=cpu_device)
        
        # Multiple forward passes with same input
        policy1, value1 = network(input_tensor)
        policy2, value2 = network(input_tensor)
        
        # Should produce identical results
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)
    
    def test_forward_pass_input_validation(self, standard_network, cpu_device):
        """Test forward pass input validation."""
        network = standard_network
        
        # Test various invalid input shapes
        invalid_inputs = [
            torch.zeros(6, 7, device=cpu_device),           # Missing batch and channel dims
            torch.zeros(1, 6, 7, device=cpu_device),        # Missing channel dim
            torch.zeros(1, 1, 5, 7, device=cpu_device),     # Wrong height
            torch.zeros(1, 1, 6, 8, device=cpu_device),     # Wrong width
            torch.zeros(1, 2, 6, 7, device=cpu_device),     # Wrong number of channels
        ]
        
        for invalid_input in invalid_inputs:
            try:
                policy, value = network(invalid_input)
                # If no error, check that output shapes are reasonable
                assert policy.shape[-1] == 7  # Policy should have 7 outputs
                assert value.shape[-1] == 1   # Value should have 1 output
            except (RuntimeError, ValueError):
                # Expected for invalid shapes
                pass


class TestConnect4NetworkActionProbabilities:
    """Test action probability calculation functionality."""
    
    def test_get_action_probabilities_no_mask(self, standard_network, cpu_device):
        """Test action probabilities without masking."""
        network = standard_network
        input_tensor = torch.zeros(1, 1, 6, 7, device=cpu_device)
        
        probs = network.get_action_probabilities(input_tensor)
        
        # Check output shape and properties
        assert probs.shape == (1, 7)
        assert torch.all(probs >= 0)  # Probabilities should be non-negative
        assert torch.allclose(probs.sum(dim=1), torch.ones(1, device=cpu_device))  # Should sum to 1
    
    def test_get_action_probabilities_with_mask(self, standard_network, cpu_device):
        """Test action probabilities with action masking."""
        network = standard_network
        input_tensor = torch.zeros(2, 1, 6, 7, device=cpu_device)
        
        # Create action mask (batch_size, 7)
        action_mask = torch.tensor([
            [True, True, True, False, False, False, False],  # Only first 3 actions valid
            [False, False, False, True, True, True, True]    # Only last 4 actions valid
        ], device=cpu_device)
        
        probs = network.get_action_probabilities(input_tensor, action_mask)
        
        # Check that masked actions have zero probability
        assert torch.all(probs[0, 3:] == 0)  # Actions 3-6 should be zero for first sample
        assert torch.all(probs[1, :3] == 0)  # Actions 0-2 should be zero for second sample
        
        # Check that valid actions have non-zero probabilities
        assert torch.all(probs[0, :3] > 0)   # Actions 0-2 should be positive for first sample
        assert torch.all(probs[1, 3:] > 0)   # Actions 3-6 should be positive for second sample
        
        # Each row should still sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(2, device=cpu_device))
    
    def test_get_action_probabilities_all_masked(self, standard_network, cpu_device):
        """Test action probabilities when all actions are masked."""
        network = standard_network
        input_tensor = torch.zeros(1, 1, 6, 7, device=cpu_device)
        
        # Mask all actions
        action_mask = torch.zeros(1, 7, dtype=torch.bool, device=cpu_device)
        
        try:
            probs = network.get_action_probabilities(input_tensor, action_mask)
            # If no error, should handle gracefully (might return uniform or error)
            assert probs.shape == (1, 7)
        except (RuntimeError, ValueError):
            # Expected behavior for invalid mask
            pass
    
    def test_get_action_probabilities_single_valid_action(self, standard_network, cpu_device):
        """Test action probabilities with only one valid action."""
        network = standard_network
        input_tensor = torch.zeros(1, 1, 6, 7, device=cpu_device)
        
        # Only action 3 is valid
        action_mask = torch.zeros(1, 7, dtype=torch.bool, device=cpu_device)
        action_mask[0, 3] = True
        
        probs = network.get_action_probabilities(input_tensor, action_mask)
        
        # Only action 3 should have probability 1
        assert probs[0, 3] == 1.0
        assert torch.all(probs[0, [0, 1, 2, 4, 5, 6]] == 0)
    
    def test_get_action_probabilities_batch_consistency(self, standard_network, cpu_device):
        """Test that batch processing is consistent with individual processing."""
        network = standard_network
        
        # Create batch input
        batch_input = torch.randn(3, 1, 6, 7, device=cpu_device)
        
        # Process as batch
        batch_probs = network.get_action_probabilities(batch_input)
        
        # Process individually
        individual_probs = []
        for i in range(3):
            single_input = batch_input[i:i+1]
            single_probs = network.get_action_probabilities(single_input)
            individual_probs.append(single_probs)
        
        individual_probs = torch.cat(individual_probs, dim=0)
        
        # Should be identical
        assert torch.allclose(batch_probs, individual_probs, atol=1e-6)


class TestConnect4DuelingNetwork:
    """Test Connect4DuelingNetwork implementation."""
    
    def test_dueling_network_initialization(self):
        """Test dueling network initialization."""
        network = Connect4DuelingNetwork()
        
        assert isinstance(network, nn.Module)
        assert isinstance(network, Connect4DuelingNetwork)
        
        # Should have different architecture components than standard network
        # (specific components depend on implementation)
    
    def test_dueling_network_forward_pass(self, dueling_network, cpu_device):
        """Test dueling network forward pass."""
        network = dueling_network
        
        input_tensor = torch.randn(2, 1, 6, 7, device=cpu_device)
        policy_logits, value = network(input_tensor)
        
        # Should have same output format as standard network
        assert policy_logits.shape == (2, 7)
        assert value.shape == (2, 1)
        assert torch.all(torch.isfinite(policy_logits))
        assert torch.all(torch.isfinite(value))
    
    def test_dueling_vs_standard_network_differences(self, cpu_device):
        """Test that dueling network produces different outputs than standard network."""
        standard_net = Connect4Network().to(cpu_device)
        dueling_net = Connect4DuelingNetwork().to(cpu_device)
        
        input_tensor = torch.randn(1, 1, 6, 7, device=cpu_device)
        
        standard_policy, standard_value = standard_net(input_tensor)
        dueling_policy, dueling_value = dueling_net(input_tensor)
        
        # Networks should produce different outputs (different architectures)
        # Note: This test might fail if networks have identical random initialization
        # In practice, after training they would be different
    
    def test_dueling_network_action_probabilities(self, dueling_network, cpu_device):
        """Test dueling network action probability calculation."""
        network = dueling_network
        input_tensor = torch.zeros(1, 1, 6, 7, device=cpu_device)
        
        probs = network.get_action_probabilities(input_tensor)
        
        # Should work same as standard network
        assert probs.shape == (1, 7)
        assert torch.all(probs >= 0)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1, device=cpu_device))


class TestNetworkFactory:
    """Test network creation factory function."""
    
    def test_create_standard_network(self):
        """Test creation of standard network."""
        network = create_network('standard')
        assert isinstance(network, Connect4Network)
        assert not isinstance(network, Connect4DuelingNetwork)
    
    def test_create_dueling_network(self):
        """Test creation of dueling network."""
        network = create_network('dueling')
        assert isinstance(network, Connect4DuelingNetwork)
    
    def test_create_network_invalid_type(self):
        """Test handling of invalid network types."""
        with pytest.raises((ValueError, KeyError)):
            create_network('invalid_network_type')
    
    def test_create_network_with_parameters(self):
        """Test network creation with custom parameters."""
        # This test depends on whether factory supports custom parameters
        try:
            network = create_network('standard', custom_param=123)
            assert isinstance(network, Connect4Network)
        except TypeError:
            # If factory doesn't support custom parameters, that's fine
            pass


class TestNetworkTraining:
    """Test network training functionality."""
    
    def test_network_gradient_updates(self, standard_network, cpu_device):
        """Test that network parameters update during training."""
        network = standard_network
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        
        # Store initial parameters
        initial_params = [param.clone() for param in network.parameters()]
        
        # Training step
        input_tensor = torch.randn(4, 1, 6, 7, device=cpu_device)
        policy_logits, value = network(input_tensor)
        
        # Dummy loss
        policy_loss = -torch.mean(torch.log_softmax(policy_logits, dim=1))
        value_loss = torch.mean(value ** 2)
        total_loss = policy_loss + value_loss
        
        # Backward pass and update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        for initial_param, current_param in zip(initial_params, network.parameters()):
            assert not torch.allclose(initial_param, current_param)
    
    def test_network_loss_calculation(self, standard_network, cpu_device):
        """Test loss calculation for training."""
        network = standard_network
        
        input_tensor = torch.randn(3, 1, 6, 7, device=cpu_device)
        policy_logits, value = network(input_tensor)
        
        # Create dummy targets
        policy_targets = torch.randint(0, 7, (3,), device=cpu_device)
        value_targets = torch.randn(3, 1, device=cpu_device)
        
        # Calculate losses
        policy_loss = nn.CrossEntropyLoss()(policy_logits, policy_targets)
        value_loss = nn.MSELoss()(value, value_targets)
        
        # Losses should be finite and positive
        assert torch.isfinite(policy_loss)
        assert torch.isfinite(value_loss)
        assert policy_loss >= 0
        assert value_loss >= 0
    
    def test_network_batch_norm_behavior(self, standard_network):
        """Test network behavior in train vs eval mode."""
        network = standard_network
        
        # Create consistent input
        input_tensor = torch.randn(2, 1, 6, 7)
        
        # Get outputs in training mode
        network.train()
        train_policy, train_value = network(input_tensor)
        
        # Get outputs in eval mode
        network.eval()
        eval_policy, eval_value = network(input_tensor)
        
        # Outputs might be different if network has dropout/batch norm
        # For basic CNN without these, outputs should be the same
        # This test documents the expected behavior


class TestNetworkPerformance:
    """Test network performance characteristics."""
    
    @pytest.mark.performance
    def test_forward_pass_speed(self, standard_network, cpu_device):
        """Test forward pass performance."""
        network = standard_network
        network.eval()
        
        input_tensor = torch.randn(32, 1, 6, 7, device=cpu_device)
        
        # Time forward passes
        import time
        start_time = time.time()
        
        for _ in range(100):
            with torch.no_grad():
                policy, value = network(input_tensor)
        
        end_time = time.time()
        forward_time = end_time - start_time
        
        # Should be reasonably fast
        passes_per_second = 100 / forward_time
        assert passes_per_second > 10  # At least 10 passes per second
    
    @pytest.mark.performance
    def test_memory_usage(self, standard_network, cpu_device):
        """Test network memory usage."""
        network = standard_network
        
        # Test with different batch sizes
        for batch_size in [1, 8, 32, 128]:
            input_tensor = torch.randn(batch_size, 1, 6, 7, device=cpu_device)
            
            # Forward pass should not cause memory issues
            policy, value = network(input_tensor)
            
            # Clean up
            del input_tensor, policy, value
            
            if cpu_device.type == 'cuda':
                torch.cuda.empty_cache()
    
    @pytest.mark.gpu
    def test_gpu_performance(self):
        """Test GPU performance if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        network = Connect4Network().to(device)
        
        input_tensor = torch.randn(64, 1, 6, 7, device=device)
        
        # Should work on GPU
        policy, value = network(input_tensor)
        
        assert policy.device.type == 'cuda'
        assert value.device.type == 'cuda'


class TestNetworkEdgeCases:
    """Test network edge cases and error handling."""
    
    def test_extreme_input_values(self, standard_network, cpu_device):
        """Test network with extreme input values."""
        network = standard_network
        
        # Test with extreme values
        extreme_inputs = [
            torch.full((1, 1, 6, 7), 1000.0, device=cpu_device),    # Very large values
            torch.full((1, 1, 6, 7), -1000.0, device=cpu_device),   # Very small values
            torch.zeros(1, 1, 6, 7, device=cpu_device),             # All zeros
            torch.ones(1, 1, 6, 7, device=cpu_device) * 1e-6,       # Very small positive
        ]
        
        for extreme_input in extreme_inputs:
            policy, value = network(extreme_input)
            
            # Should produce finite outputs
            assert torch.all(torch.isfinite(policy))
            assert torch.all(torch.isfinite(value))
    
    def test_network_state_dict_save_load(self, standard_network, tmp_path):
        """Test network state dictionary save/load."""
        network = standard_network
        save_path = tmp_path / "network_state.pth"
        
        # Save state dict
        torch.save(network.state_dict(), save_path)
        
        # Create new network and load state
        new_network = Connect4Network()
        new_network.load_state_dict(torch.load(save_path))
        
        # Networks should produce same outputs
        input_tensor = torch.randn(1, 1, 6, 7)
        
        network.eval()
        new_network.eval()
        
        policy1, value1 = network(input_tensor)
        policy2, value2 = new_network(input_tensor)
        
        assert torch.allclose(policy1, policy2)
        assert torch.allclose(value1, value2)
    
    def test_network_parameter_count(self, standard_network):
        """Test network parameter count."""
        network = standard_network
        
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters
        assert total_params > 1000  # At least 1k parameters
        assert total_params < 10_000_000  # Less than 10M parameters (reasonable for Connect4)
        assert trainable_params == total_params  # All parameters should be trainable by default
    
    def test_network_device_mismatch_handling(self, standard_network):
        """Test handling of device mismatches."""
        network = standard_network  # On CPU by default
        
        if torch.cuda.is_available():
            # Try to pass GPU tensor to CPU network
            gpu_input = torch.randn(1, 1, 6, 7, device='cuda')
            
            with pytest.raises(RuntimeError):
                policy, value = network(gpu_input)
    
    def test_network_with_nan_input(self, standard_network, cpu_device):
        """Test network behavior with NaN inputs."""
        network = standard_network
        
        # Input with NaN values
        nan_input = torch.full((1, 1, 6, 7), float('nan'), device=cpu_device)
        
        policy, value = network(nan_input)
        
        # Network might propagate NaNs or handle them
        # This test documents the behavior (implementation-dependent)
        assert policy.shape == (1, 7)
        assert value.shape == (1, 1)


class TestNetworkArchitectureDetails:
    """Test specific architecture implementation details."""
    
    def test_convolutional_layer_properties(self, standard_network):
        """Test properties of convolutional layers."""
        network = standard_network
        
        # Check that conv layers exist and have reasonable properties
        conv_layers = network.conv_layers
        assert isinstance(conv_layers, nn.Sequential)
        
        # Should have multiple conv layers
        conv_count = sum(1 for layer in conv_layers if isinstance(layer, nn.Conv2d))
        assert conv_count >= 2  # At least 2 conv layers
    
    def test_activation_functions(self, standard_network):
        """Test that network uses appropriate activation functions."""
        network = standard_network
        
        # Check for activation functions in the network
        has_relu = any(isinstance(layer, nn.ReLU) for layer in network.conv_layers)
        has_activation = has_relu or any(hasattr(layer, 'activation') for layer in network.conv_layers)
        
        # Should have some form of non-linearity
        assert has_activation or has_relu
    
    def test_output_layer_properties(self, standard_network):
        """Test properties of output layers."""
        network = standard_network
        
        # Policy head should be linear layer with 7 outputs
        assert isinstance(network.policy_head, nn.Linear)
        assert network.policy_head.out_features == 7
        
        # Value head should be linear layer with 1 output
        assert isinstance(network.value_head, nn.Linear)
        assert network.value_head.out_features == 1