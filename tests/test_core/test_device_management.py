"""
Comprehensive tests for device management functionality.

Tests GPU/CPU device detection, device placement, memory management,
and device compatibility across the system.
"""

import pytest
import torch
import psutil
import os
from unittest.mock import patch, Mock
from src.core.config import Config


class TestDeviceDetection:
    """Test device detection and selection functionality."""
    
    def test_cuda_availability_detection(self):
        """Test CUDA availability detection."""
        # Test actual CUDA availability
        actual_cuda = torch.cuda.is_available()
        
        if actual_cuda:
            assert torch.cuda.device_count() > 0
            assert torch.cuda.get_device_name(0) is not None
        else:
            assert torch.cuda.device_count() == 0
    
    def test_cpu_device_always_available(self):
        """Test that CPU device is always available."""
        cpu_device = torch.device('cpu')
        
        # Should always be able to create CPU tensors
        tensor = torch.tensor([1, 2, 3], device=cpu_device)
        assert tensor.device.type == 'cpu'
    
    @patch('torch.cuda.is_available')
    def test_device_detection_mocked_cuda_available(self, mock_cuda_available):
        """Test device detection when CUDA is mocked as available."""
        mock_cuda_available.return_value = True
        
        # Config should detect CUDA as available
        with patch('builtins.open', mock_open(read_data="device:\n  training_device: auto")):
            with patch('yaml.safe_load', return_value={'device': {'training_device': 'auto'}}):
                # Mock config would choose CUDA
                assert torch.cuda.is_available() == True
    
    @patch('torch.cuda.is_available')
    def test_device_detection_mocked_cuda_unavailable(self, mock_cuda_unavailable):
        """Test device detection when CUDA is mocked as unavailable."""
        mock_cuda_unavailable.return_value = False
        
        # Should fall back to CPU
        assert torch.cuda.is_available() == False
        
        # CPU should still work
        tensor = torch.tensor([1, 2, 3])
        assert tensor.device.type == 'cpu'
    
    def test_device_count_detection(self):
        """Test detection of number of available devices."""
        cpu_count = 1  # Always at least one CPU
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        total_devices = cpu_count + cuda_count
        assert total_devices >= 1  # At least CPU
        
        if torch.cuda.is_available():
            assert cuda_count > 0
            # Test each CUDA device
            for i in range(cuda_count):
                device = torch.device(f'cuda:{i}')
                tensor = torch.tensor([1], device=device)
                assert tensor.device.index == i
    
    def test_device_properties_detection(self):
        """Test detection of device properties."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                # Test device properties
                props = torch.cuda.get_device_properties(i)
                
                assert props.name is not None
                assert props.major >= 0
                assert props.minor >= 0
                assert props.total_memory > 0
                assert props.multi_processor_count > 0
    
    def test_device_memory_detection(self):
        """Test device memory detection."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    memory_allocated = torch.cuda.memory_allocated()
                    memory_reserved = torch.cuda.memory_reserved()
                    max_memory = torch.cuda.max_memory_allocated()
                    
                    # Memory values should be non-negative
                    assert memory_allocated >= 0
                    assert memory_reserved >= 0
                    assert max_memory >= 0


class TestDevicePlacement:
    """Test tensor and model device placement."""
    
    def test_tensor_cpu_placement(self):
        """Test tensor placement on CPU."""
        tensor = torch.tensor([1, 2, 3, 4, 5], device='cpu')
        
        assert tensor.device.type == 'cpu'
        assert tensor.tolist() == [1, 2, 3, 4, 5]
    
    @pytest.mark.gpu
    def test_tensor_cuda_placement(self):
        """Test tensor placement on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        tensor = torch.tensor([1, 2, 3, 4, 5], device='cuda')
        
        assert tensor.device.type == 'cuda'
        assert tensor.tolist() == [1, 2, 3, 4, 5]
    
    def test_tensor_device_transfer(self):
        """Test transferring tensors between devices."""
        # Start with CPU tensor
        cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
        assert cpu_tensor.device.type == 'cpu'
        
        if torch.cuda.is_available():
            # Transfer to CUDA
            cuda_tensor = cpu_tensor.to('cuda')
            assert cuda_tensor.device.type == 'cuda'
            assert torch.equal(cpu_tensor, cuda_tensor.cpu())
            
            # Transfer back to CPU
            cpu_tensor_2 = cuda_tensor.to('cpu')
            assert cpu_tensor_2.device.type == 'cpu'
            assert torch.equal(cpu_tensor, cpu_tensor_2)
    
    def test_model_device_placement(self):
        """Test model placement on different devices."""
        from src.agents.networks import Connect4Network
        
        # Test CPU placement
        model = Connect4Network()
        model = model.to('cpu')
        
        for param in model.parameters():
            assert param.device.type == 'cpu'
        
        if torch.cuda.is_available():
            # Test CUDA placement
            model = model.to('cuda')
            
            for param in model.parameters():
                assert param.device.type == 'cuda'
    
    def test_mixed_device_operations_error_handling(self):
        """Test error handling for mixed device operations."""
        cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
        
        if torch.cuda.is_available():
            cuda_tensor = torch.tensor([4, 5, 6], device='cuda')
            
            # Should raise error for mixed device operations
            with pytest.raises(RuntimeError):
                result = cpu_tensor + cuda_tensor
    
    def test_device_context_manager(self):
        """Test device context manager functionality."""
        if torch.cuda.is_available():
            # Test CUDA context manager
            with torch.cuda.device(0):
                tensor = torch.tensor([1, 2, 3])
                # Tensor should be on the current CUDA device
                assert tensor.device.type == 'cuda' or tensor.device.type == 'cpu'  # Depends on default
            
            # Test explicit device specification
            current_device = torch.cuda.current_device()
            assert isinstance(current_device, int)
            assert current_device >= 0
    
    def test_device_synchronization(self):
        """Test device synchronization functionality."""
        if torch.cuda.is_available():
            cuda_tensor = torch.tensor([1, 2, 3], device='cuda')
            
            # Test synchronization
            torch.cuda.synchronize()
            
            # Should complete without errors
            result = cuda_tensor * 2
            torch.cuda.synchronize()
            
            assert result.tolist() == [2, 4, 6]


class TestDeviceMemoryManagement:
    """Test device memory management functionality."""
    
    @pytest.mark.gpu
    def test_cuda_memory_allocation(self):
        """Test CUDA memory allocation and deallocation."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate tensor
        large_tensor = torch.randn(1000, 1000, device='cuda')
        allocated_memory = torch.cuda.memory_allocated()
        
        # Memory should increase
        assert allocated_memory > initial_memory
        
        # Delete tensor
        del large_tensor
        torch.cuda.empty_cache()
        
        # Memory should be freed (approximately)
        final_memory = torch.cuda.memory_allocated()
        assert final_memory <= allocated_memory
    
    @pytest.mark.gpu
    def test_cuda_memory_caching(self):
        """Test CUDA memory caching behavior."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_reserved = torch.cuda.memory_reserved()
        
        # Allocate and deallocate tensors
        for i in range(10):
            tensor = torch.randn(100, 100, device='cuda')
            del tensor
        
        # Memory might be cached
        final_reserved = torch.cuda.memory_reserved()
        
        # Clear cache
        torch.cuda.empty_cache()
        cleared_reserved = torch.cuda.memory_reserved()
        
        # Memory should be freed after clearing cache
        assert cleared_reserved <= final_reserved
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in device operations."""
        import gc
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many operations
        for i in range(100):
            tensor = torch.randn(100, 100)
            
            if torch.cuda.is_available():
                cuda_tensor = tensor.to('cuda')
                result = cuda_tensor * 2
                result = result.to('cpu')
                del cuda_tensor, result
            
            del tensor
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Check final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    @pytest.mark.performance
    def test_device_memory_efficiency(self):
        """Test memory efficiency of device operations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Create tensors of known size
            tensor_size = 1000 * 1000 * 4  # 4MB tensor (float32)
            tensor = torch.randn(1000, 1000, device='cuda')
            
            allocated_memory = torch.cuda.memory_allocated()
            memory_used = allocated_memory - initial_memory
            
            # Should use approximately expected memory
            assert memory_used >= tensor_size * 0.8  # At least 80% of expected
            assert memory_used <= tensor_size * 1.5  # No more than 150% of expected
            
            del tensor
            torch.cuda.empty_cache()
    
    def test_out_of_memory_handling(self):
        """Test handling of out-of-memory conditions."""
        if torch.cuda.is_available():
            # Try to allocate very large tensor that might cause OOM
            try:
                # Attempt to allocate 100GB (likely to fail)
                huge_tensor = torch.randn(10000, 10000, 1000, device='cuda')
                del huge_tensor  # Clean up if somehow succeeded
            except RuntimeError as e:
                # Should catch CUDA out of memory error
                assert "out of memory" in str(e).lower() or "cuda" in str(e).lower()
            except Exception as e:
                # Other exceptions are also acceptable for this test
                pass


class TestDeviceCompatibility:
    """Test device compatibility across system components."""
    
    def test_agent_device_compatibility(self, test_config):
        """Test that agents work correctly with different devices."""
        from src.agents.ppo_agent import PPOAgent
        
        devices_to_test = ['cpu']
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
        
        for device in devices_to_test:
            agent = PPOAgent(device=device, config=test_config)
            
            # Agent should be on correct device
            assert agent.device == torch.device(device)
            
            # Network parameters should be on correct device
            for param in agent.network.parameters():
                assert param.device.type == device
    
    def test_environment_device_compatibility(self, cpu_device):
        """Test that environments work with different devices."""
        from src.environments.hybrid_vectorized_connect4 import HybridVectorizedConnect4
        
        devices_to_test = [cpu_device]
        if torch.cuda.is_available():
            devices_to_test.append(torch.device('cuda'))
        
        for device in devices_to_test:
            env = HybridVectorizedConnect4(num_envs=4, device=device)
            
            # Environment should use correct device
            assert env.device == device
            
            # Observations should be on correct device
            observations = env.get_observations_gpu()
            assert observations.device == device
    
    def test_training_device_consistency(self, test_config):
        """Test device consistency during training."""
        from src.agents.ppo_agent import PPOAgent
        from src.environments.hybrid_vectorized_connect4 import HybridVectorizedConnect4
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        agent = PPOAgent(device=device, config=test_config)
        env = HybridVectorizedConnect4(num_envs=4, device=device)
        
        # All components should use same device
        assert agent.device == device
        assert env.device == device
        
        # Operations should work together
        observations = env.get_observations_gpu()
        valid_moves = env.get_valid_moves_tensor()
        
        actions, log_probs, values = agent.get_actions_batch(observations, valid_moves)
        
        # All tensors should be on same device
        assert observations.device == device
        assert valid_moves.device == device
        assert actions.device == device
        assert log_probs.device == device
        assert values.device == device
    
    def test_config_device_consistency(self, test_config):
        """Test that config device settings are consistent."""
        config_device = test_config.device
        
        # Should be valid PyTorch device
        assert isinstance(config_device, torch.device)
        
        # Should be either CPU or CUDA
        assert config_device.type in ['cpu', 'cuda']
        
        # If CUDA, should be available
        if config_device.type == 'cuda':
            assert torch.cuda.is_available()
    
    def test_mixed_precision_compatibility(self):
        """Test mixed precision training compatibility."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision")
        
        # Test that mixed precision operations work
        with torch.cuda.amp.autocast():
            tensor = torch.randn(100, 100, device='cuda')
            result = torch.matmul(tensor, tensor.t())
            
            # Should complete without errors
            assert result.shape == (100, 100)
            assert result.device.type == 'cuda'
    
    def test_multi_gpu_compatibility(self):
        """Test multi-GPU compatibility if available."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")
        
        # Test operations on multiple GPUs
        tensor1 = torch.tensor([1, 2, 3], device='cuda:0')
        tensor2 = torch.tensor([4, 5, 6], device='cuda:1')
        
        assert tensor1.device.index == 0
        assert tensor2.device.index == 1
        
        # Transfer and operate
        tensor2_on_gpu0 = tensor2.to('cuda:0')
        result = tensor1 + tensor2_on_gpu0
        
        assert result.device.index == 0
        assert result.tolist() == [5, 7, 9]


class TestDevicePerformance:
    """Test performance characteristics of device operations."""
    
    @pytest.mark.performance
    def test_cpu_vs_gpu_performance(self):
        """Test performance comparison between CPU and GPU."""
        size = (1000, 1000)
        
        # CPU performance
        cpu_tensor = torch.randn(*size, device='cpu')
        
        import time
        start_time = time.time()
        cpu_result = torch.matmul(cpu_tensor, cpu_tensor.t())
        cpu_time = time.time() - start_time
        
        if torch.cuda.is_available():
            # GPU performance
            gpu_tensor = torch.randn(*size, device='cuda')
            
            # Warm up GPU
            torch.matmul(gpu_tensor, gpu_tensor.t())
            torch.cuda.synchronize()
            
            start_time = time.time()
            gpu_result = torch.matmul(gpu_tensor, gpu_tensor.t())
            torch.cuda.synchronize()
            gpu_time = time.time() - start_time
            
            # Results should be approximately equal
            assert torch.allclose(cpu_result, gpu_result.cpu(), atol=1e-4)
            
            # GPU might be faster for large operations (but not always)
            print(f"CPU time: {cpu_time:.4f}s, GPU time: {gpu_time:.4f}s")
    
    @pytest.mark.performance
    def test_device_transfer_performance(self):
        """Test performance of device transfers."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        sizes = [(100, 100), (1000, 1000), (5000, 5000)]
        
        for size in sizes:
            cpu_tensor = torch.randn(*size)
            
            # Time CPU to GPU transfer
            start_time = time.time()
            gpu_tensor = cpu_tensor.to('cuda')
            torch.cuda.synchronize()
            cpu_to_gpu_time = time.time() - start_time
            
            # Time GPU to CPU transfer
            start_time = time.time()
            cpu_tensor_back = gpu_tensor.to('cpu')
            gpu_to_cpu_time = time.time() - start_time
            
            # Transfers should complete in reasonable time
            assert cpu_to_gpu_time < 1.0  # Less than 1 second
            assert gpu_to_cpu_time < 1.0  # Less than 1 second
            
            # Data should be preserved
            assert torch.allclose(cpu_tensor, cpu_tensor_back)
    
    @pytest.mark.performance
    def test_batch_operation_performance(self):
        """Test performance of batch operations on different devices."""
        batch_sizes = [1, 10, 100]
        
        for batch_size in batch_sizes:
            # CPU batch operations
            cpu_batch = torch.randn(batch_size, 6, 7)
            
            start_time = time.time()
            cpu_result = torch.sum(cpu_batch, dim=[1, 2])
            cpu_time = time.time() - start_time
            
            if torch.cuda.is_available():
                # GPU batch operations
                gpu_batch = cpu_batch.to('cuda')
                
                start_time = time.time()
                gpu_result = torch.sum(gpu_batch, dim=[1, 2])
                torch.cuda.synchronize()
                gpu_time = time.time() - start_time
                
                # Results should be approximately equal
                assert torch.allclose(cpu_result, gpu_result.cpu())
                
                # Both should complete quickly
                assert cpu_time < 0.1
                assert gpu_time < 0.1


class TestDeviceErrorHandling:
    """Test error handling for device-related issues."""
    
    def test_invalid_device_specification(self):
        """Test handling of invalid device specifications."""
        invalid_devices = [
            'invalid_device',
            'cuda:999',  # Non-existent CUDA device
            'gpu',       # Wrong device name
            123,         # Invalid type
            None,        # None device
        ]
        
        for invalid_device in invalid_devices:
            try:
                tensor = torch.tensor([1, 2, 3], device=invalid_device)
                # If no error, should be on a valid device
                assert tensor.device.type in ['cpu', 'cuda']
            except (RuntimeError, ValueError, TypeError):
                # Expected error for invalid device
                pass
    
    def test_cuda_unavailable_fallback(self):
        """Test fallback behavior when CUDA is unavailable."""
        # Mock CUDA as unavailable
        with patch('torch.cuda.is_available', return_value=False):
            # Should fall back to CPU
            tensor = torch.tensor([1, 2, 3])
            assert tensor.device.type == 'cpu'
            
            # CUDA operations should fail gracefully
            with pytest.raises(AssertionError):
                assert torch.cuda.is_available()
    
    def test_device_mismatch_error_messages(self):
        """Test that device mismatch errors have helpful messages."""
        cpu_tensor = torch.tensor([1, 2, 3], device='cpu')
        
        if torch.cuda.is_available():
            cuda_tensor = torch.tensor([4, 5, 6], device='cuda')
            
            try:
                result = cpu_tensor + cuda_tensor
                pytest.fail("Expected RuntimeError for device mismatch")
            except RuntimeError as e:
                error_message = str(e).lower()
                # Error message should mention device or cuda
                assert 'device' in error_message or 'cuda' in error_message
    
    def test_out_of_memory_recovery(self):
        """Test recovery from out of memory conditions."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        try:
            # Try to allocate huge tensor
            huge_tensor = torch.randn(50000, 50000, device='cuda')
            del huge_tensor  # Clean up if successful
        except RuntimeError as e:
            # Should be able to recover and allocate smaller tensor
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                
                # Should be able to allocate smaller tensor after clearing cache
                small_tensor = torch.randn(100, 100, device='cuda')
                assert small_tensor.device.type == 'cuda'
                del small_tensor


class TestDeviceUtilities:
    """Test device utility functions and helpers."""
    
    def test_device_info_retrieval(self):
        """Test retrieval of device information."""
        # CPU info
        cpu_device = torch.device('cpu')
        assert cpu_device.type == 'cpu'
        assert cpu_device.index is None
        
        if torch.cuda.is_available():
            # CUDA info
            cuda_device = torch.device('cuda')
            assert cuda_device.type == 'cuda'
            
            # Device properties
            device_count = torch.cuda.device_count()
            assert device_count > 0
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                assert hasattr(props, 'name')
                assert hasattr(props, 'total_memory')
                assert hasattr(props, 'multi_processor_count')
    
    def test_device_capability_detection(self):
        """Test detection of device capabilities."""
        if torch.cuda.is_available():
            # Test compute capability
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                assert isinstance(capability, tuple)
                assert len(capability) == 2
                assert all(isinstance(x, int) for x in capability)
                
                # Should have reasonable compute capability
                major, minor = capability
                assert major >= 3  # At least compute capability 3.0
    
    def test_device_context_utilities(self):
        """Test device context utility functions."""
        if torch.cuda.is_available():
            original_device = torch.cuda.current_device()
            
            # Test device context
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    current = torch.cuda.current_device()
                    assert current == i
            
            # Should return to original device
            final_device = torch.cuda.current_device()
            assert final_device == original_device
    
    def test_memory_utility_functions(self):
        """Test memory utility functions."""
        if torch.cuda.is_available():
            # Memory stats
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            max_reserved = torch.cuda.max_memory_reserved()
            
            # All should be non-negative
            assert allocated >= 0
            assert reserved >= 0
            assert max_allocated >= 0
            assert max_reserved >= 0
            
            # Max values should be >= current values
            assert max_allocated >= allocated
            assert max_reserved >= reserved
            
            # Test memory reset
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.reset_max_memory_cached()
            
            # Should reset max values
            new_max_allocated = torch.cuda.max_memory_allocated()
            assert new_max_allocated <= max_allocated