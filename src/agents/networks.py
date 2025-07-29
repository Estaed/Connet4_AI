"""
Neural Network Architectures for Connect4 RL System

This module provides PyTorch neural network architectures optimized for Connect4
board pattern recognition, including policy and value networks for PPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

from ..core.config import get_config


class Connect4Network(nn.Module):
    """
    CNN architecture for Connect4 board pattern recognition.
    
    Designed for PPO training with separate policy and value heads.
    Optimized for 6×7 Connect4 boards with action masking support.
    
    Architecture:
    - Input: (batch, 1, 6, 7) - Single channel board representation
    - Conv layers: Extract spatial patterns and features
    - Policy head: 7 outputs for column selection probabilities
    - Value head: 1 output for state value estimation
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 board_height: int = 6,
                 board_width: int = 7,
                 hidden_dim: int = 128,
                 dropout_rate: float = 0.1,
                 device: Optional[str] = None):
        """
        Initialize Connect4 neural network.
        
        Args:
            input_channels: Number of input channels (1 for basic board)
            board_height: Board height (6 for Connect4)
            board_width: Board width (7 for Connect4)  
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate for regularization
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.board_height = board_height
        self.board_width = board_width
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Load configuration
        self.config = get_config()
        
        # Device setup with CUDA 11.8 support
        if device is None:
            device = self.config.get('device.training_device', 'cpu')
        self.device = self._validate_and_setup_device(device)
        
        # Convolutional layers for spatial pattern recognition
        self.conv_layers = nn.Sequential(
            # First conv layer - detect basic patterns
            nn.Conv2d(input_channels, 64, kernel_size=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Second conv layer - combine patterns
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            # Third conv layer - high-level features
            nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )
        
        # Calculate flattened feature size after conv layers
        self.feature_size = self._calculate_conv_output_size()
        
        # Policy head - outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, board_width)  # 7 columns
        )
        
        # Value head - outputs state value estimate
        self.value_head = nn.Sequential(
            nn.Linear(self.feature_size, hidden_dim),
            nn.ReLU(inplace=True), 
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to device (GPU if available)
        self.to(self.device)
        
        # Setup logging
        self.logger = logging.getLogger(f"Network.{self.__class__.__name__}")
        self.logger.info(f"Network initialized on device: {self.device}")
        
        # Log GPU information if using CUDA
        if self.device.startswith('cuda'):
            self._log_gpu_info()
        
    def _validate_and_setup_device(self, device: str) -> str:
        """
        Validate and setup device with CUDA 11.8 support.
        
        Args:
            device: Requested device
            
        Returns:
            Validated device string
        """
        if device == 'cuda' or device.startswith('cuda:'):
            if not torch.cuda.is_available():
                logging.warning("CUDA requested but not available, falling back to CPU")
                return 'cpu'
            
            # Check CUDA version compatibility
            cuda_version = torch.version.cuda
            if cuda_version:
                logging.info(f"CUDA version: {cuda_version}")
                
            # Use specified GPU or default to cuda:0
            if device == 'cuda':
                device = 'cuda:0'
            
            # Validate GPU index
            gpu_index = int(device.split(':')[1]) if ':' in device else 0
            if gpu_index >= torch.cuda.device_count():
                logging.warning(f"GPU {gpu_index} not available, using cuda:0")
                device = 'cuda:0'
                
            return device
            
        return 'cpu'
    
    def _log_gpu_info(self) -> None:
        """Log GPU information for debugging and optimization."""
        if torch.cuda.is_available():
            gpu_index = int(self.device.split(':')[1]) if ':' in self.device else 0
            gpu_name = torch.cuda.get_device_name(gpu_index)
            gpu_memory = torch.cuda.get_device_properties(gpu_index).total_memory / 1e9
            
            self.logger.info(f"Using GPU: {gpu_name}")
            self.logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
            self.logger.info(f"PyTorch Version: {torch.__version__}")
            
            # Set memory allocation strategy for better performance
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                # Reserve 90% of GPU memory to avoid OOM errors
                torch.cuda.set_per_process_memory_fraction(0.9, gpu_index)
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after convolution layers."""
        # Create dummy input to calculate size
        dummy_input = torch.zeros(1, self.input_channels, self.board_height, self.board_width)
        with torch.no_grad():
            conv_output = self.conv_layers(dummy_input)
            # Use global average pooling
            pooled_output = F.adaptive_avg_pool2d(conv_output, (1, 1))
            return pooled_output.view(1, -1).size(1)
    
    def _init_weights(self) -> None:
        """Initialize network weights using Xavier/He initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1) 
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Tuple of (policy_logits, value):
            - policy_logits: Action probabilities (batch, 7)
            - value: State value estimates (batch, 1)
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
        
        # Ensure input has correct shape
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Extract features through conv layers
        conv_features = self.conv_layers(x)
        
        # Global average pooling to reduce spatial dimensions
        pooled_features = F.adaptive_avg_pool2d(conv_features, (1, 1))
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Generate policy logits and value estimate
        policy_logits = self.policy_head(flattened_features)
        value = self.value_head(flattened_features)
        
        return policy_logits, value.squeeze(-1)
    
    def get_action_probabilities(self, 
                               x: torch.Tensor,
                               valid_actions: Optional[torch.Tensor] = None,
                               temperature: float = 1.0) -> torch.Tensor:
        """
        Get action probabilities with optional masking and temperature scaling.
        
        Args:
            x: Input board state
            valid_actions: Binary mask for valid actions (1=valid, 0=invalid)
            temperature: Temperature for probability scaling (lower = more deterministic)
            
        Returns:
            Action probabilities (batch, 7)
        """
        policy_logits, _ = self.forward(x)
        
        # Apply temperature scaling
        if temperature != 1.0:
            policy_logits = policy_logits / temperature
        
        # Apply action masking if provided
        if valid_actions is not None:
            # Ensure valid_actions is on same device
            if valid_actions.device != policy_logits.device:
                valid_actions = valid_actions.to(policy_logits.device)
            # Set invalid actions to very negative value
            policy_logits = policy_logits.masked_fill(~valid_actions.bool(), float('-inf'))
        
        # Convert to probabilities
        action_probs = F.softmax(policy_logits, dim=-1)
        
        return action_probs
    
    def get_state_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get state value estimate.
        
        Args:
            x: Input board state
            
        Returns:
            State value estimate (batch,)
        """
        _, value = self.forward(x)
        return value
    
    def to_device(self, device: Optional[str] = None) -> 'Connect4Network':
        """
        Move network to specified device.
        
        Args:
            device: Target device (None to use current device)
            
        Returns:
            Self for chaining
        """
        if device is not None:
            self.device = self._validate_and_setup_device(device)
        
        self.to(self.device)
        
        if self.device.startswith('cuda'):
            self._log_gpu_info()
            
        return self
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get GPU memory usage information.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.device.startswith('cuda'):
            return {'device': self.device, 'memory_used_mb': 0, 'memory_total_mb': 0}
        
        gpu_index = int(self.device.split(':')[1]) if ':' in self.device else 0
        
        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(gpu_index) / 1e6  # MB
        memory_reserved = torch.cuda.memory_reserved(gpu_index) / 1e6   # MB
        memory_total = torch.cuda.get_device_properties(gpu_index).total_memory / 1e6  # MB
        
        return {
            'device': self.device,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'memory_total_mb': memory_total,
            'memory_usage_percent': (memory_reserved / memory_total) * 100
        }


class Connect4DuelingNetwork(Connect4Network):
    """
    Dueling network architecture for Connect4.
    
    Separates state value and action advantage estimation for potentially
    better learning of state values independent of action selection.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize dueling network with same parameters as base network."""
        super().__init__(*args, **kwargs)
        
        # Replace policy head with advantage head
        self.advantage_head = nn.Sequential(
            nn.Linear(self.feature_size, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.board_width)
        )
        
        # Keep value head as is
        # Move new layers to device
        self.advantage_head.to(self.device)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dueling architecture.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (q_values, value) where q_values combine value and advantage
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Ensure input has correct shape
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Extract features
        conv_features = self.conv_layers(x)
        pooled_features = F.adaptive_avg_pool2d(conv_features, (1, 1))
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        
        # Compute value and advantage
        value = self.value_head(flattened_features).squeeze(-1)
        advantage = self.advantage_head(flattened_features)
        
        # Combine using dueling formula: Q(s,a) = V(s) + A(s,a) - mean(A(s))
        q_values = value.unsqueeze(-1) + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values.squeeze(-1), value


# Utility functions for network management
def create_network(network_type: str = "standard", **kwargs) -> nn.Module:
    """
    Factory function to create neural networks.
    
    Args:
        network_type: Type of network ('standard', 'dueling')
        **kwargs: Network configuration parameters
        
    Returns:
        Initialized neural network
        
    Raises:
        ValueError: If network type is not recognized
    """
    network_registry = {
        'standard': Connect4Network,
        'dueling': Connect4DuelingNetwork,
    }
    
    if network_type.lower() not in network_registry:
        available_types = list(network_registry.keys())
        raise ValueError(f"Unknown network type '{network_type}'. Available: {available_types}")
    
    network_class = network_registry[network_type.lower()]
    return network_class(**kwargs)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'trainable_mb': trainable_params * 4 / (1024 * 1024),  # Assuming float32
        'total_mb': total_params * 4 / (1024 * 1024)
    }


def test_network_shapes(network: nn.Module, 
                       batch_size: int = 2,
                       device: Optional[str] = None) -> Dict[str, Any]:
    """
    Test network with sample inputs to verify shapes.
    
    Args:
        network: Network to test
        batch_size: Batch size for testing
        device: Device to run test on (None to use network's device)
        
    Returns:
        Dictionary with test results and shape information
    """
    if device is None:
        device = next(network.parameters()).device
    
    network = network.to(device)
    network.eval()
    
    # Create sample input (batch, 1, 6, 7)
    sample_input = torch.randn(batch_size, 1, 6, 7, device=device)
    
    with torch.no_grad():
        try:
            policy_logits, value = network(sample_input)
            
            results = {
                'success': True,
                'device': str(device),
                'input_shape': list(sample_input.shape),
                'policy_logits_shape': list(policy_logits.shape),
                'value_shape': list(value.shape),
                'expected_policy_shape': [batch_size, 7],
                'expected_value_shape': [batch_size],
                'shapes_correct': (
                    list(policy_logits.shape) == [batch_size, 7] and 
                    list(value.shape) == [batch_size]
                )
            }
            
            # Test action probabilities
            action_probs = network.get_action_probabilities(sample_input)
            results['action_probs_shape'] = list(action_probs.shape)
            results['action_probs_sum'] = action_probs.sum(dim=-1).tolist()  # Should be close to 1.0
            
            # Test with action masking
            valid_actions = torch.ones(batch_size, 7, device=device)
            valid_actions[:, -1] = 0  # Mask last column
            masked_probs = network.get_action_probabilities(sample_input, valid_actions)
            results['masked_probs_shape'] = list(masked_probs.shape)
            results['masked_last_column'] = masked_probs[:, -1].tolist()  # Should be 0
            
        except Exception as e:
            results = {
                'success': False,
                'error': str(e),
                'device': str(device),
                'input_shape': list(sample_input.shape)
            }
    
    return results


def benchmark_network_performance(network: nn.Module,
                                batch_size: int = 32,
                                num_iterations: int = 100,
                                device: Optional[str] = None) -> Dict[str, float]:
    """
    Benchmark network performance.
    
    Args:
        network: Network to benchmark
        batch_size: Batch size for testing
        num_iterations: Number of forward passes
        device: Device to run benchmark on
        
    Returns:
        Performance statistics
    """
    if device is None:
        device = next(network.parameters()).device
    
    network = network.to(device)
    network.eval()
    
    # Warm up GPU
    sample_input = torch.randn(batch_size, 1, 6, 7, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = network(sample_input)
    
    # Synchronize before timing
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = network(sample_input)
    
    # Synchronize after timing
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return {
        'total_time_seconds': total_time,
        'avg_time_per_batch_ms': (total_time / num_iterations) * 1000,
        'throughput_samples_per_second': (batch_size * num_iterations) / total_time,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'device': str(device)
    }


def setup_cuda_optimization() -> None:
    """
    Setup CUDA optimizations for better performance.
    Should be called once at the start of training.
    """
    if torch.cuda.is_available():
        # Enable cudnn benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = True
        
        # Enable cudnn deterministic mode for reproducibility (optional)
        # torch.backends.cudnn.deterministic = True
        
        # Set memory allocation strategy
        torch.cuda.empty_cache()
        
        logging.info("CUDA optimizations enabled")
        logging.info(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        logging.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logging.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")