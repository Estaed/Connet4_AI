"""
PPO Agent Implementation for Connect4 RL System

This module implements the Proximal Policy Optimization (PPO) agent with action masking
for Connect4 training. Features GPU acceleration, experience replay, and comprehensive
strategic learning capabilities.
"""

import torch
import torch.nn as nn  
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, namedtuple
import logging

from .base_agent import BaseAgent
from .networks import Connect4Network
from ..core.config import get_config


# Experience tuple for storing training data
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done', 
    'log_prob', 'value', 'valid_actions'
])


class PPOMemory:
    """
    Memory buffer for PPO training experiences.
    
    Stores trajectories of experiences and provides batch sampling
    for PPO updates with proper advantage computation.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize PPO memory buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.experiences: List[Experience] = []
        self.position = 0
        
    def store(self, 
              state: np.ndarray, 
              action: int, 
              reward: float,
              next_state: np.ndarray,
              done: bool,
              log_prob: float,
              value: float,
              valid_actions: List[int]) -> None:
        """
        Store an experience in the buffer.
        
        Args:
            state: Current board state
            action: Action taken
            reward: Reward received
            next_state: Next board state
            done: Episode termination flag
            log_prob: Log probability of the action
            value: State value estimate
            valid_actions: List of valid actions for the state
        """
        experience = Experience(
            state=state.copy(),
            action=action,
            reward=reward,
            next_state=next_state.copy(),
            done=done,
            log_prob=log_prob,
            value=value,
            valid_actions=valid_actions.copy()
        )
        
        if len(self.experiences) < self.capacity:
            self.experiences.append(experience)
        else:
            self.experiences[self.position] = experience
            
        self.position = (self.position + 1) % self.capacity
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get all experiences as batched tensors.
        
        Returns:
            Dictionary containing batched experience tensors
        """
        if not self.experiences:
            return {}
            
        # Convert experiences to numpy arrays first
        states = np.array([exp.state for exp in self.experiences])
        actions = np.array([exp.action for exp in self.experiences])
        rewards = np.array([exp.reward for exp in self.experiences])
        next_states = np.array([exp.next_state for exp in self.experiences])
        dones = np.array([exp.done for exp in self.experiences])
        log_probs = np.array([exp.log_prob for exp in self.experiences])
        values = np.array([exp.value for exp in self.experiences])
        
        return {
            'states': torch.FloatTensor(states),
            'actions': torch.LongTensor(actions),
            'rewards': torch.FloatTensor(rewards),
            'next_states': torch.FloatTensor(next_states),
            'dones': torch.BoolTensor(dones),
            'log_probs': torch.FloatTensor(log_probs),
            'values': torch.FloatTensor(values)
        }
    
    def compute_advantages(self, 
                          gamma: float = 0.99, 
                          gae_lambda: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (advantages, returns)
        """
        if not self.experiences:
            return torch.tensor([]), torch.tensor([])
            
        values = torch.FloatTensor([exp.value for exp in self.experiences])
        rewards = torch.FloatTensor([exp.reward for exp in self.experiences])
        dones = torch.BoolTensor([exp.done for exp in self.experiences])
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages and returns backwards through trajectory
        advantage = 0
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = 0  # Terminal state has no future value
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # GAE advantage
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            advantages[t] = advantage
            
            # Return (value target)
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def clear(self) -> None:
        """Clear all stored experiences."""
        self.experiences.clear()
        self.position = 0
    
    def __len__(self) -> int:
        """Return number of stored experiences."""
        return len(self.experiences)


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent for Connect4.
    
    Implements PPO algorithm with action masking, GPU acceleration,
    and strategic learning capabilities for Connect4 gameplay.
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 name: Optional[str] = None,
                 config_path: Optional[str] = None,
                 network_type: str = "standard"):
        """
        Initialize PPO agent.
        
        Args:
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
            name: Human-readable name for the agent
            config_path: Path to configuration file
            network_type: Type of network architecture ('standard', 'dueling')
        """
        super().__init__(device, name, config_path)
        
        # PPO hyperparameters from config
        self.learning_rate = self.config.get('ppo.learning_rate', 3e-4)
        self.gamma = self.config.get('ppo.gamma', 0.99)
        self.gae_lambda = self.config.get('ppo.gae_lambda', 0.95)
        self.clip_epsilon = self.config.get('ppo.clip_epsilon', 0.2)
        self.value_loss_coef = self.config.get('ppo.value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('ppo.entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('ppo.max_grad_norm', 0.5)
        self.ppo_epochs = self.config.get('ppo.ppo_epochs', 10)  # Increased for better learning
        self.batch_size = self.config.get('ppo.batch_size', 32)  # Reduced for more frequent updates
        self.n_steps = self.config.get('ppo.n_steps', 2048)  # Steps per rollout (stable-baselines3 default)
        self.memory_capacity = self.config.get('ppo.memory_capacity', 10000)
        
        # Training parameters
        self.update_frequency = self.config.get('training.update_frequency', 100)
        self.experience_count = 0
        
        # Initialize network
        from .networks import create_network
        self.network = create_network(network_type, device=self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(), 
            lr=self.learning_rate,
            eps=1e-5
        )
        
        # Learning rate scheduler (optional)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000, 
            gamma=0.95
        )
        
        # Experience memory
        self.memory = PPOMemory(capacity=self.memory_capacity)
        
        # Training metrics
        self.training_metrics = {
            'episodes_trained': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'clip_fraction': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0
        }
        
        # Set agent to training mode
        self.training = True
        
        self.logger.info(f"PPO Agent initialized on {self.device}")
        self.logger.info(f"Network parameters: {sum(p.numel() for p in self.network.parameters())}")
    
    def is_learning_agent(self) -> bool:
        """Return True as PPO is a learning agent."""
        return True
    
    def get_action(self, 
                   observation: np.ndarray, 
                   valid_actions: Optional[List[int]] = None,
                   deterministic: bool = False,
                   **kwargs) -> int:
        """
        Get action for given observation using current policy.
        
        Args:
            observation: Board state as 6x7 numpy array
            valid_actions: List of valid column indices (0-6)
            deterministic: If True, select action deterministically
            **kwargs: Additional parameters
            
        Returns:
            Column index (0-6) for piece placement
        """
        if valid_actions is None:
            valid_actions = [i for i in range(7) if observation[0, i] == 0]
        
        # Convert observation to tensor
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Create action mask
        action_mask = torch.zeros(1, 7, device=self.device)
        action_mask[0, valid_actions] = 1.0
        
        with torch.no_grad():
            # Get policy and value
            policy_logits, value = self.network(state_tensor)
            
            # Apply action masking
            masked_logits = self._apply_action_mask(policy_logits, action_mask)
            
            if deterministic:
                # Select action with highest probability
                action = masked_logits.argmax(dim=-1).item()
            else:
                # Sample from policy distribution
                action_probs = F.softmax(masked_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
        
        # Store last observation and action for training
        self.last_observation = observation.copy()
        self.last_action = action
        
        return action
    
    def get_action_with_info(self, 
                           observation: np.ndarray, 
                           valid_actions: Optional[List[int]] = None) -> Tuple[int, Dict[str, float]]:
        """
        Get action with additional information for training.
        
        Args:
            observation: Board state
            valid_actions: List of valid actions
            
        Returns:
            Tuple of (action, info_dict) where info_dict contains:
            - log_prob: Log probability of selected action
            - value: State value estimate
            - entropy: Policy entropy
        """
        if valid_actions is None:
            valid_actions = [i for i in range(7) if observation[0, i] == 0]
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Create action mask
        action_mask = torch.zeros(1, 7, device=self.device)
        action_mask[0, valid_actions] = 1.0
        
        with torch.no_grad():
            # Forward pass
            policy_logits, value = self.network(state_tensor)
            
            # Apply masking and get distribution
            masked_logits = self._apply_action_mask(policy_logits, action_mask)
            action_probs = F.softmax(masked_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # Sample action
            action = action_dist.sample()
            
            # Compute info
            log_prob = action_dist.log_prob(action)
            entropy = action_dist.entropy()
            
            info = {
                'log_prob': log_prob.item(),
                'value': value.squeeze().item(),
                'entropy': entropy.item()
            }
        
        return action.item(), info
    
    def store_experience(self, 
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool,
                        log_prob: float,
                        value: float,
                        valid_actions: List[int]) -> None:
        """
        Store experience for later training.
        
        Args:
            state: Current board state
            action: Action taken
            reward: Reward received
            next_state: Next board state
            done: Episode termination flag
            log_prob: Log probability of action
            value: State value estimate
            valid_actions: Valid actions for the state
        """
        self.memory.store(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value,
            valid_actions=valid_actions
        )
        
        self.experience_count += 1
    
    def update(self, 
               experiences: Optional[Dict[str, Any]] = None,
               **kwargs) -> Dict[str, float]:
        """
        Update agent using PPO algorithm.
        
        Args:
            experiences: Optional external experiences (uses internal memory if None)
            **kwargs: Additional update parameters
            
        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < self.batch_size:
            return {}
        
        # Get experiences from memory
        batch = self.memory.get_batch()
        if not batch:
            return {}
        
        # Move to device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.memory.compute_advantages(
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old policy for KL divergence computation
        with torch.no_grad():
            states = batch['states'].unsqueeze(1)  # Add channel dimension
            old_policy_logits, _ = self.network(states)
            old_log_probs = batch['log_probs']
        
        # PPO update loop
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        total_clip_fraction = 0.0
        total_kl_div = 0.0
        
        for epoch in range(self.ppo_epochs):
            # Forward pass
            policy_logits, values = self.network(states)
            
            # Compute policy distribution
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            
            # Get log probabilities for taken actions
            new_log_probs = action_dist.log_prob(batch['actions'])
            
            # Compute policy ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy loss (for exploration)
            entropy_loss = -action_dist.entropy().mean()
            
            # Combined loss
            total_loss = (policy_loss + 
                         self.value_loss_coef * value_loss + 
                         self.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                # Clip fraction
                clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | 
                               (ratio > 1.0 + self.clip_epsilon)).float().mean().item()
                
                # KL divergence
                kl_div = (old_log_probs - new_log_probs).mean().item()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item() 
                total_entropy_loss += entropy_loss.item()
                total_clip_fraction += clip_fraction
                total_kl_div += kl_div
        
        # Update learning rate
        self.scheduler.step()
        
        # Compute explained variance
        with torch.no_grad():
            explained_var = 1.0 - torch.var(returns - values.squeeze()) / torch.var(returns)
            explained_var = explained_var.item()
        
        # Update training metrics
        self.training_metrics.update({
            'episodes_trained': self.training_metrics['episodes_trained'] + 1,
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy_loss': total_entropy_loss / self.ppo_epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / self.ppo_epochs,
            'clip_fraction': total_clip_fraction / self.ppo_epochs,
            'kl_divergence': total_kl_div / self.ppo_epochs,
            'explained_variance': explained_var
        })
        
        # Clear memory after update
        self.memory.clear()
        
        self.logger.debug(f"PPO update completed. Policy loss: {self.training_metrics['policy_loss']:.4f}")
        
        # Return metrics in stable-baselines3 format
        metrics = {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy_loss': total_entropy_loss / self.ppo_epochs,
            'total_loss': (total_policy_loss + total_value_loss + total_entropy_loss) / self.ppo_epochs,
            'clip_fraction': total_clip_fraction / self.ppo_epochs,
            'kl_divergence': total_kl_div / self.ppo_epochs,
            'explained_variance': explained_var,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'n_updates': self.training_metrics['episodes_trained'] + 1
        }
        
        # Update internal metrics
        self.training_metrics.update(metrics)
        
        return metrics
    
    def _apply_action_mask(self, 
                          logits: torch.Tensor, 
                          valid_actions_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply action masking to policy logits.
        
        Args:
            logits: Policy logits (batch, 7)
            valid_actions_mask: Binary mask for valid actions (batch, 7)
            
        Returns:
            Masked logits with invalid actions set to -inf
        """
        # Set invalid actions to very negative value
        masked_logits = logits.masked_fill(~valid_actions_mask.bool(), float('-inf'))
        return masked_logits
    
    def set_training_mode(self, training: bool) -> None:
        """
        Set agent training mode.
        
        Args:
            training: If True, agent is in training mode
        """
        self.training = training
        self.network.train(training)
    
    def _get_save_state(self) -> Dict[str, Any]:
        """Get PPO-specific state for saving."""
        return {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_metrics': self.training_metrics,
            'experience_count': self.experience_count,
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'value_loss_coef': self.value_loss_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size
            }
        }
    
    def _set_load_state(self, state: Dict[str, Any]) -> None:
        """Set PPO-specific state from loaded data."""
        if 'network_state_dict' in state:
            self.network.load_state_dict(state['network_state_dict'])
        
        if 'optimizer_state_dict' in state:
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in state:
            self.scheduler.load_state_dict(state['scheduler_state_dict'])
        
        if 'training_metrics' in state:
            self.training_metrics.update(state['training_metrics'])
        
        if 'experience_count' in state:
            self.experience_count = state['experience_count']
        
        # Restore hyperparameters
        if 'hyperparameters' in state:
            hyper = state['hyperparameters']
            self.learning_rate = hyper.get('learning_rate', self.learning_rate)
            self.gamma = hyper.get('gamma', self.gamma)
            self.gae_lambda = hyper.get('gae_lambda', self.gae_lambda)
            self.clip_epsilon = hyper.get('clip_epsilon', self.clip_epsilon)
            self.value_loss_coef = hyper.get('value_loss_coef', self.value_loss_coef)
            self.entropy_coef = hyper.get('entropy_coef', self.entropy_coef)
            self.max_grad_norm = hyper.get('max_grad_norm', self.max_grad_norm)
            self.ppo_epochs = hyper.get('ppo_epochs', self.ppo_epochs)
            self.batch_size = hyper.get('batch_size', self.batch_size)
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get current training metrics."""
        return self.training_metrics.copy()
    
    def reset_episode(self) -> None:
        """Reset agent state for new episode."""
        super().reset_episode()
        # PPO agents don't need special episode reset
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information."""
        info = super().get_info()
        info.update({
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'experience_count': self.experience_count,
            'memory_size': len(self.memory),
            'training_metrics': self.training_metrics,
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
            'device': self.device
        })
        return info


# Update agent registry in base_agent.py
def register_ppo_agent():
    """Register PPO agent in the agent factory."""
    try:
        from .base_agent import create_agent
        # This will be imported when the registry is accessed
        pass
    except ImportError:
        pass