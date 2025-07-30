"""
Hybrid Vectorized Connect4 Environment

This module provides a hybrid implementation where:
- Game logic runs on CPU (for simplicity and consistency)  
- Neural network operations use GPU (for speed)
- Data transfers are minimized and batched efficiently
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .connect4_game import Connect4Game


class HybridVectorizedConnect4:
    """
    Hybrid vectorized Connect4 environment.
    
    - Game logic: CPU (using existing Connect4Game)
    - Neural networks: GPU 
    - Efficient data transfer between CPU/GPU
    """
    
    def __init__(self, num_envs: int = 100, device: str = 'cuda'):
        """
        Initialize hybrid vectorized Connect4 environments.
        
        Args:
            num_envs: Number of parallel environments
            device: Device for neural network operations ('cpu' or 'cuda')
        """
        self.num_envs = num_envs
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"Initializing {num_envs} environments:")
        print(f"  - Game logic: CPU")
        print(f"  - Neural networks: {self.device}")
        
        # CPU-based game environments
        self.envs = [Connect4Game() for _ in range(num_envs)]
        
        # Pre-allocated tensors for efficient GPU transfer
        self.obs_buffer = torch.zeros((num_envs, 6, 7), dtype=torch.float32, device=self.device)
        self.reward_buffer = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.done_buffer = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        
    def reset(self, env_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        Reset specified environments or all environments.
        
        Args:
            env_ids: List of environment IDs to reset (None = reset all)
            
        Returns:
            Board states as GPU tensor
        """
        if env_ids is None:
            env_ids = list(range(self.num_envs))
        
        # Reset CPU environments
        observations = []
        for env_id in env_ids:
            obs = self.envs[env_id].reset()
            observations.append(obs)
        
        # Batch transfer to GPU
        if observations:
            obs_array = np.stack(observations).astype(np.float32)
            self.obs_buffer[env_ids] = torch.from_numpy(obs_array).to(self.device)
        
        return self.obs_buffer[env_ids].clone()
    
    def get_valid_moves_batch(self) -> List[List[int]]:
        """
        Get valid moves for all environments (CPU operation).
        
        Returns:
            List of valid moves for each environment
        """
        valid_moves = []
        for env in self.envs:
            if env.game_over:
                valid_moves.append([])
            else:
                valid_moves.append(env.get_valid_moves())
        return valid_moves
    
    def get_valid_moves_tensor(self) -> torch.Tensor:
        """
        Get valid moves as GPU tensor for neural network processing.
        
        Returns:
            Boolean tensor [num_envs, 7] on GPU
        """
        valid_moves_list = self.get_valid_moves_batch()
        
        # Convert to tensor
        valid_tensor = torch.zeros((self.num_envs, 7), dtype=torch.bool, device=self.device)
        for env_id, moves in enumerate(valid_moves_list):
            if moves:  # Not game over
                valid_tensor[env_id, moves] = True
        
        return valid_tensor
    
    def step_batch(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute actions in all environments.
        
        Args:
            actions: GPU tensor of actions (will be transferred to CPU for game logic)
            
        Returns:
            Tuple of (observations, rewards, dones, info_dict) - all on GPU
        """
        # Transfer actions to CPU for game logic
        if actions.device != torch.device('cpu'):
            actions_cpu = actions.detach().cpu().numpy()
        else:
            actions_cpu = actions.numpy()
        
        # Process on CPU
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for env_id in range(self.num_envs):
            env = self.envs[env_id]
            action = int(actions_cpu[env_id])
            
            if env.game_over:
                # Environment already finished
                observations.append(env.board.copy())
                rewards.append(0.0)
                dones.append(True)
                infos.append({'already_done': True})
                continue
            
            # Validate and execute move
            if not env.is_valid_move(action):
                # Invalid move penalty
                observations.append(env.board.copy())
                rewards.append(-1.0)
                dones.append(True)
                env.game_over = True
                infos.append({'invalid_move': True, 'action': action})
                continue
            
            # Execute valid move
            env.drop_piece(action)
            observations.append(env.board.copy())
            
            # Calculate reward
            if env.game_over:
                if env.winner == env.current_player:
                    # Note: current_player switched after move, so winner is the previous player
                    rewards.append(1.0 if env.winner == 1 else -1.0)  # Win/loss from player 1 perspective
                    infos.append({'winner': env.winner})
                else:
                    rewards.append(0.0)  # Draw
                    infos.append({'draw': True})
                dones.append(True)
            else:
                rewards.append(-0.01)  # Small step penalty
                dones.append(False)
                infos.append({'continue': True})
        
        # Batch transfer to GPU
        obs_array = np.stack(observations).astype(np.float32)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array(dones, dtype=bool)
        
        # Update GPU buffers
        self.obs_buffer.copy_(torch.from_numpy(obs_array))
        self.reward_buffer.copy_(torch.from_numpy(rewards_array))
        self.done_buffer.copy_(torch.from_numpy(dones_array))
        
        # Summary info
        info_dict = {
            'num_wins': sum(1 for info in infos if info.get('winner') == 1),
            'num_losses': sum(1 for info in infos if info.get('winner') == -1),
            'num_draws': sum(1 for info in infos if 'draw' in info),
            'num_invalid': sum(1 for info in infos if 'invalid_move' in info),
            'num_active': sum(1 for done in dones if not done)
        }
        
        return self.obs_buffer.clone(), self.reward_buffer.clone(), self.done_buffer.clone(), info_dict
    
    def get_observations_gpu(self) -> torch.Tensor:
        """Get current observations as GPU tensor (efficient for neural networks)."""
        return self.obs_buffer.clone()
    
    def get_observations_cpu(self) -> np.ndarray:
        """Get current observations as CPU numpy array."""
        observations = np.array([env.board for env in self.envs])
        return observations
    
    def get_game_states(self) -> List[Dict[str, Any]]:
        """Get detailed game states (CPU operation)."""
        return [env.get_game_state() for env in self.envs]
    
    def auto_reset_finished_games(self) -> torch.Tensor:
        """
        Automatically reset finished games and return which environments were reset.
        
        Returns:
            Boolean tensor indicating which environments were reset
        """
        finished_envs = []
        for env_id, env in enumerate(self.envs):
            if env.game_over:
                finished_envs.append(env_id)
        
        if finished_envs:
            self.reset(finished_envs)
            reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            reset_mask[finished_envs] = True
            return reset_mask
        
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        total_games = sum(env.total_games if hasattr(env, 'total_games') else 0 for env in self.envs)
        finished_games = sum(1 for env in self.envs if env.game_over)
        total_moves = sum(env.move_count for env in self.envs)
        
        return {
            'num_envs': self.num_envs,
            'device': str(self.device),
            'games_finished': finished_games,
            'total_moves': total_moves,
            'avg_moves_per_game': total_moves / max(finished_games, 1),
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        }


# Example usage comparing different approaches
if __name__ == "__main__":
    import time
    
    num_envs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing hybrid approach with {num_envs} environments")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    # Initialize hybrid environment
    start_time = time.time()
    hybrid_env = HybridVectorizedConnect4(num_envs=num_envs, device=device)
    init_time = time.time() - start_time
    print(f"Initialization: {init_time:.3f}s")
    
    # Reset all environments
    start_time = time.time()
    observations = hybrid_env.reset()
    reset_time = time.time() - start_time
    print(f"Reset: {reset_time:.3f}s")
    print(f"Observations shape: {observations.shape}, device: {observations.device}")
    
    # Benchmark steps
    num_steps = 50
    start_time = time.time()
    
    for step in range(num_steps):
        # Get valid moves (CPU operation)
        valid_moves_tensor = hybrid_env.get_valid_moves_tensor()
        
        # Generate random actions (GPU operation)
        actions = torch.zeros(num_envs, dtype=torch.long, device=device)
        for env_id in range(num_envs):
            valid_cols = torch.where(valid_moves_tensor[env_id])[0]
            if len(valid_cols) > 0:
                actions[env_id] = valid_cols[torch.randint(len(valid_cols), (1,))]
        
        # Step (hybrid CPU/GPU operation)
        obs, rewards, dones, info = hybrid_env.step_batch(actions)
        
        # Auto-reset finished games
        reset_mask = hybrid_env.auto_reset_finished_games()
        
        if step % 10 == 0:
            print(f"Step {step}: Active games: {info['num_active']}, "
                  f"Wins: {info['num_wins']}, Draws: {info['num_draws']}")
    
    step_time = time.time() - start_time
    print(f"\n{num_steps} steps: {step_time:.3f}s ({step_time/num_steps*1000:.1f}ms per step)")
    print(f"Environment-steps per second: {num_envs * num_steps / step_time:.0f}")
    
    # Final statistics
    stats = hybrid_env.get_statistics()
    print(f"Final stats: {stats}")