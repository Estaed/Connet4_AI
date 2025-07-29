# Connect4 RL Training System - Simplified Project Plan

**Version:** 1.0.0  
**Date:** July 2025  
**Project Goal:** To create a straightforward, terminal-based Reinforcement Learning environment for Connect4. An AI agent will learn to play the game through self-play using the PPO algorithm with simple, modular code structure.

---

## 🏗️ Simplified Project Structure

This project is designed to be simple and manageable, with clear separation of concerns and minimal file complexity.

```
Connect4_AI/
├── src/
│   ├── environments/
│   │   ├── __init__.py
│   │   ├── connect4_env.py          # Core Gymnasium environment
│   │   └── connect4_game.py         # Pure game logic (no gym dependencies)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── ppo_agent.py             # PPO implementation with action masking
│   │   ├── networks.py              # Neural network architectures
│   │   ├── random_agent.py          # Baseline random agent
│   │   └── base_agent.py            # Abstract agent interface
│   ├── utils/
│   │   ├── __init__.py
│   │   └── checkpointing.py         # Model saving and loading
│   └── core/
│       ├── __init__.py
│       └── config.py                # Configuration management
├── configs/
│   └── default_config.yaml          # Default hyperparameters
├── scripts/
│   ├── train.py                     # Main training script
│   └── play.py                      # Game interface (random vs random, human vs human, human vs ai)
├── tests/
├── notebooks/
│   ├── strategy_analysis.ipynb      # Analyze learned strategies
│   └── training_visualization.ipynb # Visualize training progress
├── docs/
│   ├── CLAUDE.md
│   └── connect4_multiagent_rl_prd.md
├── models/                          # Saved model checkpoints
├── logs/                           # Training logs and metrics
├── requirements.txt                # Python dependencies
└── README.md
```

---

## 🎯 Core Components

### 1. Connect4 Game Logic (connect4_game.py)
- **Pure Python Game**: No dependencies on Gymnasium, just the game rules
- **Board Representation**: 6×7 NumPy array with 0 (empty), 1 (player 1), -1 (player 2)
- **Game Rules**: Drop pieces, check wins, detect draws
- **CPU Only**: Game logic always runs on CPU for simplicity

### 2. Gymnasium Environment (connect4_env.py)
- **Gym Wrapper**: Wraps the pure game logic in Gymnasium interface
- **Standard API**: Compatible with any RL library
- **Action Masking**: Prevents invalid moves
- **CPU Based**: Environment operations on CPU

### 3. Agents (agents/)
- **PPO Agent**: Main learning agent using PyTorch
- **Neural Networks**: Simple CNN for board pattern recognition
- **Random Agent**: Baseline for comparison
- **CUDA Training**: Neural networks use GPU when available for training

### 4. Training System
- **Self-Play**: Agent plays against itself
- **GPU Acceleration**: Neural network training uses CUDA when available
- **Checkpointing**: Save and load trained models
- **Simple Logging**: Basic progress tracking

---

## 🔧 Technical Specifications

### Hardware Requirements
- **Game Environment**: Always runs on CPU for consistency
- **Training**: Uses CUDA GPU when available, falls back to CPU
- **Memory**: ~2GB RAM, ~1GB GPU memory (if using GPU)

### Dependencies
```txt
# Core libraries
numpy>=1.24.0
torch>=2.0.0
gymnasium>=0.29.0

# Configuration
pyyaml>=6.0

# Utilities
tqdm>=4.65.0

# Optional for notebooks
matplotlib>=3.7.0
jupyter>=1.0.0
```

---

## 🗺️ Implementation Plan - Task Breakdown

### ✅ Phase 1: Foundation Setup

#### Task 1.1: Project Configuration
**Objective:** Set up centralized configuration system  
**Files:** `src/core/config.py`, `configs/default_config.yaml`  
**Status:** ✅ **COMPLETED**

**Details:**
- Create YAML-based configuration for all hyperparameters
- Define board dimensions, training parameters, PPO settings
- Set up device configuration (CPU for game, GPU detection for training)
- File paths and logging settings

**Verification:** Configuration loads correctly and validates parameters

---

### 🔄 Phase 2: Core Game Implementation

#### Task 2.1: Pure Game Logic
**Objective:** Implement Connect4 game without any ML dependencies  
**Files:** `src/environments/connect4_game.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Create `Connect4Game` class with basic game mechanics
- Implement board representation (6×7 NumPy array)
- Add piece dropping logic with gravity
- Win detection (horizontal, vertical, diagonal)
- Draw detection (board full)
- Valid move checking
- Basic terminal rendering for debugging

**Key Requirements:**
```python
class Connect4Game:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1
    
    def drop_piece(self, col):
        # Drop piece in column, return success/failure
    
    def check_win(self):
        # Check if current player won
    
    def is_draw(self):
        # Check if board is full
    
    def get_valid_moves(self):
        # Return list of valid columns
    
    def reset(self):
        # Reset board for new game
```

**Verification:** Can play complete games with win/draw detection

#### Task 2.2: Gymnasium Environment Wrapper
**Objective:** Wrap game logic in standard Gymnasium interface  
**Files:** `src/environments/connect4_env.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Create `Connect4Env` class inheriting from `gym.Env`
- Define observation space: `Box(low=-1, high=1, shape=(6,7), dtype=np.int8)`
- Define action space: `Discrete(7)` for column selection
- Implement `step()`, `reset()`, `render()` methods
- Add action masking in info dict
- Reward structure: +1 win, 0 draw, -1 loss, -0.01 per move

**Key Implementation:**
```python
class Connect4Env(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(6, 7), dtype=np.int8)
        self.game = Connect4Game()
    
    def step(self, action):
        # Execute action, return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # Reset environment, return initial obs and info
    
    def render(self, mode='human'):
        # Terminal-based rendering
```

**Verification:** Environment passes Gymnasium API tests

---

### 🔄 Phase 3: Agent Foundation

#### Task 3.1: Base Agent Interface
**Objective:** Create abstract base class for all agents  
**Files:** `src/agents/base_agent.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Define abstract `BaseAgent` class
- Standard interface: `get_action()`, `update()`, `save()`, `load()`
- Device management (CPU/GPU)
- Common utilities for all agents

**Implementation:**
```python
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, device='cpu'):
        self.device = device
    
    @abstractmethod
    def get_action(self, observation, valid_actions=None):
        # Return action given observation
        pass
    
    @abstractmethod
    def update(self, experiences):
        # Update agent from experiences (for learning agents)
        pass
    
    def save(self, path):
        # Save agent state
        pass
    
    def load(self, path):
        # Load agent state
        pass
```

**Verification:** Interface is clear and extensible

#### Task 3.2: Random Agent Implementation
**Objective:** Create baseline random agent for testing  
**Files:** `src/agents/random_agent.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Implement simple random agent that chooses valid moves randomly
- Use for baseline comparisons and environment testing
- No learning capability, just random valid moves

**Implementation:**
```python
class RandomAgent(BaseAgent):
    def get_action(self, observation, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(7))
        return np.random.choice(valid_actions)
    
    def update(self, experiences):
        pass  # No learning for random agent
```

**Verification:** Agent can play games without errors

#### Task 3.3: Neural Network Architecture
**Objective:** Design CNN for Connect4 board pattern recognition  
**Files:** `src/agents/networks.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Create simple CNN architecture for 6×7 board
- Separate policy and value heads
- Optimized for GPU training
- Action masking support

**Architecture:**
```python
class Connect4Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (batch, 1, 6, 7)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.policy_head = nn.Linear(128, 7)  # 7 possible moves
        self.value_head = nn.Linear(128, 1)   # State value
    
    def forward(self, x):
        conv_out = self.conv_layers(x)
        # Global average pooling
        pooled = conv_out.mean(dim=[2, 3])
        
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy_logits, value
```

**Verification:** Network processes board states correctly

---

### 🔄 Phase 4: PPO Agent Implementation

#### Task 4.1: PPO Agent Core
**Objective:** Implement PPO algorithm for Connect4  
**Files:** `src/agents/ppo_agent.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Implement PPO with action masking
- Experience collection and storage
- Policy and value loss computation
- GPU training support
- Simple memory buffer

**Key Components:**
```python
class PPOAgent(BaseAgent):
    def __init__(self, device='cpu'):
        super().__init__(device)
        self.network = Connect4Network().to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.memory = []
    
    def get_action(self, observation, valid_actions=None):
        # Get action using current policy with masking
    
    def store_experience(self, obs, action, reward, next_obs, done):
        # Store experience for training
    
    def update(self):
        # PPO update from stored experiences
```

**Verification:** Agent can learn from self-play games

#### Task 4.2: Action Masking Integration
**Objective:** Implement proper action masking for invalid moves  
**Files:** Update `src/agents/ppo_agent.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Mask invalid actions in policy distribution
- Ensure agent never chooses illegal moves
- Proper gradient flow for valid actions only

**Implementation:**
```python
def apply_action_mask(self, logits, valid_actions):
    mask = torch.full_like(logits, float('-inf'))
    mask[valid_actions] = 0
    return logits + mask
```

**Verification:** Agent never attempts invalid moves

---

### 🔄 Phase 5: Training Infrastructure

#### Task 5.1: Training Loop Implementation
**Objective:** Create main training script with self-play  
**Files:** `scripts/train.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Self-play training loop
- Experience collection from games
- Periodic model updates
- Progress logging and visualization
- Checkpoint saving

**Training Structure:**
```python
def train_agent():
    env = Connect4Env()
    agent = PPOAgent(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for episode in range(config.max_episodes):
        # Play game collecting experiences
        obs, info = env.reset()
        done = False
        
        while not done:
            valid_actions = info['valid_moves']
            action = agent.get_action(obs, valid_actions)
            next_obs, reward, done, _, info = env.step(action)
            
            agent.store_experience(obs, action, reward, next_obs, done)
            obs = next_obs
        
        # Update agent periodically
        if episode % config.update_frequency == 0:
            agent.update()
        
        # Save checkpoint
        if episode % config.checkpoint_interval == 0:
            agent.save(f'models/checkpoint_{episode}.pth')
```

**Verification:** Training runs without errors and agent improves

#### Task 5.2: Model Checkpointing
**Objective:** Implement save/load functionality  
**Files:** `src/utils/checkpointing.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Save model weights and training state
- Load models for continued training or evaluation
- Checkpoint management (keep recent, clean old)

**Implementation:**
```python
class CheckpointManager:
    def save_checkpoint(self, agent, episode, metrics, path):
        torch.save({
            'model_state_dict': agent.network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'episode': episode,
            'metrics': metrics
        }, path)
    
    def load_checkpoint(self, agent, path):
        checkpoint = torch.load(path)
        agent.network.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode'], checkpoint['metrics']
```

**Verification:** Can save and resume training from checkpoints

---

### 🔄 Phase 6: Game Interface

#### Task 6.1: Interactive Game Script
**Objective:** Create playable interface for testing agents  
**Files:** `scripts/play.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Terminal-based game interface
- Three game modes:
  1. Random vs Random (for testing)
  2. Human vs Human (for fun)
  3. Human vs AI (main feature)
- Clear board visualization
- Input validation and error handling

**Menu Structure:**
```python
def main_menu():
    print("Connect4 Game")
    print("1. Random vs Random")
    print("2. Human vs Human")
    print("3. Human vs AI")
    choice = input("Select mode (1-3): ")
    
    if choice == '1':
        play_random_vs_random()
    elif choice == '2':
        play_human_vs_human()
    elif choice == '3':
        play_human_vs_ai()
```

**Verification:** All game modes work correctly with good user experience

---

### 🔄 Phase 7: Testing and Validation

#### Task 7.1: Unit Tests
**Objective:** Create comprehensive test suite  
**Files:** `tests/test_*.py`  
**Status:** 🔄 **PENDING**

**Test Coverage:**
- Game logic correctness
- Environment API compliance
- Agent interfaces
- Training stability
- Save/load functionality

**Verification:** All tests pass with >90% code coverage

#### Task 7.2: Integration Tests
**Objective:** Test complete system integration  
**Files:** `tests/test_integration.py`  
**Status:** 🔄 **PENDING**

**Integration Tests:**
- End-to-end training pipeline
- Human vs AI gameplay
- Model loading and inference
- Configuration system

**Verification:** System works as integrated whole

---

### 🔄 Phase 8: Documentation and Polish

#### Task 8.1: Documentation
**Objective:** Create user and developer documentation  
**Files:** `README.md`, `docs/CLAUDE.md`  
**Status:** 🔄 **PENDING**

**Documentation:**
- Installation and setup guide
- Usage instructions
- Training guide
- API documentation
- Troubleshooting

**Verification:** New users can successfully train and play

#### Task 8.2: Code Quality
**Objective:** Clean up code and add polish  
**Files:** All source files  
**Status:** 🔄 **PENDING**

**Quality Improvements:**
- Code formatting and style
- Type hints
- Docstrings
- Error handling
- Performance optimization

**Verification:** Code is clean, well-documented, and maintainable

---

## 🚀 How to Run the Project

### Installation:
```bash
# Clone repository
git clone <repository-url>
cd Connect4_AI

# Install dependencies
pip install -r requirements.txt
```

### Training:
```bash
# Start training (will use GPU if available)
cd scripts
python train.py
```

### Playing:
```bash
# Launch game interface
cd scripts
python play.py
# Choose from 3 game modes
```

---

## 🏆 Success Criteria

### Learning Milestones:
- **Episode 100**: Agent makes only legal moves
- **Episode 1000**: Agent shows basic strategic play
- **Episode 5000**: Agent wins >80% vs random player
- **Episode 10000**: Agent shows advanced strategies

### Technical Goals:
- **Stability**: Training runs without crashes
- **Performance**: Reasonable training speed on both CPU/GPU
- **Usability**: Easy to install and run
- **Maintainability**: Clean, simple codebase

---

## 📝 Implementation Notes

### Key Design Principles:
1. **Simplicity First**: Keep code simple and readable
2. **Modular Design**: Clear separation of concerns
3. **CPU/GPU Flexibility**: Works on both, optimized for GPU
4. **Incremental Development**: Build and test one component at a time
5. **Practical Focus**: Prioritize working system over perfect optimization

### Development Order:
1. **Config System** (Task 1.1) ✅
2. **Game Logic** (Task 2.1)
3. **Gym Environment** (Task 2.2)
4. **Base Agents** (Tasks 3.1-3.2)
5. **Neural Networks** (Task 3.3)
6. **PPO Agent** (Tasks 4.1-4.2)
7. **Training System** (Tasks 5.1-5.2)
8. **Game Interface** (Task 6.1)
9. **Testing** (Tasks 7.1-7.2)
10. **Documentation** (Tasks 8.1-8.2)

This simplified approach focuses on creating a working, maintainable system rather than over-engineering the solution. Each task builds on the previous ones with clear verification criteria.