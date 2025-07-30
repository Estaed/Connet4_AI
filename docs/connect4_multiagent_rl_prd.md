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

### ✅ Phase 2: Core Game Implementation

#### ✅ Task 2.1: Pure Game Logic
**Objective:** Implement Connect4 game without any ML dependencies  
**Files:** `src/environments/connect4_game.py`  
**Status:** ✅ **COMPLETED**

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

#### ✅ Task 2.2: Gymnasium Environment Wrapper
**Objective:** Wrap game logic in standard Gymnasium interface  
**Files:** `src/environments/connect4_env.py`  
**Status:** ✅ **COMPLETED**

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

### ✅ Phase 3: Agent Foundation

#### ✅ Task 3.1: Base Agent Interface
**Objective:** Create abstract base class for all agents  
**Files:** `src/agents/base_agent.py`  
**Status:** ✅ **COMPLETED**

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

#### ✅ Task 3.2: Random Agent Implementation
**Objective:** Create baseline random agent for testing  
**Files:** `src/agents/random_agent.py`  
**Status:** ✅ **COMPLETED**

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

#### ✅ Task 3.3: Neural Network Architecture
**Objective:** Design CNN for Connect4 board pattern recognition  
**Files:** `src/agents/networks.py`  
**Status:** ✅ **COMPLETED**

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

### ✅ Phase 4: PPO Agent Implementation

#### ✅ Task 4.1: PPO Agent Core
**Objective:** Implement PPO algorithm for Connect4  
**Files:** `src/agents/ppo_agent.py`  
**Status:** ✅ **COMPLETED**

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

#### ✅ Task 4.2: Action Masking Integration
**Objective:** Implement proper action masking for invalid moves  
**Files:** Update `src/agents/ppo_agent.py`  
**Status:** ✅ **COMPLETED**

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

#### ✅ Task 5.1: Training Loop Implementation
**Objective:** Create main training script with self-play  
**Files:** `scripts/train.py`  
**Status:** ✅ **COMPLETED**

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

#### ✅ Task 5.2: Model Checkpointing
**Objective:** Implement save/load functionality  
**Files:** `src/utils/checkpointing.py`  
**Status:** ✅ **COMPLETED**

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

#### Task 5.3: Multi-Environment Training Optimization
**Objective:** Scale training with multiple parallel environments for faster data collection  
**Files:** Update `scripts/train.py`, create `src/training/multi_env_trainer.py`  
**Status:** 🔄 **PENDING**

**Details:**
- Parallel environment instances for accelerated experience collection
- Vectorized experience gathering from multiple simultaneous games
- Batch processing for multiple Connect4 games running concurrently
- Maintains single PPO agent while scaling data collection
- Significant training speedup through parallelization

**Implementation Structure:**
```python
class MultiEnvTrainer:
    def __init__(self, num_envs=8):
        self.num_envs = num_envs
        self.envs = [Connect4Env() for _ in range(num_envs)]
        self.agent = PPOAgent(device='cuda' if torch.cuda.is_available() else 'cpu')
        
    def collect_experiences_parallel(self):
        # Collect experiences from multiple environments simultaneously
        for env_idx in range(self.num_envs):
            # Run game in parallel, collect experiences
            pass
            
    def train_step(self):
        # Collect experiences from all environments
        # Update agent with batched experiences
        pass
```

**Benefits:**
- **Training Speed**: 4-8x faster experience collection with parallel environments
- **Data Diversity**: More diverse training experiences from simultaneous games
- **Sample Efficiency**: Better sample utilization through batch processing
- **Scalability**: Easy to scale from 1 to N environments based on hardware

**Verification:** Training runs significantly faster with multiple environments, maintains learning quality

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
2. **Game Logic** (Task 2.1) ✅
3. **Gym Environment** (Task 2.2) ✅
4. **Base Agents** (Tasks 3.1-3.2) ✅
5. **Neural Networks** (Task 3.3) ✅
6. **PPO Agent** (Tasks 4.1-4.2) ✅
7. **Training System** (Tasks 5.1-5.2)
8. **Multi-Environment Training** (Task 5.3)
9. **Game Interface** (Task 6.1)
10. **Testing** (Tasks 7.1-7.2)
11. **Documentation** (Tasks 8.1-8.2)

This simplified approach focuses on creating a working, maintainable system rather than over-engineering the solution. Each task builds on the previous ones with clear verification criteria.

---

## 🚀 Future Enhancement Phases

*These phases represent potential future improvements that maintain the project's simplicity principles. They should only be considered after completing Phases 1-8 and ensuring the core system works reliably.*

**📝 Source Attribution:** Ideas marked with 📋 are adapted from `recommendation.md` and `recommendation_gemini.md` analysis, simplified to fit project goals.

### 🔄 Phase 9: Performance & Monitoring (Future)

#### Task 9.1: Performance Benchmarking Scripts 📋
**Objective:** Create simple scripts to measure training and inference performance  
**Files:** `scripts/benchmark.py`, `src/utils/profiling.py`  
**Status:** 🔄 **FUTURE**

**Details:**
- Measure training speed (episodes/minute, games/second)
- Track memory usage during training and inference
- Simple GPU utilization monitoring
- Basic FPS measurements for environment steps
- Compare performance across different hardware configurations

**Implementation:**
```python
def benchmark_training():
    # Simple timing measurements
    start_time = time.time()
    # Run training episodes
    episodes_per_minute = episodes / ((time.time() - start_time) / 60)
    
def benchmark_inference():
    # Measure agent decision speed
    # Track memory usage
    pass
```

**Verification:** Generates clear performance reports without adding complexity

#### Task 9.2: Basic Profiling Tools 📋
**Objective:** Identify performance bottlenecks with simple tools  
**Files:** `src/utils/profiling.py`  
**Status:** 🔄 **FUTURE**

**Details:**
- Function-level timing decorators
- Memory usage tracking at key points
- GPU memory monitoring during training
- Simple bottleneck identification
- Lightweight profiling that doesn't slow down training

**Verification:** Helps identify optimization opportunities without over-engineering

#### Task 9.3: Hyperparameter Tuning Scripts 📋
**Objective:** Simple parameter exploration tools  
**Files:** `scripts/tune_hyperparameters.py`, `configs/tuning/`  
**Status:** 🔄 **FUTURE**

**Details:**
- Basic grid search for learning rates, batch sizes
- Simple random search for PPO parameters
- Configuration file generation for different parameter sets
- Basic result comparison and logging
- Focus on most impactful hyperparameters only

**Implementation:**
```python
def grid_search():
    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [32, 64, 128]
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Run training with parameters
            # Log results
            pass
```

**Verification:** Helps find better hyperparameters without complex optimization algorithms

---

### 🔄 Phase 10: Advanced Tools (Future)

#### Task 10.1: Enhanced Training Metrics
**Objective:** Better training progress tracking beyond basic logging  
**Files:** `src/utils/metrics.py`, update training scripts  
**Status:** 🔄 **FUTURE**

**Details:**
- Win rate tracking over rolling windows
- Episode length statistics
- Loss curves and training stability metrics
- Simple reward progression tracking
- Basic convergence detection

**Implementation:**
```python
class TrainingTracker:
    def __init__(self, window_size=100):
        self.win_rates = []
        self.episode_lengths = []
        self.losses = []
    
    def update(self, won, episode_length, loss):
        # Track metrics over time
        pass
    
    def get_summary(self):
        # Return simple performance summary
        pass
```

**Verification:** Provides better insight into training progress without complexity

#### Task 10.2: Model Comparison Tools
**Objective:** Compare different training runs and model versions  
**Files:** `scripts/compare_models.py`, `src/utils/comparison.py`  
**Status:** 🔄 **FUTURE**

**Details:**
- Head-to-head agent battles (model A vs model B)
- Performance comparison across multiple metrics
- Simple statistical significance testing
- Win rate confidence intervals
- Tournament-style model evaluation

**Implementation:**
```python
def compare_models(model_a, model_b, num_games=1000):
    wins_a = 0
    wins_b = 0
    draws = 0
    
    for game in range(num_games):
        result = play_game(model_a, model_b)
        # Track results
    
    return simple_statistics(wins_a, wins_b, draws)
```

**Verification:** Helps evaluate model improvements objectively

#### Task 10.3: Model Export/Import System
**Objective:** Easy sharing and versioning of trained models  
**Files:** `src/utils/model_io.py`  
**Status:** 🔄 **FUTURE**

**Details:**
- Export models with metadata (training time, performance metrics)
- Import models from different training runs
- Simple model versioning system
- Compatibility checking between model versions
- Lightweight model packaging

**Implementation:**
```python
def export_model(agent, metadata, path):
    package = {
        'model_state': agent.network.state_dict(),
        'config': agent.config,
        'metadata': metadata,
        'version': '1.0'
    }
    torch.save(package, path)

def import_model(path):
    package = torch.load(path)
    # Validate compatibility
    # Return model and metadata
    pass
```

**Verification:** Models can be easily shared and loaded across different environments

#### Task 10.4: Configuration Testing Tools
**Objective:** Validate and test different configuration setups  
**Files:** `scripts/test_configs.py`, `tests/test_configurations.py`  
**Status:** 🔄 **FUTURE**

**Details:**
- Validate configuration file syntax and values
- Test training with different config combinations
- Identify problematic parameter combinations
- Simple config file generation tools
- Basic parameter range validation

**Implementation:**
```python
def validate_config(config_path):
    config = load_config(config_path)
    
    # Check required fields
    # Validate parameter ranges
    # Test for common issues
    
    return validation_results

def test_config_training(config_path, max_episodes=10):
    # Run short training to test config
    # Return success/failure with details
    pass
```

**Verification:** Prevents configuration errors and helps find optimal settings

---

## 🚫 Explicitly Avoided Features

*Based on analysis of over-complex recommendations, these features are explicitly avoided to maintain project simplicity:*

### Complex Features to Avoid:
- **Multiple Network Architectures** - Stick to single CNN approach
- **CuPy Integration** - Unnecessary complexity for negligible gains
- **Complex Memory Management** - Use simple lists instead of sophisticated buffers
- **Multiple Experience Collection Systems** - Single unified approach only
- **Sophisticated Reward Functions** - Keep basic win/loss/draw structure
- **Advanced Logging Systems** - Basic logging and metrics only
- **Multiple Agent Types** - PPO and Random agents are sufficient
- **Performance Micro-optimizations** - Focus on correctness first
- **Complex Configuration Systems** - Simple YAML loading only
- **Manual Mixed-Precision Training** - Let PyTorch handle automatically
- **Over-engineered Device Handling** - Basic CUDA detection is enough

### Complexity Red Flags:
- Files exceeding 300 lines
- Classes with more than 15 methods
- Functions exceeding 50 lines
- Multiple inheritance patterns
- Complex design patterns
- Extensive error handling systems
- Multiple configuration formats

**Principle:** Any feature that doesn't directly contribute to the core learning objective should be carefully evaluated against the simplicity goals. When in doubt, keep it simple.