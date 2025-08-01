# Connect4 AI - Reinforcement Learning Training System

A sophisticated, terminal-based reinforcement learning environment where an AI agent learns to master Connect4 through self-play using the PPO (Proximal Policy Optimization) algorithm. The system features hybrid CPU-GPU architecture, massive parallel training capabilities, and an intuitive game interface.

## 🎯 Project Overview

This Connect4 RL system demonstrates modern reinforcement learning techniques with:

- **Self-Play Learning**: AI agent improves by playing against itself
- **Hybrid Architecture**: CPU game logic with GPU neural network acceleration  
- **Massive Scalability**: Support for up to 10,000 parallel training environments
- **Simple Design**: Clean, modular codebase that's easy to understand and modify
- **Multiple Game Modes**: Human vs AI, AI vs AI, and more
- **Advanced Model Management**: Save, load, and compare different AI models

### Key Features

🧠 **Smart AI Agent**
- PPO algorithm with action masking (prevents invalid moves)
- CNN neural network for board pattern recognition
- Learns strategic gameplay through self-play

⚡ **High-Performance Training**
- Hybrid vectorized training with 100-10,000 parallel environments
- Three difficulty levels: Small (100), Medium (1,000), Impossible (10,000)
- GPU acceleration for neural networks, CPU for game logic
- Automatic model checkpointing with compression

🎮 **Interactive Game Interface**
- 4 game modes: Human vs Human, Human vs AI, AI vs AI, Random vs Random
- Real-time AI decision visualization
- Comprehensive match statistics and win rate analysis
- Model selection and management system

📊 **Training & Analytics**
- Real-time training progress monitoring
- Detailed performance statistics and metrics
- Model comparison and evaluation tools
- Training visualization and analysis

## 🚀 Quick Setup

### Prerequisites
- Python 3.9+
- CUDA-capable (CUDA 11.8+) GPU (optional, will fallback to CPU)
- ~2GB RAM, ~1GB GPU memory (if using GPU)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Connect4_AI
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU support (optional)**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 🎮 How to Play

### Launch the Game Interface
```bash
cd scripts
python play.py

# For detailed debug logging (developers)
python play.py --debug
```

### Game Modes

**1. Human vs Human** 🎯
- Classic Connect4 gameplay between two players
- Perfect for learning the game or having fun with friends

**2. Human vs AI** 🤖
- Challenge trained AI models
- Choose from different AI skill levels
- Watch the AI "think" and make strategic decisions

**3. AI vs AI** 🏆
- Watch two AI models compete against each other
- Compare different training checkpoints
- Analyze AI strategies and improvement over time
- Configure match series (1-10 games)

**4. Random vs Random** 🎲
- Baseline gameplay with random moves
- Useful for testing and environment validation

### Additional Features

**🔧 Model Management**
- Browse and select from trained AI models
- View model statistics and training history
- Load models from different training sessions

**📈 Training Interface**
- Start new training sessions
- Choose training difficulty (Small/Medium/Impossible)
- Monitor real-time training progress
- Automatic model saving and checkpointing

**📊 Statistics & Analytics**
- View detailed match results and win rates
- Compare AI model performance
- Track training progress over time

## 🏋️ Training Your Own AI

### Quick Training Start
```bash
cd scripts
python play.py
# Select "5. Start Training" from the menu
```

### Training Difficulty Levels

**Small (100 environments)** - Perfect for beginners
- Fast training cycles
- Good for testing and experimentation
- Learns basic gameplay in ~1,000 episodes

**Medium (1,000 environments)** - Balanced performance
- Recommended for most users
- Good balance between speed and learning quality
- Develops strong strategies in ~5,000 episodes

**Impossible (10,000 environments)** - Maximum performance
- Requires powerful hardware (GPU recommended)
- Highest quality learning
- Develops advanced strategies in ~10,000 episodes

### Training Milestones
- **Episode 100**: Agent learns to make only legal moves
- **Episode 1,000**: Shows basic strategic thinking
- **Episode 5,000**: Wins 80%+ against random players
- **Episode 10,000**: Demonstrates advanced Connect4 strategies

## 🏗️ Project Architecture

### Core Components

**Game Engine** (`src/environments/`)
- `connect4_game.py`: Pure Python game logic (no ML dependencies)
- `hybrid_vectorized_connect4.py`: High-performance vectorized environments

**AI Agents** (`src/agents/`)
- `ppo_agent.py`: PPO reinforcement learning agent
- `networks.py`: CNN neural network architectures
- `random_agent.py`: Baseline random agent for comparison

**Training System** (`src/training/`)
- `hybrid_trainer.py`: Advanced multi-environment training
- `training_interface.py`: User-friendly training interface
- `training_statistics.py`: Comprehensive metrics tracking

**Utilities** (`src/utils/`)
- `model_manager.py`: AI model management and loading
- `checkpointing.py`: Model saving and restoration
- `render.py`: Game visualization and display

### Directory Structure
```
Connect4_AI/
├── src/                    # Source code
│   ├── environments/       # Game logic and environments
│   ├── agents/            # AI agents and neural networks
│   ├── training/          # Training infrastructure
│   ├── utils/             # Utilities and helpers
│   └── core/              # Configuration management
├── scripts/               # Entry points
│   ├── play.py           # Main game interface
│   └── run_tests.py      # Test runner
├── configs/               # Configuration files
├── models/                # Saved AI models
├── logs/                  # Training logs
├── tests/                 # Comprehensive test suite
└── docs/                  # Documentation
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
cd scripts
python run_tests.py
```

Test coverage includes:
- ✅ Game logic correctness
- ✅ Environment API compliance  
- ✅ Agent interfaces and behavior
- ✅ Training system stability
- ✅ Model save/load functionality
- ✅ Performance benchmarks

## 🔧 Configuration

The system uses YAML configuration files in `configs/`:

**`default_config.yaml`** - Main configuration
- Training hyperparameters (learning rate, batch size, etc.)
- Network architecture settings
- Environment configuration
- Device settings (CPU/GPU)

Modify these files to customize training behavior and model architecture.

### Debug Mode

For developers and troubleshooting, enable debug mode to see detailed initialization logs:

```bash
# Standard mode (clean interface)
python scripts/play.py

# Debug mode (detailed logging)
python scripts/play.py --debug
python scripts/play.py --debug-mode  # alias
```

**Debug mode shows:**
- Model manager initialization details
- Checkpoint manager verbose logging
- Model loading progress and metadata
- System component initialization steps

**Standard mode (default):**
- Clean, user-friendly interface
- Minimal logging for better experience
- Professional appearance for end users

## 🚀 Performance

**Training Speed**
- Small: ~100-500 episodes/minute
- Medium: ~50-200 episodes/minute  
- Impossible: ~10-50 episodes/minute (GPU recommended)

**System Requirements**
- **Minimum**: 4GB RAM, any CPU
- **Recommended**: 8GB RAM, CUDA GPU
- **Maximum**: 16GB+ RAM, RTX 3080+ GPU

## 📚 Technical Details

**Reinforcement Learning**
- Algorithm: PPO (Proximal Policy Optimization)
- Network: Convolutional Neural Network (CNN)
- Training: Self-play with experience replay
- Action Space: 7 discrete actions (column selection)
- Observation Space: 6×7 board state

**Architecture**
- **Hybrid Design**: CPU handles game logic, GPU accelerates neural networks
- **Vectorized Environments**: Massive parallel training capability
- **Action Masking**: Prevents invalid moves during training
- **Modular Design**: Clean separation between game, agent, and training

## 🤝 Contributing

This project follows clean, maintainable coding practices:

- **Type Hints**: Full typing support throughout
- **Documentation**: Comprehensive docstrings
- **Testing**: >90% test coverage
- **Code Quality**: Black formatting, mypy type checking
- **Modular Design**: Clear separation of concerns

## 📄 License

MIT License

Copyright (c) 2025 Connect4 AI Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHERS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🔗 References

- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**Ready to train your Connect4 AI champion? Run `python scripts/play.py` and start your journey!** 🚀