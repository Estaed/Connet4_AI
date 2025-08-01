# Connect4 AI - Train Your Own Connect4 Champion! 🏆

Ever wanted to build an AI that can crush you at Connect4? Well, here's your chance! This is a fun terminal-based RL system where an AI agent learns to play Connect4 by battling itself thousands of times. It's got all the cool stuff - GPU acceleration, multiple game modes, and you can even watch AIs fight each other!

## 📸 Screenshots

**Main Menu** - Clean and simple interface  
[🖼️ View Main Screen](https://github.com/Estaed/Connet4_AI/blob/master/ScreenShots/Main_Screen.PNG)

**Training in Action** - Watch your AI get smarter  
[🖼️ View Training Screen](https://github.com/Estaed/Connet4_AI/blob/master/ScreenShots/Training_Screen.PNG)

**Model Management** - Browse your AI collection  
[🖼️ View Model Management](https://github.com/Estaed/Connet4_AI/blob/master/ScreenShots/Model_Management.PNG)

**Game Environment** - Play against your creation  
[🖼️ View Game Environment](https://github.com/Estaed/Connet4_AI/blob/master/ScreenShots/Game_Environment.png)

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

### What You'll Need
- Python 3.9+ (anything newer works fine)
- A decent GPU if you want fast training (but CPU works too, just slower)
- About 2GB of RAM (nothing crazy)
- Maybe 1GB GPU memory if you're using the GPU

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
- Classic Connect4 - just you and a friend going at it
- Perfect for when you want to settle who's the real Connect4 master

**2. Human vs AI** 🤖
- Time to face your creation! Challenge the AI you trained
- Pick different skill levels and watch it think
- Pro tip: Don't get too cocky, these AIs get scary good

**3. AI vs AI** 🏆
- The ultimate showdown - watch your AIs battle each other
- Compare different versions and see which training session was better
- Grab some popcorn, it's actually pretty entertaining

**4. Random vs Random** 🎲
- Two AIs just randomly placing pieces (as dumb as it sounds)
- Good for testing stuff and making yourself feel better about your skills

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

**Small (100 environments)** - Baby's first AI
- Quick and easy, great for testing
- Your AI will learn basic moves pretty fast (~1,000 episodes)
- Perfect if you just want to see it work

**Medium (1,000 environments)** - The sweet spot
- Best for most people - not too slow, not too demanding
- Develops some serious strategies in about 5,000 episodes
- Recommended unless you're feeling ambitious

**Impossible (10,000 environments)** - Beast mode
- Hope you've got a good GPU because this will put it to work
- Creates genuinely scary-good AIs after 10,000+ episodes
- Only try this if you want to create a Connect4 monster

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

## 🤝 Want to Help Out?

This project is pretty clean and well-organized (if I do say so myself):

- **Type Hints**: Everything's properly typed so your IDE won't yell at you
- **Documentation**: Lots of comments and docs so you know what's going on
- **Testing**: Tons of tests so things don't randomly break
- **Code Quality**: Formatted with Black, linted, all that good stuff
- **Clean Code**: Each part does its own thing, no spaghetti here

## 📄 License

Hey, this is free stuff! 🎉

You can use this code however you want, whenever you want, for whatever you want. Want to modify it? Go for it! Want to sell it? Sure! Want to use it in your own projects? Absolutely!

No strings attached, no complicated legal stuff. Just take it and have fun with it. If you make something cool, I'd love to hear about it, but you don't have to.

Basically: **Do whatever you want with this code!** ✨

## 🔗 References

- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**Ready to create your Connect4 AI overlord? Just run `python scripts/play.py` and let's see what you can build!** 🚀

Have fun, and don't blame me if your AI becomes unbeatable! 😄
