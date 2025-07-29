# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 PROJECT OVERVIEW

**Connect4 RL Training System** - A straightforward, terminal-based reinforcement learning environment for Connect4 where an AI agent learns to play the game through self-play using the PPO algorithm with simple, modular code structure.

### Project Context
This is a simplified Connect4 RL project focused on:
- **Simple Implementation**: Clear, manageable code structure with minimal complexity
- **Self-Play Learning**: Agent learns through playing against itself
- **PPO Algorithm**: Proximal Policy Optimization with action masking
- **CPU/GPU Flexibility**: Works on CPU, uses GPU when available for neural networks
- **Terminal Interface**: Simple text-based game interface and training visualization

The project aims to create a working RL system that's easy to understand and modify.

## 🔄 **DEVELOPMENT ENVIRONMENT SETUP**

**CRITICAL**: Activate the virtual environment and ensure GPU support for Python operations:

```bash
# Navigate to project directory 
cd Connect4_AI

# Activate virtual environment (Windows)
.\venv\Scripts\Activate.ps1

# Verify activation - should see (venv) in terminal prompt
# Verify GPU support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**When to use venv:**
- ✅ Running training: `python scripts/train.py`
- ✅ Running tests: `pytest`
- ✅ Code quality tools: `black .`, `mypy .`, `flake8 .`, `isort .`
- ✅ Installing dependencies: `pip install -r requirements.txt`
- ✅ GPU benchmarking: `python scripts/benchmark_env.py`

**When venv is NOT needed:**
- ❌ File editing/reading
- ❌ Git operations: `git add`, `git commit`, `git push`
- ❌ Basic file operations

**Hardware Requirements:**
- **Game Environment**: Always runs on CPU for consistency
- **Training**: Uses CUDA GPU when available, falls back to CPU
- **Memory**: ~2GB RAM, ~1GB GPU memory (if using GPU)

### 🚨 **CRITICAL QUESTIONS RULE**

**When you have REALLY IMPORTANT questions about the project or during task execution:**

- **YOU CAN ASK QUESTIONS even if it stops the generation progress**
- **Important questions include:**
  - Architecture decisions that affect multiple components
  - Unclear requirements that could lead to wrong implementation
  - Breaking changes that might affect existing functionality
  - Technical approach questions when multiple valid options exist
  - Data model or API design questions
  - Integration concerns with existing systems
- **When to ask**: If the question is critical for correct implementation
- **How to ask**: Stop generation, ask the question clearly, wait for clarification
- **Priority**: Getting the right answer is more important than continuous generation

**Example scenarios where you SHOULD ask:**
- "Should I use vectorized environments or sequential for this implementation? This affects training speed"
- "Should I implement custom CUDA kernels or stick with PyTorch operations? This impacts performance"
- "The task mentions GPU optimization but I see multiple approaches - which method do you prefer?"

**Remember**: It's better to ask and get it right than to implement the wrong solution.

### 🎯 **TASK-BASED DEVELOPMENT WORKFLOW**

#### Before Starting ANY Development Task:

1. **📋 ANALYZE THE TASK**
   - Read the task description completely
   - Understand requirements and acceptance criteria
   - Check dependencies and prerequisites
   - Identify which files/modules will be affected

2. **🧠 THINKING LEVELS FOR PROBLEM COMPLEXITY - UPDATED RULES**
   - **DEFAULT LEVEL**: Always start with **"think"** as the baseline for ALL tasks
   - **think**: Default level - ALL tasks start here, straightforward implementation
   - **think hard**: Multi-step tasks, moderate complexity  
   - **think harder**: Complex analysis, integration challenges, architecture decisions
   - **ultrathink**: Critical system changes, cross-project impacts, complex debugging
   - **ESCALATION RULE**: For critical/complex problems, escalate one level higher than normal
     - Normal "think" task → Use "think hard" if critical
     - Normal "think hard" task → Use "think harder" if critical  
     - Normal "think harder" task → Use "ultrathink" if critical
   - **MANDATORY GEMINI VALIDATION**: Always use Gemini CLI to check everything is okay
   - **CRITICAL**: Always explain what you're thinking and planning to the user
   - **Stay visible**: Communicate your process and get user approval before proceeding

### 🧠 **MANDATORY PRE-COMMIT VALIDATION**

**🚨 CRITICAL RULE**: Before every commit and push to GitHub, ALWAYS validate your work based on complexity.

#### **Validation by Complexity - UPDATED RULES:**
- **DEFAULT**: Always start with **"think"** level validation as baseline
- **think**: Default level - basic review and testing + MANDATORY Gemini CLI validation
- **think hard**: Multi-step changes - check integrations and test coverage + MANDATORY Gemini CLI validation
- **think harder**: Complex changes - validate assumptions and end-to-end workflows + MANDATORY Gemini CLI validation
- **ultrathink**: Critical changes - comprehensive review of all potential impacts + MANDATORY Gemini CLI validation
- **ESCALATION RULE**: For critical problems, escalate validation one level higher
- **GEMINI PRINCIPLE**: Claude is smarter for code writing, Gemini handles big structures and massive token analysis

#### **Pre-Commit Process - UPDATED WITH MANDATORY GEMINI:**
1. **Review all changes made** - What files were modified and why?
2. **Check integration points** - Do the changes properly connect with existing systems?
3. **Validate assumptions** - Are there hidden dependencies or requirements we missed?
4. **Test end-to-end flows** - Do complete user workflows actually work?
5. **MANDATORY GEMINI CLI VALIDATION** - Use Gemini to analyze entire codebase for gaps, issues, and completeness
6. **Identify potential gaps** - What could we have overlooked? (Enhanced by Gemini analysis)
7. **Verify non-obvious connections** - Could this change affect seemingly unrelated functionality? (Gemini checks all files)

#### **Red Flags That Require Deeper Thinking - UPDATED WITH ESCALATION:**
- ✅ Changes to configuration systems (think hard → escalate to think harder if critical)
- ✅ Integration of multiple components (think hard → escalate to think harder if critical)
- ✅ Updates to core functionality (think hard → escalate to think harder if critical)
- ✅ Environment or deployment changes (think harder → escalate to ultrathink if critical)
- ✅ Cross-project modifications (think harder → escalate to ultrathink if critical)
- ✅ Complex refactoring tasks (think harder → escalate to ultrathink if critical)
- ✅ **ALL TASKS**: Start with "think" level, apply escalation rule for critical scenarios

3. **🏗️ CHECK & PLAN FOLDER STRUCTURE**
   - **BE FLEXIBLE**: Don't force exact folder structure from documentation
   - **USE EXISTING**: If there's a logical existing folder, use it
   - **CREATE WHEN NEEDED**: Only create new folders if absolutely necessary
   - **THINK LOGICALLY**: Place files where they make the most sense
   - **EXAMPLE**: If task says "create in `agents/` but there's already a `src/training/` folder for training code, use `src/training/`"

4. **💻 WRITE THE CODE**
   - **ALWAYS SHOW CODE FIRST**: Display the code you plan to write before implementing
   - **CRITICAL**: When using sequential thinking, user can't accept/reject - show code blocks clearly
   - **GET CONFIRMATION**: Ask user to confirm before proceeding with file edits
   - Follow the task requirements
   - Implement with proper error handling
   - Add comprehensive docstrings
   - Use type hints throughout

5. **🧪 TEST THE CODE**
   - Write unit tests for new functionality
   - Run existing tests to ensure no regressions
   - Test manually if needed
   - Ensure >80% test coverage
   - **🚀 GPU TESTING**: When testing GPU code, verify CUDA availability and fallback to CPU
   - **⚡ PERFORMANCE TESTING**: Benchmark critical training loops for speed targets

6. **📚 UPDATE DOCUMENTATION**
   - **ALWAYS update README.md and relevant docs/**.md files**

7. **🧠 VALIDATION & COMMIT - UPDATED WITH NEW RULES**
   - **DEFAULT LEVEL**: Start with "think" level validation for ALL tasks
   - **ESCALATION**: Apply escalation rule for critical/complex scenarios
   - **MANDATORY GEMINI**: Always use Gemini CLI for comprehensive codebase validation
   - **Check for**: Missing integrations, overlooked dependencies, false assumptions
   - **Validate**: That fixes actually work end-to-end, not just in isolation
   - **Gemini Analysis**: Let Gemini handle massive token analysis and structural checks

8. **🚀 COMMIT & PUSH TO GITHUB**
   - Stage all changes: `git add .`
   - Write descriptive commit message (NO CLAUDE ATTRIBUTION)
   - Push to GitHub: `git push origin master`
   - Verify the push was successful

**🚫 COMMIT MESSAGE RULES:**
- **DO NOT add Claude Code attribution**: No "🤖 Generated with [Claude Code]" text
- **DO NOT add Co-Authored-By**: No "Co-Authored-By: Claude <noreply@anthropic.com>" text  
- **Keep it clean**: Only include the actual commit message content
- **Use HEREDOC format** for multi-line messages without attribution:
```bash
git commit -m "$(cat <<'EOF'
✅ COMPLETED Task X.Y.Z: Brief Description

Detailed description of changes made.

Key features:
- Feature 1 description
- Feature 2 description
- Feature 3 description

Files created/modified:
- path/to/file1.py
- path/to/file2.py
EOF
)"
```

#### Example Task Development Flow:
```bash
# 1. Check the task (e.g., "Add PPO agent implementation")
# 2. Plan: "This goes in src/agents/ or maybe existing src/training/"
cd Connect4_AI
.\venv\Scripts\Activate.ps1

# 3. Check existing structure
ls src/
# Found: agents/ exists - use it!

# 4. Write code in src/agents/ppo_agent.py
# 5. Write tests in tests/test_agents.py
# 6. Update README.md and docs/architecture.md
# 7. Commit and push
git add .
git commit -m "Add PPO agent with GPU support and action masking"
git push origin master
```

## 🏗️ ARCHITECTURE & IMPLEMENTATION

### Tech Stack & Dependencies
- **Core RL Framework**: PyTorch with optional CUDA support
- **Environment**: Gymnasium-compatible Connect4 environment
- **Algorithms**: PPO (Proximal Policy Optimization) with action masking
- **Game Logic**: Pure Python Connect4 implementation (CPU-based)
- **Neural Networks**: Simple CNN for board pattern recognition
- **Configuration**: YAML-based configuration management
- **Testing**: pytest with basic functionality tests
- **Code Quality**: black, mypy, flake8, isort

### Project Structure (from PRD)
```
Connect4_AI/
├── src/
│   ├── environments/     # Connect4 game logic and Gym wrapper
│   ├── agents/          # PPO agent, networks, random agent
│   ├── utils/           # Checkpointing utilities
│   └── core/            # Configuration management
├── configs/             # YAML configuration files
├── scripts/             # Training and play scripts
├── tests/               # Unit tests
├── notebooks/           # Analysis notebooks
├── models/              # Saved model checkpoints
└── logs/                # Training logs
```

### Core Components
The system provides these key components:
- `Connect4Game` - Pure game logic without ML dependencies
- `Connect4Env` - Gymnasium wrapper for the game
- `PPOAgent` - PPO implementation with CNN networks
- `RandomAgent` - Baseline random agent for testing
- `CheckpointManager` - Model saving and loading

### 📁 **FOLDER STRUCTURE FLEXIBILITY RULES**

#### Core Principle: **LOGICAL PLACEMENT OVER RIGID STRUCTURE**

1. **🔍 FIRST**: Check what folders already exist
2. **🤔 THINK**: Where does this logically belong?
3. **📂 USE**: Existing folders when they make sense
4. **📝 DOCUMENT**: Update README/docs when you make structural decisions

#### Common Flexibility Examples:
- **RL Agents** → Could go in `src/agents/`, `src/training/`, or `src/models/`
- **Utilities** → Could go in `src/utils/`, `src/helpers/`, or `src/common/`
- **Neural Networks** → Could go in `src/agents/networks.py`, `src/models/`, or `src/networks/`
- **Tests** → Always in `tests/` but mirror the source structure

#### When to Create New Folders:
- ✅ When existing folders don't make logical sense
- ✅ When you have 3+ related files that form a logical group
- ✅ When project documentation specifically requires it
- ❌ Don't create for 1-2 files that fit elsewhere

## 🎯 PROJECT PURPOSE & GOALS

### Primary Purpose
Create a straightforward Connect4 reinforcement learning system that demonstrates RL concepts:
- **Self-play learning** where agent learns by playing against itself
- **Simple architecture** with clear separation of concerns
- **Educational focus** - easy to understand and modify
- **CPU-first design** with optional GPU acceleration for neural networks
- **Terminal-based interface** for training and gameplay

### Training Objectives
- **Legal move mastery** - Agent makes only valid moves (Episode 100)
- **Basic strategy** - Shows strategic play (Episode 1000)
- **Strong performance** - Wins 80%+ vs random player (Episode 5000)
- **Advanced tactics** - Demonstrates sophisticated strategies (Episode 10000)

### Current Implementation Status (from PRD)
- ✅ **Phase 1**: Project configuration system completed
- 🔄 **Phase 2**: Core game implementation (pending)
- 🔄 **Phase 3**: Agent foundation (pending)
- 🔄 **Phase 4**: PPO agent implementation (pending)
- 🔄 **Phase 5**: Training infrastructure (pending)
- 🔄 **Phase 6**: Game interface (pending)
- 🔄 **Phase 7**: Testing and validation (pending)
- 🔄 **Phase 8**: Documentation and polish (pending)

## 🐍 **CODE STANDARDS**

### Python Requirements
- **Python 3.9+** required for modern type hints and PyTorch compatibility
- **Type hints mandatory** - all functions and classes must have proper typing
- **Docstrings required** - comprehensive documentation for all public APIs
- **PEP 8 compliance** - enforced via black formatter
- **GPU-first design** - all tensor operations should support CUDA

### 🎯 **SIMPLIFIED DESIGN PRINCIPLES**
- **Simplicity First**: Keep code simple and readable
- **Modular Design**: Clear separation of concerns
- **CPU/GPU Flexibility**: Works on both, optimized for GPU when available
- **Incremental Development**: Build and test one component at a time
- **Practical Focus**: Prioritize working system over perfect optimization

### 🧠 **NETWORK ARCHITECTURE (from PRD)**
Simple CNN architecture for Connect4 board pattern recognition:
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
```

### ⚡ **IMPLEMENTATION REQUIREMENTS**
- **Game Logic**: Pure Python on CPU for consistency
- **Board Representation**: 6×7 NumPy array with 0 (empty), 1 (player 1), -1 (player 2)
- **Action Masking**: Prevent invalid moves in policy distribution
- **Reward Structure**: +1 win, 0 draw, -1 loss, -0.01 per move

### 📋 **FOLLOW PRD IMPLEMENTATION PLAN**
All development must follow the granular, phased implementation plan outlined in `connect4_multiagent_rl_prd.md`. Start with Phase 1 (Core Environment) and proceed through tasks as specified, paying close attention to instructions and verification criteria for each task.
- **Performance critical** - profile hot paths and optimize bottlenecks

### 🧪 **Testing Requirements**
- pytest framework with GPU testing support
- Unit tests for core functionality (environment, agents, training)
- Performance benchmarks with speed targets
- Maintain >80% test coverage
- Mock GPU operations for CI/CD environments
- Property-based testing for game logic invariants

### 📦 **Dependencies Management**
- requirements.txt with GPU-enabled PyTorch
- Pin dependency versions for reproducibility
- Separate requirements for CPU vs GPU environments
- Regular dependency updates with performance testing

### 🚀 **Development Workflow Examples**

#### Working on Connect4 Environment:
```bash
cd Connect4_AI
.\venv\Scripts\Activate.ps1  # Windows
# Verify: (venv) appears in prompt and GPU available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Now develop environment features
```

#### Working on RL Agents:
```bash
cd Connect4_AI
.\venv\Scripts\Activate.ps1  # Windows
# Verify: (venv) appears in prompt
# Run quick test to ensure agent loads on GPU
python -c "from src.agents.ppo_agent import PPOAgent; print('Agent OK')"
# Now develop agent features
```

#### Training and Testing:
```bash
cd Connect4_AI
.\venv\Scripts\Activate.ps1  # Windows
# Quick training test
python scripts/train.py --device cuda --num-envs 1000 --total-games 10000
# Benchmark performance
python scripts/benchmark_env.py
```

### 🔍 **Code Quality Checklist - UPDATED FOR RL PROJECT**
Before any commit to Connect4 RL project:
- [ ] **Correct virtual environment is active**
- [ ] **Currently in Connect4_AI project directory**
- [ ] **Task analyzed and understood completely**
- [ ] **Applied "think" level as baseline (escalate if critical)**
- [ ] **Folder structure decision documented**
- [ ] **GPU compatibility verified** (CUDA available check)
- [ ] Tests pass: `pytest`
- [ ] Code formatted: `black .`
- [ ] Type checking: `mypy .`
- [ ] Linting: `flake8 .`
- [ ] Import sorting: `isort .`
- [ ] **Performance benchmarks meet targets** (if applicable)
- [ ] **README.md updated** if structure changed
- [ ] **Relevant .md files updated** in docs/
- [ ] **🧠 THINKING LEVEL VALIDATION**: Applied appropriate thinking level with escalation rule
- [ ] **🔍 MANDATORY GEMINI CLI VALIDATION**: Used Gemini to analyze entire codebase
- [ ] **Committed and pushed to GitHub**

### 📚 **DOCUMENTATION UPDATE REQUIREMENTS**

#### Always Update When:
- ✅ **Folder structure changes** → Update README.md
- ✅ **New modules added** → Update architecture docs
- ✅ **API endpoints change** → Update API documentation  
- ✅ **Dependencies change** → Update requirements and docs
- ✅ **Configuration changes** → Update setup instructions

### 🔍 **GEMINI CLI FOR LARGE CODEBASE ANALYSIS**

When Claude Code's context window is insufficient for large-scale analysis, use the Gemini CLI with its massive context capacity.

#### **When to Use Gemini CLI:**
- Analyzing entire codebases or large directories
- Comparing multiple large files  
- Understanding project-wide patterns or architecture
- Working with files totaling more than 100KB
- Verifying if specific features, patterns, or security measures are implemented
- Checking for coding patterns across the entire codebase

#### **File and Directory Inclusion Syntax:**

Use the `@` syntax to include files and directories. Paths are relative to your current working directory:

```bash
# Single file analysis
gemini -p "@src/services/stats_service.py Explain this service's architecture"

# Multiple files
gemini -p "@pyproject.toml @requirements.txt Analyze the project dependencies"

# Entire directory
gemini -p "@src/ Summarize the MCP server architecture"

# Multiple directories
gemini -p "@src/ @tests/ Analyze test coverage for the source code"

# Current directory and subdirectories
gemini -p "@./ Give me an overview of this entire LoL MCP server project"

# Or use --all_files flag
gemini --all_files -p "Analyze the project structure and dependencies"
```

#### **Project-Specific Examples:**

```bash
# Check MCP implementation completeness
gemini -p "@src/mcp_server/ @src/services/ Are all MCP tools properly implemented? List missing functionality"

# Verify scraping implementation  
gemini -p "@src/data_sources/ @src/services/ Is web scraping properly implemented for all LoL data types?"

# Check error handling patterns
gemini -p "@src/ Is proper error handling implemented across all services? Show examples"

# Verify async patterns
gemini -p "@src/ Are async/await patterns consistently used? List any blocking operations"

# Check test coverage
gemini -p "@src/services/ @tests/ Is the champion stats service fully tested? List test gaps"

# Verify specific implementations
gemini -p "@src/ @config/ Is Redis caching implemented? List all cache-related functions"
```

#### **Important Notes:**
- Paths in `@` syntax are relative to your current working directory
- The CLI includes file contents directly in the context
- Gemini's context window can handle entire codebases that exceed Claude's limits
- Be specific about what you're looking for to get accurate results

### 🎯 **MAIN PROJECT: Connect4 Multi-Agent RL**

This project provides a high-performance reinforcement learning system for Connect4.

#### **Architecture Overview**:
- **`src/environments/`**: Connect4 Gymnasium environment with GPU optimization
- **`src/agents/`**: PPO agents with CNN networks and action masking
- **`src/training/`**: Self-play training orchestration and experience collection
- **`src/utils/`**: Visualization, profiling, and utility functions
- **`src/core/`**: Configuration management and device selection
- **`configs/`**: YAML configuration files for different training setups
- **`tests/`**: Unit tests with performance benchmarking
- **`scripts/`**: Training, evaluation, and benchmarking scripts

#### **Key Development Commands**:
```bash
# Activate virtual environment (ALWAYS FIRST)
.\venv\Scripts\Activate.ps1

# Verify GPU support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Testing
pytest                           # Run all tests
pytest tests/test_environment.py # Test environment
pytest -v --benchmark            # Include performance tests

# Code Quality
black .                   # Format code
mypy .                    # Type checking  
flake8 .                  # Linting
isort .                   # Import sorting

# Training and Playing
python scripts/train.py    # Start training (uses GPU if available)
python scripts/play.py     # Interactive game interface
```

#### **Current Development Status**:
- **Phase**: Initial implementation following PRD task breakdown
- **Target**: Functional RL system with good learning performance
- **Focus**: Core game logic, Gymnasium environment, and PPO agent

#### **Key Implementation Requirements (from PRD)**:
- **CPU Game Logic**: Pure Python game implementation without ML dependencies
- **Gymnasium Wrapper**: Standard RL environment interface
- **Simple Training**: Self-play with periodic weight updates
- **Terminal Interface**: Text-based game rendering and user interaction
- **Modular Design**: Clean separation between game, environment, and agent

### Dependencies (from PRD)
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

## 🚨 **CRITICAL DEVELOPMENT RULES**

1. **Environment Management**: Always activate virtual environment for Python operations
2. **Simplicity First**: Prioritize clear, readable code over complex optimizations
3. **CPU/GPU Flexibility**: Design for CPU with optional GPU acceleration
4. **Code Quality**: Run all quality checks before committing
5. **Documentation**: Update README.md and relevant docs when making changes
6. **Incremental Development**: Build and test one component at a time
7. **Follow PRD**: Implement according to the detailed task breakdown in the PRD
8. **Ask When Uncertain**: Clarify requirements before implementing major changes

### 📚 **Documentation References**
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Gymnasium**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **CUDA Programming**: https://docs.nvidia.com/cuda/
- **PPO Algorithm**: https://arxiv.org/abs/1707.06347
- **Connect4 Strategy**: https://en.wikipedia.org/wiki/Connect_Four