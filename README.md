# DQN CartPole-v1 Project

This project implements a Deep Q-Network (DQN) using `stable-baselines3` to solve the `CartPole-v1` environment (Maximum Reward: 500).

## Features
- **Aggressive Training**: Configured with `train_freq=1` to match high-performance tutorial standards.
- **Selective Rendering**: No window during training (high speed); visual rendering during evaluation.
- **Custom Reward Wrapper**: Includes a `CenterRewardWrapper` that punishes the agent for moving toward the viewport boundaries, effectively teaching it to stay centered.
- **Model Saving**: Automatically saves the trained agent for future use.
- **TensorBoard Integration**: Built-in logging for real-time visualization of loss and reward curves.

## Installation

Ensure you have `uv` installed, then run:
```powershell
uv sync
```

## Usage

### 1. Training and Testing
Run the main script to start training (100k steps) followed by evaluation:
```powershell
uv run .\main.py
```

### 2. Testing Only
If you already have a saved model (`dqn_cartpole.zip`) and want to see it run without training:
```powershell
uv run .\main.py --test
```

### 3. Visualization (TensorBoard)
To see the training statistics (Loss, Rewards, Exploration) in real-time:
```powershell
uv run python -m tensorboard.main --logdir ./dqn_logs/
```

## Troubleshooting & Key Learnings

### 1. The "100 Reward" Ceiling
The agent was initially stuck at exactly 100 reward. This was caused by two issues:
- **Premature Exploration Cutoff**: The agent stopped trying new things before finding the 500-step path.
- **Viewport Exit**: The agent balanced the pole but drifted off-screen.
- **Fix**: Added `CenterRewardWrapper` and increased `exploration_fraction` to 0.3.

### 2. High Loss (Millions)
Calculated targets in DQN can create feedback loops where weights (and thus loss) explode into millions.
- **Fix**: Lowered the learning rate (`1e-3` or `1e-4`) and used `target_update_interval=500` to stabilize the "moving goalposts."

### 3. TensorBoard `pkg_resources` Error
Modern Python environments sometimes lack `setuptools` modules required by TensorBoard.
- **Fix**: Pin `setuptools<70` in `requirements.txt` to maintain compatibility with older tools like TensorBoard.
