# F1 Racing Reinforcement Learning

A reinforcement learning environment for training an autonomous F1 racing agent using the Advantage Actor-Critic (A2C) algorithm with vectorized environments.

## Overview

This project implements a custom Gymnasium environment where an agent learns to:
- Navigate an oval racing track with curves
- Avoid obstacles placed on the track
- Pass through checkpoints to complete laps
- Optimize racing line and speed

The environment uses a 15-dimensional state space with distance sensors and a 5-action discrete control scheme.

## Project Structure

```
f1_rl/
├── f1_game.py          # Core game logic, physics, and Pygame rendering
├── f1_env.py           # Gymnasium-compatible environment wrapper
├── train_a2c.py        # A2C training script with vectorized environments
├── play.py             # Visualization and manual control
├── models/             # Saved model checkpoints
└── requirements.txt    # Python dependencies
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/f1_rl.git
cd f1_rl
pip install -r requirements.txt
```

## Usage

### Training

Train the A2C agent with parallel environments:

```bash
python train_a2c.py --updates 500 --n_envs 8
```

**Training Options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--updates` | 500 | Number of training updates |
| `--n_envs` | 8 | Parallel environments |
| `--n_steps` | 128 | Steps per update per environment |
| `--gamma` | 0.99 | Discount factor |
| `--actor_lr` | 0.0003 | Actor learning rate |
| `--critic_lr` | 0.001 | Critic learning rate |

### Evaluation

Watch the trained agent:

```bash
python play.py --episodes 5
```

### Manual Control

Drive the car manually using keyboard:

```bash
python play.py --manual
```

**Controls:**
- `W` / `Up Arrow` - Accelerate
- `A` / `Left Arrow` - Turn left
- `D` / `Right Arrow` - Turn right
- `S` / `Down Arrow` - Brake
- `Space` - Coast
- `R` - Reset episode
- `ESC` - Exit

## Environment Specification

### Observation Space

15-dimensional continuous vector:

| Index | Description |
|-------|-------------|
| 0-4 | Wall distance sensors (5 directions) |
| 5-6 | Velocity components (vx, vy) |
| 7 | Car rotation angle |
| 8-9 | Direction to next checkpoint |
| 10-14 | Obstacle proximity sensors |

### Action Space

5 discrete actions:

| Action | Description |
|--------|-------------|
| 0 | Accelerate forward |
| 1 | Turn left and accelerate |
| 2 | Turn right and accelerate |
| 3 | Brake |
| 4 | Coast (no input) |

### Reward Structure

| Event | Reward |
|-------|--------|
| Pass checkpoint | +50 |
| Complete lap | +200 |
| Collision (wall/obstacle) | -100 |
| Time step penalty | -0.1 |

## Algorithm

The agent uses **Advantage Actor-Critic (A2C)** with:
- Generalized Advantage Estimation (GAE) for variance reduction
- Entropy bonus for exploration
- Vectorized environments for parallel data collection
- Separate actor and critic neural networks

## Dependencies

- gymnasium
- pygame
- torch
- numpy
- matplotlib
- tqdm

## License

MIT License

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) - Mnih et al., 2016
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) - Schulman et al., 2015
