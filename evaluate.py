"""
Evaluation script for the trained DQN agent on Highway-Env.

Usage:
    python evaluate.py              # with rendering (default, as required by spec)
    python evaluate.py --no-render  # headless mode (for environments without display)
"""

import gymnasium
import highway_env
import numpy as np
import torch
import random
import os
import sys

from dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
no_render = "--no-render" in sys.argv
render_mode = None if no_render else "human"

env_name = "highway-v0"
env = gymnasium.make(
    env_name,
    config={
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 3,
        "ego_spacing": 1.5,
    },
    render_mode=render_mode,
)

# ---------------------------------------------------------------------------
# Load the trained agent
# ---------------------------------------------------------------------------
agent = DQNAgent(state_dim=25, action_dim=5, hidden_dim=256)
weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "dqn_highway.pt")
agent.load(weight_path)
print(f"Loaded model from {weight_path}")

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
state, _ = env.reset()
state = state.reshape(-1)
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0

while episode <= 10:
    episode_steps += 1

    # Greedy action selection (no exploration)
    action = agent.select_action(state, evaluate=True)

    state, reward, done, truncated, _ = env.step(action)
    state = state.reshape(-1)
    if not no_render:
        env.render()

    episode_return += reward

    if done or truncated:
        print(
            f"Episode Num: {episode}  Episode T: {episode_steps}  "
            f"Return: {episode_return:.3f}  Crash: {done}"
        )

        state, _ = env.reset()
        state = state.reshape(-1)
        episode += 1
        episode_steps = 0
        episode_return = 0

env.close()
