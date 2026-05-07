"""
Manual control baseline for the Highway-Env autonomous driving task.

Usage:
    python manual_control.py

Controls the ego vehicle via keyboard for 10 episodes.
Logs episode returns and crash status to results/manual_control_results.json
for comparison in the report.

Keyboard controls (as defined by highway-env):
    Arrow keys: LEFT / RIGHT / UP (faster) / DOWN (slower)
"""

import gymnasium
import highway_env
import numpy as np
import json
import os


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env_name = "highway-v0"
env = gymnasium.make(
    env_name,
    config={
        "manual_control": True,
        "lanes_count": 3,
        "ego_spacing": 1.5,
    },
    render_mode="human",
)

# ---------------------------------------------------------------------------
# Episode loop with logging
# ---------------------------------------------------------------------------
num_episodes = 10
episode_returns = []
episode_crashes = []

env.reset()
done, truncated = False, False

episode = 1
episode_steps = 0
episode_return = 0.0

while episode <= num_episodes:
    episode_steps += 1

    # In manual control mode, env.step() ignores the action argument;
    # the vehicle is controlled via keyboard input processed by the renderer.
    _, reward, done, truncated, _ = env.step(env.action_space.sample())
    env.render()

    episode_return += reward

    if done or truncated:
        print(
            f"Episode Num: {episode}  Episode T: {episode_steps}  "
            f"Return: {episode_return:.3f}  Crash: {done}"
        )
        episode_returns.append(episode_return)
        episode_crashes.append(done)

        env.reset()
        episode += 1
        episode_steps = 0
        episode_return = 0.0

env.close()

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results = {
    "episode_returns": episode_returns,
    "episode_crashes": [bool(c) for c in episode_crashes],
    "mean_return": float(np.mean(episode_returns)),
    "std_return": float(np.std(episode_returns)),
    "crash_rate": float(np.mean(episode_crashes)),
}
results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)
with open(os.path.join(results_dir, "manual_control_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"\nManual Control Summary:")
print(f"  Mean Return: {results['mean_return']:.3f} +/- {results['std_return']:.3f}")
print(f"  Crash Rate:  {results['crash_rate']:.1%}")
print(f"Results saved to results/manual_control_results.json")
