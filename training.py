"""
Training script for the Double DQN agent on Highway-Env.

Usage:
    python training.py

The script:
    1. Creates the fast training environment (highway-fast-v0).
    2. Instantiates a DQN agent with a Dueling architecture.
    3. Runs an epsilon-greedy training loop for MAX_STEPS environment steps.
    4. Periodically evaluates the agent (greedy policy) on a separate evaluation
       environment to track learning progress.
    5. Saves the final model weights to weights/dqn_highway.pt.
    6. Saves training logs (episode returns, evaluation returns, losses) to
       results/training_log.json for report plotting.
"""

import gymnasium
import highway_env
import numpy as np
import torch
import random
import json
import os

from dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
MAX_STEPS = int(2e4)
EVAL_INTERVAL = 50       # Evaluate every N episodes
EVAL_EPISODES = 5        # Number of episodes per evaluation run
SAVE_DIR = os.path.join(os.path.dirname(__file__), "weights")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Environment creation helpers
# ---------------------------------------------------------------------------
def make_train_env():
    """Training environment: fast simulation, 50 vehicles, duration 40."""
    return gymnasium.make(
        "highway-fast-v0",
        config={
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,
            "vehicles_count": 50,
            "lanes_count": 3,
        },
    )


def make_eval_env():
    """Evaluation environment: same config as training but standard sim speed."""
    return gymnasium.make(
        "highway-fast-v0",
        config={
            "action": {"type": "DiscreteMetaAction"},
            "duration": 40,
            "vehicles_count": 50,
            "lanes_count": 3,
        },
    )


# ---------------------------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------------------------
def evaluate_agent(agent, num_episodes=EVAL_EPISODES):
    """Run the agent greedily for num_episodes and return mean return and crash rate."""
    eval_env = make_eval_env()
    returns = []
    crashes = []

    for ep in range(num_episodes):
        state, _ = eval_env.reset()
        state = state.reshape(-1)
        done, truncated = False, False
        ep_return = 0.0

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            state, reward, done, truncated, _ = eval_env.step(action)
            state = state.reshape(-1)
            ep_return += reward

        returns.append(ep_return)
        crashes.append(float(done))

    eval_env.close()
    return np.mean(returns), np.std(returns), np.mean(crashes)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    env = make_train_env()

    agent = DQNAgent(
        state_dim=25,
        action_dim=5,
        hidden_dim=256,
        lr=5e-4,
        gamma=0.99,
        buffer_capacity=15000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=10000,
        tau=0.005,
    )

    # Logging containers
    training_returns = []      # Return per training episode
    training_crashes = []      # Whether each training episode ended in crash
    eval_returns = []          # (step, mean_return, std_return, crash_rate)
    losses = []                # (step, loss) sampled periodically

    state, _ = env.reset(seed=SEED)
    state = state.reshape(-1)
    done, truncated = False, False

    episode = 1
    episode_steps = 0
    episode_return = 0.0

    for t in range(MAX_STEPS):
        episode_steps += 1

        # Select action (epsilon-greedy)
        action = agent.select_action(state, evaluate=False)

        # Step the environment
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = next_state.reshape(-1)

        # Store transition and train
        agent.store_transition(state, action, reward, next_state, float(done))
        loss = agent.train_step()

        # Log loss periodically
        if loss is not None and t % 100 == 0:
            losses.append((t, loss))

        state = next_state
        episode_return += reward

        if done or truncated:
            print(
                f"Total T: {t}  Episode: {episode}  Steps: {episode_steps}  "
                f"Return: {episode_return:.3f}  Epsilon: {agent.epsilon:.3f}"
            )

            training_returns.append(episode_return)
            training_crashes.append(float(done))

            # Periodic evaluation
            if episode % EVAL_INTERVAL == 0:
                mean_ret, std_ret, crash_rate = evaluate_agent(agent)
                eval_returns.append({
                    "step": t,
                    "episode": episode,
                    "mean_return": float(mean_ret),
                    "std_return": float(std_ret),
                    "crash_rate": float(crash_rate),
                })
                print(
                    f"  [EVAL] Mean Return: {mean_ret:.3f} +/- {std_ret:.3f}  "
                    f"Crash Rate: {crash_rate:.1%}"
                )

            # Reset environment
            state, _ = env.reset()
            state = state.reshape(-1)
            episode += 1
            episode_steps = 0
            episode_return = 0.0

    env.close()

    # Save model weights
    weight_path = os.path.join(SAVE_DIR, "dqn_highway.pt")
    agent.save(weight_path)
    print(f"\nModel saved to {weight_path}")

    # Save training logs
    log = {
        "training_returns": training_returns,
        "training_crashes": training_crashes,
        "eval_returns": eval_returns,
        "losses": losses,
    }
    log_path = os.path.join(RESULTS_DIR, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {log_path}")

    # Final evaluation
    print("\n--- Final Evaluation (10 episodes) ---")
    mean_ret, std_ret, crash_rate = evaluate_agent(agent, num_episodes=10)
    print(f"Mean Return: {mean_ret:.3f} +/- {std_ret:.3f}")
    print(f"Crash Rate:  {crash_rate:.1%}")

    final_eval = {
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "crash_rate": float(crash_rate),
    }
    with open(os.path.join(RESULTS_DIR, "final_eval.json"), "w") as f:
        json.dump(final_eval, f, indent=2)


if __name__ == "__main__":
    main()
