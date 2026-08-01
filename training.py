"""
Training script for the Double DQN agent on Highway-Env.

Usage:
    python training.py                 # fresh training (or auto-resume if checkpoint exists)
    python training.py --no-resume     # force fresh training, overwriting any checkpoint
    python training.py --resume        # explicitly require a checkpoint to resume from

The script periodically saves a full training checkpoint (weights + optimizer +
replay buffer + RNG state + training log). If training is interrupted, running
the script again will pick up from the last checkpoint.

Key changes from v2:
- Trains on highway-v0 directly (same env as evaluate.py) to eliminate the
  sim-frequency transfer gap.
- 50k training steps with matching epsilon decay.
- Warmup of 1000 steps before gradient updates.
- Larger replay buffer (50k) for more diverse experience.
- Checkpoint every CHECKPOINT_INTERVAL steps.
"""

import gymnasium
import highway_env
import numpy as np
import torch
import random
import json
import shutil
import time
import os
import sys

from dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 0


def set_seeds(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ---------------------------------------------------------------------------
# Hyperparameters and paths
# ---------------------------------------------------------------------------
MAX_STEPS = int(5e4)
UPDATES_PER_STEP = 2            # Gradient updates per environment step.
                                # Env stepping dominates wall-clock time on
                                # highway-v0 (sim_frequency=15), so extra
                                # gradient updates are nearly free and improve
                                # sample efficiency.
EVAL_INTERVAL = 50              # Evaluate every N episodes
EVAL_EPISODES = 10              # Episodes per evaluation
CHECKPOINT_INTERVAL = 2000      # Save full checkpoint every N environment steps
CHECKPOINT_TIME_INTERVAL = 300  # ...or every N seconds, whichever comes first
                                # (5 min: important on Colab free tier, where a
                                # session can be killed at any moment)

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FINAL_WEIGHTS_PATH = os.path.join(SAVE_DIR, "dqn_highway.pt")
BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "dqn_highway_best.pt")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pt")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.json")
FINAL_EVAL_PATH = os.path.join(RESULTS_DIR, "final_eval.json")


def eval_score(mean_return, crash_rate):
    """
    Composite score for model selection. Heavily penalizes crashes because
    the domain is safety-critical: a policy with 30% crash rate at 29 return
    is worse than a policy with 0% crash rate at 27 return.

    score = mean_return - 10 * crash_rate
    A collision costs 10 return points of equivalent mean improvement.
    """
    return mean_return - 10.0 * crash_rate


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
ENV_CONFIG = {
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "ego_spacing": 1.5,
    "vehicles_count": 50,
    "duration": 40,
}


def make_train_env():
    return gymnasium.make("highway-v0", config=ENV_CONFIG)


def make_eval_env():
    return gymnasium.make("highway-v0", config=ENV_CONFIG)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_SEED_BASE = 12345          # Fixed seeds for evaluation episodes, disjoint
                                # from the training seed. Every evaluation uses
                                # episode seeds EVAL_SEED_BASE + i, so all
                                # checkpoints are compared on the SAME traffic
                                # configurations. This removes environment
                                # sampling noise from model selection: score
                                # differences between checkpoints then reflect
                                # only the policy, not luck.


def evaluate_agent(agent, num_episodes=EVAL_EPISODES):
    eval_env = make_eval_env()
    returns = []
    crashes = []

    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=EVAL_SEED_BASE + ep)
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


TEST_SEED_BASE = 90000          # TEST set: disjoint from the selection seeds
                                # above. Used exactly ONCE, on the finally
                                # chosen model, and never for any selection
                                # decision. This is the number reported as the
                                # agent's performance. Reporting a
                                # selection-set score instead would be optimistic
                                # (selecting the argmax over many noisy
                                # evaluations inflates the apparent score, the
                                # "winner's curse"); the test set gives an
                                # unbiased estimate.


def evaluate_agent_test(agent, num_episodes=100):
    """Final unbiased evaluation on a large TEST seed set, disjoint from the
    seeds used for checkpoint selection. Larger N shrinks the crash-rate
    standard error to about sqrt(p(1-p)/100) ~ 0.04 at p=0.2."""
    eval_env = make_eval_env()
    returns = []
    crashes = []

    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=TEST_SEED_BASE + ep)
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
# Log persistence
# ---------------------------------------------------------------------------
def save_log(training_returns, training_crashes, eval_returns, losses, current_step):
    """Persist the training log alongside checkpoints."""
    log = {
        "training_returns": training_returns,
        "training_crashes": training_crashes,
        "eval_returns": eval_returns,
        "losses": losses,
        "current_step": current_step,
    }
    # Atomic write: temp file then rename, so a kill mid-write never
    # corrupts the log (which would break resume).
    tmp_path = LOG_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp_path, LOG_PATH)


def load_log():
    """Load a previously saved training log, or return empty structures."""
    if not os.path.exists(LOG_PATH):
        return [], [], [], [], 0
    with open(LOG_PATH) as f:
        log = json.load(f)
    return (
        log.get("training_returns", []),
        log.get("training_crashes", []),
        log.get("eval_returns", []),
        log.get("losses", []),
        log.get("current_step", 0),
    )


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def main():
    # Argument parsing (simple flags only)
    force_fresh = "--no-resume" in sys.argv
    require_resume = "--resume" in sys.argv

    # Candidate checkpoints, newest first. A Colab VM kill can lose the most
    # recent write; the rotation in save_checkpoint guarantees that if any
    # save ever completed, at least one of these is a complete file.
    CHECKPOINT_CANDIDATES = [
        CHECKPOINT_PATH,
        CHECKPOINT_PATH + ".tmp",
        CHECKPOINT_PATH + ".prev",
    ]

    checkpoint_exists = any(os.path.exists(p) for p in CHECKPOINT_CANDIDATES)
    resume = checkpoint_exists and not force_fresh

    if require_resume and not checkpoint_exists:
        print(f"Error: --resume specified but no checkpoint found at {CHECKPOINT_PATH}")
        sys.exit(1)

    if force_fresh and checkpoint_exists:
        print(f"--no-resume specified: ignoring existing checkpoint at {CHECKPOINT_PATH}")

    set_seeds(SEED)

    env = make_train_env()

    agent = DQNAgent(
        state_dim=25,
        action_dim=5,
        hidden_dim=128,
        lr=5e-4,
        gamma=0.99,
        buffer_capacity=50000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=80000,   # in GRADIENT steps = 40k env steps x UPDATES_PER_STEP
        target_update_freq=400,      # in GRADIENT steps = 200 env steps x UPDATES_PER_STEP
        learning_starts=1000,
    )

    # Resume or fresh start
    if resume:
        loaded = False
        for cand in CHECKPOINT_CANDIDATES:
            if not os.path.exists(cand):
                continue
            print(f"Resuming from checkpoint: {cand}")
            try:
                agent.load_checkpoint(cand)
                loaded = True
                break
            except Exception as e:
                print(f"WARNING: {cand} failed to load ({e}). Trying next candidate...")
        if not loaded:
            resume = False

    # GUARD: never silently start fresh over evidence of a longer run.
    # If no checkpoint loaded but the training log records real progress,
    # a fresh start would overwrite the log, the best-model weights, and
    # the checkpoint rotation -- destroying everything recoverable. Abort
    # with recovery instructions instead. --no-resume overrides explicitly.
    if not resume and not force_fresh:
        _, _, _, _, logged_step = load_log()
        if logged_step > 0:
            print("=" * 70)
            print(f"ABORTING: training_log.json records progress up to step "
                  f"{logged_step}, but no loadable checkpoint was found.")
            print("A fresh start now would OVERWRITE the log, the best-model")
            print("weights, and any recoverable checkpoint revisions.")
            print()
            print("Recovery options:")
            print("  1. Google Drive web UI > weights/checkpoint.pt >")
            print("     'Manage versions' > download an earlier revision and")
            print("     put it back as weights/checkpoint.pt, then re-run.")
            print("  2. Same for weights/dqn_highway_best.pt: that file alone")
            print("     is a usable trained model (copy to dqn_highway.pt).")
            print("  3. Check the Drive Trash for checkpoint files.")
            print()
            print("To intentionally discard the previous run and start over:")
            print("     python training.py --no-resume")
            print("=" * 70)
            sys.exit(1)

    if resume:
        training_returns, training_crashes, eval_returns, losses, start_step = load_log()
        episode = len(training_returns) + 1

        # Try to recover previous best score from the eval history
        best_score = -float("inf")
        for ev in eval_returns:
            s = eval_score(ev["mean_return"], ev["crash_rate"])
            if s > best_score:
                best_score = s
        if best_score == -float("inf"):
            best_score = None
        print(f"  Resumed at step {start_step}, episode {episode}, "
              f"epsilon {agent.epsilon:.3f}, buffer size {len(agent.replay_buffer)}")
        if best_score is not None:
            print(f"  Best eval score so far: {best_score:.3f}")
    else:
        print("Starting fresh training")
        training_returns = []
        training_crashes = []
        eval_returns = []
        losses = []
        start_step = 0
        episode = 1
        best_score = None

    print(f"Device: {agent.device}")
    print(f"Training on highway-v0 for {MAX_STEPS} steps total "
          f"({MAX_STEPS - start_step} remaining)")
    print("-" * 50)

    if start_step >= MAX_STEPS:
        print("Training already complete. Skipping to final evaluation.")
    else:
        state, _ = env.reset(seed=SEED + start_step)  # Offset seed to avoid repeating same trajectories
        state = state.reshape(-1)
        episode_steps = 0
        episode_return = 0.0
        last_checkpoint_time = time.time()
        t = start_step  # defined before the loop so the interrupt handler is always safe

        try:
            for t in range(start_step, MAX_STEPS):
                episode_steps += 1

                action = agent.select_action(state, evaluate=False)
                next_state, reward, done, truncated, _ = env.step(action)
                next_state = next_state.reshape(-1)

                agent.store_transition(state, action, reward, next_state, float(done))
                loss = None
                for _ in range(UPDATES_PER_STEP):
                    loss = agent.train_step()

                if loss is not None and t % 100 == 0:
                    losses.append((t, loss))

                state = next_state
                episode_return += reward

                if done or truncated:
                    print(
                        f"Total T: {t}  Episode: {episode}  Steps: {episode_steps}  "
                        f"Return: {episode_return:.3f}  Epsilon: {agent.epsilon:.3f}"
                    )

                    training_returns.append(float(episode_return))
                    training_crashes.append(float(done))

                    if episode % EVAL_INTERVAL == 0:
                        mean_ret, std_ret, crash_rate = evaluate_agent(agent)
                        current_score = eval_score(mean_ret, crash_rate)
                        eval_returns.append({
                            "step": t,
                            "episode": episode,
                            "mean_return": float(mean_ret),
                            "std_return": float(std_ret),
                            "crash_rate": float(crash_rate),
                            "score": float(current_score),
                        })
                        print(
                            f"  [EVAL] Mean Return: {mean_ret:.3f} +/- {std_ret:.3f}  "
                            f"Crash Rate: {crash_rate:.1%}  Score: {current_score:.3f}"
                        )

                        # Save best model if this eval is the best so far.
                        # Only track after the warmup: `learning_starts` counts
                        # environment transitions, so compare it against the
                        # environment step counter t, not agent.total_steps,
                        # which counts gradient updates.
                        if t > agent.learning_starts:
                            if best_score is None or current_score > best_score:
                                best_score = current_score
                                agent.save(BEST_WEIGHTS_PATH)
                                print(f"  [BEST] New best model saved (score {current_score:.3f})")

                    state, _ = env.reset()
                    state = state.reshape(-1)
                    episode += 1
                    episode_steps = 0
                    episode_return = 0.0

                # Periodic checkpoint: by step count OR by elapsed time,
                # whichever fires first. Time-based saving matters on Colab
                # free tier, where sessions can be killed without warning.
                step_due = (t + 1) % CHECKPOINT_INTERVAL == 0
                time_due = (time.time() - last_checkpoint_time) >= CHECKPOINT_TIME_INTERVAL
                if step_due or time_due:
                    reason = "step" if step_due else "time"
                    print(f"  [CHECKPOINT] Saving at step {t + 1} ({reason}-triggered)...")
                    agent.save_checkpoint(CHECKPOINT_PATH)
                    save_log(training_returns, training_crashes, eval_returns, losses, t + 1)
                    last_checkpoint_time = time.time()

        except KeyboardInterrupt:
            # Ignore further SIGINTs while saving: a second Stop press was
            # previously able to kill the save itself and lose the state.
            import signal
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\n\nTraining interrupted. Saving checkpoint -- do NOT stop again...")
            try:
                agent.save_checkpoint(CHECKPOINT_PATH)
                save_log(training_returns, training_crashes, eval_returns, losses, t + 1)
                print(f"Checkpoint saved to {CHECKPOINT_PATH}")
            except BaseException as e:
                # Last resort: weights-only save is fast and better than nothing
                print(f"Full checkpoint save failed ({e}); saving weights only...")
                try:
                    agent.save(os.path.join(SAVE_DIR, "emergency_weights.pt"))
                    print("Weights saved to weights/emergency_weights.pt")
                except BaseException as e2:
                    print(f"Emergency save also failed: {e2}")
            print("Re-run 'python training.py' to resume.")
            env.close()
            sys.exit(0)

    env.close()

    # Save final weights (last step, lightweight)
    LAST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "dqn_highway_last.pt")
    agent.save(LAST_WEIGHTS_PATH)
    print(f"\nLast model saved to {LAST_WEIGHTS_PATH}")

    # Save final training log
    save_log(training_returns, training_crashes, eval_returns, losses, MAX_STEPS)
    print(f"Training log saved to {LOG_PATH}")

    # Remove the resumable checkpoint since training is complete
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print(f"Removed intermediate checkpoint {CHECKPOINT_PATH}")

    # ---- Model selection, then unbiased test ----
    # Selection (best vs last) is decided on the SELECTION set. This is a
    # legitimate use of that set: it is what selection is for. The number we
    # REPORT, however, is the chosen model's score on the disjoint TEST set,
    # which was never used for any decision. Reporting the selection-set score
    # instead would be optimistically biased (winner's curse). Both are printed
    # so the gap between them is visible.
    from dqn_agent import DQNAgent as _DQNAgent

    print("\n--- Model selection (on SELECTION seeds) ---")

    sel_last_m, sel_last_s, sel_last_c = evaluate_agent(agent, num_episodes=20)
    sel_last_score = eval_score(sel_last_m, sel_last_c)
    print(f"[Last]  selection: {sel_last_m:.3f} +/- {sel_last_s:.3f}, "
          f"crash {sel_last_c:.1%}, score {sel_last_score:.3f}")

    chosen = "last"
    chosen_path = LAST_WEIGHTS_PATH
    if os.path.exists(BEST_WEIGHTS_PATH):
        best_agent = _DQNAgent(state_dim=25, action_dim=5, hidden_dim=128,
                               device=str(agent.device))
        best_agent.load(BEST_WEIGHTS_PATH)
        sel_best_m, sel_best_s, sel_best_c = evaluate_agent(best_agent, num_episodes=20)
        sel_best_score = eval_score(sel_best_m, sel_best_c)
        print(f"[Best]  selection: {sel_best_m:.3f} +/- {sel_best_s:.3f}, "
              f"crash {sel_best_c:.1%}, score {sel_best_score:.3f}")
        if sel_best_score >= sel_last_score:
            chosen = "best"
            chosen_path = BEST_WEIGHTS_PATH

    shutil.copyfile(chosen_path, FINAL_WEIGHTS_PATH)
    print(f"\nSelected model: {chosen}  ->  {FINAL_WEIGHTS_PATH}")

    # ---- Unbiased TEST-set evaluation of the chosen model (100 episodes) ----
    print("\n--- TEST evaluation of the chosen model (100 held-out episodes) ---")
    final_agent = _DQNAgent(state_dim=25, action_dim=5, hidden_dim=128,
                            device=str(agent.device))
    final_agent.load(FINAL_WEIGHTS_PATH)
    mean_ret, std_ret, crash_rate = evaluate_agent_test(final_agent, num_episodes=100)
    print(f"  Mean Return: {mean_ret:.3f} +/- {std_ret:.3f}")
    print(f"  Crash Rate:  {crash_rate:.1%}")
    print(f"  Score:       {eval_score(mean_ret, crash_rate):.3f}")
    print("  (This TEST number is the reported result. It is measured on seeds")
    print("   never used for selection, so it is an unbiased estimate.)")

    final_eval = {
        "mean_return": float(mean_ret),
        "std_return": float(std_ret),
        "crash_rate": float(crash_rate),
        "score": float(eval_score(mean_ret, crash_rate)),
        "chosen": chosen,
        "n_test_episodes": 100,
        "note": "mean_return/crash_rate are on the TEST seed set (90000+i), "
                "disjoint from selection seeds. Selection used seeds 12345+i.",
    }
    with open(FINAL_EVAL_PATH, "w") as f:
        json.dump(final_eval, f, indent=2)


if __name__ == "__main__":
    main()
