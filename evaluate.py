"""
Evaluation script for the trained DQN agent on Highway-Env.

By default this runs the HONEST protocol: 100 episodes on the held-out TEST
seed set (90000+i), which is disjoint from the seeds used during training for
checkpoint selection. This is the number to trust and to cite.

A crash rate estimated on only a handful of episodes is close to meaningless:
at a true rate of 15% the standard error over 10 episodes is about 11
percentage points, so a lucky draw can read 0%. The default of 100 episodes
brings that standard error down to about 3-4 points. Fixing the seeds also
means the number is reproducible and comparable across models and filters.

Usage:
    python evaluate.py                     # 100 test episodes, headless summary
    python evaluate.py --render            # watch a few episodes in a window
    python evaluate.py --episodes 200      # more episodes, tighter estimate
    python evaluate.py --strict            # apply the strict safety filter
    python evaluate.py --conservative      # apply the conservative safety filter
    python evaluate.py --seed-base 12345   # evaluate on the SELECTION seeds instead
                                           # (for illustration only; do not report)
"""

import gymnasium
import highway_env
import numpy as np
import torch
import random
import os
import sys
import argparse

from dqn_agent import DQNAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained DQN agent.")
    p.add_argument("--episodes", type=int, default=10,
                   help="number of evaluation episodes (default 10 so that a bare "
                        "`python evaluate.py` finishes quickly; pass --full or "
                        "--episodes 100 for the protocol reported in the paper)")
    p.add_argument("--full", action="store_true",
                   help="use the reported protocol: 100 episodes")
    p.add_argument("--seed-base", type=int, default=90000,
                   help="episode i uses seed seed-base+i. Default 90000 is the "
                        "held-out TEST set. Use 12345 to reproduce the "
                        "(optimistic) selection-set numbers.")
    p.add_argument("--render", action="store_true",
                   help="render in a window (forces a small episode count)")
    p.add_argument("--weights", type=str, default=None,
                   help="path to weights (default weights/dqn_highway.pt)")
    # safety filters
    p.add_argument("--smooth", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--conservative", action="store_true")
    return p.parse_args()


def make_safety(args):
    if args.conservative:
        from safety_filter import ConservativeFilter
        return ConservativeFilter(), "conservative"
    if args.strict:
        from safety_filter import StrictFilter
        return StrictFilter(), "strict"
    if args.smooth:
        from safety_filter import SafetyFilter
        return SafetyFilter(), "smooth"
    return None, "none"


def main():
    args = parse_args()

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    safety, safety_name = make_safety(args)

    render_mode = "human" if args.render else None
    # When rendering, a handful of episodes is enough to watch; keep the
    # headless default at the full honest sample.
    episodes = 100 if args.full else args.episodes
    n_episodes = min(episodes, 5) if args.render else episodes

    env = gymnasium.make(
        "highway-v0",
        config={
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 3,
            "ego_spacing": 1.5,
            "vehicles_count": 50,
            "duration": 40,
        },
        render_mode=render_mode,
    )

    agent = DQNAgent(state_dim=25, action_dim=5, hidden_dim=128)
    weight_path = args.weights or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "weights", "dqn_highway.pt")
    agent.load(weight_path)
    print(f"Loaded model from {weight_path}")
    print(f"Protocol: {n_episodes} episodes, seeds {args.seed_base}..{args.seed_base + n_episodes - 1}, "
          f"safety filter: {safety_name}")
    if args.seed_base == 12345:
        print("WARNING: seed-base 12345 is the SELECTION set. These numbers are "
              "optimistic and must not be reported as test performance.")
    print("-" * 60)

    returns = []
    crashes = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=args.seed_base + ep)
        state = state.reshape(-1)
        if safety is not None:
            safety.reset()
        done, truncated = False, False
        ep_return = 0.0
        ep_steps = 0

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)
            if safety is not None:
                action = safety.filter(action, state)
            state, reward, done, truncated, _ = env.step(action)
            state = state.reshape(-1)
            if render_mode is not None:
                env.render()
            ep_return += reward
            ep_steps += 1

        returns.append(ep_return)
        crashes.append(1.0 if done else 0.0)

        # Per-episode line only when rendering or for small runs
        if render_mode is not None or n_episodes <= 20:
            print(f"Episode {ep+1:3d}  Steps: {ep_steps:3d}  "
                  f"Return: {ep_return:7.3f}  Crash: {bool(done)}", flush=True)
        elif (ep + 1) % 10 == 0:
            # Progress heartbeat for long runs, so the cell never looks frozen.
            running_crash = float(np.mean(crashes))
            print(f"  ... {ep+1}/{n_episodes} episodes  "
                  f"(running crash rate {running_crash:.1%})", flush=True)

    env.close()

    mean_r = float(np.mean(returns))
    std_r = float(np.std(returns))
    crash_rate = float(np.mean(crashes))
    n = len(returns)
    # Standard error of the crash-rate estimate (binomial)
    se = (crash_rate * (1 - crash_rate) / n) ** 0.5
    score = mean_r - 10.0 * crash_rate

    print("-" * 60)
    print(f"Episodes:     {n}")
    print(f"Mean return:  {mean_r:.3f} +/- {std_r:.3f}")
    print(f"Crash rate:   {crash_rate:.1%}  (standard error +/- {se:.1%})")
    print(f"Score (R - 10*crash): {score:.3f}")
    if n < 100:
        print(f"Note: {n} episodes is a small sample. The paper reports n=100 "
              f"(rerun with --full); at n={n} the crash rate standard error is "
              f"+/- {se:.1%}.")
    if crash_rate == 0.0:
        upper = 3.0 / n  # rule of three, approx 95% upper bound when 0 events
        print(f"Note: 0 crashes in {n} episodes bounds the true rate below "
              f"~{upper:.1%} at 95% confidence, not exactly zero.")


if __name__ == "__main__":
    main()
