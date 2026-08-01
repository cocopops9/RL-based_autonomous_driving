"""
Render the demonstration animations used in the README.

Produces one GIF per policy over a fixed set of seeds, so the four clips are
directly comparable: the same traffic, four different drivers.

Usage:
    python make_media.py                 # all four clips
    python make_media.py --policy agent  # just one
"""
import argparse
import os

import gymnasium
import highway_env
import imageio
import numpy as np

from dqn_agent import DQNAgent
from your_baseline import heuristic_action

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {"action": {"type": "DiscreteMetaAction"},
       "lanes_count": 3, "ego_spacing": 1.5,
       "vehicles_count": 50, "duration": 40}
SEEDS = [90000, 90001, 90002, 90003]
FPS = 12

POLICIES = ["baseline", "agent", "agent_strict", "agent_conservative"]


def build(policy):
    """Return (kind, callable_or_agent, filter) for a policy name."""
    if policy == "baseline":
        return "raw", heuristic_action, None
    agent = DQNAgent(state_dim=25, action_dim=5, hidden_dim=128)
    agent.load(os.path.join(HERE, "weights", "dqn_highway.pt"))
    filt = None
    if policy == "agent_strict":
        from safety_filter import StrictFilter
        filt = StrictFilter()
    elif policy == "agent_conservative":
        from safety_filter import ConservativeFilter
        filt = ConservativeFilter()
    return "net", agent, filt


def record(policy, outdir):
    kind, pol, filt = build(policy)
    env = gymnasium.make("highway-v0", config=CFG, render_mode="rgb_array")
    frames, returns, crashes = [], [], []

    for seed in SEEDS:
        obs, _ = env.reset(seed=seed)
        if filt is not None:
            filt.reset()
        done = trunc = False
        ep_return = 0.0
        while not (done or trunc):
            if kind == "raw":
                action = pol(obs)
            else:
                flat = obs.reshape(-1)
                action = pol.select_action(flat, evaluate=True)
                if filt is not None:
                    action = filt.filter(action, flat)
            obs, reward, done, trunc, _ = env.step(action)
            frames.append(env.render())
            ep_return += reward
        returns.append(ep_return)
        crashes.append(done)
        # Hold the final frame briefly so episode boundaries are visible.
        for _ in range(FPS // 2):
            frames.append(frames[-1])

    env.close()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{policy}.gif")
    imageio.mimsave(path, frames, fps=FPS, loop=0)
    print(f"{policy:20s} {len(frames):4d} frames  "
          f"returns {[round(r, 1) for r in returns]}  "
          f"crashes {sum(crashes)}/{len(SEEDS)}  -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=POLICIES, default=None)
    parser.add_argument("--outdir", default=os.path.join(HERE, "media"))
    args = parser.parse_args()
    targets = [args.policy] if args.policy else POLICIES
    for policy in targets:
        record(policy, args.outdir)


if __name__ == "__main__":
    main()
