"""
Visual comparison of the heuristic baseline and the trained DQN agent.

Opens a graphical window and runs episodes with rendering, so you can watch the
car drive and take screenshots for the report.

Usage:
    python visualize.py baseline        # watch the heuristic baseline
    python visualize.py agent           # watch the trained DQN agent
    python visualize.py both            # baseline episodes, then agent episodes

    # options
    python visualize.py agent --episodes 5
    python visualize.py agent --seed 90000     # start from a specific test seed

On Ubuntu with a Conda Python, if the window fails with a libGL/iris error,
launch with the software renderer:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 LIBGL_ALWAYS_SOFTWARE=1 \
        python visualize.py both
"""

import sys
import argparse
import numpy as np
import gymnasium
import highway_env

from your_baseline import heuristic_action

ENV_CONFIG = {
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 3,
    "ego_spacing": 1.5,
    "vehicles_count": 50,
    "duration": 40,
}


def make_env():
    return gymnasium.make("highway-v0", config=ENV_CONFIG, render_mode="human")


def run_baseline(episodes, seed):
    env = make_env()
    print("\n=== HEURISTIC BASELINE ===")
    for ep in range(episodes):
        s, _ = env.reset(seed=(seed + ep) if seed is not None else None)
        done = trunc = False
        ret = 0.0
        while not (done or trunc):
            a = heuristic_action(s)
            s, r, done, trunc, _ = env.step(a)
            env.render()
            ret += r
        print(f"  baseline ep {ep+1}: return={ret:.2f}  crash={done}")
    env.close()


def run_agent(episodes, seed, weights):
    import torch  # imported here so baseline-only runs need no torch
    from dqn_agent import DQNAgent
    env = make_env()
    agent = DQNAgent(state_dim=25, action_dim=5, hidden_dim=128, device="cpu")
    agent.load(weights)
    print(f"\n=== DQN AGENT ({weights}) ===")
    for ep in range(episodes):
        s, _ = env.reset(seed=(seed + ep) if seed is not None else None)
        s = s.reshape(-1)
        done = trunc = False
        ret = 0.0
        while not (done or trunc):
            a = agent.select_action(s, evaluate=True)
            s, r, done, trunc, _ = env.step(a)
            s = s.reshape(-1)
            env.render()
            ret += r
        print(f"  agent ep {ep+1}: return={ret:.2f}  crash={done}")
    env.close()



def record_clip(mode, episodes, seed, weights, out_path):
    """Record `episodes` consecutive episodes to an mp4 (no window)."""
    import imageio
    import gymnasium, highway_env
    cfg = {"action": {"type": "DiscreteMetaAction"},
           "lanes_count": 3, "ego_spacing": 1.5, "vehicles_count": 50, "duration": 40}
    fps = 15

    def policy_for(m):
        if m == "baseline":
            from your_baseline import heuristic_action
            return ("raw", (lambda o2d: heuristic_action(o2d)), None)
        from dqn_agent import DQNAgent
        ag = DQNAgent(state_dim=25, action_dim=5, hidden_dim=128)
        ag.load(weights)
        return ("net", ag, None)

    kind, pol, _ = policy_for("baseline" if mode == "baseline" else "agent")
    env = gymnasium.make("highway-v0", config=cfg, render_mode="rgb_array")
    frames = []
    for k in range(episodes):
        obs, _ = env.reset(seed=(seed + k) if seed is not None else None)
        done = trunc = False
        while not (done or trunc):
            if kind == "raw":
                a = pol(obs)
            else:
                a = pol.select_action(obs.reshape(-1), evaluate=True)
            obs, _, done, trunc, _ = env.step(a)
            frames.append(env.render())
        for _ in range(fps // 2):
            frames.append(frames[-1])
    env.close()
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"saved {out_path}: {episodes} episodes, {len(frames)} frames "
          f"(~{len(frames)/fps:.0f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["baseline", "agent", "both"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=90000,
                   help="starting seed; use the TEST seeds (90000+) to reproduce "
                        "the reported episodes. Pass -1 for random.")
    p.add_argument("--weights", type=str, default="weights/dqn_highway.pt")
    p.add_argument("--record", type=str, default=None,
                   help="if set, save an mp4 of N episodes to this path instead of "
                        "opening a window (needs imageio)")
    args = p.parse_args()

    seed = None if args.seed == -1 else args.seed

    if args.record is not None:
        record_clip(args.mode, args.episodes, seed, args.weights, args.record)
        return

    if args.mode in ("baseline", "both"):
        run_baseline(args.episodes, seed)
    if args.mode in ("agent", "both"):
        run_agent(args.episodes, seed, args.weights)


if __name__ == "__main__":
    main()
