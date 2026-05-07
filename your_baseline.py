"""
Heuristic baseline policy for the Highway-Env autonomous driving task.

Design rationale:
    With 50 vehicles on a 3-lane highway but only 4 visible in the observation,
    aggressive lane changes are dangerous (blind spots). The heuristic therefore
    adopts a DEFENSIVE strategy:

    1. Default action: IDLE (maintain current speed/lane).
    2. If a same-lane vehicle is close ahead, SLOW DOWN.
    3. Only accelerate (FASTER) when the road ahead is clearly free.
    4. Only change lanes as a last resort, and only when the target lane
       has NO visible vehicles within a wide safety margin.
    5. Prefer right-lane merges (reward bonus) over left, but only when safe.

    This policy avoids collisions reliably while still capturing the
    high-speed reward when conditions allow.

Observation format (5x5 matrix, row 0 = ego, rows 1-4 = nearest vehicles):
    [presence, x, y, vx, vy]
    - Ego: absolute normalized values
    - Others: relative to ego

Actions:
    0 = LANE_LEFT,  1 = IDLE,  2 = LANE_RIGHT,  3 = FASTER,  4 = SLOWER
"""

import gymnasium
import highway_env
import numpy as np
import json
import os


# Action constants
LANE_LEFT = 0
IDLE = 1
LANE_RIGHT = 2
FASTER = 3
SLOWER = 4


def heuristic_action(obs):
    """
    Defensive heuristic policy.

    Given the raw 5x5 observation (NOT flattened), returns a discrete action.
    """
    ego = obs[0]
    others = obs[1:]

    # -- Thresholds (calibrated against default Kinematics normalization) --
    # Relative y ~ 0.333 per lane; relative x units ~ fraction of 100m clip.
    SAME_LANE_Y = 0.15            # |rel_y| < this -> same lane
    CLOSE_DIST = 0.10             # imminent collision range
    MEDIUM_DIST = 0.25            # comfortable following distance
    LANE_CLEAR_FRONT = 0.30       # safety window ahead for lane change
    LANE_CLEAR_REAR = 0.15        # safety window behind for lane change

    # Adjacent lane relative-y bands
    RIGHT_LANE_Y = (0.15, 0.55)
    LEFT_LANE_Y = (-0.55, -0.15)

    # ---- 1. Scan for same-lane vehicles ahead ----
    closest_ahead_x = float('inf')
    for v in others:
        if v[0] < 0.5:
            continue
        rel_x, rel_y = v[1], v[2]
        if abs(rel_y) < SAME_LANE_Y and rel_x > 0:
            closest_ahead_x = min(closest_ahead_x, rel_x)

    # ---- 2. Decision logic ----

    # Case A: imminent collision - must slow down or change lanes
    if closest_ahead_x < CLOSE_DIST:
        # Try lane change only if target is very clear
        right_ok = _is_lane_clear(others, RIGHT_LANE_Y, LANE_CLEAR_FRONT, LANE_CLEAR_REAR)
        left_ok = _is_lane_clear(others, LEFT_LANE_Y, LANE_CLEAR_FRONT, LANE_CLEAR_REAR)
        if right_ok:
            return LANE_RIGHT
        elif left_ok:
            return LANE_LEFT
        else:
            return SLOWER

    # Case B: vehicle ahead at medium range - slow down preemptively
    if closest_ahead_x < MEDIUM_DIST:
        return SLOWER

    # Case C: road is clear - accelerate, and merge right if possible
    ego_y = ego[2]
    # Ego y ~ 0.0 = leftmost lane, ~0.333 = middle, ~0.667 = rightmost
    # Try to merge right for the reward bonus
    if ego_y < 0.55:
        right_ok = _is_lane_clear(others, RIGHT_LANE_Y, LANE_CLEAR_FRONT, LANE_CLEAR_REAR)
        if right_ok:
            return LANE_RIGHT

    return FASTER


def _is_lane_clear(others, target_y_range, x_front, x_rear):
    """
    Check that NO visible vehicle occupies the target lane within the
    longitudinal safety window [-x_rear, +x_front] relative to ego.
    """
    y_min, y_max = target_y_range
    for v in others:
        if v[0] < 0.5:
            continue
        rel_x, rel_y = v[1], v[2]
        if y_min < rel_y < y_max and -x_rear < rel_x < x_front:
            return False
    return True


# ---------------------------------------------------------------------------
# Standalone evaluation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    np.random.seed(0)

    render = "--render" in sys.argv

    env_name = "highway-v0"
    env = gymnasium.make(
        env_name,
        config={
            "action": {"type": "DiscreteMetaAction"},
            "lanes_count": 3,
            "ego_spacing": 1.5,
        },
        render_mode="human" if render else None,
    )

    num_episodes = 10
    episode_returns = []
    episode_crashes = []

    state, _ = env.reset(seed=0)
    done, truncated = False, False
    episode = 1
    episode_steps = 0
    episode_return = 0.0

    while episode <= num_episodes:
        episode_steps += 1
        action = heuristic_action(state)
        state, reward, done, truncated, _ = env.step(action)
        if render:
            env.render()
        episode_return += reward

        if done or truncated:
            print(
                f"Episode Num: {episode}  Episode T: {episode_steps}  "
                f"Return: {episode_return:.3f}  Crash: {done}"
            )
            episode_returns.append(episode_return)
            episode_crashes.append(done)

            state, _ = env.reset()
            episode += 1
            episode_steps = 0
            episode_return = 0.0

    env.close()

    results = {
        "episode_returns": episode_returns,
        "episode_crashes": [bool(c) for c in episode_crashes],
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "crash_rate": float(np.mean(episode_crashes)),
    }
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBaseline Summary:")
    print(f"  Mean Return: {results['mean_return']:.3f} +/- {results['std_return']:.3f}")
    print(f"  Crash Rate:  {results['crash_rate']:.1%}")
    print(f"Results saved to results/baseline_results.json")
