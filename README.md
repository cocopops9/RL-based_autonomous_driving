# Autonomous Driving RL Project

Reinforcement Learning course 2025/2026 — University of Padova

## Project Structure

```
rl_project_ad-main/
├── dqn_agent.py          # Double DQN agent with Dueling architecture (core module)
├── training.py           # Training loop (fills skeleton gaps)
├── evaluate.py           # Evaluation script (loads weights, runs 10 episodes)
├── your_baseline.py      # Heuristic baseline policy (rule-based, no learning)
├── manual_control.py     # Manual keyboard control with logging
├── plot_results.py       # Report figure generation
├── weights/              # Saved model checkpoints (after training)
│   └── dqn_highway.pt
├── results/              # Logs and plots (after training/evaluation)
│   ├── training_log.json
│   ├── baseline_results.json
│   ├── manual_control_results.json
│   └── *.pdf / *.png
└── README.md
```

## Setup

```bash
pip install gymnasium highway-env torch numpy matplotlib
```

**Recommended:** Use Anaconda with Python 3.10+.

## Usage

### 1. Run the heuristic baseline
```bash
python your_baseline.py
```
Runs 10 episodes with the rule-based policy and saves results to `results/baseline_results.json`.

### 2. Run manual control baseline
```bash
python manual_control.py
```
Use arrow keys to control the vehicle. Results saved to `results/manual_control_results.json`.

### 3. Train the DQN agent
```bash
python training.py
```
Trains a Double DQN agent for 20,000 steps. Saves:
- Model weights to `weights/dqn_highway.pt`
- Training logs to `results/training_log.json`

### 4. Evaluate the trained agent
```bash
python evaluate.py
```
Loads the trained model and runs 10 rendered episodes.

### 5. Generate report plots
```bash
python plot_results.py
```
Generates PDF/PNG figures from training logs and baseline results.

## Algorithm: Double DQN with Dueling Architecture

**Why DQN?** The action space is discrete (5 actions) and the state space is
continuous (25-dimensional flattened observation). DQN is the canonical
algorithm for this setting.

**Why Double DQN?** Standard DQN suffers from overestimation bias in the
Q-value targets. Double DQN (van Hasselt et al., 2016) decouples action
selection (online network) from action evaluation (target network), reducing
this bias.

**Why Dueling?** The Dueling architecture (Wang et al., 2016) decomposes
Q(s,a) = V(s) + A(s,a) - mean(A). In highway driving, many actions have
similar values (e.g., IDLE and FASTER when the lane is clear), so separating
value from advantage accelerates learning.

### Hyperparameters

| Parameter           | Value   | Rationale                                    |
|---------------------|---------|----------------------------------------------|
| Hidden dim          | 256     | Sufficient for 25-dim state                  |
| Learning rate       | 5e-4    | Standard for DQN                             |
| Gamma               | 0.99    | Long-horizon highway driving                 |
| Buffer size         | 15,000  | ~750 episodes of transitions                 |
| Batch size          | 64      | Standard                                     |
| Epsilon decay       | 10,000  | Linear from 1.0 to 0.05                      |
| Tau (soft update)   | 0.005   | Smooth target updates                        |
| Max steps           | 20,000  | Sufficient for convergence                   |

## Heuristic Baseline

Defensive rule-based policy with three priority levels:
1. **Imminent collision** (vehicle <0.10 ahead): lane change if safe, else brake
2. **Vehicle at medium range** (<0.25 ahead): slow down
3. **Road clear**: accelerate, merge right when safe

Performance: ~27 mean return, ~20% crash rate over 20 episodes.

## Environment

- **Highway-Env** (`highway-fast-v0` for training, `highway-v0` for evaluation)
- State: 5×5 kinematics matrix (5 nearest vehicles × 5 features), flattened to 25
- Actions: LANE_LEFT(0), IDLE(1), LANE_RIGHT(2), FASTER(3), SLOWER(4)
- Reward: high_speed(0.4) + right_lane(0.1) + collision(-1), normalized

## Seed

All random seeds fixed to 0 for reproducibility.
