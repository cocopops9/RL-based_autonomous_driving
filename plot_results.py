"""
Plot generation for the RL project report.

Usage:
    python plot_results.py

Reads training logs and baseline results from the results/ directory and
produces publication-quality figures saved as PDF files in results/.
"""

import json
import os
import os
import numpy as np
import matplotlib
# The ICML template asks for Type-1 fonts and explicitly links this fix for
# matplotlib, whose default Type-3 subsetting is what the instruction targets.
# 42 selects TrueType, which is what the linked fix produces; Type-1 is not
# reachable from matplotlib without switching to a PS/dvi toolchain.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# The report includes these at \columnwidth (about 3.25 in) in a two-column
# layout. Drawing them at 8 in wide meant a 2.5x downscale that made every
# label illegible, so they are drawn at final size with explicit point sizes.
COLUMN_WIDTH_IN = 3.25
matplotlib.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 5.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
import matplotlib.pyplot as plt


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def smoothed(data, window=10):
    """Simple moving average for smoother curves."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def _matched_reference(default_results):
    """Reference line for the during-training figures.

    The greedy series is collected in the training environment
    (ego_spacing 2.0) on the selection seeds, so the baseline drawn against
    it must be measured the same way. Falls back to the test-seed value if
    the matched measurement is unavailable.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "baseline_training_env.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f), True
    return default_results, False


def plot_training_curve(training_log, baseline_results):
    """
    Plot training episode returns with a smoothed line, plus the baseline
    mean as a horizontal reference.
    """
    returns = training_log["training_returns"]
    episodes = np.arange(1, len(returns) + 1)
    smoothed_returns = smoothed(returns, window=20)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.3))

    # Raw episode returns (light)
    ax.plot(episodes, returns, alpha=0.25, color="C0", linewidth=0.8, label="Episode return")
    # Smoothed
    offset = len(returns) - len(smoothed_returns)
    ax.plot(episodes[offset:], smoothed_returns, color="C0", linewidth=2, label="Smoothed (window=20)")

    # Baseline horizontal line
    if baseline_results is not None:
        ref, matched = _matched_reference(baseline_results)
        tag = "same env + seeds" if matched else "test seeds"
        ax.axhline(
            y=ref["mean_return"],
            color="C1", linestyle="--", linewidth=1.5,
            label=f"Heuristic baseline ({ref['mean_return']:.2f}, {tag})",
        )

    # Overlay the greedy evaluation series. The training curve includes
    # epsilon-greedy exploration and so understates the deployed policy;
    # the greedy series is the honest learning-versus-baseline comparison.
    if training_log.get("eval_returns"):
        ev_x = [e["episode"] for e in training_log["eval_returns"]]
        ev_y = [e["mean_return"] for e in training_log["eval_returns"]]
        ax.plot(ev_x, ev_y, color="tab:green", marker="o", markersize=3,
                linewidth=1.3, label="Greedy evaluation (selection seeds)")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Episode Return")
    ax.set_title("DQN Training Progress vs. Baseline")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "training_curve.pdf"), dpi=150)
    fig.savefig(os.path.join(RESULTS_DIR, "training_curve.png"), dpi=150)
    plt.close(fig)
    print("Saved training_curve.pdf / .png")


def plot_eval_returns(training_log, baseline_results):
    """Plot periodic evaluation returns during training, highlighting the best checkpoint."""
    evals = training_log.get("eval_returns", [])
    if not evals:
        print("No evaluation data to plot.")
        return

    episodes = [e["episode"] for e in evals]
    means = [e["mean_return"] for e in evals]
    stds = [e["std_return"] for e in evals]

    # Identify best-score checkpoint if score field is present
    def _score(e):
        if "score" in e:
            return e["score"]
        return e["mean_return"] - 10.0 * e["crash_rate"]

    best_idx = max(range(len(evals)), key=lambda i: _score(evals[i]))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.3))
    ax.errorbar(episodes, means, yerr=stds, fmt="-o", color="C0",
                capsize=3, linewidth=1.5, markersize=4, label="Eval mean return")

    # Highlight best checkpoint
    ax.plot(episodes[best_idx], means[best_idx], marker="*",
            color="gold", markersize=18, markeredgecolor="black",
            markeredgewidth=0.8, zorder=5,
            label=f"Best model (ep {episodes[best_idx]})")

    if baseline_results is not None:
        ref, matched = _matched_reference(baseline_results)
        tag = "same env + seeds" if matched else "test seeds"
        ax.axhline(
            y=ref["mean_return"],
            color="C1", linestyle="--", linewidth=1.5,
            label=f"Heuristic baseline ({ref['mean_return']:.2f}, {tag})",
        )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Mean Evaluation Return")
    ax.set_title("Evaluation Performance During Training")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "eval_returns.pdf"), dpi=150)
    fig.savefig(os.path.join(RESULTS_DIR, "eval_returns.png"), dpi=150)
    plt.close(fig)
    print("Saved eval_returns.pdf / .png")


def plot_loss_curve(training_log):
    """Plot the training loss over time steps."""
    losses = training_log.get("losses", [])
    if not losses:
        print("No loss data to plot.")
        return

    steps = [l[0] for l in losses]
    values = [l[1] for l in losses]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.1))
    ax.plot(steps, values, color="C2", linewidth=0.8, alpha=0.6)
    # Smoothed
    if len(values) > 10:
        sm = smoothed(values, window=10)
        offset = len(values) - len(sm)
        ax.plot(steps[offset:], sm, color="C2", linewidth=2, label="Smoothed loss")
        ax.legend()

    ax.set_xlabel("Environment Step")
    ax.set_ylabel("Huber Loss")
    ax.set_title("DQN Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.pdf"), dpi=150)
    fig.savefig(os.path.join(RESULTS_DIR, "loss_curve.png"), dpi=150)
    plt.close(fig)
    print("Saved loss_curve.pdf / .png")


def plot_crash_rate(training_log, baseline_results):
    """Plot the crash rate over training episodes (smoothed)."""
    crashes = training_log.get("training_crashes", [])
    if not crashes:
        return

    episodes = np.arange(1, len(crashes) + 1)
    window = 20
    if len(crashes) > window:
        crash_smooth = smoothed(crashes, window=window)
    else:
        crash_smooth = crashes

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.1))
    offset = len(crashes) - len(crash_smooth)
    ax.plot(episodes[offset:], crash_smooth, color="C3", linewidth=2, label=f"Crash rate (window={window})")

    if baseline_results is not None:
        ax.axhline(
            y=baseline_results["crash_rate"],
            color="C1", linestyle="--", linewidth=1.5,
            label=f"Heuristic baseline ({baseline_results['crash_rate']:.2f})",
        )

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Crash Rate")
    ax.set_title("Collision Rate During Training")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "crash_rate.pdf"), dpi=150)
    fig.savefig(os.path.join(RESULTS_DIR, "crash_rate.png"), dpi=150)
    plt.close(fig)
    print("Saved crash_rate.pdf / .png")


def main():
    training_log = load_json("training_log.json")
    baseline_results = load_json("baseline_results.json")

    if training_log is None:
        print("No training log found. Run training.py first.")
        return

    plot_training_curve(training_log, baseline_results)
    plot_eval_returns(training_log, baseline_results)
    plot_loss_curve(training_log)
    plot_crash_rate(training_log, baseline_results)

    print("\nAll plots generated in results/")


if __name__ == "__main__":
    main()
