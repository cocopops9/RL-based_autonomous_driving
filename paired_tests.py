"""
Recompute every statistical test reported in the paper.

All policies are evaluated on the identical seed set, so the crash outcomes
are paired and the correct instruments are McNemar's exact test on the
discordant episodes and a paired t-test on the per-episode return
differences, not the two-sample tests one would use on independent runs.

Input is results/episode_vectors.json, which stores the per-episode return
and crash flag for every configuration. That file is written by
run_final_eval.py, so the tests here are auditable without re-running the
evaluations (roughly 500 episodes of simulation).

No SciPy: the exact binomial tail, the Student t survival function and the
Wilson interval are implemented directly, so the only dependency is the
standard library.

Usage:
    python paired_tests.py
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS = os.path.join(HERE, "results", "episode_vectors.json")
OUTPUT = os.path.join(HERE, "results", "paired_tests.json")


def mcnemar_exact(a_crashes, b_crashes):
    """Two-sided exact McNemar test on paired binary outcomes.

    Returns (b10, b01, p) where b10 counts episodes that a crashed and b did
    not, and b01 the reverse. Concordant pairs carry no information about the
    difference and are correctly ignored.
    """
    b10 = sum(1 for x, y in zip(a_crashes, b_crashes) if x and not y)
    b01 = sum(1 for x, y in zip(a_crashes, b_crashes) if y and not x)
    disc = b10 + b01
    if disc == 0:
        return b10, b01, 1.0
    lo = min(b10, b01)
    tail = sum(math.comb(disc, k) for k in range(lo + 1)) / 2 ** disc
    return b10, b01, min(2 * tail, 1.0)


def _t_sf(t, df):
    """Survival function of Student's t, via the regularized incomplete beta."""
    x = df / (df + t * t)
    return 0.5 * _betainc(df / 2.0, 0.5, x)


def _betainc(a, b, x):
    """Regularized incomplete beta via its continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(200):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    result = front * (f - 1.0)
    return result if a >= b else result  # symmetric use is avoided below


def paired_t(a_returns, b_returns):
    """Paired t-test on per-episode return differences."""
    diffs = [x - y for x, y in zip(a_returns, b_returns)]
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("inf")
    p = 2 * _t_sf(abs(t), n - 1)
    return mean, t, min(max(p, 0.0), 1.0)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def holm(pvalues):
    """Holm step-down adjusted p-values, returned in the input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    with open(VECTORS) as f:
        vec = json.load(f)

    base = vec["baseline"]
    comparisons = ["raw", "smooth", "strict", "conservative"]

    results = {"vs_baseline": {}}
    raw_p = []
    for name in comparisons:
        arm = vec[name]
        n = min(len(base["crashes"]), len(arm["crashes"]))
        b10, b01, p = mcnemar_exact(base["crashes"][:n], arm["crashes"][:n])
        mean_diff, t, tp = paired_t(arm["returns"][:n], base["returns"][:n])
        results["vs_baseline"][name] = {
            "baseline_only_crashes": b10,
            "policy_only_crashes": b01,
            "mcnemar_exact_p": round(p, 6),
            "return_diff": round(mean_diff, 3),
            "paired_t": round(t, 2),
            "paired_t_p": round(tp, 6),
            "n": n,
        }
        raw_p.append(p)

    for name, adj in zip(comparisons, holm(raw_p)):
        results["vs_baseline"][name]["holm_adjusted_p"] = round(adj, 6)
        results["vs_baseline"][name]["significant_at_5pct"] = bool(adj < 0.05)

    # Filter against filter.
    b10, b01, p = mcnemar_exact(vec["strict"]["crashes"], vec["conservative"]["crashes"])
    results["conservative_vs_strict"] = {
        "strict_only_crashes": b10, "conservative_only_crashes": b01,
        "mcnemar_exact_p": round(p, 6),
    }

    # The trivial policy shares its seeds with the agent, so the comparison is
    # paired on the 45 seeds it covers, not a two-sample test.
    brake = vec["always_brake"]
    n = len(brake["crashes"])
    b10, b01, p = mcnemar_exact(brake["crashes"], vec["raw"]["crashes"][:n])
    mean_diff, t, tp = paired_t(vec["raw"]["returns"][:n], brake["returns"][:n])
    results["always_brake_vs_raw"] = {
        "brake_only_crashes": b10, "raw_only_crashes": b01,
        "mcnemar_exact_p": round(p, 6),
        "return_diff": round(mean_diff, 3), "paired_t": round(t, 2),
        "paired_t_p": round(tp, 6), "n": n,
        "note": "Shared seeds 90000+i; paired test, not two-sample.",
    }

    # Speed-cap probe: small sample, so report the interval rather than a test.
    probe = vec["speed_cap_probe"]
    k, n = sum(probe["crashes"]), len(probe["crashes"])
    lo, hi = wilson_interval(k, n)
    strict_same = sum(vec["strict"]["crashes"][:n])
    results["speed_cap_probe"] = {
        "crashes": k, "n": n,
        "crash_rate": round(k / n, 4),
        "wilson_95_ci": [round(lo, 4), round(hi, 4)],
        "strict_crashes_same_seeds": strict_same,
        "note": "Underpowered: fails to show a benefit from the cap, too small to exclude one.",
    }

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    for name in comparisons:
        r = results["vs_baseline"][name]
        print(f"{name:13s} vs baseline: {r['baseline_only_crashes']:2d}/{r['policy_only_crashes']:2d} "
              f"McNemar p={r['mcnemar_exact_p']:.4f} Holm={r['holm_adjusted_p']:.4f} "
              f"return {r['return_diff']:+.2f} (t={r['paired_t']:.2f})")
    r = results["conservative_vs_strict"]
    print(f"conservative vs strict: {r['strict_only_crashes']}/{r['conservative_only_crashes']} "
          f"p={r['mcnemar_exact_p']:.4f}")
    r = results["always_brake_vs_raw"]
    print(f"always-brake vs raw (n={r['n']}): {r['brake_only_crashes']}/{r['raw_only_crashes']} "
          f"p={r['mcnemar_exact_p']:.4f}, return {r['return_diff']:+.2f}")
    r = results["speed_cap_probe"]
    print(f"speed-cap probe: {r['crashes']}/{r['n']} = {r['crash_rate']:.1%}, "
          f"Wilson [{r['wilson_95_ci'][0]:.1%}, {r['wilson_95_ci'][1]:.1%}], "
          f"strict on same seeds: {r['strict_crashes_same_seeds']}")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
