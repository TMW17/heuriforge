"""Statistics for comparing two heuristics on the same instances (paired samples).

- Wilcoxon signed-rank test: is the difference systematic, or could it be luck?
- Holm's correction: several comparisons at once must not inflate false positives.
- Vargha-Delaney A12: effect size, the probability that the first heuristic does better
  on a randomly chosen instance (0.5 = no difference; 0.56 / 0.64 / 0.71 are the usual
  small / medium / large thresholds).
- Bootstrap confidence interval of the mean gap difference, in percentage points.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def wilcoxon_p(a: list[float], b: list[float]) -> float:
    """Two-sided p-value of the Wilcoxon signed-rank test on paired values."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.size == 0 or np.allclose(diff, 0.0):
        return 1.0
    return float(sps.wilcoxon(diff, zero_method="zsplit").pvalue)


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def a12_lower_is_better(a: list[float], b: list[float]) -> float:
    """Probability that a value of ``a`` is lower (better) than one of ``b``; ties count half."""
    x = np.asarray(a, dtype=float)[:, None]
    y = np.asarray(b, dtype=float)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def a12_label(a12: float) -> str:
    d = abs(a12 - 0.5)
    if d < 0.06:
        return "negligible"
    if d < 0.14:
        return "small"
    if d < 0.21:
        return "medium"
    return "large"


def bootstrap_ci(
    a: list[float], b: list[float], n: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) and its percentile bootstrap confidence interval."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(diff.mean()), float(lo), float(hi)


def compare(a: list[float], b: list[float], seed: int = 0) -> dict:
    """Everything needed to report "a vs b" on paired gaps (lower is better)."""
    mean, lo, hi = bootstrap_ci(a, b, seed=seed)
    a12 = a12_lower_is_better(a, b)
    wins = int(np.sum(np.asarray(a) < np.asarray(b) - 1e-9))
    losses = int(np.sum(np.asarray(a) > np.asarray(b) + 1e-9))
    return {
        "mean_diff": mean,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": wilcoxon_p(a, b),
        "a12": a12,
        "effect": a12_label(a12),
        "wins": wins,
        "losses": losses,
        "ties": len(a) - wins - losses,
    }
