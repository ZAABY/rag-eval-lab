"""Ranking metrics and bootstrap confidence intervals."""
from __future__ import annotations

import math
import random


def recall_at_k(ranked, relevant, k):
    return 1.0 if any(c in relevant for c in ranked[:k]) else 0.0


def mrr_at_k(ranked, relevant, k=10):
    for i, c in enumerate(ranked[:k], start=1):
        if c in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k=10):
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:k], start=1) if c in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean. Returns (mean, low, high)."""
    values = list(values)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return sum(values) / n, lo, hi


def paired_bootstrap_diff(a, b, n_boot=2000, alpha=0.05, seed=0):
    """CI for mean(a) - mean(b) over the same questions."""
    assert len(a) == len(b)
    diffs = [x - y for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
