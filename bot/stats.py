"""Forecast scores and the tests that say whether one forecaster beats another.

Scores: Brier, log loss, hit rate and calibration (expected calibration error over ten equal-width bins) for
probabilities; MAE and RMSE for values. Comparisons: the Diebold-Mariano test on a series of loss differences,
with a Newey-West variance and the Harvey-Leybourne-Newbold small-sample correction, and a moving-block
bootstrap of the mean difference; Holm's step-down correction across several comparisons.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from scipy import stats as sps

EPS = 1e-6


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    yv = np.asarray(y, float)
    return -(yv * np.log(q) + (1 - yv) * np.log(1 - q))


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    """{"ece": expected calibration error, "table": [{lo, hi, n, mean_p, rate}]} over equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    table, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        mp, rate = float(p[m].mean()), float(y[m].mean())
        ece += m.sum() / len(p) * abs(mp - rate)
        table.append({"lo": b / bins, "hi": (b + 1) / bins, "n": int(m.sum()), "mean_p": mp, "rate": rate})
    return {"ece": float(ece), "table": table}


def hit_rate(p: np.ndarray, y: np.ndarray) -> float:
    """Share of rows whose side of 0.5 was right. A probability of exactly 0.5 calls neither side, so it scores
    half a hit rather than counting as a call of "down" (as in evaluation/harness)."""
    p, up = np.asarray(p, float), np.asarray(y, float) > 0.5
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == up)))


def scores(p: np.ndarray, y: np.ndarray, value: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
    """Every score of one forecaster on resolved rows."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    err = np.asarray(value, float) - np.asarray(actual, float)
    return {
        "n": len(p),
        "brier": float(brier(p, y).mean()),
        "log_loss": float(log_loss(p, y).mean()),
        "hit_rate": hit_rate(p, y),
        "ece": float(calibration(p, y)["ece"]),
        "mae": float(np.abs(err).mean()),
        "rmse": float(math.sqrt((err**2).mean())),
        "event_rate": float(y.mean()),
    }


def newey_west(d: np.ndarray, lags: int) -> float:
    """The long-run variance of a series (Bartlett weights)."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    v = float(d @ d) / n
    for k in range(1, min(lags, n - 1) + 1):
        v += 2 * (1 - k / (lags + 1)) * float(d[k:] @ d[:-k]) / n
    return v


def diebold_mariano(loss_a: Sequence[float], loss_b: Sequence[float], h: int = 1) -> Dict[str, float]:
    """Test of equal expected loss. d = loss_a - loss_b: a negative mean says A is better. Returns
    {mean_diff, stat, p} (two-sided, Student t with n-1 degrees of freedom after the HLN correction)."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 3:
        return {"mean_diff": float(d.mean()) if n else float("nan"), "stat": float("nan"), "p": float("nan"), "n": n}
    lags = max(h - 1, int(n ** (1 / 3)))
    var = newey_west(d, lags) / n
    if var <= 0:
        return {"mean_diff": float(d.mean()), "stat": float("nan"), "p": 1.0 if d.mean() == 0 else 0.0, "n": n}
    stat = d.mean() / math.sqrt(var)
    hln = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(2 * sps.t.sf(abs(stat), n - 1)), "n": n}


def block_bootstrap(
    loss_a: Sequence[float], loss_b: Sequence[float], reps: int = 2000, block: int = 0, seed: int = 0
) -> Dict[str, float]:
    """Moving-block bootstrap of the mean loss difference: {mean_diff, lo, hi (95%), p (two-sided)}. A fixed
    seed keeps the report reproducible."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    if n < 3:
        return {
            "mean_diff": float(d.mean()) if n else float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "p": float("nan"),
        }
    block = block or max(1, round(n ** (1 / 3)))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(reps, math.ceil(n / block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    means = d[idx].mean(axis=1)
    centered = means - d.mean()  # the null: no difference, same spread
    p = float((np.abs(centered) >= abs(d.mean())).mean())
    return {
        "mean_diff": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "p": max(p, 1 / reps),
    }


def holm(pvalues: Mapping[str, float]) -> Dict[str, float]:
    """Holm-adjusted p-values (NaN stays NaN)."""
    items = sorted(((p, k) for k, p in pvalues.items() if not math.isnan(p)))
    m = len(items)
    out = {k: float("nan") for k in pvalues}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def by_period(keys: Sequence[str], values: Sequence[float]) -> List[float]:
    """Mean of values per key, in key order: a cross-section of same-day stock losses becomes one number a day,
    so the test's series is not inflated by 500 correlated rows a day."""
    acc: Dict[str, List[float]] = {}
    for k, v in zip(keys, values):
        acc.setdefault(k, []).append(v)
    return [float(np.mean(acc[k])) for k in sorted(acc)]
