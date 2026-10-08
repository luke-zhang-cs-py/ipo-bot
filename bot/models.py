"""The models and their baselines. Plain numpy, so a fit is deterministic and its parameters fit in a JSON row.

Each model has two heads on the same standardized features: a logistic regression for the event's
probability (the stock closes higher next session; the IPO's first close is 20% or more above its offer) and a
ridge regression for the value (the next-day log return; the first-day return). A missing feature is set to
its training mean (0 once standardized), and standardized features are clipped to +-5, so one wild input
cannot swing a prediction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

CLIP = 5.0
# What a baseline says before it has outcomes to go on
PRIOR_UP = 0.5  # a stock closes higher next session
PRIOR_POP = 0.25  # an IPO's first close is 20% or more above its offer (about the long-run US rate)
PRIOR_FIRST_DAY = 0.15  # an IPO's first-day return


def fit_logit(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-penalized logistic regression by Newton's method. x includes no intercept; returns [b0, b1, ...]."""
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0  # the intercept is not shrunk
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        grad = X.T @ (p - y) + pen * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def fit_ridge(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x])
    pen = np.full(X.shape[1], l2)
    pen[0] = 0.0
    return np.linalg.solve(X.T @ X + np.diag(pen) + 1e-9 * np.eye(X.shape[1]), X.T @ y)


@dataclass
class Model:
    """Standardization and both heads' weights, for one feature list."""

    kind: str
    features: Sequence[str]
    mean: np.ndarray
    std: np.ndarray
    w_prob: np.ndarray
    w_value: np.ndarray
    n_train: int
    trained_through: str

    def _x(self, df: pd.DataFrame) -> np.ndarray:
        x = df[list(self.features)].to_numpy(dtype=float)
        z = (x - self.mean) / self.std
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=CLIP, neginf=-CLIP), -CLIP, CLIP)

    def prob(self, df: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([np.ones(len(df)), self._x(df)])
        return 1 / (1 + np.exp(-np.clip(X @ self.w_prob, -30, 30)))

    def value(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(df)), self._x(df)]) @ self.w_value

    def params(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "features": list(self.features),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "w_prob": self.w_prob.tolist(),
            "w_value": self.w_value.tolist(),
            "n_train": self.n_train,
            "trained_through": self.trained_through,
        }

    @property
    def model_id(self) -> str:
        digest = hashlib.sha256(json.dumps(self.params(), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.kind}-{self.trained_through}-{digest}"

    @classmethod
    def from_params(cls, p: Dict[str, Any]) -> Model:
        return cls(
            p["kind"],
            p["features"],
            np.array(p["mean"]),
            np.array(p["std"]),
            np.array(p["w_prob"]),
            np.array(p["w_value"]),
            int(p["n_train"]),
            p["trained_through"],
        )


def fit(kind: str, df: pd.DataFrame, features: Sequence[str], through: str, l2: float = 1.0) -> Model:
    """Fit both heads on the rows of df that have an outcome. Raises ValueError with too few rows."""
    train = df[df["y"].notna() & df["ret"].notna()]
    if len(train) < 30 or train["y"].nunique() < 2:
        raise ValueError(f"{kind}: {len(train)} training rows with an outcome; need 30 with both outcomes")
    x = train[list(features)].to_numpy(dtype=float)
    seen = ~np.isnan(x)
    n = seen.sum(axis=0)
    mean = np.where(n > 0, np.where(seen, x, 0.0).sum(axis=0) / np.maximum(n, 1), 0.0)  # an empty column: 0
    std = np.sqrt(np.where(seen, (x - mean) ** 2, 0.0).sum(axis=0) / np.maximum(n, 1))
    std = np.where(~np.isfinite(std) | (std < 1e-12), 1.0, std)
    m = Model(
        kind, list(features), mean, std, np.zeros(len(features) + 1), np.zeros(len(features) + 1), len(train), through
    )
    z = m._x(train)
    m.w_prob = fit_logit(z, train["y"].to_numpy(dtype=float), l2=l2)
    m.w_value = fit_ridge(z, train["ret"].to_numpy(dtype=float), l2=l2)
    return m


# ---------------------------------------------------------------------------- baselines


def base_rate(
    y: Sequence[float],
    groups: Sequence[Any],
    pred_at: Sequence[str],
    known_at: Sequence[Any],
    prior: float,
    min_obs: int = 20,
) -> np.ndarray:
    """Each row's event rate within its group, counting only outcomes known strictly before the row's
    prediction time (a stock's same-day outcomes are not known yet when its prediction is made). With fewer
    than min_obs such outcomes, the pooled rate; with fewer than that too, `prior`. Times are sortable text;
    a missing known time means the outcome is not known yet."""
    yv = np.asarray(y, dtype=float)
    order = sorted((str(k), i) for i, k in enumerate(known_at) if k is not None and not np.isnan(yv[i]))
    out = np.empty(len(yv))
    sums: Dict[Any, float] = {}
    ns: Dict[Any, int] = {}
    tot, n_tot, j = 0.0, 0, 0
    for i in sorted(range(len(yv)), key=lambda i: str(pred_at[i])):
        while j < len(order) and order[j][0] < str(pred_at[i]):
            k = order[j][1]
            sums[groups[k]] = sums.get(groups[k], 0.0) + yv[k]
            ns[groups[k]] = ns.get(groups[k], 0) + 1
            tot += yv[k]
            n_tot += 1
            j += 1
        g = groups[i]
        out[i] = sums[g] / ns[g] if ns.get(g, 0) >= min_obs else (tot / n_tot if n_tot >= min_obs else prior)
    return out


def recent_mean(
    values: Sequence[float],
    pred_at: Sequence[str],
    known_at: Sequence[Any],
    prior: float,
    window: int = 20,
    min_obs: int = 5,
) -> np.ndarray:
    """The mean of the last `window` outcomes known before each row's prediction time ("the average recent
    IPO"); `prior` with fewer than min_obs."""
    v = np.asarray(values, dtype=float)
    known = sorted((str(k), i) for i, k in enumerate(known_at) if k is not None and not np.isnan(v[i]))
    out = np.empty(len(v))
    for i in range(len(v)):
        past = [v[k] for t, k in known if t < str(pred_at[i])][-window:]
        out[i] = float(np.mean(past)) if len(past) >= min_obs else prior
    return out
