"""real_setup's fast logistic fit against ipo_eval.Logit, so the two cannot drift apart (run with the harness)."""
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import ipo_eval  # noqa: E402
import real_setup  # noqa: E402


def test_fit_logit_matches_ipo_eval():
    rng = random.Random(4)
    names = ["a", "b", "c", "d"]
    feats = [{k: rng.gauss(0, 1) for k in names} for _ in range(300)]
    ys = [int(f["a"] - 0.7 * f["c"] + rng.gauss(0, 1) > 0) for f in feats]
    for l2 in (0.1, 1.0, 30.0):
        slow = ipo_eval.Logit(names, l2).fit(feats, ys)
        X = np.array([[f[k] for k in names] for f in feats])
        fast = real_setup.fit_logit(X, np.array(ys, float), l2)
        assert np.max(np.abs(fast(X) - np.array([slow.prob(f) for f in feats]))) < 1e-9
