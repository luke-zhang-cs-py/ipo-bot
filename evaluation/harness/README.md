# Point-in-time harness

`bot_benchmark.py`, `demo_setup.py` and `test_bot_benchmark.py` are a point-in-time backtesting harness: every
row of data carries the time it became public, bots only ever see a view cut at their prediction time, every
bot predicts the same events, and the tests check fairness, leaks (a scrambled-future test that must catch two
planted cheats), identity masking, calibration, paired significance (Diebold-Mariano with a block bootstrap and
Holm's correction), stability by year, ranking skill and edge after costs.

`real_setup.py` plugs ipo-bot into it. The default, `real`, puts IPO and stock events in one store so every test
applies: the candidate is the revision-only IPO model (the better of ipo-bot's two on 2015-2023) with the
benchmark's blend for stocks; the rivals pair the 14-feature IPO model with each single stock algorithm. `ipo` and
`market` run each half alone. The 2024-on IPO holdout stays sealed throughout.

```bash
pip install -r evaluation/harness/requirements.txt
cd evaluation/harness
python test_bot_benchmark.py                                       # the default: real data, IPO and stock events
HARNESS_SETUP=ipo python test_bot_benchmark.py                     # IPO models only
HARNESS_SETUP=market python test_bot_benchmark.py                  # stock algorithms only
HARNESS_SETUP=demo python -m pytest -q test_bot_benchmark.py       # the synthetic demo
```

The IPO setup needs the dataset first (`python evaluation/ipo_data.py`); the market setup uses the benchmark's
price cache, fetched on first use.

## Results, 8 October 2026

| Setup | Passed | Failed | Skipped |
|---|---|---|---|
| **real (the default)**: revision-only IPO model + blend, 868 IPOs and 6,150 stock-months | **10** | **5** | **0** |
| ipo: the 14-feature model alone | 8 | 4 | 3 |
| market: the blend alone | 8 | 5 | 2 |
| demo (synthetic) | 14 | 0 | 1 |

What changed the real setup's markers: the IPO candidate is now the revision-only model (Brier 0.1977 against the
14-feature model's 0.2017 on 2015-2023, and calibrated: error 1.1% overall against the 5% limit); IPO and stock
events share one store, so the IPO-only, ranking and cost tests all run; and the candidate declares its true
knowledge cutoff (it is fitted from nothing inside the walk-forward).

Passing: every fairness and leak test, the knowledge cutoff, beating both baselines, calibration.

Failing, and why these are findings rather than bugs:
- Beats every rival / on IPOs alone: the revision model is better than the 14-feature model (Brier -0.0062 on IPO
  days) but not significantly (one-sided p = 0.074, Holm 0.37; 95% CI -0.0153 to +0.0026). Six variants (sparser,
  more regularised, nonlinear, initial-range revision, recalibrated) were tried on 2015-2023; none beat the
  revision model significantly, so no further change was made.
- Years and kinds: on stock-months the blend ties every single algorithm year by year (differences within
  +/-0.0007 Brier), so it can't win 75% of years.
- Ranking and costs: the stock blend has no edge (IC -0.031; long-short -0.89% a month after costs), as the
  benchmark found (0 of 36).

These five can't be made to pass by tuning on the same years without the test no longer meaning anything. What
could genuinely move them: the sealed 2024-2026 IPOs (about 330 more, scored once), or new information for
stocks (fundamentals), not more settings.

Speed: runs that took 57 s per IPO walk-forward take 15.5 s (one pass over each view, numpy fits, plain-string
columns in the store); the scoreboard and the tests share their runs; the full default run takes about 8 minutes.
