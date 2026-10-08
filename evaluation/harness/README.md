# Point-in-time harness

`bot_benchmark.py`, `demo_setup.py` and `test_bot_benchmark.py` are a point-in-time backtesting harness: every
row of data carries the time it became public, bots only ever see a view cut at their prediction time, every
bot predicts the same events, and the tests check fairness, leaks (a scrambled-future test that must catch two
planted cheats), identity masking, calibration, paired significance (Diebold-Mariano with a block bootstrap and
Holm's correction), stability by year, ranking skill and edge after costs.

`real_setup.py` plugs ipo-bot into it: `ipo` runs the pre-listing IPO model from `evaluation/ipo_eval.py` on the
real IPO dataset (2015-2023; the 2024-on holdout stays sealed), and `market` runs the benchmark's stock
algorithms on its 30 US large caps, monthly.

```bash
pip install -r evaluation/harness/requirements.txt
cd evaluation/harness
python -m pytest -q test_bot_benchmark.py                          # the synthetic demo
HARNESS_SETUP=ipo python test_bot_benchmark.py                     # scoreboard, then every test
HARNESS_SETUP=market python test_bot_benchmark.py
```

The IPO setup needs the dataset first (`python evaluation/ipo_data.py`); the market setup uses the benchmark's
price cache, fetched on first use.

## Results, 8 October 2026

| Setup | Passed | Failed | Skipped |
|---|---|---|---|
| demo (synthetic) | 14 | 0 | 1 (no knowledge cutoff to check) |
| ipo: ipo_eval's model, 868 IPOs 2015-2023 | 8 | 4 | 3 (knowledge cutoff; ranking and long-short need stock events) |
| market: the benchmark blend, 6,150 stock-months 2009-2026 | 8 | 5 | 2 (knowledge cutoff; IPO-only claims need IPO events) |

Every fairness and leak test passes in both real setups: reproducible, no change when the future is scrambled,
both planted cheats caught, identical predictions with names masked.

IPO: the model beats the base rate (Brier -0.0309, 95% CI -0.0479 to -0.0127, Holm p = 0.0001) and a coin flip
(-0.0476), but not its own revision-only version, which scores better (Brier 0.1977 against 0.2017; the model
wins 2 of 9 years against it). Calibration error 5.4%, against a 5.0% limit. The extra features add nothing
the price revision doesn't already carry.

Market: no edge. The blend doesn't beat the base rate (p = 0.20) or a coin flip (Holm p = 0.18); its mean
information coefficient is -0.031 (t = -1.46), and a long-short of its top and bottom fifths loses 0.89% a month
after costs. The same answer as the benchmark's 0 of 36.
