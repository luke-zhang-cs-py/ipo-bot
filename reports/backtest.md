# Backtest

Walk-forward, out of sample, run `20261010T115905Z-weekly-0fae17` at 2026-10-10T11:59:05Z. Stocks: next-session direction (close up) of S&P 500 members; IPOs: a first close 20% or more above the offer. Base rate, random walk and average recent IPO are the baselines; DM is Diebold-Mariano (HLN-corrected), the bootstrap is a moving-block one.

## Stocks: next-day direction

974,606 out-of-sample predictions, 2017-01-03 to 2026-10-08.

| forecaster | Brier | log loss | hit rate | ECE | MAE | RMSE |
|---|---|---|---|---|---|---|
| model | 0.2497 | 0.6925 | 51.7% | 0.007 | 0.0136 | 0.0208 |
| base | 0.2498 | 0.6927 | 51.8% | 0.010 | 0.0134 | 0.0207 |
| rw | 0.2500 | 0.6931 | 50.0% | 0.020 | 0.0134 | 0.0207 |

| comparison | mean loss difference | DM p | DM p (Holm) | bootstrap 95% CI | bootstrap p |
|---|---|---|---|---|---|
| model vs base: brier | -0.00010 | 0.541 | 1 | [-0.00042, +0.00023] | 0.534 |
| model vs base: log_loss | -0.00020 | 0.533 | 1 | [-0.00084, +0.00046] | 0.528 |
| model vs base: abs_error | +0.00012 | 3.44e-06 | 2.06e-05 | [+0.00007, +0.00017] | 0.0005 |
| model vs rw: brier | -0.00031 | 0.259 | 1 | [-0.00081, +0.00024] | 0.262 |
| model vs rw: log_loss | -0.00062 | 0.261 | 1 | [-0.00162, +0.00048] | 0.265 |
| model vs rw: abs_error | +0.00012 | 3.44e-06 | 2.06e-05 | [+0.00007, +0.00017] | 0.0005 |

The model does not beat the best baseline (base) on Brier at the 5% level after Holm.
A negative difference means the model's loss is lower.

## IPOs: first-day pop

No resolved out-of-sample predictions yet.

## Leakage test (scrambled future)

- stocks: passed at cutoffs 2021-05-20, 2024-01-30, 2026-03-27 (largest change 0)
- IPOs: passed at cutoffs  (largest change 0)
