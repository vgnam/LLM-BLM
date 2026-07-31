# December 2025 cutoff backtest

## Design

- Information/formation window: 2 January through 31 December 2025 (250 common
  trading days). No 2026 realized return is included in an LLM prompt or used
  to form a portfolio.
- Test window: 2 January through 30 July 2026, the latest completed US session
  available at run time (144 common trading days).
- Policy: form each portfolio once at the cutoff and hold the initial weights
  for the complete test window. The archived headline tables below used the
  same adjusted-close panels, a historical 0.15 maximum asset weight, and a 10
  bp entry cost. The tracked canonical config is now `max_weight=1.0`; its
  already-computed no-cap comparison is reported separately below.
- Five datasets, each containing 15 named assets: US technology, US financials,
  US healthcare, US industrial/energy, and cross-asset ETFs.
- The two LLM methods use `deepseek-v4-flash` through OpenCode Go with thinking
  disabled. Each absolute asset view and each relative pair view is called 30
  times. Absolute LLM-BLM uses the sample mean as `Q` and sample variance as
  diagonal `Omega`. RelView-BL uses the mean of the 30 oriented pairwise
  probabilities and an abstention threshold of 0.60.
- BL No Views uses an equal-cap equilibrium prior with no LLM view. MVO uses
  the 2025 sample mean and covariance. Equal Weight assigns 1/15 to every asset.

## Cumulative return by dataset

| Dataset | MVO | BL No Views | Absolute LLM-BLM | RelView-BL | Equal Weight | Winner |
|---|---:|---:|---:|---:|---:|---|
| US technology | +49.66% | +49.55% | **+64.35%** | +49.55% | +18.98% | Absolute LLM-BLM |
| US financials | +7.55% | +4.99% | +7.68% | +4.99% | **+8.08%** | Equal Weight |
| US healthcare | **+10.90%** | +9.94% | -11.32% | +9.94% | +0.31% | MVO |
| US industrial/energy | +25.98% | +27.19% | **+29.61%** | +27.19% | +26.18% | Absolute LLM-BLM |
| Cross-asset ETFs | +6.61% | **+20.91%** | +5.57% | **+20.91%** | +12.67% | BL No Views / RelView-BL |

Median cumulative return across the five datasets was +20.91% for BL No Views
and RelView-BL, +12.67% for Equal Weight, +10.90% for MVO, and +7.68% for
Absolute LLM-BLM. Absolute beat BL No Views in three of five datasets, but its
mean edge was -3.34 percentage points because of large shortfalls in healthcare
and cross-asset ETFs.

## Equal-dataset aggregate

This portfolio averages the five already net-of-cost dataset portfolio returns
each day. It therefore gives every dataset equal weight and is not an additional
asset-level optimization.

| Method | Cumulative return | Annualized return | Annualized volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| MVO | +20.59% | +38.77% | 19.19% | 1.80 | -10.47% |
| BL No Views | **+23.17%** | **+44.01%** | 19.94% | **1.93** | -8.53% |
| Absolute LLM-BLM | +17.88% | +33.36% | 17.50% | 1.73 | -9.27% |
| RelView-BL | **+23.17%** | **+44.01%** | 19.94% | **1.93** | -8.53% |
| Equal Weight | +13.57% | +24.94% | **12.28%** | 1.87 | **-7.16%** |

The no-LLM BL baseline beat Absolute LLM-BLM by 5.29 percentage points in the
aggregate. RelView-BL exactly equals BL No Views in this experiment: all 150
candidate pair views were below the 0.60 confidence threshold, so RelView
correctly abstained from adding any LLM constraint. This is a valid RelView
outcome, but it does not provide evidence that relative LLM views improve BL at
this 144-day horizon.

## No-abstention ablation

Setting the threshold to 0.50 accepts all 150 saved pair views. This reuses the
same calls and changes only the view filter:

| Dataset | BL No Views | RelView-BL, threshold 0.50 | RelView edge |
|---|---:|---:|---:|
| US technology | +49.55% | +42.88% | -6.67 pp |
| US financials | +4.99% | +10.84% | +5.84 pp |
| US healthcare | +9.94% | -6.10% | -16.05 pp |
| US industrial/energy | +27.19% | +29.27% | +2.08 pp |
| Cross-asset ETFs | +20.91% | +11.19% | -9.72 pp |

The aggregate RelView return falls to +17.85%, with Sharpe 1.65 and maximum
drawdown -9.13%. It trails BL No Views by 5.32 percentage points and is almost
identical in cumulative return to Absolute LLM-BLM (+17.88%). The 0.60 filter
therefore protected the portfolio from weak views in this sample; it was not
hiding a broad RelView advantage. Because the observed confidence range is only
0.5003--0.5500, an intermediate threshold must be selected on a separate
validation period rather than tuned against these realized test returns.

## Temperature 1.0 with diversified system prompts

A second collection uses temperature 1.0 and 30 deterministic but differently
framed system prompts per asset/pair. Every successful sample stores its prompt
variant ID and SHA-256 hash. Across the study, all 2,250 absolute prompt hashes
and all 4,500 relative prompt hashes are unique. Thinking remains disabled.

| Relative-call diagnostic | Temperature 0.3, one prompt | Temperature 1.0, prompt ensemble |
|---|---:|---:|
| Mean pair confidence | 0.5409 | 0.5302 |
| Maximum pair confidence | 0.5500 | 0.5957 |
| Mean within-pair probability SD | 0.0139 | 0.0382 |
| Mean majority-vote share | 0.938 | 0.812 |
| Views accepted at 0.60 | 0/150 | 0/150 |

The ensemble increases disagreement and extends the upper confidence tail, but
does not produce a pair at or above 0.60. Consequently, threshold-0.60
RelView-BL remains identical to BL No Views. Absolute LLM-BLM improves from
+17.88% to +18.79% aggregate return; Sharpe improves from 1.73 to 1.91 and
maximum drawdown improves from -9.27% to -8.83%. It still trails BL No Views
at +23.17%.

Temperature 1.0 reduces absolute structured-output reliability: 611 malformed
JSON attempts were discarded and replaced, requiring 2,861 attempts for 2,250
valid absolute samples, versus 45 discarded attempts out of 2,295 at
temperature 0.3. All 4,500 relative calls in each scenario were valid on their
first attempt. The reported comparison therefore uses equal valid sample counts
but should retain this retry/selection caveat.

Using the temperature-1 ensemble responses, RelView threshold sensitivity is:

| Threshold | Accepted views | Aggregate return | Sharpe | Max drawdown |
|---:|---:|---:|---:|---:|
| 0.500 | 150 | +19.18% | 1.72 | -9.15% |
| 0.525 | 93 | +18.74% | 1.65 | -9.74% |
| 0.550 | 14 | +21.51% | 1.79 | -9.96% |
| 0.575 | 1 | +21.04% | 1.72 | -10.12% |
| 0.600 | 0 | **+23.17%** | **1.93** | **-8.53%** |

None of the thresholds with accepted LLM views beats BL No Views. This grid is
a diagnostic, not a valid threshold-selection exercise, because choosing its
best row using the same realized test window would be look-ahead overfitting.

## Removing the per-asset concentration cap

Setting `max_weight=1.0` removes the 0.15 cap for every method. With temperature
1.0 prompt-ensemble responses:

| Weight constraint | Threshold | Accepted views | BL No Views | RelView-BL | RelView max drawdown |
|---|---:|---:|---:|---:|---:|
| 0.15 cap | 0.60 | 0 | +23.17% | +23.17% | -8.53% |
| No cap | 0.60 | 0 | +35.99% | +35.99% | -16.31% |
| 0.15 cap | 0.50 | 150 | +23.17% | +19.18% | -9.15% |
| No cap | 0.50 | 150 | +35.99% | +12.26% | -13.41% |

Removing the cap cannot separate RelView from BL when no view is accepted:
both methods still optimize the same prior. At threshold 0.50, the methods do
separate, but the unconstrained optimizer produces nearly single-asset
portfolios. In US technology, BL No Views assigns 99.28% to MU and 0.72% to AMD,
while RelView assigns 100% to NVDA. Their technology test returns are +206.03%
and +4.60%, respectively. This concentration drives much of the aggregate gap
and shows that the 0.15 cap acts as important regularization rather than merely
hiding view effects.

Annualized figures summarize only 144 realized trading days and should not be
read as stable long-run estimates. The five datasets also share the same dates,
so they are diversified asset universes, not five independent time samples.

## Leakage boundary

The data pipeline enforces input-level separation: formation data end on 31
December 2025, realized data begin in January 2026, and the saved validation
report confirms the boundary. Prompts omit observation dates and the data-source
name, but retain the real ticker, company/fund name, sector, and asset type.

This does **not** prove the LLM evaluation is free of model-memory leakage. The
calls were made after part of the 2026 test period had already occurred, and the
served model's exact training/refresh cutoff is not independently known. It
could therefore recognize the assets and use memorized 2026 information despite
the point-in-time instruction. A strict leakage test would need a model frozen
before January 2026, anonymized asset identities, or genuinely forward data that
did not exist when calls were made. MVO, BL No Views, and Equal Weight do not
have this LLM-memory risk.

## Reusable artifacts

All main tables exist in both CSV and Parquet:

- `summary/daily_returns_long`: 3,600 rows with dataset, method, date, daily
  portfolio return, and cumulative return.
- `summary/daily_returns_wide`: one row per test date and one return column per
  dataset/method combination.
- `summary/weights_long`: 375 initial asset weights.
- `summary/method_metrics`: all 25 dataset/method metric rows.
- `summary/equal_dataset_portfolio_daily` and
  `summary/equal_dataset_portfolio_metrics`: the five-method aggregate.
- `{dataset}/results/by_method/`: standalone daily, weight, and metric files for
  each method on each dataset.
- `{dataset}/data/`: the exact formation and realized adjusted-close return
  panels used by the run.
- `{dataset}/responses_absolute/` and `{dataset}/responses_relative/`: every
  saved LLM sample, allowing the comparison to be recomputed without API calls.
- `summary/data_catalog.json`: paths, units, row counts, dates, configuration,
  methods, and dataset IDs.
- `summary/validation_report.json`: 2,250 absolute samples, 4,500 probability
  samples, and zero validation errors.
- `ablations/no_abstention/`: complete per-dataset and aggregate artifacts for
  threshold 0.50. `ablations/threshold_comparison/` contains ready-to-plot
  RelView daily paths and metric tables for thresholds 0.60 and 0.50.
- `ablations/temperature_1_prompt_ensemble/`: temperature-1 responses and main
  threshold-0.60 results. The corresponding `*_no_abstention`, `*_threshold_*`,
  and `*_threshold_sensitivity` directories contain the complete sensitivity
  outputs in CSV and Parquet.
- `ablations/temperature_prompt_comparison/`: ready-to-plot method paths,
  metrics, and pair-probability diagnostics for temperature 0.3 versus 1.0.
- `ablations/max_weight_comparison/`: capped/no-cap daily paths, metrics, and
  weights for thresholds 0.60 and 0.50, in both CSV and Parquet.

## Reproduce or recompute

```powershell
py -3.10 run_cutoff_backtest.py
py -3.10 run_cutoff_backtest.py --skip-collect
py -3.10 validate_cutoff_backtest.py
```

Collect and validate the temperature-1 prompt ensemble in its own output root:

```powershell
py -3.10 run_cutoff_backtest.py --temperature 1 --prompt-ensemble --responses-root experiments/cutoff_2025_12/ablations/temperature_1_prompt_ensemble --results-root experiments/cutoff_2025_12/ablations/temperature_1_prompt_ensemble
py -3.10 validate_cutoff_backtest.py --responses-root experiments/cutoff_2025_12/ablations/temperature_1_prompt_ensemble --results-root experiments/cutoff_2025_12/ablations/temperature_1_prompt_ensemble
```

The first command collects missing views and evaluates the study. The second
reuses saved views and data without an API key. The third checks date boundaries,
sample counts, portfolio constraints, table dimensions, and CSV/Parquet equality.
