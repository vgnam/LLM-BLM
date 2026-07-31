# December 2025 cutoff backtest

## Design

- Information/formation window: 2 January through 31 December 2025 (250 common
  trading days). No 2026 realized return is included in an LLM prompt or used
  to form a portfolio.
- Test window: 2 January through 30 July 2026, the latest completed US session
  available at run time (144 common trading days).
- Policy: form each portfolio once at the cutoff and hold the initial weights
  for the complete test window. All methods use the same adjusted-close return
  panels, 0.15 maximum asset weight, and a 10 bp entry cost.
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

## Reproduce or recompute

```powershell
py -3.10 run_cutoff_backtest.py
py -3.10 run_cutoff_backtest.py --skip-collect
py -3.10 validate_cutoff_backtest.py
```

The first command collects missing views and evaluates the study. The second
reuses saved views and data without an API key. The third checks date boundaries,
sample counts, portfolio constraints, table dimensions, and CSV/Parquet equality.
