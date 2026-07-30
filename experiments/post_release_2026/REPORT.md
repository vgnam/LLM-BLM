# Post-release multi-dataset comparison

## Design

- Model: `deepseek-v4-flash` through OpenCode Go, thinking disabled.
- Five datasets with 15 named assets each: US technology, US financials,
  US healthcare, US industrial/energy, and cross-asset ETFs.
- The model receives the real ticker and full asset name, but no observation
  dates or data-source label.
- Formation months: May and June 2026. Each portfolio is evaluated on the first
  10 trading days of the following month (June and July 2026), for 20 test days.
- Both methods use 30 calls per asset/pair, the same return panels, equal-cap
  Black--Litterman prior, optimizer, 0.15 maximum weight, and 10 bps trading cost.
- RelView uses 30 candidate pairs and an abstention threshold of 0.60.

The windows begin after the public DeepSeek V4 release on 24 April 2026. This
reduces training-cutoff leakage, but cannot prove that the provider did not
silently update the served model after release.

## Main results: rolling isotonic calibration

| Dataset | Absolute LLM-BLM | RelView-BL | Equal Weight | Winner |
|---|---:|---:|---:|---|
| US technology | -17.55% | -12.17% | -9.46% | Equal Weight |
| US financials | +5.61% | +11.41% | +11.43% | Equal Weight |
| US healthcare | +0.51% | +0.03% | +2.11% | Equal Weight |
| US industrial/energy | -1.12% | -0.70% | +2.29% | Equal Weight |
| Cross-asset ETFs | -2.85% | -4.45% | -0.80% | Equal Weight |

Median cumulative return across datasets was -1.12% for Absolute LLM-BLM,
-0.70% for RelView-BL, and +2.11% for Equal Weight. RelView beat Absolute in
three of five datasets, but did not beat Equal Weight in any dataset. An
equal-weight combination of all five dataset portfolios returned -3.21%,
-1.18%, and +1.02%, respectively.

The main RelView run accepted 66 of 300 pair-period views. With only 30 prior
calibration observations available in the second formation month, isotonic
calibration was unstable: some datasets accepted all 30 second-month views and
others accepted none.

## No-calibration ablation

Using the same saved LLM calls with raw probabilities and the same 0.60
threshold, Equal Weight won four datasets and RelView won the financials
dataset. Median returns were -1.12% for Absolute, -1.26% for RelView, and
+2.11% for Equal Weight. The equal-weight combination returned -3.21%, -0.67%,
and +1.02%, respectively.

This ablation accepted only 12 of 300 pair-period views. It confirms that the
failure to consistently beat Equal Weight is not caused solely by the short
isotonic calibration history.

## Interpretation

Across these five post-release datasets, RelView was generally better than the
absolute-view baseline in the primary specification, but neither LLM method
beat a simple equal-weight portfolio consistently. The observation window is
only 20 trading days and all datasets share the same calendar dates, so the
five results are not independent and annualized statistics are not reliable.
The defensible conclusion is therefore negative but preliminary: the new data
does not reproduce the strong RelView advantage seen in the contaminated
2024--2025 historical test.

Machine-readable outputs are in `summary/` and `summary_no_calibration/`.
