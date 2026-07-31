# Post-release multi-dataset comparison

## Design

- Model: `deepseek-v4-flash` through OpenCode Go, thinking disabled.
- Five datasets with 15 named assets each: US technology, US financials,
  US healthcare, US industrial/energy, and cross-asset ETFs.
- The model receives the real ticker and full asset name, but no observation
  dates or data-source label.
- Formation months: May and June 2026. Each portfolio is evaluated on the first
  10 trading days of the following month (June and July 2026), for 20 test days.
- Both LLM methods use 30 calls per asset/pair. All methods use the same return
  panels, equal-cap Black--Litterman prior, optimizer, 0.15 maximum weight, and
  10 bps trading cost. `BL No Views` uses the equilibrium prior without LLM
  `P`, `Q`, or `Omega`. `MVO` uses the formation-window sample mean and
  covariance without BL or LLM inputs.
- RelView uses 30 candidate pairs and an abstention threshold of 0.60.

The windows begin after the public DeepSeek V4 release on 24 April 2026. This
reduces training-cutoff leakage, but cannot prove that the provider did not
silently update the served model after release.

## Main results: rolling isotonic calibration

| Dataset | MVO | BL No Views | Absolute LLM-BLM | RelView-BL | Equal Weight | Winner |
|---|---:|---:|---:|---:|---:|---|
| US technology | -16.93% | -12.95% | -17.55% | -12.17% | -9.46% | Equal Weight |
| US financials | +6.65% | +13.25% | +5.61% | +11.41% | +11.43% | BL No Views |
| US healthcare | -1.13% | +0.78% | +0.51% | +0.03% | +2.11% | Equal Weight |
| US industrial/energy | -2.38% | -0.69% | -1.12% | -0.70% | +2.29% | Equal Weight |
| Cross-asset ETFs | -3.08% | -3.97% | -2.85% | -4.45% | -0.80% | Equal Weight |

Median cumulative return across datasets was -2.38% for MVO, -0.69% for BL No
Views, -1.12% for Absolute LLM-BLM, -0.70% for RelView-BL, and +2.11% for Equal Weight.
RelView beat BL No Views in only one of five datasets and trailed it by 0.46
percentage points per dataset on average. Absolute beat BL No Views in one of
five and trailed it by 2.36 points on average. An equal-weight combination of
all five dataset portfolios returned -3.47%, -0.78%, -3.21%, -1.18%, and
+1.02%, respectively. RelView beat MVO in four of five datasets by an average
of 2.20 percentage points and a median of 1.68 points.

The main RelView run accepted 66 of 300 pair-period views. With only 30 prior
calibration observations available in the second formation month, isotonic
calibration was unstable: some datasets accepted all 30 second-month views and
others accepted none.

## No-calibration ablation

Using the same saved LLM calls with raw probabilities and the same 0.60
threshold, Equal Weight won four datasets and BL No Views won financials.
Median returns were -2.38% for MVO, -0.69% for BL No Views, -1.12% for
Absolute, -1.26% for RelView, and +2.11% for Equal Weight. The equal-weight
combination returned -3.47%, -0.78%, -3.21%, -0.67%, and +1.02%, respectively.
RelView beat BL No Views in
two of five datasets; its mean edge was +0.04 points but its median edge was
-0.57 points, so the result was driven by the technology dataset.

This ablation accepted only 12 of 300 pair-period views. It confirms that the
failure to consistently beat Equal Weight is not caused solely by the short
isotonic calibration history.

## Interpretation

Across these five post-release datasets, sample-mean MVO was unstable and was
beaten by RelView in four of five datasets. Adding absolute LLM views hurt the
no-view BL baseline materially. RelView was better than Absolute but did not
improve on BL No Views reliably, and none of the optimized variants beat a
simple equal-weight portfolio consistently. The observation window is only 20
trading days and all datasets share the same calendar dates, so the five
results are not independent and annualized statistics are not reliable. The
defensible conclusion is therefore negative but preliminary: the new data does
not reproduce the strong RelView advantage seen in the contaminated 2024--2025
historical test.

Machine-readable outputs are in `summary/` and `summary_no_calibration/`.
