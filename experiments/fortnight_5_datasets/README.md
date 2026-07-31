# Five new fortnight datasets

These five disjoint 15-asset universes do not reuse assets from the previous
five-dataset manifest. They use the same validation/test calendar as the paper
protocol: five 10-trading-day validation windows and twenty 10-trading-day
test windows from September 2024 through June 2025 (200 test trading days,
less than one year).

All methods use the paper optimizer convention, long-only fully invested
weights with `max_weight=1.0`, market-cap equilibrium priors, and 10 bps
turnover costs. DeepSeek V4 Flash is queried 30 times at temperature 1 with
thinking disabled and a unique system-prompt variant per call. The relative
`decisive_v3` score is intentionally confidence-forced and is not a calibrated
real-world probability. Because the model postdates this historical period,
all LLM results remain vulnerable to training-memory leakage.
