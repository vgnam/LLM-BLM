# Paper-protocol reproduction

This experiment targets the data and walk-forward protocol described in
arXiv:2504.14345v2: 50 large S&P 500 constituents, June 2024 through June
2025, a June-August validation phase, a September-June test phase, and
10-trading-day lookback/holding windows. Only complete holding windows are
evaluated, giving 20 test rebalances and matching the paper's 100,000 raw
views per model when `N=100`.

The optimizer uses the paper objective `variance - 0.1 * expected_return`, is
long-only and fully invested, and has no additional concentration cap
(`max_weight=1.0`). Turnover costs are 10 bps per rebalance.

Known reproduction limits are machine-readable in `config.json`:

- the paper names an exact 2025-03-26 ranking but does not publish the list;
  the tracked reconstruction uses a March 2025 top-50 index-weight table posted
  on 2025-04-01, the nearest public snapshot found. This fixes the public
  repository's material omission of Berkshire Hathaway but remains a
  five-calendar-day snapshot approximation;
- the paper does not specify how sector return series were constructed, so
  this implementation uses adjusted-close returns of the corresponding GICS
  Select Sector SPDR proxy;
- DeepSeek V4 Flash is not one of the four paper models and its knowledge
  cutoff does not precede the backtest, so its outputs are a leakage-sensitive
  comparison, not a replication of the paper's reported model returns;
- `comparison_repeats=30` preserves the requested repeated-call comparison;
  `paper_exact_repeats=100` is available for the paper call-count convention.

The current public GitHub code is an older monthly/all-constituent pipeline and
does not implement the top-50 fortnight protocol described by arXiv v2. This
reproduction therefore treats the paper text and LaTeX prompt figures as the
protocol authority, and uses the public repository only for historical company
metadata. The optimizer convention is tested explicitly: the paper objective
is `variance - 0.1 * expected_return`, not the repository extension's legacy
`-expected_return + 0.1 * variance` utility convention.

The 20 test windows are not chosen after seeing performance. They are the 20
complete non-overlapping 10-trading-day holding blocks beginning with the first
September 2024 session; incomplete days after the twentieth block are excluded.
This count independently matches the paper's raw-view table:
`20 periods * 50 stocks * N=100 = 100,000` views per model.
