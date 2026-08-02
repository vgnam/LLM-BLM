# NVIDIA NIM: 2025 views, 2026 backtest

This experiment forms portfolios once from daily observations dated in 2025
and evaluates the fixed weights on the available 2026 sessions. It uses three
different datasets: US technology equities, US financial equities, and a
cross-asset ETF universe.

The method inventory is BL, MVO, EW, and one BLM-LLM plus one RelViewBL variant
for every configured NVIDIA NIM model. A requested model that is unavailable
is retained in `summary/method_inventory.csv`, `summary/method_metrics.csv`,
and `model_status.json`; it is never silently replaced by another model.

Run from the repository root (the API key is intentionally not stored here):

```powershell
$env:NVIDIA_API_KEY = "<your NVIDIA API key>"
py -3.10 run_nvidia_nim_experiment.py
```

Use `--skip-llm` to regenerate and test only the deterministic BL, MVO, and EW
artifacts. Successful LLM calls are checkpointed below each dataset's
`responses/` directory, so rerunning the command resumes rather than discards
completed calls.

Outputs include CSV and Parquet daily returns, NAV, weights, portfolio metrics,
per-method JSON files, PNG NAV plots, an equal-dataset aggregate, a data
catalog, model availability details, and a concise Markdown report.
