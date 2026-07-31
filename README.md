# Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model 

 > This is an official implementation of the paper [Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model](https://arxiv.org/abs/2504.14345), presented at ICLR 2025 Workshop on Advances in Financial AI.

![model](figure/model.png)




## Project Structure

```
.
├── run.py                  # Main file to run LLMs and collect their views
├── collect_relative_views.py # Collects sparse pairwise LLM views
├── collect_absolute_views.py # Collects absolute views with the same LLM/provider
├── relview_bl.py           # RelView-BL method implementation
├── evaluate_relview.py     # Runs one leak-free RelView-BL period
├── evaluate_absolute_bl.py # Runs comparable absolute-view LLM-BLM
├── portfolio_backtest.py   # Shared realized-return metrics
├── baselines.py           # Implementation of baseline portfolio strategies
├── calculate_llm_returns.py # Calculates returns for LLM-based portfolios
├── evaluate_multiple.py    # Evaluates multiple portfolio strategies
├── responses/             # Stores LLM predictions and views
├── responses_portfolios/  # Contains baseline portfolio weights and returns
├── results/              # Final evaluation results
└── yfinance/             # Downloaded stock price data
```

## Workflow Description

### 1. Data Collection and LLM Views (`run.py`)
- Downloads S&P 500 stock price data using yfinance API
- Data is stored in the `yfinance/` directory
- Queries different LLM models (Qwen, LLaMA, Gemma, GPT) for stock return predictions
- LLM responses are stored in `responses/` directory as JSON files

### 2. Baseline Portfolio Construction (`baselines.py`)
- Implements two baseline portfolio strategies:
  1. Equal-weighted portfolio
  2. Mean-variance optimized portfolio
- Processes data monthly from June 2024 to February 2025
- Portfolio weights and returns are stored in `responses_portfolios/`

### 3. Portfolio Evaluation
The evaluation process is split into two main components:

#### a. LLM Returns Calculation (`calculate_llm_returns.py`)
- Processes the LLM-based portfolio weights
- Calculates portfolio returns for each LLM strategy
- Results are stored in `results/` directory

#### b. Multiple Strategy Evaluation (`evaluate_multiple.py`)
- Implements Black-Litterman portfolio optimization using LLM views
- Processes multiple time periods
- Generates final performance metrics and comparisons
- Stores final evaluation results in `results/` directory

### 4. RelView-BL (relative LLM views)

`relview_bl.py` adds the consistency-calibrated relative-view method without
changing the original absolute-return baseline. The pipeline:

1. selects a sparse set of pairs using return correlation, sector, and market-cap similarity;
2. asks the LLM which asset is more likely to outperform and repeats each comparison;
3. calibrates probabilities using only closed, earlier periods (isotonic or temperature scaling);
4. rejects low-confidence or unsupported comparisons;
5. projects pairwise logits onto global latent asset scores to remove cyclic contradictions;
6. constructs relative Black--Litterman `P`, `q`, and `Omega`;
7. solves a long-only portfolio with optional turnover cost and an asset-weight cap.

The uncertainty diagonal combines normalized probability entropy, disagreement
between repeated LLM calls, and rolling pairwise calibration error. If the
calibration history is smaller than `--min-calibration-samples`, calibration
falls back to the raw probability for that period.

![model](figure/cumulative_returns2.png)
![model](figure/boxplot_all2.png)
![model](figure/compare_weight2.png)

## File Descriptions

### Main Files
- `run.py`: Main entry point for collecting LLM views on stock returns
- `baselines.py`: Implements baseline portfolio construction strategies
- `calculate_llm_returns.py`: Calculates returns for LLM-based portfolios
- `evaluate_multiple.py`: Evaluates and compares different portfolio strategies

### Directories
- `responses/`: Contains JSON files with LLM predictions for each stock
- `responses_portfolios/`: Stores baseline portfolio weights and returns
- `results/`: Contains final evaluation results and performance metrics
- `yfinance/`: Stores downloaded stock price data and returns

## Usage

1. Run LLM predictions:
```bash
python run.py --model_name [qwen|llama|gemma|gpt]
```

2. Generate baseline portfolios:
```bash
python baselines.py
```

3. Calculate returns and evaluate strategies:
```bash
python calculate_llm_returns.py
python evaluate_multiple.py
``` 

### Run RelView-BL

Set your OpenCode Go API key, then collect 30--50 pairwise views with DeepSeek
V4 Flash. The collector uses `deepseek-v4-flash` through OpenCode Go and sends
`thinking.type=disabled` by default. For the proposed 20-stock scope, pass the
same `--universe universe.json` to both commands:

```powershell
$env:OPENCODE_GO_API_KEY = "..."
py collect_relative_views.py --returns yfinance/returns_2024-06-01_2024-06-30.csv --universe universe.json --market-caps market_caps.json --max-pairs 50 --repeats 30 --probability-estimator mean --thinking disabled --horizon-days 10 --output responses_relative/deepseek-v4-flash_2024-06.json
```

With `--probability-estimator mean` (the default), every call returns one
probability. Probabilities are first oriented as `P(asset_a > asset_b)` and
then averaged. For example, `0.90`, `0.60`, and `0.20` produce a final
probability of `0.5667`; their dispersion supplies the disagreement term in
`Omega`. Use `--probability-estimator votes` only for a vote-frequency ablation.

Use `--metadata companies.csv` and `--context context_2024-06.json` to include
point-in-time company, news, macro, or earnings information. `context_*.json`
is a JSON object keyed by ticker. To use curated competitor/shared-event pairs
instead of automatic selection, pass `--pairs pairs.csv`, with `asset_a` and
`asset_b` columns.

Saved or hand-built views use this schema (repeated probabilities are optional):

```json
{
  "views": [{
    "asset_a": "NVDA",
    "asset_b": "AMD",
    "preferred_asset": "NVDA",
    "probability": 0.72,
    "probability_samples": [0.70, 0.74, 0.71],
    "horizon_days": 10,
    "evidence": ["stronger data-center guidance"]
  }]
}
```

Evaluate the views using calibration observations from earlier, already closed
periods only:

```powershell
py evaluate_relview.py --returns yfinance/returns_2024-06-01_2024-06-30.csv --universe universe.json --views responses_relative/deepseek-v4-flash_2024-06.json --market-caps market_caps.json --history results/calibration_history.json --calibration isotonic --abstention-threshold 0.60 --max-weight 0.1 --output results/relview_2024-06.json
```

For an offline walk-forward backtest, add the realized next-window returns only
after that window has closed. The evaluator first builds the portfolio from old
history, then appends current outcomes for future periods:

```powershell
py evaluate_relview.py --returns yfinance/returns_2024-06-01_2024-06-30.csv --universe universe.json --views responses_relative/deepseek-v4-flash_2024-06.json --market-caps market_caps.json --history results/calibration_history.json --realized-returns yfinance/returns_2024-07-01_2024-07-31.csv --history-output results/calibration_history.json --output results/relview_2024-06.json
```

The diagnostics JSON contains accepted/rejected views, raw cycle count, latent
scores, posterior returns, and portfolio weights. A companion weights CSV is
created automatically. The reusable entry point for experiments and ablations
is `run_relview_bl()` in `relview_bl.py`.

### Same-model Absolute LLM-BLM comparison

For a controlled comparison, collect absolute views with the same DeepSeek V4
Flash model, 30 calls, universe, information set, and disabled thinking:

```powershell
py collect_absolute_views.py --returns yfinance/returns_2024-06-01_2024-06-30.csv --universe universe.json --repeats 30 --thinking disabled --horizon-days 10 --temperature 0.3 --output responses/deepseek-v4-flash_2024-06-01_2024-06-30.json
```

Evaluate Absolute LLM-BLM on July realized returns:

```powershell
py evaluate_absolute_bl.py --returns yfinance/returns_2024-06-01_2024-06-30.csv --views responses/deepseek-v4-flash_2024-06-01_2024-06-30.json --universe universe.json --market-caps market_caps.json --tau 0.025 --risk-aversion 0.1 --market-risk-aversion 2.5 --max-weight 0.1 --realized-returns yfinance/returns_2024-07-01_2024-07-31.csv --evaluation-days 10 --weights-output results/absolute_deepseek_2024-06_weights.csv --returns-output results/absolute_deepseek_2024-06_returns.csv --output results/absolute_deepseek_2024-06.json
```

Evaluate RelView-BL on the identical period. The operational default delta is
`0.60`; keep it explicit in experiments and select it on a validation period,
not the final test period:

```powershell
py evaluate_relview.py --returns yfinance/returns_2024-06-01_2024-06-30.csv --universe universe.json --views responses_relative/deepseek-v4-flash_2024-06.json --market-caps market_caps.json --calibration none --abstention-threshold 0.60 --tau 0.025 --risk-aversion 0.1 --market-risk-aversion 2.5 --max-weight 0.1 --realized-returns yfinance/returns_2024-07-01_2024-07-31.csv --evaluation-days 10 --history-output results/calibration_history.json --weights-output results/relview_deepseek_2024-06_weights.csv --returns-output results/relview_deepseek_2024-06_returns.csv --output results/relview_deepseek_2024-06.json
```

Both evaluator JSON files now include cumulative return, annualized return and
volatility, Sharpe ratio, maximum drawdown, turnover, and transaction cost. The
daily return paths are saved separately for plotting and multi-period analysis.

### December 2025 static-cutoff study

Run all five 15-asset datasets with formation data ending on 31 December 2025,
then hold the portfolios through the latest configured 2026 session:

```powershell
py -3.10 run_cutoff_backtest.py
```

Recompute every method from saved data and LLM responses without an API key,
then validate all CSV, Parquet, JSON, date-boundary, and sample-count artifacts:

```powershell
py -3.10 run_cutoff_backtest.py --skip-collect
py -3.10 validate_cutoff_backtest.py
```

Run a no-abstention RelView ablation from the same saved LLM calls without
overwriting the main 0.60-threshold results:

```powershell
py -3.10 run_cutoff_backtest.py --skip-collect --abstention-threshold 0.5 --results-root experiments/cutoff_2025_12/ablations/no_abstention
py -3.10 validate_cutoff_backtest.py --results-root experiments/cutoff_2025_12/ablations/no_abstention
```

See `experiments/cutoff_2025_12/REPORT.md` for the methodology, leakage caveat,
results, and reusable artifact catalog.
