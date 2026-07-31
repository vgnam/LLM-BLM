"""Build reusable six-dataset comparisons and a human-readable experiment report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_PAPER_DATA_ROOT = Path("experiments/paper_reproduction/paper_sp500_top50")
DEFAULT_PAPER_RUN_ROOT = DEFAULT_PAPER_DATA_ROOT / "deepseek_comparison"
DEFAULT_FIVE_ROOT = Path("experiments/fortnight_5_datasets")
DEFAULT_MANIFEST = DEFAULT_FIVE_ROOT / "datasets.json"
DEFAULT_PREVIOUS_MANIFEST = Path("experiments/post_release_2026/datasets.json")
DEFAULT_OUTPUT = DEFAULT_FIVE_ROOT / "summary"
METHODS = (
    "MVO",
    "BL_No_Views",
    "Equal_Weight",
    "Absolute_LLM_BLM",
    "RelView_BL",
)


def manifest_tickers(manifest: dict[str, Any]) -> list[str]:
    tickers: list[str] = []
    for dataset in manifest.get("datasets", []):
        records = dataset.get("tickers", dataset.get("assets", []))
        for record in records:
            ticker = record.get("ticker") if isinstance(record, dict) else record
            tickers.append(str(ticker))
    return tickers


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_frame(frame: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    frame.to_parquet(base.with_suffix(".parquet"), index=False)


def phase_from_period(period_id: str) -> str:
    return str(period_id).split("_", 1)[0]


def relative_diagnostics(run_root: Path, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "responses_relative").glob("*.json")):
        response = load_json(path)
        views = [item for item in response.get("views", []) if item.get("status") == "ok"]
        aggregate = np.asarray([float(item["probability"]) for item in views], dtype=float)
        samples = np.asarray([
            float(value)
            for item in views
            for value in item.get("reported_probability_samples_a", [])
        ], dtype=float)
        call_confidence = np.maximum(samples, 1.0 - samples)
        vote_agreement = np.asarray([
            max(int(value) for value in item.get("votes", {}).values())
            / max(1, int(item.get("successful_repeats", 0)))
            for item in views
        ], dtype=float)
        rows.append({
            "Period": path.stem,
            "Phase": phase_from_period(path.stem),
            "Pair_Views": int(len(aggregate)),
            "Successful_Calls": int(len(samples)),
            "Aggregate_Confidence_Min": float(np.min(aggregate)),
            "Aggregate_Confidence_Mean": float(np.mean(aggregate)),
            "Aggregate_Confidence_Max": float(np.max(aggregate)),
            "Aggregate_Confidence_Sum": float(np.sum(aggregate)),
            "Mean_Abs_From_0_5": float(np.mean(np.abs(aggregate - 0.5))),
            "Abs_From_0_5_Sum": float(np.sum(np.abs(aggregate - 0.5))),
            "Near_Half_Below_0_55": int(np.sum(aggregate < 0.55)),
            "Accepted_At_Threshold": int(np.sum(aggregate >= threshold)),
            "Accepted_Share": float(np.mean(aggregate >= threshold)),
            "Per_Call_Confidence_Min": float(np.min(call_confidence)),
            "Per_Call_Confidence_Mean": float(np.mean(call_confidence)),
            "Per_Call_Confidence_Max": float(np.max(call_confidence)),
            "Per_Call_Confidence_Sum": float(np.sum(call_confidence)),
            "Mean_Directional_Agreement": float(np.mean(vote_agreement)),
            "Directional_Agreement_Sum": float(np.sum(vote_agreement)),
            "Threshold": float(threshold),
            "Probability_Semantics": response.get("probability_semantics"),
        })
    return pd.DataFrame(rows)


def absolute_diagnostics(run_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_root / "responses_absolute").glob("*.json")):
        response = load_json(path)
        assets = [item for item in response.values() if isinstance(item, dict)]
        samples = np.asarray([
            float(value)
            for item in assets
            for value in item.get("expected_return", [])
        ], dtype=float)
        rows.append({
            "Period": path.stem,
            "Phase": phase_from_period(path.stem),
            "Assets": int(len(assets)),
            "Successful_Calls": int(len(samples)),
            "Expected_Daily_Return_Min": float(np.min(samples)),
            "Expected_Daily_Return_Mean": float(np.mean(samples)),
            "Expected_Daily_Return_Max": float(np.max(samples)),
            "Expected_Daily_Return_Std": float(np.std(samples, ddof=1)),
            "Retry_Errors": int(sum(len(item.get("errors", [])) for item in assets)),
            "Attempted_Calls": int(sum(int(item.get("attempted_calls", 0)) for item in assets)),
        })
    return pd.DataFrame(rows)


def add_dataset(frame: pd.DataFrame, dataset: str, data_kind: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "Data_Kind", data_kind)
    result.insert(0, "Dataset", dataset)
    return result


def dataset_artifacts(
    dataset: str,
    data_kind: str,
    data_root: Path,
    run_root: Path,
) -> dict[str, pd.DataFrame]:
    results = run_root / "results"
    summary = load_json(results / "summary.json")
    methods = set(summary.get("summary", {}))
    if methods != set(METHODS):
        raise ValueError(f"{dataset} methods {sorted(methods)} != {sorted(METHODS)}")
    threshold = float(summary["config"]["abstention_threshold"])
    metrics = pd.DataFrame([
        {"Method": method, **values}
        for method, values in summary["summary"].items()
    ])
    metrics["Return_Rank"] = metrics["cumulative_return"].rank(
        ascending=False, method="min"
    ).astype(int)
    metrics["Sharpe_Rank"] = metrics["sharpe"].rank(
        ascending=False, method="min"
    ).astype(int)
    bl = metrics.set_index("Method").loc["BL_No_Views"]
    metrics["Cumulative_Return_Delta_vs_BL"] = metrics["cumulative_return"] - float(
        bl["cumulative_return"]
    )
    metrics["Sharpe_Delta_vs_BL"] = metrics["sharpe"] - float(bl["sharpe"])

    daily = pd.read_csv(results / "daily_returns_long.csv")
    periods = pd.read_csv(results / "period_metrics.csv")
    weights = pd.read_csv(results / "weights_long.csv")
    grouped = weights.groupby(["Method", "Period"], sort=False)["Weight"]
    concentration = grouped.agg(
        Max_Weight="max",
        HHI=lambda values: float(np.sum(np.square(values))),
        Positive_Assets=lambda values: int(np.sum(np.asarray(values) > 1e-8)),
    ).reset_index()
    concentration["Effective_Assets"] = 1.0 / concentration["HHI"]
    weight_diagnostics = concentration.groupby("Method", as_index=False).agg(
        Periods=("Period", "nunique"),
        Maximum_Observed_Weight=("Max_Weight", "max"),
        Mean_Period_Max_Weight=("Max_Weight", "mean"),
        Mean_HHI=("HHI", "mean"),
        Mean_Effective_Assets=("Effective_Assets", "mean"),
        Minimum_Positive_Assets=("Positive_Assets", "min"),
    )
    return {
        "metrics": add_dataset(metrics, dataset, data_kind),
        "daily": add_dataset(daily, dataset, data_kind),
        "periods": add_dataset(periods, dataset, data_kind),
        "weights": add_dataset(weights, dataset, data_kind),
        "weight_diagnostics": add_dataset(weight_diagnostics, dataset, data_kind),
        "relative": add_dataset(relative_diagnostics(run_root, threshold), dataset, data_kind),
        "absolute": add_dataset(absolute_diagnostics(run_root), dataset, data_kind),
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(output)


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def weighted_probability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    test = frame[frame["Phase"] == "test"].copy()
    rows: list[dict[str, Any]] = []
    for dataset, group in test.groupby("Dataset", sort=False):
        pairs = int(group["Pair_Views"].sum())
        calls = int(group["Successful_Calls"].sum())
        rows.append({
            "Dataset": dataset,
            "Pairs": pairs,
            "Calls": calls,
            "Aggregate_Confidence_Mean": float(group["Aggregate_Confidence_Sum"].sum() / pairs),
            "Near_Half_Share": float(group["Near_Half_Below_0_55"].sum() / pairs),
            "Accepted_Share": float(group["Accepted_At_Threshold"].sum() / pairs),
            "Per_Call_Confidence_Mean": float(group["Per_Call_Confidence_Sum"].sum() / calls),
            "Directional_Agreement_Mean": float(group["Directional_Agreement_Sum"].sum() / pairs),
        })
    return pd.DataFrame(rows)


def win_counts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = {
        "All_6": metrics,
        "New_5": metrics[metrics["Data_Kind"] == "new_dataset"],
    }
    for scope, subset in scopes.items():
        for metric in ("cumulative_return", "sharpe"):
            winners = subset.loc[
                subset.groupby("Dataset")[metric].transform("max") == subset[metric],
                "Method",
            ].value_counts()
            for method in METHODS:
                rows.append({
                    "Scope": scope,
                    "Metric": metric,
                    "Method": method,
                    "Wins": int(winners.get(method, 0)),
                })
    return pd.DataFrame(rows)


def build_report(
    metrics: pd.DataFrame,
    relative_summary: pd.DataFrame,
    wins: pd.DataFrame,
    weight_diagnostics: pd.DataFrame,
) -> str:
    metric_rows = []
    for item in metrics.itertuples(index=False):
        metric_rows.append([
            item.Dataset,
            item.Method,
            pct(item.cumulative_return),
            f"{item.sharpe:.3f}",
            pct(item.annualized_volatility),
            pct(item.max_drawdown),
            pct(item.Cumulative_Return_Delta_vs_BL),
        ])
    probability_rows = [
        [
            item.Dataset,
            item.Pairs,
            item.Calls,
            f"{item.Aggregate_Confidence_Mean:.4f}",
            pct(item.Near_Half_Share),
            pct(item.Accepted_Share),
            f"{item.Per_Call_Confidence_Mean:.4f}",
            pct(item.Directional_Agreement_Mean),
        ]
        for item in relative_summary.itertuples(index=False)
    ]
    win_rows = [
        [item.Scope, item.Metric, item.Method, item.Wins]
        for item in wins.itertuples(index=False)
        if item.Wins > 0
    ]
    concentration_rows = [
        [
            item.Dataset,
            item.Method,
            pct(item.Maximum_Observed_Weight),
            pct(item.Mean_Period_Max_Weight),
            f"{item.Mean_Effective_Assets:.2f}",
            item.Minimum_Positive_Assets,
        ]
        for item in weight_diagnostics.itertuples(index=False)
    ]
    return f"""# Paper-aligned and five-dataset comparison

## Protocol

- The paper reconstruction is run first on 50 reconstructed March-2025 S&P 500 leaders.
- Five additional mutually disjoint 15-asset datasets (also disjoint from the earlier five-universe study) use the same five validation and twenty test periods. They may overlap the paper top-50 universe.
- Every test contains 200 trading days from September 2024 through June 2025, so it is shorter than one year.
- All optimizers are long-only and fully invested with `max_weight=1.0`; there is no 15% concentration cap.
- Absolute LLM-BLM and RelView-BL both use DeepSeek V4 Flash, temperature 1, disabled reasoning, 30 calls, and validation-selected tau.
- RelView `decisive_v3` forces every call to report a ranking confidence of at least 0.55. The stored aggregate remains the arithmetic mean after orienting every call as `P(A>B)`, so directional disagreement may legitimately move the mean toward 0.5.
- Returns include 10 bps transaction costs based on the L1 change in portfolio weights at each rebalance.

## Test performance

{markdown_table(
    ["Dataset", "Method", "Cumulative return", "Sharpe", "Annualized volatility", "Max drawdown", "Return delta vs BL"],
    metric_rows,
)}

## Relative-score diagnostics on test periods

The values below are confidence-forced ranking scores, not calibrated real-world probabilities.

{markdown_table(
    ["Dataset", "Pairs", "Calls", "Mean aggregate", "Aggregate < 0.55", "Accepted at 0.60", "Mean per-call", "Directional agreement"],
    probability_rows,
)}

## Method wins

{markdown_table(["Scope", "Metric", "Method", "Wins"], win_rows)}

## Portfolio concentration with no 15% cap

{markdown_table(
    ["Dataset", "Method", "Maximum weight", "Mean period maximum", "Mean effective assets", "Minimum positive assets"],
    concentration_rows,
)}

## Interpretation constraints

DeepSeek V4 Flash postdates the historical test interval. Its outputs may remember information from that interval, so LLM-based returns are leakage-sensitive and are not an out-of-sample replication of the four models in the paper. The five new universes share the same calendar, so they are asset-universe robustness checks rather than five independent time samples. The exact 26 March 2025 paper universe was not published; the paper dataset uses the nearest public March-2025 reconstruction. The paper also does not publish its sector-return construction, so adjusted-close GICS sector ETF proxies are used.

## Reusable artifacts

CSV and Parquet tables in this directory contain combined daily returns, period metrics, long portfolio weights, method metrics, concentration diagnostics, absolute-view diagnostics, and relative-score diagnostics. `data_catalog.json` records their paths and units.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-data-root", type=Path, default=DEFAULT_PAPER_DATA_ROOT)
    parser.add_argument("--paper-run-root", type=Path, default=DEFAULT_PAPER_RUN_ROOT)
    parser.add_argument("--five-root", type=Path, default=DEFAULT_FIVE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--previous-manifest", type=Path, default=DEFAULT_PREVIOUS_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    new_tickers = manifest_tickers(manifest)
    if len(manifest.get("datasets", [])) != 5:
        raise ValueError("the new-data manifest must contain exactly five datasets")
    if any(len(dataset.get("tickers", [])) != 15 for dataset in manifest["datasets"]):
        raise ValueError("every new dataset must contain exactly 15 tickers")
    if len(new_tickers) != len(set(new_tickers)):
        raise ValueError("the five new datasets are not mutually disjoint")
    if args.previous_manifest.exists():
        previous = set(manifest_tickers(load_json(args.previous_manifest)))
        reused = sorted(set(new_tickers).intersection(previous))
        if reused:
            raise ValueError(f"new datasets reuse previous-study assets: {reused}")
    specifications = [
        (
            "paper_sp500_top50",
            "paper_reconstruction",
            args.paper_data_root,
            args.paper_run_root,
        )
    ]
    specifications.extend(
        (
            str(dataset["id"]),
            "new_dataset",
            args.five_root / str(dataset["id"]),
            args.five_root / str(dataset["id"]) / "deepseek_comparison",
        )
        for dataset in manifest["datasets"]
    )
    parts: dict[str, list[pd.DataFrame]] = {
        name: [] for name in (
            "metrics", "daily", "periods", "weights", "weight_diagnostics",
            "relative", "absolute",
        )
    }
    for dataset, data_kind, data_root, run_root in specifications:
        artifacts = dataset_artifacts(dataset, data_kind, data_root, run_root)
        for name, frame in artifacts.items():
            parts[name].append(frame)
    combined = {name: pd.concat(frames, ignore_index=True) for name, frames in parts.items()}
    relative_summary = weighted_probability_summary(combined["relative"])
    wins = win_counts(combined["metrics"])
    output_names = {
        "metrics": "method_metrics_all",
        "daily": "daily_returns_long_all",
        "periods": "period_metrics_all",
        "weights": "weights_long_all",
        "weight_diagnostics": "weight_diagnostics",
        "relative": "relative_probability_diagnostics",
        "absolute": "absolute_view_diagnostics",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for key, output_name in output_names.items():
        save_frame(combined[key], args.output / output_name)
    save_frame(relative_summary, args.output / "relative_probability_summary")
    save_frame(wins, args.output / "method_win_counts")
    (args.output / "REPORT.md").write_text(
        build_report(
            combined["metrics"], relative_summary, wins,
            combined["weight_diagnostics"],
        ),
        encoding="utf-8",
    )
    (args.output / "data_catalog.json").write_text(json.dumps({
        "datasets": [item[0] for item in specifications],
        "test_trading_days_per_dataset": 200,
        "methods": list(METHODS),
        "tables": {
            **{name: f"{name}.{{csv,parquet}}" for name in output_names.values()},
            "relative_probability_summary": "relative_probability_summary.{csv,parquet}",
            "method_win_counts": "method_win_counts.{csv,parquet}",
        },
        "report": "REPORT.md",
        "units": {
            "returns": "decimal net return after transaction costs",
            "weights": "portfolio fraction",
            "LLM expected returns": "decimal daily return",
            "relative probability": "confidence-forced ranking score, not calibrated probability",
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved six-dataset analysis to {args.output}", flush=True)


if __name__ == "__main__":
    main()
