"""Run and summarize the five new paper-calendar fortnight datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONFIG = Path("experiments/fortnight_5_datasets/config.json")
DEFAULT_MANIFEST = Path("experiments/fortnight_5_datasets/datasets.json")
DEFAULT_ROOT = Path("experiments/fortnight_5_datasets")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def summarize(root: Path, datasets: list[dict[str, Any]]) -> None:
    metric_rows: list[dict[str, Any]] = []
    daily_parts: list[pd.DataFrame] = []
    period_parts: list[pd.DataFrame] = []
    weight_parts: list[pd.DataFrame] = []
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        results = root / dataset_id / "deepseek_comparison" / "results"
        summary = load_json(results / "summary.json")
        metric_rows.extend({
            "Dataset": dataset_id,
            "Method": method,
            **metrics,
        } for method, metrics in summary["summary"].items())
        daily = pd.read_csv(results / "daily_returns.csv")
        daily.insert(0, "Dataset", dataset_id)
        daily_parts.append(daily)
        periods = pd.read_csv(results / "period_metrics.csv")
        periods.insert(0, "Dataset", dataset_id)
        period_parts.append(periods)
        weights = pd.read_csv(results / "weights_long.csv")
        weights.insert(0, "Dataset", dataset_id)
        weight_parts.append(weights)

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_method_metrics": pd.DataFrame(metric_rows),
        "daily_returns": pd.concat(daily_parts, ignore_index=True),
        "period_metrics": pd.concat(period_parts, ignore_index=True),
        "weights_long": pd.concat(weight_parts, ignore_index=True),
    }
    for name, frame in tables.items():
        frame.to_csv(summary_dir / f"{name}.csv", index=False)
        frame.to_parquet(summary_dir / f"{name}.parquet", index=False)
    (summary_dir / "data_catalog.json").write_text(json.dumps({
        "datasets": [dataset["id"] for dataset in datasets],
        "dataset_count": len(datasets),
        "methods": sorted(tables["dataset_method_metrics"]["Method"].unique().tolist()),
        "tables": {name: f"summary/{name}.{{csv,parquet}}" for name in tables},
        "units": {
            "daily returns": "decimal net return",
            "weights": "portfolio fraction",
            "turnover cost": "10 bps times L1 weight change at each rebalance",
        },
    }, indent=2), encoding="utf-8")
    print(f"Saved cross-dataset reusable tables to {summary_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--retry-calls", type=int, default=60)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    requested = set(args.datasets or [])
    datasets = [
        dataset for dataset in manifest["datasets"]
        if not requested or dataset["id"] in requested
    ]
    missing = requested - {dataset["id"] for dataset in datasets}
    if missing:
        raise ValueError(f"unknown datasets: {sorted(missing)}")
    if not args.skip_prepare:
        command = [
            sys.executable, "prepare_fortnight_datasets.py",
            "--config", str(args.config), "--manifest", str(args.manifest),
            "--root", str(args.root), "--datasets",
            *[str(dataset["id"]) for dataset in datasets],
        ]
        run(command)

    for dataset in datasets:
        dataset_root = args.root / str(dataset["id"])
        command = [
            sys.executable, "run_paper_reproduction.py",
            "--config", str(args.config),
            "--data-root", str(dataset_root),
            "--run-root", str(dataset_root / "deepseek_comparison"),
            "--methods", "absolute", "relative",
            "--workers", str(args.workers),
            "--retry-calls", str(args.retry_calls),
        ]
        if args.skip_collect:
            command.append("--skip-collect")
        if args.skip_backtest:
            command.append("--skip-backtest")
        if args.force:
            command.append("--force")
        if args.dry_run:
            command.append("--dry-run")
        run(command)
    if not args.dry_run and not args.skip_backtest and not args.skip_summary:
        summarize(args.root, datasets)


if __name__ == "__main__":
    main()
