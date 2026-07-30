"""Prepare, collect, backtest, and summarize the post-release multi-dataset study."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("experiments/post_release_2026/datasets.json")
DEFAULT_ROOT = Path("experiments/post_release_2026")


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("datasets"), list) or not value["datasets"]:
        raise ValueError("manifest needs a non-empty datasets list")
    return value


def run(command: list[str]) -> None:
    print("RUN", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def write_dataset_inputs(root: Path, dataset: dict[str, Any]) -> Path:
    dataset_root = root / str(dataset["id"])
    dataset_root.mkdir(parents=True, exist_ok=True)
    assets = dataset["assets"]
    tickers = [str(item["ticker"]) for item in assets]
    (dataset_root / "universe.json").write_text(
        json.dumps(tickers, indent=2), encoding="utf-8"
    )
    (dataset_root / "equal_caps.json").write_text(
        json.dumps({ticker: 1.0 for ticker in tickers}, indent=2), encoding="utf-8"
    )
    with (dataset_root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Symbol", "Name", "Sector", "Asset_Type"])
        writer.writeheader()
        for item in assets:
            writer.writerow({
                "Symbol": item["ticker"],
                "Name": item["name"],
                "Sector": item["sector"],
                "Asset_Type": item["asset_type"],
            })
    return dataset_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--datasets", nargs="*", help="Optional dataset IDs")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--overwrite-returns", action="store_true")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--retry-calls", type=int, default=60)
    parser.add_argument("--calibration", choices=["none", "temperature", "isotonic"])
    parser.add_argument("--output-name", default="comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    config = manifest["experiment"]
    selected = set(args.datasets or [])
    datasets = [
        item for item in manifest["datasets"]
        if not selected or str(item["id"]) in selected
    ]
    unknown = selected - {str(item["id"]) for item in datasets}
    if unknown:
        raise ValueError(f"unknown dataset IDs: {sorted(unknown)}")

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        print(f"\n=== {dataset_id}: {dataset['description']} ===", flush=True)
        dataset_root = write_dataset_inputs(args.root, dataset)
        returns_dir = dataset_root / "returns"
        absolute_dir = dataset_root / "responses_absolute"
        relative_dir = dataset_root / "responses_relative"
        universe = dataset_root / "universe.json"
        metadata = dataset_root / "metadata.csv"
        caps = dataset_root / "equal_caps.json"
        output_prefix = dataset_root / "results" / args.output_name

        if not args.skip_prepare:
            command = [
                sys.executable, "prepare_monthly_returns.py",
                "--start-month", str(config["start_month"]),
                "--end-month", str(config["realized_end_month"]),
                "--universe", str(universe),
                "--output-dir", str(returns_dir),
            ]
            if args.overwrite_returns:
                command.append("--overwrite")
            run(command)

        if not args.skip_collect:
            run([
                sys.executable, "collect_walkforward_views.py",
                "--start-month", str(config["start_month"]),
                "--end-month", str(config["end_month"]),
                "--returns-dir", str(returns_dir),
                "--universe", str(universe),
                "--metadata", str(metadata),
                "--market-caps", str(caps),
                "--absolute-dir", str(absolute_dir),
                "--relative-dir", str(relative_dir),
                "--methods", "absolute", "relative",
                "--repeats", str(config["repeats"]),
                "--min-successful-calls", "20",
                "--workers", str(args.workers),
                "--retry-calls", str(args.retry_calls),
                "--horizon-days", str(config["evaluation_days"]),
                "--temperature", "0.3",
                "--max-pairs", str(config["max_pairs"]),
                "--probability-estimator", "mean",
                "--thinking", "disabled",
            ])

        if not args.skip_backtest:
            run([
                sys.executable, "backtest_compare.py",
                "--start-month", str(config["start_month"]),
                "--end-month", str(config["end_month"]),
                "--returns-dir", str(returns_dir),
                "--absolute-dir", str(absolute_dir),
                "--relative-dir", str(relative_dir),
                "--universe", str(universe),
                "--market-caps", str(caps),
                "--evaluation-days", str(config["evaluation_days"]),
                "--calibration", str(args.calibration or config.get("calibration", "isotonic")),
                "--abstention-threshold", str(config["abstention_threshold"]),
                "--transaction-cost-bps", str(config["transaction_cost_bps"]),
                "--max-weight", "0.15",
                "--output-prefix", str(output_prefix),
            ])


if __name__ == "__main__":
    main()
