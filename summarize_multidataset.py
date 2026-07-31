"""Create consolidated tables for the post-release multi-dataset study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtest import evaluate_realized_portfolio
from run_multidataset_experiment import DEFAULT_MANIFEST, DEFAULT_ROOT, load_manifest


METHODS = ("MVO", "BL_No_Views", "Absolute_LLM_BLM", "RelView_BL", "Equal_Weight")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--result-name", default="comparison")
    parser.add_argument("--output-name", default="summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    accepted_total = 0
    rejected_total = 0
    calibrations: set[str] = set()

    for dataset in manifest["datasets"]:
        dataset_id = str(dataset["id"])
        result_root = args.root / dataset_id / "results"
        summary_payload = json.loads(
            (result_root / f"{args.result_name}_summary.json").read_text(encoding="utf-8")
        )
        calibrations.add(str(summary_payload["config"]["calibration"]))
        periods = pd.read_csv(result_root / f"{args.result_name}_periods.csv")
        daily = pd.read_csv(result_root / f"{args.result_name}_daily.csv")
        daily.insert(0, "Dataset", dataset_id)
        daily_frames.append(daily)
        accepted_total += int(periods["Accepted_Relative_Views"].sum())
        rejected_total += int(periods["Rejected_Relative_Views"].sum())
        for method in METHODS:
            metrics = summary_payload["summary"][method]
            rows.append({
                "Dataset": dataset_id,
                "Description": dataset["description"],
                "Method": method,
                **metrics,
            })

    table = pd.DataFrame(rows)
    all_daily = pd.concat(daily_frames, ignore_index=True)
    meta = all_daily.groupby("Date")[[f"{method}_Return" for method in METHODS]].mean()
    aggregate: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        synthetic = pd.DataFrame({method: meta[f"{method}_Return"]})
        _, metrics = evaluate_realized_portfolio(synthetic, {method: 1.0})
        method_rows = table[table["Method"] == method]
        metrics["turnover"] = float(method_rows["total_turnover"].mean())
        metrics["transaction_cost_bps"] = float(
            manifest["experiment"]["transaction_cost_bps"]
        )
        aggregate[method] = metrics

    cumulative = table.pivot(index="Dataset", columns="Method", values="cumulative_return")
    wins = {method: int((cumulative.idxmax(axis=1) == method).sum()) for method in METHODS}
    medians = {method: float(cumulative[method].median()) for method in METHODS}
    rel_minus_abs = cumulative["RelView_BL"] - cumulative["Absolute_LLM_BLM"]
    rel_minus_bl = cumulative["RelView_BL"] - cumulative["BL_No_Views"]
    absolute_minus_bl = cumulative["Absolute_LLM_BLM"] - cumulative["BL_No_Views"]
    rel_minus_mvo = cumulative["RelView_BL"] - cumulative["MVO"]
    rel_minus_equal = cumulative["RelView_BL"] - cumulative["Equal_Weight"]
    report = {
        "experiment": manifest["experiment"],
        "result_name": args.result_name,
        "calibration": sorted(calibrations),
        "dataset_count": len(manifest["datasets"]),
        "method_wins": wins,
        "median_cumulative_return": medians,
        "relview_minus_absolute": {
            "mean": float(rel_minus_abs.mean()),
            "median": float(rel_minus_abs.median()),
            "positive_datasets": int((rel_minus_abs > 0).sum()),
        },
        "relview_minus_bl_no_views": {
            "mean": float(rel_minus_bl.mean()),
            "median": float(rel_minus_bl.median()),
            "positive_datasets": int((rel_minus_bl > 0).sum()),
        },
        "absolute_minus_bl_no_views": {
            "mean": float(absolute_minus_bl.mean()),
            "median": float(absolute_minus_bl.median()),
            "positive_datasets": int((absolute_minus_bl > 0).sum()),
        },
        "relview_minus_mvo": {
            "mean": float(rel_minus_mvo.mean()),
            "median": float(rel_minus_mvo.median()),
            "positive_datasets": int((rel_minus_mvo > 0).sum()),
        },
        "relview_minus_equal": {
            "mean": float(rel_minus_equal.mean()),
            "median": float(rel_minus_equal.median()),
            "positive_datasets": int((rel_minus_equal > 0).sum()),
        },
        "accepted_relative_views": accepted_total,
        "rejected_relative_views": rejected_total,
        "equal_weight_across_datasets": aggregate,
    }

    output = args.root / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "method_metrics.csv", index=False)
    cumulative.reset_index().to_csv(output / "cumulative_returns.csv", index=False)
    all_daily.to_csv(output / "daily_returns.csv", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
