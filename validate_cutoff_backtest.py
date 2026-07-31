"""Validate the December 2025 cutoff study and all reusable artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_cutoff_backtest import DEFAULT_CONFIG, METHODS, load_config
from run_multidataset_experiment import load_manifest


DEFAULT_ROOT = Path("experiments/cutoff_2025_12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def finite(values: Any) -> bool:
    array = np.asarray(values, dtype=float)
    return bool(np.isfinite(array).all())


def load_pair(base: Path, errors: list[str]) -> pd.DataFrame:
    csv_path = base.with_suffix(".csv")
    parquet_path = base.with_suffix(".parquet")
    if not csv_path.exists() or not parquet_path.exists():
        errors.append(f"{base}: missing CSV or Parquet file")
        return pd.DataFrame()
    csv_frame = pd.read_csv(csv_path)
    parquet_frame = pd.read_parquet(parquet_path)
    csv_comparison = csv_frame.copy()
    parquet_comparison = parquet_frame.copy()
    if "Date" in csv_comparison and "Date" in parquet_comparison:
        csv_comparison["Date"] = pd.to_datetime(csv_comparison["Date"])
        parquet_comparison["Date"] = pd.to_datetime(parquet_comparison["Date"])
    try:
        pd.testing.assert_frame_equal(
            csv_comparison,
            parquet_comparison,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"{base}: CSV/Parquet mismatch: {str(exc).splitlines()[0]}")
    return csv_frame


def validate_dates(
    frame: pd.DataFrame,
    label: str,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
    errors: list[str],
) -> pd.DatetimeIndex:
    if "Date" not in frame:
        errors.append(f"{label}: Date column missing")
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(pd.to_datetime(frame["Date"], errors="coerce"))
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.has_duplicates:
        errors.append(f"{label}: dates are invalid, duplicated, or unsorted")
    if len(dates) and (dates.min() < lower or dates.max() > upper):
        errors.append(f"{label}: dates fall outside [{lower.date()}, {upper.date()}]")
    return dates


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest = load_manifest(Path(config["dataset_manifest"]))
    repeats = int(config["repeats"])
    max_pairs = int(config["max_pairs"])
    cutoff = pd.Timestamp(config["cutoff_date"])
    formation_start = pd.Timestamp(config["formation_start"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])
    expected_methods = set(METHODS)
    errors: list[str] = []
    absolute_files = 0
    relative_files = 0
    absolute_samples = 0
    probability_samples = 0
    accepted_views = 0
    rejected_views = 0
    realized_date_sets: list[set[pd.Timestamp]] = []

    for dataset in manifest["datasets"]:
        dataset_id = str(dataset["id"])
        root = args.root / dataset_id
        assets = [str(item["ticker"]) for item in dataset["assets"]]
        if len(assets) != 15 or len(set(assets)) != 15:
            errors.append(f"{dataset_id}: expected 15 unique assets")

        formation = load_pair(root / "data" / "formation_returns", errors)
        realized = load_pair(root / "data" / "realized_returns", errors)
        formation_dates = validate_dates(
            formation, f"{dataset_id}/formation", formation_start, cutoff, errors
        )
        realized_dates = validate_dates(
            realized, f"{dataset_id}/realized", test_start, test_end, errors
        )
        realized_date_sets.append(set(realized_dates))
        for label, frame in (("formation", formation), ("realized", realized)):
            if list(frame.columns) != ["Date", *assets]:
                errors.append(f"{dataset_id}/{label}: wrong asset columns or order")
            elif not finite(frame[assets].to_numpy()):
                errors.append(f"{dataset_id}/{label}: non-finite returns")
        if len(formation_dates) < 100 or len(realized_dates) < 10:
            errors.append(f"{dataset_id}: insufficient formation or realized history")

        absolute_path = (
            root / "responses_absolute" / f"{config['model']}_cutoff_2025-12.json"
        )
        response = json.loads(absolute_path.read_text(encoding="utf-8"))
        absolute_files += 1
        if set(response) != set(assets):
            errors.append(f"{absolute_path}: wrong asset keys")
        for asset in assets:
            item = response.get(asset, {})
            samples = item.get("expected_return", [])
            absolute_samples += len(samples)
            if (
                len(samples) != repeats
                or int(item.get("successful_repeats", 0)) != repeats
                or not finite(samples)
                or item.get("model") != config["model"]
                or item.get("thinking") != "disabled"
                or int(item.get("horizon_days", -1)) != len(realized_dates)
            ):
                errors.append(f"{absolute_path}: invalid samples or metadata for {asset}")

        relative_path = (
            root / "responses_relative" / f"{config['model']}_cutoff_2025-12.json"
        )
        payload = json.loads(relative_path.read_text(encoding="utf-8"))
        views = payload.get("views", [])
        relative_files += 1
        if (
            payload.get("model") != config["model"]
            or payload.get("thinking") != "disabled"
            or payload.get("probability_estimator") != "mean"
            or payload.get("cutoff_date") != config["cutoff_date"]
            or int(payload.get("horizon_days", -1)) != len(realized_dates)
            or int(payload.get("pairs_requested", -1)) != max_pairs
            or len(views) != max_pairs
        ):
            errors.append(f"{relative_path}: invalid top-level metadata")
        for view in views:
            samples = view.get("probability_samples_a", [])
            probability_samples += len(samples)
            valid_probabilities = (
                len(samples) == repeats
                and finite(samples)
                and all(0.0 <= float(value) <= 1.0 for value in samples)
            )
            if (
                view.get("status") != "ok"
                or int(view.get("successful_repeats", 0)) != repeats
                or int(view.get("horizon_days", -1)) != len(realized_dates)
                or not valid_probabilities
                or not math.isclose(
                    float(view.get("probability_a", math.nan)),
                    float(np.mean(samples)) if samples else math.nan,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                errors.append(
                    f"{relative_path}: invalid pair {view.get('asset_a')}/{view.get('asset_b')}"
                )

        results = root / "results"
        daily = load_pair(results / "daily_long", errors)
        weights = load_pair(results / "weights_long", errors)
        metrics = load_pair(results / "method_metrics", errors)
        load_pair(results / "daily_wide", errors)
        if (
            len(daily) != len(realized_dates) * len(METHODS)
            or set(daily.get("Method", [])) != expected_methods
            or set(daily.get("Dataset", [])) != {dataset_id}
        ):
            errors.append(f"{dataset_id}: invalid daily result dimensions")
        if (
            len(weights) != len(assets) * len(METHODS)
            or set(weights.get("Method", [])) != expected_methods
            or set(weights.get("Asset", [])) != set(assets)
            or not finite(weights.get("Weight", []))
        ):
            errors.append(f"{dataset_id}: invalid weight dimensions")
        else:
            totals = weights.groupby("Method")["Weight"].sum()
            if not np.allclose(totals.to_numpy(), 1.0, rtol=1e-10, atol=1e-10):
                errors.append(f"{dataset_id}: method weights do not sum to one")
            if (weights["Weight"] < -1e-10).any() or (
                weights["Weight"] > float(config["max_weight"]) + 1e-8
            ).any():
                errors.append(f"{dataset_id}: method weights violate bounds")
        if len(metrics) != len(METHODS) or set(metrics.get("Method", [])) != expected_methods:
            errors.append(f"{dataset_id}: invalid method metric dimensions")

        diagnostics = json.loads(
            (results / "relview_diagnostics.json").read_text(encoding="utf-8")
        )
        accepted = len(diagnostics.get("accepted_views", []))
        rejected = len(diagnostics.get("rejected_views", []))
        accepted_views += accepted
        rejected_views += rejected
        if accepted + rejected != max_pairs:
            errors.append(f"{dataset_id}: RelView diagnostics do not cover all pairs")

        for method in METHODS:
            method_daily = load_pair(results / "by_method" / f"{method}_daily", errors)
            method_weights = load_pair(results / "by_method" / f"{method}_weights", errors)
            metric_path = results / "by_method" / f"{method}_metrics.json"
            if (
                len(method_daily) != len(realized_dates)
                or len(method_weights) != len(assets)
                or not metric_path.exists()
            ):
                errors.append(f"{dataset_id}/{method}: incomplete method-specific artifacts")

    if realized_date_sets and any(dates != realized_date_sets[0] for dates in realized_date_sets[1:]):
        errors.append("datasets do not share the same realized trading dates")
    test_days = len(realized_date_sets[0]) if realized_date_sets else 0
    dataset_count = len(manifest["datasets"])
    summary = args.root / "summary"
    summary_daily = load_pair(summary / "daily_returns_long", errors)
    summary_weights = load_pair(summary / "weights_long", errors)
    summary_metrics = load_pair(summary / "method_metrics", errors)
    aggregate_daily = load_pair(summary / "equal_dataset_portfolio_daily", errors)
    aggregate_metrics = load_pair(summary / "equal_dataset_portfolio_metrics", errors)
    load_pair(summary / "daily_returns_wide", errors)
    expected_summary_rows = dataset_count * len(METHODS) * test_days
    if len(summary_daily) != expected_summary_rows:
        errors.append("summary daily table has the wrong row count")
    if len(summary_weights) != dataset_count * len(METHODS) * 15:
        errors.append("summary weight table has the wrong row count")
    if len(summary_metrics) != dataset_count * len(METHODS):
        errors.append("summary metric table has the wrong row count")
    if len(aggregate_daily) != len(METHODS) * test_days:
        errors.append("aggregate daily table has the wrong row count")
    if len(aggregate_metrics) != len(METHODS):
        errors.append("aggregate metric table has the wrong row count")

    report = {
        "dataset_count": dataset_count,
        "assets_per_dataset": 15,
        "formation_days": int(len(formation_dates)) if "formation_dates" in locals() else 0,
        "test_days": test_days,
        "absolute_response_files": absolute_files,
        "relative_response_files": relative_files,
        "absolute_samples": absolute_samples,
        "probability_samples": probability_samples,
        "accepted_relative_views": accepted_views,
        "rejected_relative_views": rejected_views,
        "validation_errors": errors,
    }
    print(json.dumps(report, indent=2))
    (summary / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
