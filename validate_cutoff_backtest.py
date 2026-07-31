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
from prompt_ensemble import ENSEMBLE_NAME


DEFAULT_ROOT = Path("experiments/cutoff_2025_12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Optional separate ablation output root; data and responses still come from --root",
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        help="Optional separate response root; formation/realized data still come from --root",
    )
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
    results_root = args.results_root or args.root
    responses_root = args.responses_root or args.root
    result_catalog_path = results_root / "summary" / "data_catalog.json"
    result_catalog = json.loads(result_catalog_path.read_text(encoding="utf-8"))
    result_config = result_catalog.get("config", config)
    manifest = load_manifest(Path(result_config["dataset_manifest"]))
    repeats = int(result_config["repeats"])
    max_pairs = int(result_config["max_pairs"])
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
    absolute_attempts = 0
    absolute_errors = 0
    relative_attempts = 0
    relative_errors = 0
    accepted_views = 0
    rejected_views = 0
    realized_date_sets: list[set[pd.Timestamp]] = []

    for dataset in manifest["datasets"]:
        dataset_id = str(dataset["id"])
        root = args.root / dataset_id
        response_root = responses_root / dataset_id
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
            response_root / "responses_absolute" / f"{result_config['model']}_cutoff_2025-12.json"
        )
        response = json.loads(absolute_path.read_text(encoding="utf-8"))
        absolute_files += 1
        if set(response) != set(assets):
            errors.append(f"{absolute_path}: wrong asset keys")
        for asset in assets:
            item = response.get(asset, {})
            samples = item.get("expected_return", [])
            absolute_samples += len(samples)
            absolute_attempts += int(item.get("attempted_calls", 0))
            absolute_errors += len(item.get("errors", []))
            if (
                len(samples) != repeats
                or int(item.get("successful_repeats", 0)) != repeats
                or not finite(samples)
                or item.get("model") != result_config["model"]
                or item.get("thinking") != "disabled"
                or int(item.get("horizon_days", -1)) != len(realized_dates)
            ):
                errors.append(f"{absolute_path}: invalid samples or metadata for {asset}")
            if result_config.get("prompt_ensemble", False):
                variant_ids = item.get("prompt_variant_ids", [])
                prompt_hashes = item.get("system_prompt_sha256", [])
                if (
                    item.get("prompt_ensemble") != ENSEMBLE_NAME
                    or not math.isclose(
                        float(item.get("temperature", math.nan)),
                        float(result_config["temperature"]),
                    )
                    or len(variant_ids) != repeats
                    or len(set(map(int, variant_ids))) != repeats
                    or len(prompt_hashes) != repeats
                    or len(set(map(str, prompt_hashes))) != repeats
                ):
                    errors.append(f"{absolute_path}: invalid prompt ensemble for {asset}")

        relative_path = (
            response_root / "responses_relative" / f"{result_config['model']}_cutoff_2025-12.json"
        )
        payload = json.loads(relative_path.read_text(encoding="utf-8"))
        views = payload.get("views", [])
        relative_files += 1
        if (
            payload.get("model") != result_config["model"]
            or payload.get("thinking") != "disabled"
            or payload.get("probability_estimator") != "mean"
            or payload.get("cutoff_date") != result_config["cutoff_date"]
            or int(payload.get("horizon_days", -1)) != len(realized_dates)
            or int(payload.get("pairs_requested", -1)) != max_pairs
            or len(views) != max_pairs
        ):
            errors.append(f"{relative_path}: invalid top-level metadata")
        if result_config.get("prompt_ensemble", False) and (
            payload.get("prompt_ensemble") != ENSEMBLE_NAME
            or not math.isclose(
                float(payload.get("temperature", math.nan)),
                float(result_config["temperature"]),
            )
        ):
            errors.append(f"{relative_path}: invalid prompt-ensemble metadata")
        for view in views:
            samples = view.get("probability_samples_a", [])
            probability_samples += len(samples)
            relative_attempts += int(view.get("attempted_calls", 0))
            relative_errors += len(view.get("errors", []))
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
            if result_config.get("prompt_ensemble", False):
                variant_ids = view.get("prompt_variant_ids", [])
                prompt_hashes = view.get("system_prompt_sha256", [])
                if (
                    view.get("prompt_ensemble") != ENSEMBLE_NAME
                    or len(variant_ids) != repeats
                    or len(set(map(int, variant_ids))) != repeats
                    or len(prompt_hashes) != repeats
                    or len(set(map(str, prompt_hashes))) != repeats
                ):
                    errors.append(
                        f"{relative_path}: invalid prompt ensemble for "
                        f"{view.get('asset_a')}/{view.get('asset_b')}"
                    )

        results = results_root / dataset_id / "results"
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
    summary = results_root / "summary"
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

    catalog_path = summary / "data_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    result_threshold = float(catalog.get("config", {}).get("abstention_threshold", math.nan))
    if not 0.5 <= result_threshold <= 1.0:
        errors.append("result catalog has an invalid abstention threshold")

    report = {
        "dataset_count": dataset_count,
        "assets_per_dataset": 15,
        "formation_days": int(len(formation_dates)) if "formation_dates" in locals() else 0,
        "test_days": test_days,
        "absolute_response_files": absolute_files,
        "relative_response_files": relative_files,
        "absolute_samples": absolute_samples,
        "probability_samples": probability_samples,
        "absolute_attempts": absolute_attempts,
        "absolute_errors": absolute_errors,
        "relative_attempts": relative_attempts,
        "relative_errors": relative_errors,
        "accepted_relative_views": accepted_views,
        "rejected_relative_views": rejected_views,
        "abstention_threshold": result_threshold,
        "temperature": float(result_config["temperature"]),
        "prompt_ensemble": bool(result_config.get("prompt_ensemble", False)),
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
