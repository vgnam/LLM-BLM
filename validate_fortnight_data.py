"""Validate prepared fortnight data and, optionally, completed response caches/results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prepare_paper_reproduction import sha256
from prompt_ensemble import ENSEMBLE_NAME


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--require-results", action="store_true")
    return parser.parse_args()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    config = load_json(args.config)
    if float(config.get("max_weight", -1)) != 1.0:
        errors.append("config max_weight must be 1.0 for the uncapped paper comparison")
    if config.get("objective_convention") != "paper_variance_minus_return":
        errors.append("config does not use the paper variance-minus-return objective")
    manifest = load_json(args.root / "data_manifest.json")
    assets = [str(item) for item in load_json(args.root / "universe.json")]
    caps = load_json(args.root / "market_caps.json")
    metadata = pd.read_csv(args.root / "metadata.csv")
    periods = pd.read_csv(args.root / "periods.csv")
    stock_panel = pd.read_csv(
        args.root / "data" / "stock_returns.csv", parse_dates=["Date"]
    ).set_index("Date")
    market_panel = pd.read_csv(
        args.root / "data" / "market_returns.csv", parse_dates=["Date"]
    ).set_index("Date")["Market_Return"]
    sector_panel = pd.read_csv(
        args.root / "data" / "sector_returns.csv", parse_dates=["Date"]
    ).set_index("Date")
    metadata_by_asset = metadata.set_index("Symbol").to_dict(orient="index")
    if len(assets) != len(set(assets)):
        errors.append("universe contains duplicates")
    if set(assets) != set(caps):
        errors.append("market caps do not match universe")
    if set(assets) != set(metadata["Symbol"].astype(str)):
        errors.append("metadata does not match universe")
    expected_counts = {"validation": 5, "test": 20}
    actual_counts = periods["phase"].value_counts().to_dict()
    if actual_counts != expected_counts:
        errors.append(f"period counts {actual_counts} != {expected_counts}")
    for relative, expected_hash in manifest.get("files", {}).items():
        path = args.root / relative
        if not path.exists():
            errors.append(f"missing hashed file {relative}")
        elif sha256(path) != expected_hash:
            errors.append(f"hash mismatch {relative}")

    expected_days = int(config["lookback_trading_days"])
    expected_holding = int(config["holding_trading_days"])
    realized_dates_by_phase: dict[str, list[pd.Timestamp]] = {
        "validation": [], "test": [],
    }
    for period in periods.itertuples(index=False):
        period_root = args.root / "periods" / period.period_id
        try:
            formation = pd.read_csv(period_root / "formation_returns.csv", parse_dates=["Date"])
            realized = pd.read_csv(period_root / "realized_returns.csv", parse_dates=["Date"])
            context = load_json(period_root / "context.json")
        except Exception as error:
            errors.append(f"{period.period_id} unreadable: {error}")
            continue
        if len(formation) != expected_days or len(realized) != expected_holding:
            errors.append(
                f"{period.period_id} lengths formation={len(formation)} realized={len(realized)}"
            )
        if list(formation.columns[1:]) != assets or list(realized.columns[1:]) != assets:
            errors.append(f"{period.period_id} asset order mismatch")
        if formation["Date"].max() >= realized["Date"].min():
            errors.append(f"{period.period_id} formation overlaps realized data")
        if (
            str(formation["Date"].min().date()) != str(period.formation_start)
            or str(formation["Date"].max().date()) != str(period.formation_end)
            or str(realized["Date"].min().date()) != str(period.test_start)
            or str(realized["Date"].max().date()) != str(period.test_end)
        ):
            errors.append(f"{period.period_id} period boundary metadata mismatch")
        if str(formation["Date"].max().date()) != str(period.reference_date):
            errors.append(f"{period.period_id} reference date mismatch")
        realized_dates_by_phase[str(period.phase)].extend(realized["Date"].tolist())
        formation_dates = pd.DatetimeIndex(formation["Date"])
        realized_dates = pd.DatetimeIndex(realized["Date"])
        try:
            expected_formation = stock_panel.loc[formation_dates, assets].to_numpy(dtype=float)
            expected_realized = stock_panel.loc[realized_dates, assets].to_numpy(dtype=float)
            if not np.allclose(
                formation[assets].to_numpy(dtype=float), expected_formation,
                rtol=0.0, atol=1e-15,
            ):
                errors.append(f"{period.period_id} formation values differ from stock panel")
            if not np.allclose(
                realized[assets].to_numpy(dtype=float), expected_realized,
                rtol=0.0, atol=1e-15,
            ):
                errors.append(f"{period.period_id} realized values differ from stock panel")
        except KeyError as error:
            errors.append(f"{period.period_id} date missing from stock panel: {error}")
        if set(context) != set(assets):
            errors.append(f"{period.period_id} context universe mismatch")
        for asset in assets:
            item = context.get(asset, {})
            sector = metadata_by_asset.get(asset, {}).get("GICS Sector")
            if (
                len(item.get("sector_returns", [])) != expected_days
                or len(item.get("market_returns", [])) != expected_days
                or item.get("reference_date") != str(period.reference_date)
            ):
                errors.append(f"{period.period_id}/{asset} context mismatch")
                break
            expected_sector = (
                100.0 * sector_panel.loc[formation_dates, sector]
            ).to_numpy(dtype=float)
            expected_market = (
                100.0 * market_panel.loc[formation_dates]
            ).to_numpy(dtype=float)
            if not np.allclose(
                np.asarray(item["sector_returns"], dtype=float), expected_sector,
                rtol=0.0, atol=1e-12,
            ) or not np.allclose(
                np.asarray(item["market_returns"], dtype=float), expected_market,
                rtol=0.0, atol=1e-12,
            ):
                errors.append(f"{period.period_id}/{asset} context values differ from proxy panels")
                break

    expected_realized_counts = {"validation": 50, "test": 200}
    for phase, expected_count in expected_realized_counts.items():
        dates = realized_dates_by_phase[phase]
        if len(dates) != expected_count or len(set(dates)) != expected_count:
            errors.append(
                f"{phase} realized calendar has {len(dates)} rows/"
                f"{len(set(dates))} unique dates, expected {expected_count}"
            )

    if args.run_root:
        repeats = int(args.repeats or config["comparison_repeats"])
        expected_ensemble = ENSEMBLE_NAME if config["prompt_ensemble"] else "single_prompt"
        absolute_hashes: list[str] = []
        relative_hashes: list[str] = []
        for period in periods.itertuples(index=False):
            absolute_path = args.run_root / "responses_absolute" / f"{period.period_id}.json"
            relative_path = args.run_root / "responses_relative" / f"{period.period_id}.json"
            if not absolute_path.exists():
                errors.append(f"missing absolute response {period.period_id}")
            else:
                absolute = load_json(absolute_path)
                for asset in assets:
                    item = absolute.get(asset, {})
                    samples = item.get("expected_return", [])
                    hashes = item.get("system_prompt_sha256", [])
                    variant_ids = item.get("prompt_variant_ids", [])
                    if (
                        len(samples) != repeats
                        or not all(np.isfinite(float(sample)) for sample in samples)
                        or int(item.get("successful_repeats", -1)) != repeats
                        or int(item.get("attempted_calls", 0)) < repeats
                        or item.get("model") != config["model"]
                        or item.get("thinking") != config["thinking"]
                        or not close(item.get("temperature", -1), config["temperature"])
                        or item.get("prompt_mode") != config["absolute_prompt_mode"]
                        or item.get("prompt_ensemble") != expected_ensemble
                        or item.get("stored_return_units") != "decimal_daily_return"
                        or item.get("model_output_units") != "percentage_daily_return"
                        or (
                            config["prompt_ensemble"]
                            and (
                                len(hashes) != repeats
                                or len(set(hashes)) != repeats
                                or len(variant_ids) != repeats
                                or len(set(variant_ids)) != repeats
                            )
                        )
                    ):
                        errors.append(f"invalid absolute response {period.period_id}/{asset}")
                        break
                    absolute_hashes.extend(hashes)
            if not relative_path.exists():
                errors.append(f"missing relative response {period.period_id}")
            else:
                relative = load_json(relative_path)
                views = relative.get("views", [])
                if (
                    len(views) != int(config["max_pairs"])
                    or relative.get("model") != config["model"]
                    or relative.get("thinking") != config["thinking"]
                    or not close(relative.get("temperature", -1), config["temperature"])
                    or int(relative.get("repeats", -1)) != repeats
                    or relative.get("probability_estimator") != "mean"
                    or relative.get("prompt_mode") != config["relative_prompt_mode"]
                    or relative.get("prompt_ensemble") != expected_ensemble
                    or relative.get("probability_semantics") != (
                        "confidence-forced ranking score; not externally calibrated probability"
                    )
                    or any(int(item.get("successful_repeats", 0)) != repeats for item in views)
                    or (
                        config["prompt_ensemble"]
                        and any(len(item.get("system_prompt_sha256", [])) != repeats for item in views)
                    )
                ):
                    errors.append(f"invalid relative response {period.period_id}")
                for view_index, item in enumerate(views):
                    samples = [float(value) for value in item.get("probability_samples_a", [])]
                    reported = [float(value) for value in item.get("reported_probability_samples_a", [])]
                    probability_a = float(item.get("probability_a", float("nan")))
                    probability = float(item.get("probability", float("nan")))
                    hashes = item.get("system_prompt_sha256", [])
                    variant_ids = item.get("prompt_variant_ids", [])
                    invalid_view = (
                        len(samples) != repeats
                        or len(reported) != repeats
                        or not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in samples)
                        or samples != reported
                        or not close(probability_a, np.mean(samples) if samples else float("nan"))
                        or not close(probability, max(probability_a, 1.0 - probability_a))
                        or item.get("probability_estimator") != "mean"
                        or int(item.get("attempted_calls", 0)) < repeats
                        or (
                            str(config["relative_prompt_mode"]).startswith("decisive_")
                            and any(max(value, 1.0 - value) < 0.55 - 1e-12 for value in reported)
                        )
                        or (
                            config["prompt_ensemble"]
                            and (
                                len(hashes) != repeats
                                or len(set(hashes)) != repeats
                                or len(variant_ids) != repeats
                                or len(set(variant_ids)) != repeats
                            )
                        )
                    )
                    if invalid_view:
                        errors.append(
                            f"invalid relative aggregation {period.period_id}/view_{view_index + 1:02d}"
                        )
                        break
                relative_hashes.extend(
                    prompt_hash
                    for item in views
                    for prompt_hash in item.get("system_prompt_sha256", [])
                )
        if config["prompt_ensemble"]:
            if len(absolute_hashes) != len(set(absolute_hashes)):
                errors.append("absolute system prompt hashes are not globally unique")
            if len(relative_hashes) != len(set(relative_hashes)):
                errors.append("relative system prompt hashes are not globally unique")
        if args.require_results:
            results_root = args.run_root / "results"
            for name in (
                "daily_returns", "daily_returns_long", "period_metrics",
                "weights_long", "method_metrics",
            ):
                for suffix in (".csv", ".parquet"):
                    if not (results_root / f"{name}{suffix}").exists():
                        errors.append(f"missing result {name}{suffix}")
            summary_path = results_root / "summary.json"
            if not summary_path.exists():
                errors.append("missing result summary.json")
            required_paths = [
                results_root / "daily_returns.csv",
                results_root / "daily_returns_long.csv",
                results_root / "period_metrics.csv",
                results_root / "weights_long.csv",
                results_root / "method_metrics.csv",
                summary_path,
            ]
            if all(path.exists() for path in required_paths):
                methods = [
                    "MVO", "BL_No_Views", "Equal_Weight",
                    "Absolute_LLM_BLM", "RelView_BL",
                ]
                daily = pd.read_csv(results_root / "daily_returns.csv")
                daily_long = pd.read_csv(results_root / "daily_returns_long.csv")
                period_metrics = pd.read_csv(results_root / "period_metrics.csv")
                weights = pd.read_csv(results_root / "weights_long.csv")
                method_metrics = pd.read_csv(results_root / "method_metrics.csv")
                summary = load_json(summary_path)
                expected_daily_columns = {"Date", *[f"{method}_Return" for method in methods]}
                if (
                    len(daily) != 200
                    or daily["Date"].nunique() != 200
                    or set(daily.columns) != expected_daily_columns
                    or not np.isfinite(daily.drop(columns="Date").to_numpy(dtype=float)).all()
                ):
                    errors.append("daily_returns does not contain five methods over 200 unique days")
                if (
                    len(daily_long) != 200 * len(methods)
                    or set(daily_long.get("Method", [])) != set(methods)
                    or not np.isfinite(daily_long["Portfolio_Return"].to_numpy(dtype=float)).all()
                ):
                    errors.append("daily_returns_long is invalid")
                if (
                    len(period_metrics) != 20
                    or period_metrics["Period"].nunique() != 20
                    or set(period_metrics["Phase"]) != {"test"}
                    or not period_metrics["Accepted_Relative_Views"].between(
                        0, int(config["max_pairs"])
                    ).all()
                ):
                    errors.append("period_metrics is not a complete 20-period test")
                expected_weight_rows = 20 * len(methods) * len(assets)
                numeric_weights = weights.get("Weight", pd.Series(dtype=float)).to_numpy(dtype=float)
                sums = weights.groupby(["Period", "Method"])["Weight"].sum()
                if (
                    len(weights) != expected_weight_rows
                    or set(weights.get("Method", [])) != set(methods)
                    or set(weights.get("Asset", [])) != set(assets)
                    or not np.isfinite(numeric_weights).all()
                    or np.any(numeric_weights < -1e-8)
                    or np.any(numeric_weights > float(config["max_weight"]) + 1e-8)
                    or not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-7)
                ):
                    errors.append("weights_long violates row, universe, or portfolio constraints")
                if set(method_metrics.get("Method", [])) != set(methods):
                    errors.append("method_metrics does not contain all five methods")
                if (
                    set(summary.get("summary", {})) != set(methods)
                    or int(summary.get("run", {}).get("test_periods", -1)) != 20
                    or int(summary.get("run", {}).get("test_trading_days", -1)) != 200
                    or int(summary.get("run", {}).get("repeats", -1)) != repeats
                    or float(summary.get("config", {}).get("max_weight", -1)) != 1.0
                ):
                    errors.append("summary.json run metadata is invalid")

    print(json.dumps({
        "root": str(args.root),
        "universe_count": len(assets),
        "period_counts": actual_counts,
        "errors": errors,
        "error_count": len(errors),
    }, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
