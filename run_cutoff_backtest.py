"""Static 2025-12 cutoff experiment with reusable per-method artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_compare import absolute_weights
from collect_absolute_views import collect_absolute_views
from collect_relative_views import (
    OPENCODE_GO_BASE_URL,
    _metadata_lookup,
    collect_pairwise_views,
)
from collect_walkforward_views import valid_absolute, valid_relative
from env_utils import load_env_file
from portfolio_backtest import evaluate_realized_portfolio
from prepare_monthly_returns import download_chart_close
from relview_bl import (
    RelViewConfig,
    implied_equilibrium_returns,
    optimize_portfolio,
    run_relview_bl,
    select_candidate_pairs,
)
from run_multidataset_experiment import load_manifest, write_dataset_inputs


DEFAULT_CONFIG = Path("experiments/cutoff_2025_12/config.json")
METHODS = ("MVO", "BL_No_Views", "Absolute_LLM_BLM", "RelView_BL", "Equal_Weight")


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_manifest", "formation_start", "cutoff_date", "test_start", "test_end",
        "model", "repeats", "minimum_successful_calls", "max_pairs", "thinking",
        "calibration", "abstention_threshold", "risk_aversion", "market_risk_aversion",
        "max_weight", "transaction_cost_bps",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"cutoff config missing fields: {missing}")
    if pd.Timestamp(value["cutoff_date"]) >= pd.Timestamp(value["test_start"]):
        raise ValueError("cutoff_date must precede test_start")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_table(frame: pd.DataFrame, base: Path) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = base.with_suffix(".csv")
    parquet_path = base.with_suffix(".parquet")
    frame.to_csv(csv_path, index=False)
    frame.to_parquet(parquet_path, index=False)
    return {"csv": str(csv_path), "parquet": str(parquet_path)}


def prepare_dataset_data(
    root: Path,
    assets: list[str],
    formation_start: pd.Timestamp,
    cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    overwrite: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = root / "data"
    formation_path = data_dir / "formation_returns.csv"
    realized_path = data_dir / "realized_returns.csv"
    if formation_path.exists() and realized_path.exists() and not overwrite:
        formation = pd.read_csv(formation_path, parse_dates=["Date"]).set_index("Date")
        realized = pd.read_csv(realized_path, parse_dates=["Date"]).set_index("Date")
        if list(formation.columns) == assets and list(realized.columns) == assets:
            return formation, realized

    close = download_chart_close(
        assets,
        formation_start - pd.Timedelta(days=7),
        test_end + pd.Timedelta(days=2),
    ).reindex(columns=assets)
    returns = close.pct_change(fill_method=None)
    formation = returns[(returns.index >= formation_start) & (returns.index <= cutoff)]
    realized = returns[(returns.index >= test_start) & (returns.index <= test_end)]
    formation = formation.dropna(axis=0, how="any")
    realized = realized.dropna(axis=0, how="any")
    if len(formation) < 100 or len(realized) < 10:
        raise RuntimeError(
            f"insufficient complete data: formation={len(formation)}, realized={len(realized)}"
        )
    data_dir.mkdir(parents=True, exist_ok=True)
    formation.index.name = "Date"
    realized.index.name = "Date"
    formation.reset_index().to_csv(formation_path, index=False)
    realized.reset_index().to_csv(realized_path, index=False)
    formation.reset_index().to_parquet(data_dir / "formation_returns.parquet", index=False)
    realized.reset_index().to_parquet(data_dir / "realized_returns.parquet", index=False)
    return formation, realized


def collect_views(
    dataset_root: Path,
    formation: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    config: dict[str, Any],
    api_key: str,
    workers: int,
    retry_calls: int,
    force: bool,
    horizon_days: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assets = formation.columns.astype(str).tolist()
    metadata = _metadata_lookup(metadata_frame)
    caps = {asset: 1.0 for asset in assets}
    absolute_path = dataset_root / "responses_absolute" / f"{config['model']}_cutoff_2025-12.json"
    relative_path = dataset_root / "responses_relative" / f"{config['model']}_cutoff_2025-12.json"
    minimum = int(config["minimum_successful_calls"])

    if force or not valid_absolute(absolute_path, assets, minimum):
        absolute = collect_absolute_views(
            formation,
            str(config["model"]),
            os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL),
            api_key,
            metadata,
            {},
            int(config["repeats"]),
            horizon_days,
            float(config["temperature"]),
            str(config["thinking"]),
            workers,
            retry_calls,
        )
        incomplete = [
            asset for asset in assets
            if len(absolute[asset].get("expected_return", [])) < minimum
        ]
        if incomplete:
            raise RuntimeError(f"absolute cutoff collection incomplete: {incomplete}")
        atomic_json(absolute_path, absolute)
    else:
        absolute = json.loads(absolute_path.read_text(encoding="utf-8"))

    if force or not valid_relative(relative_path, int(config["max_pairs"]), minimum):
        pairs = select_candidate_pairs(
            formation,
            metadata_frame,
            caps,
            max_pairs=int(config["max_pairs"]),
        )
        views = collect_pairwise_views(
            formation,
            pairs,
            str(config["model"]),
            os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL),
            api_key,
            metadata,
            {},
            int(config["repeats"]),
            horizon_days,
            float(config["temperature"]),
            str(config["probability_estimator"]),
            str(config["thinking"]),
            workers,
            retry_calls,
        )
        incomplete = [
            f"{view.get('asset_a')}/{view.get('asset_b')}"
            for view in views
            if view.get("status") != "ok"
            or int(view.get("successful_repeats", 0)) < minimum
        ]
        if incomplete:
            raise RuntimeError(f"relative cutoff collection incomplete: {incomplete}")
        atomic_json(relative_path, {
            "model": config["model"],
            "cutoff_date": config["cutoff_date"],
            "horizon_days": horizon_days,
            "repeats": config["repeats"],
            "probability_estimator": config["probability_estimator"],
            "thinking": config["thinking"],
            "pairs_requested": len(pairs),
            "views": views,
        })
    else:
        views = json.loads(relative_path.read_text(encoding="utf-8"))["views"]
    return absolute, views


def evaluate_dataset(
    dataset_id: str,
    dataset_root: Path,
    formation: pd.DataFrame,
    realized: pd.DataFrame,
    absolute: dict[str, Any],
    views: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assets = formation.columns.astype(str).tolist()
    covariance = formation.cov().to_numpy(dtype=float)
    equal = np.full(len(assets), 1.0 / len(assets))
    prior = implied_equilibrium_returns(covariance, equal, float(config["market_risk_aversion"]))

    mvo, _ = optimize_portfolio(
        formation.mean().to_numpy(dtype=float), covariance,
        risk_aversion=float(config["risk_aversion"]),
        max_weight=float(config["max_weight"]),
    )
    bl, _ = optimize_portfolio(
        prior, covariance,
        risk_aversion=float(config["risk_aversion"]),
        max_weight=float(config["max_weight"]),
    )
    absolute_method, _ = absolute_weights(
        absolute, assets, prior, covariance,
        float(config["tau"]), 1e-8,
        float(config["risk_aversion"]), float(config["max_weight"]),
        None, 0.0,
    )
    relative_result = run_relview_bl(
        formation,
        prior,
        views,
        [],
        RelViewConfig(
            calibration=str(config["calibration"]),
            abstention_threshold=float(config["abstention_threshold"]),
            tau=float(config["tau"]),
            risk_aversion=float(config["risk_aversion"]),
            max_weight=float(config["max_weight"]),
        ),
    )
    relative = relative_result.weights.reindex(assets).to_numpy(dtype=float)
    weights = {
        "MVO": mvo,
        "BL_No_Views": bl,
        "Absolute_LLM_BLM": absolute_method,
        "RelView_BL": relative,
        "Equal_Weight": equal,
    }

    daily_wide = pd.DataFrame({"Date": realized.index.astype(str)})
    daily_long_parts: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    results_dir = dataset_root / "results"
    by_method = results_dir / "by_method"

    for method in METHODS:
        daily, metrics = evaluate_realized_portfolio(
            realized,
            pd.Series(weights[method], index=assets),
            turnover=1.0,
            transaction_cost_bps=float(config["transaction_cost_bps"]),
        )
        daily_wide[f"{method}_Return"] = daily["Portfolio_Return"]
        daily_wide[f"{method}_Cumulative_Return"] = daily["Cumulative_Return"]
        method_daily = daily.assign(Dataset=dataset_id, Method=method)[
            ["Dataset", "Method", "Date", "Portfolio_Return", "Cumulative_Return"]
        ]
        daily_long_parts.append(method_daily)
        method_weights = pd.DataFrame({
            "Dataset": dataset_id,
            "Method": method,
            "Asset": assets,
            "Weight": weights[method],
        })
        weight_rows.extend(method_weights.to_dict(orient="records"))
        metric_rows.append({"Dataset": dataset_id, "Method": method, **metrics})
        save_table(method_daily, by_method / f"{method}_daily")
        save_table(method_weights, by_method / f"{method}_weights")
        atomic_json(by_method / f"{method}_metrics.json", metrics)

    daily_long = pd.concat(daily_long_parts, ignore_index=True)
    weights_long = pd.DataFrame(weight_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    save_table(daily_wide, results_dir / "daily_wide")
    save_table(daily_long, results_dir / "daily_long")
    save_table(weights_long, results_dir / "weights_long")
    save_table(metrics_frame, results_dir / "method_metrics")
    atomic_json(results_dir / "relview_diagnostics.json", {
        "calibration_method": relative_result.calibrator.fitted_method,
        "accepted_views": relative_result.matrices.accepted_views,
        "rejected_views": relative_result.matrices.rejected_views,
        "raw_cycle_count": relative_result.matrices.raw_cycle_count,
        "consistency_rmse": relative_result.matrices.consistency_rmse,
        "latent_scores": relative_result.matrices.latent_scores.to_dict(),
    })
    return daily_long, weights_long, metrics_frame


def aggregate_results(
    root: Path,
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    summary_dir = root / "summary"
    save_table(daily, summary_dir / "daily_returns_long")
    save_table(weights, summary_dir / "weights_long")
    save_table(metrics, summary_dir / "method_metrics")
    wide = daily.pivot_table(
        index="Date", columns=["Dataset", "Method"], values="Portfolio_Return"
    ).sort_index()
    wide.columns = [f"{dataset}__{method}" for dataset, method in wide.columns]
    save_table(wide.reset_index(), summary_dir / "daily_returns_wide")

    meta = daily.groupby(["Date", "Method"], as_index=False)["Portfolio_Return"].mean()
    aggregate_daily_parts: list[pd.DataFrame] = []
    aggregate_metrics: list[dict[str, Any]] = []
    for method in METHODS:
        method_returns = meta[meta["Method"] == method].set_index("Date")[["Portfolio_Return"]]
        evaluated, values = evaluate_realized_portfolio(method_returns, {"Portfolio_Return": 1.0})
        # The per-dataset series already include the same one-off entry cost.
        # Preserve that fact in the aggregate metric metadata without charging it twice.
        values["turnover"] = 1.0
        values["transaction_cost_bps"] = float(config["transaction_cost_bps"])
        evaluated.insert(0, "Method", method)
        aggregate_daily_parts.append(evaluated)
        aggregate_metrics.append({"Method": method, **values})
    aggregate_daily = pd.concat(aggregate_daily_parts, ignore_index=True)
    aggregate_metrics_frame = pd.DataFrame(aggregate_metrics)
    save_table(aggregate_daily, summary_dir / "equal_dataset_portfolio_daily")
    save_table(aggregate_metrics_frame, summary_dir / "equal_dataset_portfolio_metrics")

    catalog = {
        "config": config,
        "methods": list(METHODS),
        "datasets": sorted(daily["Dataset"].astype(str).unique().tolist()),
        "dataset_count": int(daily["Dataset"].nunique()),
        "test_days": int(daily["Date"].nunique()),
        "actual_test_start": str(daily["Date"].min()),
        "actual_test_end": str(daily["Date"].max()),
        "formats": ["csv", "parquet", "json"],
        "units": {
            "Portfolio_Return": "decimal daily return, net of the entry cost on the first test day",
            "Cumulative_Return": "compounded decimal return",
            "Weight": "portfolio fraction",
        },
        "row_counts": {
            "daily_long": int(len(daily)),
            "weights_long": int(len(weights)),
            "method_metrics": int(len(metrics)),
            "aggregate_daily": int(len(aggregate_daily)),
            "aggregate_metrics": int(len(aggregate_metrics_frame)),
        },
        "tables": {
            "daily_long": "summary/daily_returns_long.{csv,parquet}",
            "daily_wide": "summary/daily_returns_wide.{csv,parquet}",
            "weights_long": "summary/weights_long.{csv,parquet}",
            "method_metrics": "summary/method_metrics.{csv,parquet}",
            "aggregate_daily": "summary/equal_dataset_portfolio_daily.{csv,parquet}",
            "aggregate_metrics": "summary/equal_dataset_portfolio_metrics.{csv,parquet}",
            "per_dataset": "{dataset}/results/ and {dataset}/results/by_method/",
        },
    }
    atomic_json(summary_dir / "data_catalog.json", catalog)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=Path("experiments/cutoff_2025_12"))
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Optional separate output root for an ablation; inputs and saved views still come from --root",
    )
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--retry-calls", type=int, default=60)
    parser.add_argument("--force-views", action="store_true")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        help="Override the RelView threshold from config (use 0.5 to accept every valid pair)",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    config = load_config(args.config)
    if args.abstention_threshold is not None:
        if not 0.5 <= args.abstention_threshold <= 1.0:
            raise ValueError("--abstention-threshold must be in [0.5, 1]")
        config = {**config, "abstention_threshold": float(args.abstention_threshold)}
    results_root = args.results_root or args.root
    manifest = load_manifest(Path(config["dataset_manifest"]))
    selected = set(args.datasets or [])
    datasets = [item for item in manifest["datasets"] if not selected or item["id"] in selected]
    if selected - {item["id"] for item in datasets}:
        raise ValueError(f"unknown dataset IDs: {sorted(selected)}")
    api_key = os.getenv("OPENCODE_GO_API_KEY")
    if not args.skip_collect and not api_key:
        raise ValueError("OPENCODE_GO_API_KEY is not set")

    formation_start = pd.Timestamp(config["formation_start"])
    cutoff = pd.Timestamp(config["cutoff_date"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])
    all_daily: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        print(f"\n=== {dataset_id}: cutoff {cutoff.date()} ===", flush=True)
        dataset_root = write_dataset_inputs(args.root, dataset)
        result_dataset_root = results_root / dataset_id
        assets = [str(item["ticker"]) for item in dataset["assets"]]
        formation, realized = prepare_dataset_data(
            dataset_root, assets, formation_start, cutoff, test_start, test_end,
            args.overwrite_data,
        )
        print(f"formation rows={len(formation)}, realized rows={len(realized)}")
        if args.prepare_only:
            continue
        metadata = pd.read_csv(dataset_root / "metadata.csv")
        horizon_days = len(realized)
        if args.skip_collect:
            absolute = json.loads(
                (dataset_root / "responses_absolute" / f"{config['model']}_cutoff_2025-12.json")
                .read_text(encoding="utf-8")
            )
            views = json.loads(
                (dataset_root / "responses_relative" / f"{config['model']}_cutoff_2025-12.json")
                .read_text(encoding="utf-8")
            )["views"]
        else:
            absolute, views = collect_views(
                dataset_root, formation, metadata, config, str(api_key),
                args.workers, args.retry_calls, args.force_views, horizon_days,
            )
        daily, weights, metrics = evaluate_dataset(
            dataset_id, result_dataset_root, formation, realized, absolute, views, config
        )
        all_daily.append(daily)
        all_weights.append(weights)
        all_metrics.append(metrics)

    if args.prepare_only:
        print(f"Prepared cutoff data under {args.root}")
        return
    aggregate_results(
        results_root,
        pd.concat(all_daily, ignore_index=True),
        pd.concat(all_weights, ignore_index=True),
        pd.concat(all_metrics, ignore_index=True),
        config,
    )
    print(f"Saved reusable CSV/Parquet/JSON artifacts under {results_root}")


if __name__ == "__main__":
    main()
