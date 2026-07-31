"""Collect and backtest the paper-aligned 10-trading-day walk-forward study."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_compare import absolute_weights
from collect_absolute_views import collect_absolute_views
from collect_relative_views import (
    OPENCODE_GO_BASE_URL,
    ProviderUsageLimitError,
    _metadata_lookup,
    collect_pairwise_views,
)
from env_utils import load_env_file
from portfolio_backtest import evaluate_realized_portfolio
from prompt_ensemble import ENSEMBLE_NAME
from relview_bl import (
    RelViewConfig,
    implied_equilibrium_returns,
    optimize_portfolio,
    run_relview_bl,
    select_candidate_pairs,
)


DEFAULT_CONFIG = Path("experiments/paper_reproduction/config.json")
DEFAULT_DATA_ROOT = Path("experiments/paper_reproduction/paper_sp500_top50")
DEFAULT_RUN_ROOT = Path("experiments/paper_reproduction/paper_sp500_top50/deepseek_comparison")
BASELINE_METHODS = ("MVO", "BL_No_Views", "Equal_Weight")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_returns(path: Path, assets: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    missing = [asset for asset in assets if asset not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing assets: {missing}")
    frame = frame[assets].astype(float)
    if frame.isna().any().any():
        raise ValueError(f"{path} contains missing returns")
    return frame


def absolute_cache_matches(
    path: Path,
    assets: list[str],
    config: dict[str, Any],
    repeats: int,
    minimum: int,
) -> bool:
    if not path.exists():
        return False
    try:
        value = load_json(path)
        expected_ensemble = ENSEMBLE_NAME if config["prompt_ensemble"] else "single_prompt"
        return all(
            asset in value
            and len(value[asset].get("expected_return", [])) >= minimum
            and int(value[asset].get("successful_repeats", -1)) == repeats
            and value[asset].get("model") == config["model"]
            and value[asset].get("thinking") == config["thinking"]
            and float(value[asset].get("temperature", -1)) == float(config["temperature"])
            and value[asset].get("prompt_ensemble") == expected_ensemble
            and value[asset].get("prompt_mode") == config["absolute_prompt_mode"]
            for asset in assets
        )
    except Exception:
        return False


def relative_cache_matches(
    path: Path,
    assets: list[str],
    config: dict[str, Any],
    repeats: int,
    minimum: int,
) -> bool:
    if not path.exists():
        return False
    try:
        value = load_json(path)
        expected_ensemble = ENSEMBLE_NAME if config["prompt_ensemble"] else "single_prompt"
        valid = [
            item for item in value.get("views", [])
            if item.get("status") == "ok"
            and int(item.get("successful_repeats", -1)) == repeats
            and int(item.get("successful_repeats", 0)) >= minimum
            and item.get("prompt_mode") == config["relative_prompt_mode"]
        ]
        allowed_assets = set(assets)
        return (
            value.get("model") == config["model"]
            and int(value.get("repeats", -1)) == repeats
            and value.get("thinking") == config["thinking"]
            and float(value.get("temperature", -1)) == float(config["temperature"])
            and value.get("prompt_ensemble") == expected_ensemble
            and value.get("prompt_mode") == config["relative_prompt_mode"]
            and len(valid) == int(config["max_pairs"])
            and all(
                item.get("asset_a") in allowed_assets
                and item.get("asset_b") in allowed_assets
                for item in valid
            )
        )
    except Exception:
        return False


def collect_periods(
    data_root: Path,
    run_root: Path,
    periods: pd.DataFrame,
    assets: list[str],
    metadata_frame: pd.DataFrame,
    caps: dict[str, float],
    config: dict[str, Any],
    methods: set[str],
    repeats: int,
    workers: int,
    retry_calls: int,
    force: bool,
    dry_run: bool,
) -> None:
    minimum = min(repeats, int(config["minimum_successful_calls"]))
    absolute_dir = run_root / "responses_absolute"
    relative_dir = run_root / "responses_relative"
    pending: list[tuple[str, str]] = []
    for period in periods.itertuples(index=False):
        if "absolute" in methods and (
            force or not absolute_cache_matches(
                absolute_dir / f"{period.period_id}.json", assets, config, repeats, minimum
            )
        ):
            pending.append((period.period_id, "absolute"))
        if "relative" in methods and (
            force or not relative_cache_matches(
                relative_dir / f"{period.period_id}.json", assets, config, repeats, minimum
            )
        ):
            pending.append((period.period_id, "relative"))
    calls = sum(
        (len(assets) if method == "absolute" else int(config["max_pairs"])) * repeats
        for _, method in pending
    )
    print(f"Pending response files: {len(pending)}; estimated base API calls: {calls}", flush=True)
    if dry_run or not pending:
        return

    load_env_file()
    api_key = os.getenv("OPENCODE_GO_API_KEY")
    if not api_key:
        raise ValueError("OPENCODE_GO_API_KEY is not set")
    metadata = _metadata_lookup(metadata_frame)
    base_url = os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL)
    for task_number, (period_id, method) in enumerate(pending, start=1):
        period_root = data_root / "periods" / period_id
        formation = load_returns(period_root / "formation_returns.csv", assets)
        context = load_json(period_root / "context.json")
        print(f"[{task_number}/{len(pending)}] {period_id} {method}", flush=True)
        if method == "absolute":
            value = collect_absolute_views(
                formation, str(config["model"]), base_url, api_key, metadata, context,
                repeats, int(config["holding_trading_days"]), float(config["temperature"]),
                str(config["thinking"]), workers, retry_calls,
                bool(config["prompt_ensemble"]), str(config["absolute_prompt_mode"]),
            )
            incomplete = [
                asset for asset in assets
                if len(value[asset].get("expected_return", [])) < minimum
            ]
            if incomplete:
                raise RuntimeError(f"{period_id} absolute incomplete: {incomplete}")
            atomic_json(absolute_dir / f"{period_id}.json", value)
        else:
            pairs = select_candidate_pairs(
                formation, metadata_frame, caps, max_pairs=int(config["max_pairs"])
            )
            views = collect_pairwise_views(
                formation, pairs, str(config["model"]), base_url, api_key, metadata, context,
                repeats, int(config["holding_trading_days"]), float(config["temperature"]),
                str(config["probability_estimator"]), str(config["thinking"]), workers,
                retry_calls, bool(config["prompt_ensemble"]),
                str(config["relative_prompt_mode"]),
            )
            incomplete = [
                f"{item.get('asset_a')}/{item.get('asset_b')}"
                for item in views
                if item.get("status") != "ok"
                or int(item.get("successful_repeats", 0)) < minimum
            ]
            if incomplete:
                raise RuntimeError(f"{period_id} relative incomplete: {incomplete}")
            atomic_json(relative_dir / f"{period_id}.json", {
                "model": config["model"],
                "period_id": period_id,
                "reference_date": str(context[assets[0]]["reference_date"]),
                "horizon_days": config["holding_trading_days"],
                "repeats": repeats,
                "probability_estimator": config["probability_estimator"],
                "thinking": config["thinking"],
                "temperature": config["temperature"],
                "prompt_ensemble": ENSEMBLE_NAME if config["prompt_ensemble"] else "single_prompt",
                "prompt_mode": config["relative_prompt_mode"],
                "probability_semantics": (
                    "confidence-forced ranking score; not externally calibrated probability"
                    if str(config["relative_prompt_mode"]).startswith("decisive_")
                    else "model-reported probability"
                ),
                "pairs_requested": len(pairs),
                "views": views,
            })


def tau_candidates_from_validation(
    data_root: Path,
    run_root: Path,
    validation: pd.DataFrame,
    assets: list[str],
) -> tuple[float, list[float]]:
    ratios: list[float] = []
    for period in validation.itertuples(index=False):
        formation = load_returns(
            data_root / "periods" / period.period_id / "formation_returns.csv", assets
        )
        response = load_json(run_root / "responses_absolute" / f"{period.period_id}.json")
        variances = np.asarray([
            np.var(np.asarray(response[asset]["expected_return"], dtype=float), ddof=1)
            for asset in assets
        ])
        omega_matrix_mean = float(np.sum(variances) / (len(assets) ** 2))
        covariance_mean = float(np.mean(formation.cov().to_numpy(dtype=float)))
        if omega_matrix_mean > 0 and covariance_mean > 0:
            ratios.append(omega_matrix_mean / covariance_mean)
    if not ratios:
        raise RuntimeError("could not derive tau_init from validation views")
    tau_init = float(np.mean(ratios))
    return tau_init, [factor * tau_init for factor in (0.5, 0.75, 1.0, 1.25, 1.5)]


def turnover(weights: np.ndarray, previous: np.ndarray | None) -> float:
    return 1.0 if previous is None else float(np.sum(np.abs(weights - previous)))


def backtest_periods(
    data_root: Path,
    run_root: Path,
    periods: pd.DataFrame,
    assets: list[str],
    caps: dict[str, float],
    config: dict[str, Any],
    tau: float,
    include_absolute: bool,
    include_relative: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    cap_values = np.asarray([float(caps[asset]) for asset in assets], dtype=float)
    cap_weights = cap_values / cap_values.sum()
    previous: dict[str, np.ndarray | None] = {}
    daily_frames: list[pd.DataFrame] = []
    period_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    methods = list(BASELINE_METHODS)
    if include_absolute:
        methods.append("Absolute_LLM_BLM")
    if include_relative:
        methods.append("RelView_BL")
    for method in methods:
        previous[method] = None

    for period in periods.itertuples(index=False):
        period_root = data_root / "periods" / period.period_id
        formation = load_returns(period_root / "formation_returns.csv", assets)
        realized = load_returns(period_root / "realized_returns.csv", assets)
        covariance = formation.cov().to_numpy(dtype=float)
        prior = implied_equilibrium_returns(
            covariance, cap_weights, float(config["market_risk_aversion"])
        )
        convention = str(config["objective_convention"])
        weights_by_method: dict[str, np.ndarray] = {}
        weights_by_method["MVO"], _ = optimize_portfolio(
            formation.mean().to_numpy(dtype=float), covariance,
            risk_aversion=float(config["risk_aversion"]),
            previous_weights=previous["MVO"], max_weight=float(config["max_weight"]),
            objective_convention=convention,
        )
        weights_by_method["BL_No_Views"], _ = optimize_portfolio(
            prior, covariance, risk_aversion=float(config["risk_aversion"]),
            previous_weights=previous["BL_No_Views"], max_weight=float(config["max_weight"]),
            objective_convention=convention,
        )
        weights_by_method["Equal_Weight"] = np.full(len(assets), 1.0 / len(assets))
        accepted_relative = None
        if include_absolute:
            response = load_json(run_root / "responses_absolute" / f"{period.period_id}.json")
            weights_by_method["Absolute_LLM_BLM"], _ = absolute_weights(
                response, assets, prior, covariance, tau, 1e-12,
                float(config["risk_aversion"]), float(config["max_weight"]),
                previous["Absolute_LLM_BLM"], 0.0, convention,
            )
        if include_relative:
            views = load_json(
                run_root / "responses_relative" / f"{period.period_id}.json"
            )["views"]
            relative_result = run_relview_bl(
                formation, prior, views, [],
                RelViewConfig(
                    calibration=str(config["calibration"]),
                    abstention_threshold=float(config["abstention_threshold"]),
                    tau=tau,
                    risk_aversion=float(config["risk_aversion"]),
                    max_weight=float(config["max_weight"]),
                    objective_convention=convention,
                ),
                previous["RelView_BL"],
            )
            weights_by_method["RelView_BL"] = relative_result.weights.reindex(assets).to_numpy()
            accepted_relative = len(relative_result.matrices.accepted_views)

        period_daily = pd.DataFrame({"Date": realized.index.astype(str)})
        row: dict[str, Any] = {
            "Period": period.period_id,
            "Phase": period.phase,
            "Test_Start": str(realized.index[0].date()),
            "Test_End": str(realized.index[-1].date()),
            "Tau": tau,
            "Accepted_Relative_Views": accepted_relative,
        }
        for method in methods:
            weights = weights_by_method[method]
            method_turnover = turnover(weights, previous[method])
            daily, metrics = evaluate_realized_portfolio(
                realized, pd.Series(weights, index=assets), turnover=method_turnover,
                transaction_cost_bps=float(config["transaction_cost_bps"]),
            )
            period_daily[f"{method}_Return"] = daily["Portfolio_Return"].to_numpy()
            row.update({f"{method}_{key}": value for key, value in metrics.items()})
            weight_rows.extend({
                "Period": period.period_id,
                "Phase": period.phase,
                "Method": method,
                "Asset": asset,
                "Weight": float(weight),
            } for asset, weight in zip(assets, weights))
            previous[method] = weights
        daily_frames.append(period_daily)
        period_rows.append(row)

    daily_all = pd.concat(daily_frames, ignore_index=True)
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        returns = pd.DataFrame(
            {method: daily_all[f"{method}_Return"].to_numpy()},
            index=daily_all["Date"],
        )
        _, metrics = evaluate_realized_portfolio(returns, {method: 1.0})
        turnovers = [float(row[f"{method}_turnover"]) for row in period_rows]
        metrics["transaction_cost_bps"] = float(config["transaction_cost_bps"])
        metrics["total_turnover"] = float(np.sum(turnovers))
        metrics["average_period_turnover"] = float(np.mean(turnovers))
        summary[method] = metrics
    return daily_all, pd.DataFrame(period_rows), pd.DataFrame(weight_rows), summary


def save_frame(frame: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    frame.to_parquet(base.with_suffix(".parquet"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--methods", nargs="+", choices=["absolute", "relative"], default=["absolute", "relative"])
    parser.add_argument("--phases", nargs="+", choices=["validation", "test"], default=["validation", "test"])
    parser.add_argument("--periods", nargs="*", help="Optional exact period IDs, for example validation_01")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--paper-exact", action="store_true", help="Use N=100; disables the diversified prompt ensemble")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--retry-calls", type=int, default=60)
    parser.add_argument(
        "--wait-on-rate-limit",
        action="store_true",
        help="Wait for the provider reset time and resume the first incomplete period",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.paper_exact:
        config["prompt_ensemble"] = False
    repeats = int(
        args.repeats
        or config["paper_exact_repeats" if args.paper_exact else "comparison_repeats"]
    )
    if repeats < 2:
        raise ValueError("repeats must be at least 2")
    assets = [str(item) for item in load_json(args.data_root / "universe.json")]
    caps = {str(key): float(value) for key, value in load_json(args.data_root / "market_caps.json").items()}
    metadata_frame = pd.read_csv(args.data_root / "metadata.csv")
    periods = pd.read_csv(args.data_root / "periods.csv")
    selected = periods[periods["phase"].isin(args.phases)].copy()
    if args.periods:
        requested = set(args.periods)
        selected = selected[selected["period_id"].isin(requested)].copy()
        missing = requested - set(selected["period_id"])
        if missing:
            raise ValueError(f"unknown or phase-excluded period IDs: {sorted(missing)}")
    methods = set(args.methods)
    if not args.skip_collect:
        while True:
            try:
                collect_periods(
                    args.data_root, args.run_root, selected, assets, metadata_frame, caps,
                    config, methods, repeats, args.workers, args.retry_calls, args.force,
                    args.dry_run,
                )
                break
            except ProviderUsageLimitError as error:
                if not args.wait_on_rate_limit:
                    raise
                print(
                    "Provider usage limit reached; resuming in "
                    f"{error.retry_after_seconds} seconds: {error}",
                    flush=True,
                )
                time.sleep(error.retry_after_seconds)
    if args.dry_run or args.skip_backtest:
        return

    validation = periods[periods["phase"] == "validation"]
    test = periods[periods["phase"] == "test"]
    include_absolute = "absolute" in methods
    include_relative = "relative" in methods
    tau_init = None
    tau_grid: list[float] = []
    validation_scores: list[dict[str, float]] = []
    if include_absolute:
        tau_init, tau_grid = tau_candidates_from_validation(
            args.data_root, args.run_root, validation, assets
        )
        for tau in tau_grid:
            _, _, _, summary = backtest_periods(
                args.data_root, args.run_root, validation, assets, caps, config, tau,
                include_absolute=True, include_relative=False,
            )
            validation_scores.append({
                "tau": tau,
                "sharpe": float(summary["Absolute_LLM_BLM"]["sharpe"]),
            })
        selected_tau = max(validation_scores, key=lambda item: item["sharpe"])["tau"]
    else:
        selected_tau = float(config.get("tau", 0.025))

    daily, period_metrics, weights, summary = backtest_periods(
        args.data_root, args.run_root, test, assets, caps, config, selected_tau,
        include_absolute=include_absolute, include_relative=include_relative,
    )
    results = args.run_root / "results"
    save_frame(daily, results / "daily_returns")
    daily_long = daily.melt(
        id_vars=["Date"], var_name="Method", value_name="Portfolio_Return"
    )
    daily_long["Method"] = daily_long["Method"].str.replace(r"_Return$", "", regex=True)
    save_frame(daily_long, results / "daily_returns_long")
    save_frame(period_metrics, results / "period_metrics")
    save_frame(weights, results / "weights_long")
    method_metrics = pd.DataFrame([
        {"Method": method, **metrics} for method, metrics in summary.items()
    ])
    save_frame(method_metrics, results / "method_metrics")
    atomic_json(results / "summary.json", {
        "config": config,
        "run": {
            "repeats": repeats,
            "paper_exact_call_count": bool(args.paper_exact),
            "methods": sorted(methods),
            "test_periods": len(test),
            "test_trading_days": len(daily),
            "selected_tau": selected_tau,
            "tau_init": tau_init,
            "tau_grid": tau_grid,
            "validation_scores": validation_scores,
            "relative_tau_note": "RelView uses the absolute-view validation-selected tau for comparability",
        },
        "summary": summary,
        "artifacts": {
            "daily_wide": "results/daily_returns.{csv,parquet}",
            "daily_long": "results/daily_returns_long.{csv,parquet}",
            "period_metrics": "results/period_metrics.{csv,parquet}",
            "weights_long": "results/weights_long.{csv,parquet}",
            "method_metrics": "results/method_metrics.{csv,parquet}",
        },
    })
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved reusable results to {results}", flush=True)


if __name__ == "__main__":
    main()
