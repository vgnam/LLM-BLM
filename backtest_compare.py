"""Leak-free walk-forward comparison of Absolute LLM-BLM and RelView-BL."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from collect_walkforward_views import monthly_periods
from evaluate_relview import load_json, load_records, load_returns, load_universe
from portfolio_backtest import evaluate_realized_portfolio
from relview_bl import (
    RelViewConfig,
    black_litterman_posterior,
    calibration_observations_from_realized_returns,
    implied_equilibrium_returns,
    optimize_portfolio,
    run_relview_bl,
)


def next_month_path(returns_dir: Path, formation_month: str) -> Path:
    next_start = pd.Timestamp(f"{formation_month}-01") + pd.DateOffset(months=1)
    end_day = calendar.monthrange(next_start.year, next_start.month)[1]
    return returns_dir / f"returns_{next_start:%Y-%m-%d}_{next_start:%Y-%m}-{end_day:02d}.csv"


def turnover(current: np.ndarray, previous: np.ndarray | None) -> float:
    return float(np.sum(np.abs(current - previous))) if previous is not None else 1.0


def absolute_weights(
    response: dict[str, Any],
    assets: list[str],
    prior: np.ndarray,
    covariance: np.ndarray,
    tau: float,
    omega_epsilon: float,
    risk_aversion: float,
    max_weight: float,
    previous: np.ndarray | None,
    turnover_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    missing = [
        asset for asset in assets
        if asset not in response or len(response[asset].get("expected_return", [])) < 2
    ]
    if missing:
        raise ValueError(f"absolute views need at least two calls for: {missing}")
    samples = [np.asarray(response[asset]["expected_return"], dtype=float) for asset in assets]
    q = np.asarray([np.mean(item) for item in samples], dtype=float)
    omega = np.diag([
        max(float(np.var(item, ddof=1)), omega_epsilon) for item in samples
    ])
    posterior = black_litterman_posterior(
        prior, covariance, np.eye(len(assets)), q, omega, tau
    )
    weights, _ = optimize_portfolio(
        posterior,
        covariance,
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        previous_weights=previous,
        max_weight=max_weight,
    )
    return weights, posterior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Absolute LLM-BLM and RelView-BL walk-forward")
    parser.add_argument("--start-month", default="2024-06")
    parser.add_argument("--end-month", default="2025-05")
    parser.add_argument("--returns-dir", type=Path, default=Path("yfinance"))
    parser.add_argument("--absolute-dir", type=Path, default=Path("responses"))
    parser.add_argument("--relative-dir", type=Path, default=Path("responses_relative"))
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--universe", type=Path, default=Path("universe.json"))
    parser.add_argument("--market-caps", type=Path, default=Path("market_caps.json"))
    parser.add_argument("--output-prefix", type=Path, default=Path("results/walkforward_comparison"))
    parser.add_argument("--evaluation-days", type=int, default=10)
    parser.add_argument("--calibration", choices=["none", "temperature", "isotonic"], default="isotonic")
    parser.add_argument("--min-calibration-samples", type=int, default=20)
    parser.add_argument("--abstention-threshold", type=float, default=0.60)
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--consistency-lambda", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--omega-epsilon", type=float, default=1e-8)
    parser.add_argument("--risk-aversion", type=float, default=0.1)
    parser.add_argument("--market-risk-aversion", type=float, default=2.5)
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.1)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evaluation_days <= 0:
        raise ValueError("evaluation-days must be positive")
    assets = load_universe(args.universe)
    market_caps = load_json(args.market_caps)
    missing_caps = [asset for asset in assets if asset not in market_caps]
    if missing_caps:
        raise ValueError(f"market caps missing assets: {missing_caps}")

    history: list[dict[str, Any]] = []
    previous_absolute: np.ndarray | None = None
    previous_relative: np.ndarray | None = None
    daily_frames: list[pd.DataFrame] = []
    period_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for formation_month, start, end in monthly_periods(args.start_month, args.end_month):
        formation_path = args.returns_dir / f"returns_{start}_{end}.csv"
        realized_path = next_month_path(args.returns_dir, formation_month)
        absolute_path = args.absolute_dir / f"{args.model}_{start}_{end}.json"
        relative_path = args.relative_dir / f"{args.model}_{formation_month}.json"
        missing_files = [
            path for path in (formation_path, realized_path, absolute_path, relative_path)
            if not path.exists()
        ]
        if missing_files:
            raise FileNotFoundError(f"missing files for {formation_month}: {missing_files}")

        formation = load_returns(formation_path)
        realized = load_returns(realized_path).iloc[:args.evaluation_days]
        missing_assets = [
            asset for asset in assets
            if asset not in formation.columns or asset not in realized.columns
        ]
        if missing_assets:
            raise ValueError(f"{formation_month} returns missing assets: {missing_assets}")
        formation = formation[assets]
        realized = realized[assets]
        covariance = formation.cov().to_numpy(dtype=float)
        caps = np.asarray([float(market_caps[asset]) for asset in assets], dtype=float)
        prior = implied_equilibrium_returns(
            covariance, caps / caps.sum(), args.market_risk_aversion
        )

        absolute_response = load_json(absolute_path)
        absolute, _ = absolute_weights(
            absolute_response, assets, prior, covariance, args.tau,
            args.omega_epsilon, args.risk_aversion, args.max_weight,
            previous_absolute, args.turnover_penalty,
        )
        relative_views = load_records(relative_path, "views")
        rel_config = RelViewConfig(
            calibration=args.calibration,
            min_calibration_samples=args.min_calibration_samples,
            abstention_threshold=args.abstention_threshold,
            min_evidence=args.min_evidence,
            consistency_lambda=args.consistency_lambda,
            omega_epsilon=args.omega_epsilon,
            tau=args.tau,
            risk_aversion=args.risk_aversion,
            turnover_penalty=args.turnover_penalty,
            max_weight=args.max_weight,
        )
        relative_result = run_relview_bl(
            formation, prior, relative_views, history, rel_config, previous_relative
        )
        relative = relative_result.weights.to_numpy(dtype=float)
        equal = np.full(len(assets), 1.0 / len(assets))

        absolute_turnover = turnover(absolute, previous_absolute)
        relative_turnover = turnover(relative, previous_relative)
        method_data: dict[str, tuple[np.ndarray, float]] = {
            "Absolute_LLM_BLM": (absolute, absolute_turnover),
            "RelView_BL": (relative, relative_turnover),
            "Equal_Weight": (equal, 0.0 if period_rows else 1.0),
        }
        period_daily = pd.DataFrame({"Date": realized.index.astype(str)})
        period_metrics: dict[str, dict[str, float | int]] = {}
        for method, (method_weights, method_turnover) in method_data.items():
            daily, metrics = evaluate_realized_portfolio(
                realized,
                pd.Series(method_weights, index=assets),
                turnover=method_turnover,
                transaction_cost_bps=args.transaction_cost_bps,
            )
            period_daily[f"{method}_Return"] = daily["Portfolio_Return"]
            period_metrics[method] = metrics
            weight_rows.append({
                "Formation_Month": formation_month,
                "Method": method,
                **dict(zip(assets, method_weights.tolist())),
            })
        daily_frames.append(period_daily)
        period_rows.append({
            "Formation_Month": formation_month,
            "Test_Start": str(realized.index[0]),
            "Test_End": str(realized.index[-1]),
            "Calibration_Method": relative_result.calibrator.fitted_method,
            "Calibration_History_Size": len(history),
            "Accepted_Relative_Views": len(relative_result.matrices.accepted_views),
            "Rejected_Relative_Views": len(relative_result.matrices.rejected_views),
            "Raw_Cycles": relative_result.matrices.raw_cycle_count,
            **{
                f"{method}_{metric}": value
                for method, metrics in period_metrics.items()
                for metric, value in metrics.items()
            },
        })

        # Outcomes enter calibration only after this period's portfolio is fixed.
        history.extend(calibration_observations_from_realized_returns(relative_views, realized))
        previous_absolute = absolute
        previous_relative = relative
        print(
            f"{formation_month}: accepted {len(relative_result.matrices.accepted_views)} relative views; "
            f"history now {len(history)}"
        )

    daily_all = pd.concat(daily_frames, ignore_index=True)
    summary: dict[str, dict[str, float | int]] = {}
    for method in ("Absolute_LLM_BLM", "RelView_BL", "Equal_Weight"):
        column = f"{method}_Return"
        synthetic = pd.DataFrame({method: daily_all[column].to_numpy()}, index=daily_all["Date"])
        _, metrics = evaluate_realized_portfolio(synthetic, {method: 1.0})
        method_periods = [row for row in period_rows]
        turnovers = [float(row[f"{method}_turnover"]) for row in method_periods]
        # The concatenated daily series is already net of each rebalance cost.  The
        # helper therefore receives no second cost charge above; restore the
        # actual run-level metadata instead of exposing its synthetic 0/0 inputs.
        metrics["turnover"] = float(np.sum(turnovers))
        metrics["transaction_cost_bps"] = float(args.transaction_cost_bps)
        metrics["average_period_turnover"] = float(np.mean(turnovers))
        metrics["total_turnover"] = float(np.sum(turnovers))
        summary[method] = metrics

    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    daily_path = prefix.with_name(f"{prefix.name}_daily.csv")
    periods_path = prefix.with_name(f"{prefix.name}_periods.csv")
    weights_path = prefix.with_name(f"{prefix.name}_weights.csv")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    daily_all.to_csv(daily_path, index=False)
    pd.DataFrame(period_rows).to_csv(periods_path, index=False)
    pd.DataFrame(weight_rows).to_csv(weights_path, index=False)
    summary_path.write_text(json.dumps({
        "config": vars(args) | {"returns_dir": str(args.returns_dir), "absolute_dir": str(args.absolute_dir),
                                "relative_dir": str(args.relative_dir), "universe": str(args.universe),
                                "market_caps": str(args.market_caps), "output_prefix": str(args.output_prefix)},
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {daily_path}, {periods_path}, {weights_path}, and {summary_path}")


if __name__ == "__main__":
    main()
