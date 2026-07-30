"""Run one leak-free RelView-BL portfolio period from saved pairwise views."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_backtest import evaluate_realized_portfolio
from relview_bl import (
    RelViewConfig,
    calibration_diagnostics,
    calibration_observations_from_realized_returns,
    implied_equilibrium_returns,
    run_relview_bl,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(path: Path | None, key: str) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a list or an object with a '{key}' list")
    return [item for item in value if isinstance(item, dict) and item.get("status", "ok") == "ok"]


def load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" in frame.columns:
        frame = frame.set_index("Date")
    return frame.select_dtypes(include=[np.number]).dropna(axis=1, how="any")


def load_universe(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        value = load_json(path)
        if isinstance(value, dict):
            value = value.get("assets", value.get("universe", []))
        if not isinstance(value, list):
            raise ValueError("universe JSON must be a list or contain an assets list")
        return [str(item) for item in value]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = next((item for item in ("Symbol", "symbol", "ticker", "asset") if item in frame.columns), None)
        if column is None:
            raise ValueError("universe CSV needs a Symbol, symbol, ticker, or asset column")
        return frame[column].astype(str).tolist()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate consistency-calibrated relative LLM views")
    parser.add_argument("--returns", type=Path, required=True, help="Formation-window returns CSV")
    parser.add_argument("--views", type=Path, required=True, help="Pairwise view JSON")
    parser.add_argument("--market-caps", type=Path, required=True, help="Ticker-to-market-cap JSON")
    parser.add_argument("--universe", type=Path, help="Optional JSON/CSV/TXT asset universe")
    parser.add_argument("--output", type=Path, required=True, help="Output diagnostics JSON")
    parser.add_argument("--weights-output", type=Path, help="Optional weights CSV")
    parser.add_argument("--history", type=Path, help="Past calibration observations only")
    parser.add_argument("--previous-weights", type=Path, help="Optional prior-period weights CSV")
    parser.add_argument("--realized-returns", type=Path, help="Closed next-period returns used after evaluation")
    parser.add_argument("--history-output", type=Path, help="Append realized outcomes to this JSON")
    parser.add_argument("--calibration", choices=["none", "temperature", "isotonic"], default="isotonic")
    parser.add_argument("--min-calibration-samples", type=int, default=20)
    parser.add_argument("--abstention-threshold", type=float, default=0.60)
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--consistency-lambda", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.5)
    parser.add_argument("--disagreement-weight", type=float, default=0.3)
    parser.add_argument("--calibration-error-weight", type=float, default=0.2)
    parser.add_argument("--omega-epsilon", type=float, default=1e-8)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--risk-aversion", type=float, default=0.1)
    parser.add_argument("--market-risk-aversion", type=float, default=2.5)
    parser.add_argument("--turnover-penalty", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.1)
    parser.add_argument("--returns-output", type=Path, help="Optional realized daily returns CSV")
    parser.add_argument("--annual-risk-free-rate", type=float, default=0.0)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--evaluation-days", type=int, help="Use only the first N realized trading days")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = load_returns(args.returns)
    if args.universe:
        universe = load_universe(args.universe)
        missing = [asset for asset in universe if asset not in returns.columns]
        if missing:
            raise ValueError(f"universe assets missing from returns: {missing}")
        returns = returns[universe]
    views = load_records(args.views, "views")
    history = load_records(args.history, "observations")
    market_caps = load_json(args.market_caps)

    eligible_assets = [
        asset for asset in returns.columns
        if asset in market_caps and np.isfinite(float(market_caps[asset])) and float(market_caps[asset]) > 0
    ]
    returns = returns[eligible_assets]
    if not len(eligible_assets):
        raise ValueError("no return columns have a valid market cap")
    covariance = returns.cov().to_numpy(dtype=float)
    cap_values = np.asarray([float(market_caps[asset]) for asset in eligible_assets], dtype=float)
    market_weights = cap_values / cap_values.sum()
    prior = implied_equilibrium_returns(covariance, market_weights, args.market_risk_aversion)

    previous_weights = None
    if args.previous_weights:
        previous_frame = pd.read_csv(args.previous_weights)
        if {"asset", "weight"}.issubset(previous_frame.columns):
            previous_weights = dict(zip(previous_frame["asset"].astype(str), previous_frame["weight"]))
        elif len(previous_frame) == 1:
            previous_weights = previous_frame.iloc[0].to_dict()
        else:
            raise ValueError("previous weights CSV needs asset/weight columns or one wide row")

    config = RelViewConfig(
        calibration=args.calibration,
        min_calibration_samples=args.min_calibration_samples,
        abstention_threshold=args.abstention_threshold,
        min_evidence=args.min_evidence,
        consistency_lambda=args.consistency_lambda,
        entropy_weight=args.entropy_weight,
        disagreement_weight=args.disagreement_weight,
        calibration_error_weight=args.calibration_error_weight,
        omega_epsilon=args.omega_epsilon,
        tau=args.tau,
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
        max_weight=args.max_weight,
    )
    result = run_relview_bl(returns, prior, views, history, config, previous_weights)
    output = {
        "config": asdict(config),
        "calibration": {
            "requested_method": config.calibration,
            "fitted_method": result.calibrator.fitted_method,
            "history_size": len(history),
            "training_brier": result.calibrator.training_brier,
            "diagnostics": calibration_diagnostics(history, result.calibrator),
        },
        "optimizer_message": result.optimizer_message,
        "raw_cycle_count": result.matrices.raw_cycle_count,
        "projected_cycle_count": 0,
        "consistency_rmse": result.matrices.consistency_rmse,
        "accepted_view_count": len(result.matrices.accepted_views),
        "rejected_view_count": len(result.matrices.rejected_views),
        "accepted_views": result.matrices.accepted_views,
        "rejected_views": result.matrices.rejected_views,
        "latent_scores": result.matrices.latent_scores.to_dict(),
        "posterior_returns": result.posterior_returns.to_dict(),
        "weights": result.weights.to_dict(),
    }
    weights_output = args.weights_output or args.output.with_name(f"{args.output.stem}_weights.csv")
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    result.weights.rename_axis("asset").reset_index().to_csv(weights_output, index=False)

    if args.realized_returns:
        realized = load_returns(args.realized_returns)
        if args.evaluation_days is not None:
            if args.evaluation_days <= 0:
                raise ValueError("--evaluation-days must be positive")
            realized = realized.iloc[:args.evaluation_days]
        missing_realized = [asset for asset in eligible_assets if asset not in realized.columns]
        if missing_realized:
            raise ValueError(f"realized returns are missing assets: {missing_realized}")
        realized = realized[eligible_assets]
        if previous_weights is None:
            turnover = 1.0
        else:
            previous_series = pd.Series(previous_weights, dtype=float).loc[eligible_assets]
            turnover = float(np.sum(np.abs(result.weights - previous_series)))
        daily, metrics = evaluate_realized_portfolio(
            realized,
            result.weights,
            annual_risk_free_rate=args.annual_risk_free_rate,
            turnover=turnover,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        returns_output = args.returns_output or args.output.with_name(f"{args.output.stem}_returns.csv")
        returns_output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(returns_output, index=False)
        output["backtest"] = metrics
        output["returns_output"] = str(returns_output)
        new_observations = calibration_observations_from_realized_returns(views, realized)
        combined_history = history + new_observations
        history_output = args.history_output or args.output.with_name("calibration_history.json")
        history_output.parent.mkdir(parents=True, exist_ok=True)
        with history_output.open("w", encoding="utf-8") as handle:
            json.dump({"observations": combined_history}, handle, indent=2, ensure_ascii=False)
        print(f"Appended {len(new_observations)} closed-period observations to {history_output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    print(
        f"Saved {len(result.matrices.accepted_views)} accepted views, portfolio diagnostics to {args.output}, "
        f"and weights to {weights_output}"
    )
    if "backtest" in output:
        print(json.dumps(output["backtest"], indent=2))


if __name__ == "__main__":
    main()
