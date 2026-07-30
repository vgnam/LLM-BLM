"""Evaluate original absolute-view LLM-BLM on the same setup as RelView-BL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_relview import load_json, load_returns, load_universe
from portfolio_backtest import evaluate_realized_portfolio
from relview_bl import (
    black_litterman_posterior,
    implied_equilibrium_returns,
    optimize_portfolio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate absolute LLM-BLM on one period")
    parser.add_argument("--returns", type=Path, required=True, help="Formation-window returns")
    parser.add_argument("--views", type=Path, required=True, help="Absolute LLM response JSON")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--market-caps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-output", type=Path)
    parser.add_argument("--realized-returns", type=Path)
    parser.add_argument("--returns-output", type=Path)
    parser.add_argument("--tau", type=float, default=0.025)
    parser.add_argument("--omega-epsilon", type=float, default=1e-8)
    parser.add_argument("--risk-aversion", type=float, default=0.1)
    parser.add_argument("--market-risk-aversion", type=float, default=2.5)
    parser.add_argument("--max-weight", type=float, default=0.1)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument("--evaluation-days", type=int, help="Use only the first N realized trading days")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    returns = load_returns(args.returns)
    universe = load_universe(args.universe)
    market_caps = load_json(args.market_caps)
    views: dict[str, Any] = load_json(args.views)
    missing = [asset for asset in universe if asset not in returns.columns or asset not in market_caps]
    if missing:
        raise ValueError(f"universe assets missing from returns or market caps: {missing}")
    returns = returns[universe]
    covariance = returns.cov().to_numpy(dtype=float)
    cap_values = np.asarray([float(market_caps[asset]) for asset in universe], dtype=float)
    market_weights = cap_values / cap_values.sum()
    prior = implied_equilibrium_returns(covariance, market_weights, args.market_risk_aversion)

    missing_views = [
        asset for asset in universe
        if asset not in views or len(views[asset].get("expected_return", [])) < 2
    ]
    if missing_views:
        raise ValueError(f"assets need at least two successful absolute LLM calls: {missing_views}")
    samples = [np.asarray(views[asset]["expected_return"], dtype=float) for asset in universe]
    q = np.asarray([float(np.mean(item)) for item in samples], dtype=float)
    omega_diagonal = np.asarray([
        max(float(np.var(item, ddof=1)), args.omega_epsilon) for item in samples
    ])
    P = np.eye(len(universe), dtype=float)
    omega = np.diag(omega_diagonal)
    posterior = black_litterman_posterior(prior, covariance, P, q, omega, args.tau)
    weights, optimizer_message = optimize_portfolio(
        posterior,
        covariance,
        risk_aversion=args.risk_aversion,
        max_weight=args.max_weight,
    )
    weight_series = pd.Series(weights, index=universe, name="weight")
    output: dict[str, Any] = {
        "method": "Absolute LLM-BLM",
        "model": views[universe[0]].get("model", "unknown"),
        "thinking": views[universe[0]].get("thinking", "unknown"),
        "tau": args.tau,
        "omega_epsilon": args.omega_epsilon,
        "risk_aversion": args.risk_aversion,
        "market_risk_aversion": args.market_risk_aversion,
        "max_weight": args.max_weight,
        "optimizer_message": optimizer_message,
        "assets": universe,
        "q": dict(zip(universe, q.tolist())),
        "omega_diagonal": dict(zip(universe, omega_diagonal.tolist())),
        "posterior_returns": dict(zip(universe, posterior.tolist())),
        "weights": weight_series.to_dict(),
    }

    if args.realized_returns:
        realized = load_returns(args.realized_returns)
        if args.evaluation_days is not None:
            if args.evaluation_days <= 0:
                raise ValueError("--evaluation-days must be positive")
            realized = realized.iloc[:args.evaluation_days]
        realized = realized[universe]
        daily, metrics = evaluate_realized_portfolio(
            realized,
            weight_series,
            transaction_cost_bps=args.transaction_cost_bps,
            turnover=1.0,
        )
        output["backtest"] = metrics
        returns_output = args.returns_output or args.output.with_name(f"{args.output.stem}_returns.csv")
        returns_output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(returns_output, index=False)
        output["returns_output"] = str(returns_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    weights_output = args.weights_output or args.output.with_name(f"{args.output.stem}_weights.csv")
    weights_output.parent.mkdir(parents=True, exist_ok=True)
    weight_series.rename_axis("asset").reset_index().to_csv(weights_output, index=False)
    print(f"Saved Absolute LLM-BLM diagnostics to {args.output} and weights to {weights_output}")
    if "backtest" in output:
        print(json.dumps(output["backtest"], indent=2))


if __name__ == "__main__":
    main()
