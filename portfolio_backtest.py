"""Shared realized-return evaluation for portfolio methods."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def evaluate_realized_portfolio(
    realized_returns: pd.DataFrame,
    weights: Sequence[float] | Mapping[str, float] | pd.Series,
    annual_risk_free_rate: float = 0.0,
    turnover: float = 0.0,
    transaction_cost_bps: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if isinstance(weights, Mapping) or isinstance(weights, pd.Series):
        weight_series = pd.Series(weights, dtype=float)
    else:
        values = np.asarray(weights, dtype=float)
        if len(values) != len(realized_returns.columns):
            raise ValueError("weight count must match realized return columns")
        weight_series = pd.Series(values, index=realized_returns.columns, dtype=float)
    missing = [asset for asset in weight_series.index if asset not in realized_returns.columns]
    if missing:
        raise ValueError(f"realized returns are missing weighted assets: {missing}")
    daily = realized_returns[weight_series.index].mul(weight_series, axis=1).sum(axis=1).astype(float)
    if len(daily) and transaction_cost_bps:
        daily.iloc[0] -= float(turnover) * float(transaction_cost_bps) / 10_000.0
    wealth = (1.0 + daily).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    days = len(daily)
    cumulative_return = float(wealth.iloc[-1] - 1.0) if days else 0.0
    annualized_return = (
        float((1.0 + cumulative_return) ** (252.0 / days) - 1.0)
        if days and cumulative_return > -1.0 else -1.0
    )
    daily_volatility = float(daily.std(ddof=1)) if days > 1 else 0.0
    annualized_volatility = daily_volatility * np.sqrt(252.0)
    excess_mean = float(daily.mean()) - float(annual_risk_free_rate) / 252.0 if days else 0.0
    sharpe = excess_mean / daily_volatility * np.sqrt(252.0) if daily_volatility > 0 else 0.0
    frame = pd.DataFrame({
        "Date": realized_returns.index.astype(str),
        "Portfolio_Return": daily.to_numpy(),
        "Cumulative_Return": wealth.to_numpy() - 1.0,
    })
    metrics: dict[str, float | int] = {
        "trading_days": days,
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": float(annualized_volatility),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()) if days else 0.0,
        "turnover": float(turnover),
        "transaction_cost_bps": float(transaction_cost_bps),
    }
    return frame, metrics
