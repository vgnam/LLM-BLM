"""GPT-OSS-20B walk-forward rebalance backtest entirely within calendar 2025.

Every N trading days (N in {30, 60}) the deterministic baselines (BL / MVO / EW)
and the two LLM Black-Litterman methods (Absolute LLM-BLM and Pairwise RelView-BL)
are re-formed from the trailing ~252 trading sessions (about one trading year)
before the rebalance cutoff, then held unchanged for the next N sessions. The
holding-period daily returns are chained into one continuous investable NAV path
per method per holding period.

Formation data for the trailing window is loaded from the repository's yfinance
monthly files (2024 H2) plus the 2025 daily formation returns produced by
experiments/cutoff_2025_12, so an early-2025 rebalance still has several months
of prior history. LLM views are checkpointed per (dataset, holding period,
rebalance index) so a crash or user interrupt resumes rather than discards
completed calls.

Run from the repository root with the NVIDIA API key exported:
    $env:NVIDIA_API_KEY = "<key>"
    py run_nvidia_nim_walkforward_2025.py --dry-run
    py run_nvidia_nim_walkforward_2025.py
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from backtest_compare import absolute_weights
from collect_absolute_views import collect_absolute_views
from collect_relative_views import _metadata_lookup, collect_pairwise_views
from portfolio_backtest import evaluate_realized_portfolio
from relview_bl import (
    RelViewConfig,
    implied_equilibrium_returns,
    optimize_portfolio,
    run_relview_bl,
    select_candidate_pairs,
)


BASELINE_METHODS = ("BL", "MVO", "EW")
METHOD_PLOT_ORDER = ["BL", "MVO", "EW"]
DEFAULT_CONFIG = Path("experiments/nvidia_nim_2025_walkforward/config.json")
DEFAULT_OUTPUT = Path("experiments/nvidia_nim_2025_walkforward")
_CONFIG_CACHE = {"max_weight": 0.15, "slug": "", "model_label": ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-calls", type=int, default=6)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--force-views", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--holding-periods", nargs="+", type=int, default=None,
        help="Override the configured holding periods (positive integers).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Override the configured dataset ids.",
    )
    return parser.parse_args()


# ---------- helpers -----------------------------------------------------------


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_frame(frame: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    frame.to_parquet(base.with_suffix(".parquet"), index=False)


def method_names(slug: str) -> tuple[str, str]:
    return f"BLM_LLM__{slug}", f"RelViewBL__{slug}"


def display_method_name(method: str) -> str:
    label = _CONFIG_CACHE.get("model_label") or "LLM"
    if method.startswith("RelViewBL__"):
        return f"PairBL ({label})"
    if method.startswith("BLM_LLM__"):
        return f"LLM-BLM ({label})"
    return method


def load_manifest(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(Path(str(config["dataset_manifest"])))
    by_id = {str(item["id"]): item for item in manifest["datasets"]}
    selected = [str(item) for item in config["datasets"]]
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"dataset manifest is missing: {missing}")
    return [by_id[item] for item in selected]


def monthly_files(returns_dir: Path, start_month: str, end_month: str) -> list[Path]:
    starts = pd.date_range(f"{start_month}-01", f"{end_month}-01", freq="MS")
    paths = []
    for start in starts:
        month_end = start + pd.offsets.MonthEnd(0)
        path = returns_dir / f"returns_{start:%Y-%m-%d}_{month_end:%Y-%m-%d}.csv"
        if path.exists():
            paths.append(path)
    return paths


def load_dataset_returns(
    config: Mapping[str, Any], dataset: Mapping[str, Any], output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return (combined trailing history, 2025-only panel, provenance).

    `trailing` spans yfinance 2024 H2 through the end of 2025 and supplies the
    one-year look-back window for early rebalances. `panel_2025` is the subset of
    rows dated in 2025 and drives both the rebalance schedule and realized holds.
    """
    dataset_id = str(dataset["id"])
    source = Path(str(config["source_experiment_root"])) / dataset_id
    formation_path = source / "data" / "formation_returns.csv"
    metadata_path = source / "metadata.csv"
    if not formation_path.exists():
        raise FileNotFoundError(f"missing source formation returns: {formation_path}")
    returns_2025 = pd.read_csv(formation_path, parse_dates=["Date"]).set_index("Date").sort_index()
    assets = [str(item["ticker"]) for item in dataset["assets"]]

    yfinance_dir = Path(str(config.get("yfinance_dir", "yfinance")))
    start_month = str(config.get("yfinance_start_month", "2024-06"))
    end_month = str(config.get("yfinance_end_month", "2025-07"))
    frames = [returns_2025]
    for path in monthly_files(yfinance_dir, start_month, end_month):
        try:
            frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
        except Exception:
            continue
        # The yfinance monthly files only cover the equity universe; skip files
        # that carry none of this dataset's assets (e.g. ETF datasets) instead
        # of injecting all-NaN rows into the trailing window.
        present = [asset for asset in assets if asset in frame.columns]
        if not present:
            continue
        frames.append(frame[present])
    trailing = pd.concat(frames)
    trailing = trailing[~trailing.index.duplicated(keep="first")].sort_index()
    trailing = trailing[[asset for asset in assets if asset in trailing.columns]]
    missing = [asset for asset in assets if asset not in trailing.columns]
    if missing:
        raise ValueError(f"{dataset_id}: combined history missing assets: {missing}")
    trailing = trailing[assets]
    # Remove rows where every asset is missing so trailing windows do not
    # contain degenerate all-NaN dates.
    trailing = trailing.dropna(how="all")

    year = int(config["formation_year"])
    panel_2025 = trailing.loc[
        (trailing.index >= pd.Timestamp(f"{year}-01-01")) &
        (trailing.index <= pd.Timestamp(f"{year}-12-31"))
    ]
    if panel_2025.empty:
        raise ValueError(f"{dataset_id}: no {year} returns")

    metadata = pd.read_csv(metadata_path) if metadata_path.exists() else pd.DataFrame()
    target = output_root / dataset_id / "data"
    target.mkdir(parents=True, exist_ok=True)
    trailing.to_csv(target / "trailing_returns.csv", index=True)
    panel_2025.to_csv(target / "returns_2025.csv", index=True)
    if metadata_path.exists():
        import shutil
        shutil.copy2(metadata_path, target / "metadata.csv")
    provenance = {
        "dataset": dataset_id,
        "source": str(formation_path),
        "trailing_first": str(trailing.index.min().date()),
        "trailing_last": str(trailing.index.max().date()),
        "trailing_days": int(len(trailing)),
        "first_2025": str(panel_2025.index.min().date()),
        "last_2025": str(panel_2025.index.max().date()),
        "panel_2025_days": int(len(panel_2025)),
        "assets": assets,
    }
    atomic_json(target / "provenance.json", provenance)
    return trailing, panel_2025, metadata, provenance


def rebalance_indices(n_days: int, total_days: int, min_formation: int) -> list[int]:
    if n_days <= 0:
        raise ValueError("holding period must be positive")
    if min_formation < 2:
        raise ValueError("min_formation_days must be at least 2")
    indices: list[int] = []
    t = min_formation - 1
    while t + 1 + n_days <= total_days:
        indices.append(t)
        t += n_days
    return indices


def turnover(current: np.ndarray, previous: np.ndarray | None) -> float:
    return float(np.sum(np.abs(current - previous))) if previous is not None else 1.0


def baseline_weights(
    formation: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    assets = formation.columns.astype(str).tolist()
    covariance = formation.cov().to_numpy(dtype=float)
    equal = np.full(len(assets), 1.0 / len(assets))
    prior = implied_equilibrium_returns(
        covariance, equal, float(config["market_risk_aversion"])
    )
    bl, _ = optimize_portfolio(
        prior, covariance, risk_aversion=float(config["risk_aversion"]),
        max_weight=float(config["max_weight"]),
    )
    mvo, _ = optimize_portfolio(
        formation.mean().to_numpy(dtype=float), covariance,
        risk_aversion=float(config["risk_aversion"]),
        max_weight=float(config["max_weight"]),
    )
    return {"BL": bl, "MVO": mvo, "EW": equal}, prior, covariance


# ---------- LLM views --------------------------------------------------------


def rebalance_views_complete(
    absolute_path: Path, relative_path: Path, assets: list[str], config: Mapping[str, Any]
) -> bool:
    if not absolute_path.exists() or not relative_path.exists():
        return False
    try:
        absolute = read_json(absolute_path)
        relative = read_json(relative_path)
        minimum = int(config["minimum_successful_calls"])
        max_pairs = int(config["max_pairs"])
        views = relative.get("views", [])
        return (
            all(
                len(absolute.get(asset, {}).get("expected_return", [])) >= minimum
                for asset in assets
            )
            and len(views) >= max_pairs
            and all(
                item.get("status") == "ok"
                and int(item.get("successful_repeats", 0)) >= minimum
                for item in views
            )
        )
    except Exception:
        return False


def collect_rebalance_views(
    dataset_root: Path,
    holding_days: int,
    rebalance_idx: int,
    formation: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    workers: int,
    retry_calls: int,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    assets = formation.columns.astype(str).tolist()
    response_root = dataset_root / "responses" / str(model["slug"]) / f"holding_{holding_days}"
    response_root.mkdir(parents=True, exist_ok=True)
    tag = f"rebalance_{rebalance_idx:03d}_{formation.index[-1].date()}"
    absolute_path = response_root / f"absolute_{tag}.json"
    relative_path = response_root / f"relative_{tag}.json"

    if not force and rebalance_views_complete(absolute_path, relative_path, assets, config):
        return read_json(absolute_path), read_json(relative_path)["views"]
    if force:
        absolute_checkpoint = None
        relative_checkpoint = None
    else:
        absolute_checkpoint = absolute_path
        relative_checkpoint = relative_path

    api_key = str(os.environ[str(config["api_key_env"])])
    metadata = _metadata_lookup(metadata_frame)
    horizon_days = holding_days
    absolute = collect_absolute_views(
        formation, str(model["api_model"]), str(config["base_url"]), api_key,
        metadata, {}, int(config["repeats"]), horizon_days, float(config["temperature"]),
        "disabled", workers, retry_calls, False, "generic", absolute_path,
    )
    atomic_json(absolute_path, absolute)
    pairs = select_candidate_pairs(
        formation, metadata_frame, {asset: 1.0 for asset in assets},
        max_pairs=int(config["max_pairs"]),
    )
    views = collect_pairwise_views(
        formation, pairs, str(model["api_model"]), str(config["base_url"]), api_key,
        metadata, {}, int(config["repeats"]), horizon_days, float(config["temperature"]),
        str(config.get("probability_estimator", "mean")), "disabled", workers,
        retry_calls, False, "calibrated", relative_path,
    )
    payload = {
        "requested_model": model["requested_model"],
        "model": model["api_model"],
        "base_url": config["base_url"],
        "rebalance_index": rebalance_idx,
        "formation_end": str(formation.index[-1].date()),
        "formation_days": int(len(formation)),
        "horizon_days": horizon_days,
        "repeats": config["repeats"],
        "probability_estimator": config.get("probability_estimator", "mean"),
        "temperature": config["temperature"],
        "pairs_requested": len(pairs),
        "views": views,
    }
    atomic_json(relative_path, payload)
    return absolute, views


def llm_weights(
    formation: pd.DataFrame,
    absolute: dict[str, Any],
    views: list[dict[str, Any]],
    prior: np.ndarray,
    covariance: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    assets = formation.columns.astype(str).tolist()
    absolute_portfolio, _ = absolute_weights(
        absolute, assets, prior, covariance, float(config["tau"]), 1e-8,
        float(config["risk_aversion"]), float(config["max_weight"]), None, 0.0,
    )
    relview = run_relview_bl(
        formation, prior, views, [],
        RelViewConfig(
            calibration=str(config["calibration"]),
            abstention_threshold=float(config["abstention_threshold"]),
            tau=float(config["tau"]),
            risk_aversion=float(config["risk_aversion"]),
            max_weight=float(config["max_weight"]),
        ),
    )
    diagnostics = {
        "calibration_method": relview.calibrator.fitted_method,
        "accepted_view_count": len(relview.matrices.accepted_views),
        "rejected_view_count": len(relview.matrices.rejected_views),
        "raw_cycle_count": relview.matrices.raw_cycle_count,
        "consistency_rmse": relview.matrices.consistency_rmse,
    }
    return (
        absolute_portfolio,
        relview.weights.reindex(assets).to_numpy(dtype=float),
        diagnostics,
    )


# ---------- backtest ----------------------------------------------------------


def run_holding_period(
    holding_days: int,
    trailing: pd.DataFrame,
    panel_2025: pd.DataFrame,
    metadata: pd.DataFrame,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    dataset_root: Path,
    workers: int,
    retry_calls: int,
    force_views: bool,
    skip_llm: bool,
) -> dict[str, Any]:
    assets = panel_2025.columns.astype(str).tolist()
    total = len(panel_2025)
    min_formation = int(config["min_formation_days"])
    lookback = int(config["formation_lookback_days"])
    rebalances = rebalance_indices(holding_days, total, min_formation)
    n_reb = len(rebalances)

    previous: dict[str, np.ndarray | None] = {m: None for m in BASELINE_METHODS}
    abs_method, rel_method = method_names(str(model["slug"]))
    previous[abs_method] = None
    previous[rel_method] = None
    daily_frames: list[pd.DataFrame] = []
    weight_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for position, rebalance_idx in enumerate(rebalances):
        cutoff_date = panel_2025.index[rebalance_idx]
        formation = trailing.loc[:cutoff_date].tail(lookback)
        realized = panel_2025.iloc[rebalance_idx + 1 : rebalance_idx + 1 + holding_days]
        if len(realized) != holding_days:
            continue
        holding_start = realized.index[0]
        holding_end = realized.index[-1]

        baselines, prior, covariance = baseline_weights(formation, config)
        method_weights: dict[str, np.ndarray] = dict(baselines)

        if skip_llm:
            method_weights[abs_method] = previous[abs_method] if previous[abs_method] is not None else baselines["BL"]
            method_weights[rel_method] = previous[rel_method] if previous[rel_method] is not None else baselines["BL"]
        else:
            try:
                absolute, views = collect_rebalance_views(
                    dataset_root, holding_days, rebalance_idx, formation, metadata,
                    model, config, workers, retry_calls, force_views,
                )
                abs_portfolio, rel_portfolio, diagnostics = llm_weights(
                    formation, absolute, views, prior, covariance, config
                )
                method_weights[abs_method] = abs_portfolio
                method_weights[rel_method] = rel_portfolio
                diag_rows.append({
                    "Rebalance": position,
                    "Formation_End": str(cutoff_date.date()),
                    "Formation_Days": int(len(formation)),
                    "Holding_Start": str(holding_start.date()),
                    "Holding_End": str(holding_end.date()),
                    "Accepted_Views": diagnostics["accepted_view_count"],
                    "Rejected_Views": diagnostics["rejected_view_count"],
                    "Calibration": diagnostics["calibration_method"],
                })
            except Exception as error:
                detail = str(error)[:1000]
                print(
                    f"[{holding_days}d rb#{position} {cutoff_date.date()}] "
                    f"LLM views failed: {detail}",
                    flush=True,
                )
                errors.append({
                    "Holding_Days": holding_days,
                    "Rebalance": position,
                    "Formation_End": str(cutoff_date.date()),
                    "Detail": detail,
                })
                bl_fallback = baselines["BL"]
                method_weights[abs_method] = previous[abs_method] if previous[abs_method] is not None else bl_fallback
                method_weights[rel_method] = previous[rel_method] if previous[rel_method] is not None else bl_fallback

        period_daily = pd.DataFrame({"Date": realized.index.astype(str)})
        for method, weights in method_weights.items():
            daily, _ = evaluate_realized_portfolio(
                realized, pd.Series(weights, index=assets),
                annual_risk_free_rate=float(config["annual_risk_free_rate"]),
                turnover=turnover(weights, previous[method]),
                transaction_cost_bps=float(config["transaction_cost_bps"]),
            )
            period_daily[f"{method}_Return"] = daily["Portfolio_Return"].to_numpy()
            weight_rows.append({
                "Holding_Days": holding_days,
                "Rebalance": position,
                "Formation_End": str(cutoff_date.date()),
                "Holding_Start": str(holding_start.date()),
                "Method": method,
                **dict(zip(assets, weights.tolist())),
            })
            previous[method] = weights
        period_daily.insert(0, "Holding_Days", holding_days)
        period_daily.insert(1, "Rebalance", position)
        period_daily.insert(2, "Formation_End", str(cutoff_date.date()))
        daily_frames.append(period_daily)
        print(
            f"[{holding_days}d] rb {position + 1}/{n_reb}: formed {cutoff_date.date()} "
            f"({len(formation)}d history), held {holding_start.date()}->{holding_end.date()}",
            flush=True,
        )

    daily_all = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    return {
        "holding_days": holding_days,
        "rebalances_attempted": n_reb,
        "daily": daily_all,
        "weights": pd.DataFrame(weight_rows),
        "diagnostics": pd.DataFrame(diag_rows),
        "errors": errors,
        "assets": assets,
    }


# ---------- metrics and plots ------------------------------------------------


def chained_nav(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    daily = daily.copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    for col in [c for c in daily.columns if c.endswith("_Return")]:
        method = col[: -len("_Return")]
        daily[f"{method}_NAV"] = (1.0 + daily[col]).cumprod()
    return daily.sort_values("Date").reset_index(drop=True)


def drawdown_stats(nav: np.ndarray) -> dict[str, float]:
    """Drawdown series plus longest episode length and time-to-recovery."""
    dd = nav / np.maximum.accumulate(nav) - 1.0
    max_dd = float(np.min(dd)) if len(dd) else 0.0
    if len(dd) == 0:
        return {"max_drawdown": 0.0, "avg_drawdown": 0.0, "max_drawdown_days": 0,
                "drawdown_recovery_days": 0, "drawdown_count": 0}
    in_dd = dd < 0.0
    episodes: list[int] = []
    current = 0
    for flag in in_dd:
        if flag:
            current += 1
        else:
            if current:
                episodes.append(current)
            current = 0
    if current:
        episodes.append(current)
    episodes = episodes or [0]
    # time-to-recovery: days from the max-drawdown trough back to a new high
    trough_idx = int(np.argmin(dd))
    trough_nav = nav[trough_idx]
    recovery = 0
    for idx in range(trough_idx, len(nav)):
        if nav[idx] >= nav[: trough_idx + 1].max():
            recovery = idx - trough_idx
            break
    return {
        "max_drawdown": max_dd,
        "avg_drawdown": float(np.mean(dd[in_dd])) if in_dd.any() else 0.0,
        "max_drawdown_days": int(max(episodes)),
        "drawdown_recovery_days": int(recovery),
        "drawdown_count": int(len(episodes)),
    }


def var_cvar(values: np.ndarray, level: float) -> tuple[float, float]:
    """Historical VaR and CVaR (expected shortfall) at a probability level."""
    var = float(np.quantile(values, 1.0 - level))
    tail = values[values <= var]
    cvar = float(np.mean(tail)) if len(tail) else var
    return var, cvar


def method_metrics(daily_nav: pd.DataFrame, method: str, annual_rf: float) -> dict[str, Any]:
    col = f"{method}_Return"
    nav_col = f"{method}_NAV"
    if col not in daily_nav.columns:
        return {}
    values = daily_nav[col].to_numpy(dtype=float)
    nav = daily_nav[nav_col].to_numpy(dtype=float)
    days = len(values)
    if days == 0:
        return {}
    daily_rf = annual_rf / 252.0
    downside = np.minimum(values - daily_rf, 0.0)
    downside_dev = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252.0))
    excess = float((np.mean(values) - daily_rf) * 252.0)
    std_daily = float(np.std(values, ddof=1)) if days > 1 else 0.0
    cumulative = float(nav[-1] - 1.0)
    annualized = float(nav[-1] ** (252.0 / days) - 1.0) if cumulative > -1.0 else -1.0
    vol = float(np.std(values, ddof=1) * np.sqrt(252.0)) if days > 1 else 0.0
    sharpe = (float(np.mean(values) - daily_rf) / std_daily * np.sqrt(252.0)) if std_daily > 0 else 0.0
    sortino = excess / downside_dev if downside_dev > 0 else 0.0
    dd_stats = drawdown_stats(nav)
    calmar = annualized / abs(dd_stats["max_drawdown"]) if dd_stats["max_drawdown"] < 0 else 0.0
    gains = values[values > 0]
    losses = values[values < 0]
    gain_loss_ratio = float(np.mean(gains) / abs(np.mean(losses))) if len(gains) and len(losses) and np.mean(losses) != 0 else 0.0
    profit_factor = float(np.sum(gains) / abs(np.sum(losses))) if np.sum(losses) != 0 else float("inf")
    n_reb = int(daily_nav["Rebalance"].nunique()) if "Rebalance" in daily_nav.columns else 0
    var_95, cvar_95 = var_cvar(values, 0.95)
    var_99, cvar_99 = var_cvar(values, 0.99)
    worst_5 = float(np.mean(np.sort(values)[: max(int(days * 0.05), 1)]))
    return {
        "Method": method,
        "Display": display_method_name(method),
        "trading_days": days,
        "rebalances": n_reb,
        "final_nav": float(nav[-1]),
        "cumulative_return": cumulative,
        "annualized_return": annualized,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        **dd_stats,
        "max_drawdown_duration_pct": float(dd_stats["max_drawdown_days"] / days) if days else 0.0,
        "downside_deviation": downside_dev,
        "ulcer_index": float(np.sqrt(np.mean(np.square(np.minimum(dd_stats["max_drawdown"], 0.0))))) if dd_stats["max_drawdown"] != 0.0 else 0.0,
        "mean_daily_return": float(np.mean(values)),
        "best_day": float(np.max(values)),
        "worst_day": float(np.min(values)),
        "positive_day_ratio": float(np.mean(values > 0.0)),
        "gain_loss_ratio": gain_loss_ratio,
        "profit_factor": profit_factor,
        "daily_var_95": var_95,
        "daily_cvar_95": cvar_95,
        "daily_var_99": var_99,
        "daily_cvar_99": cvar_99,
        "worst_5_percent_avg": worst_5,
    }


def method_palette(methods: list[str]) -> dict[str, str]:
    palette = sns.color_palette("Set1", n_colors=max(len(methods), 5)).as_hex()
    return {method: palette[i % len(palette)] for i, method in enumerate(methods)}


def plot_nav(daily_nav: pd.DataFrame, title: str, output: Path, methods: list[str]) -> None:
    if daily_nav.empty:
        return
    palette = method_palette(methods)
    fig, ax = plt.subplots(figsize=(13, 7))
    for method in methods:
        nav_col = f"{method}_NAV"
        if nav_col not in daily_nav.columns:
            continue
        sns.lineplot(data=daily_nav, x="Date", y=nav_col, ax=ax,
                     label=display_method_name(method), color=palette[method],
                     linewidth=2.4 if "LLM" in method else 1.8)
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (initial capital = 1)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    sns.despine()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_drawdown(daily_nav: pd.DataFrame, title: str, output: Path, methods: list[str]) -> None:
    if daily_nav.empty:
        return
    palette = method_palette(methods)
    fig, ax = plt.subplots(figsize=(13, 6))
    for method in methods:
        nav_col = f"{method}_NAV"
        if nav_col not in daily_nav.columns:
            continue
        nav = daily_nav[nav_col].to_numpy(dtype=float)
        dd = nav / np.maximum.accumulate(nav) - 1.0
        sns.lineplot(x=daily_nav["Date"], y=dd, ax=ax,
                     label=display_method_name(method), color=palette[method],
                     linewidth=1.6)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.grid(alpha=0.25)
    sns.despine()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, title: str, output: Path) -> None:
    if metrics.empty:
        return
    metric_names = [
        "cumulative_return", "annualized_return", "sharpe", "sortino",
        "max_drawdown", "annualized_volatility",
        "max_drawdown_days", "drawdown_recovery_days",
        "downside_deviation", "gain_loss_ratio",
        "daily_cvar_95", "daily_cvar_99",
    ]
    palette = method_palette(metrics["Method"].tolist())
    fig, axes = plt.subplots(4, 3, figsize=(18, 15))
    axes = axes.ravel()
    for ax, metric in zip(axes, metric_names):
        order = metrics.sort_values(metric, ascending=False)["Method"].tolist()
        colors = [palette[m] for m in order]
        sns.barplot(data=metrics, x="Method", y=metric, ax=ax,
                    hue="Method", hue_order=order, palette=colors, legend=False)
        ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        if metric in {"cumulative_return", "annualized_return", "max_drawdown",
                      "annualized_volatility", "downside_deviation", "daily_cvar_95",
                      "daily_cvar_99"}:
            ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    sns.despine()
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_weight_heatmap(weights: pd.DataFrame, title: str, output: Path, assets: list[str]) -> None:
    if weights.empty:
        return
    long = weights.melt(
        id_vars=["Display_Label"], value_vars=assets,
        var_name="Asset", value_name="Weight",
    )
    matrix = long.pivot(index="Display_Label", columns="Asset", values="Weight").reindex(columns=assets)
    fig, ax = plt.subplots(figsize=(14, 0.36 * len(matrix.index) + 2.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "Weight"}, vmin=0.0,
                vmax=max(float(config_max_weight()), 0.15), ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Method / rebalance")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


_CONFIG_CACHE = {"max_weight": 0.15, "slug": "", "model_label": ""}


def set_config_max_weight(value: float) -> None:
    _CONFIG_CACHE["max_weight"] = float(value)


def set_model_context(config: Mapping[str, Any]) -> None:
    models = config.get("models") or []
    if models:
        model = dict(models[0])
        _CONFIG_CACHE["slug"] = str(model.get("slug", ""))
        requested = str(model.get("requested_model", ""))
        label = requested.split("/")[-1].replace("-instruct", "")
        tokens = label.replace("-", " ").split()
        parts = [
            part.upper()
            if (part.isalpha() and len(part) <= 4) or part[:1].isdigit()
            else part.capitalize()
            for part in tokens
        ]
        label = "-".join(parts)
        _CONFIG_CACHE["model_label"] = label or "LLM"


def pretty_dataset_label(dataset_id: str) -> str:
    """Human-readable universe label for titles and captions."""
    labels = {
        "us_technology": "US Technology Equities",
        "us_financials": "US Financial Equities",
        "cross_asset_etfs": "Cross-Asset ETFs",
    }
    return labels.get(dataset_id, dataset_id.replace("_", " ").title())


def plot_title(kind: str, dataset_id: str, holding_days: int) -> str:
    """Standard plot title: '<Universe> — <kind>, <N>-Day Rebalance, 2025'."""
    model_label = _CONFIG_CACHE.get("model_label") or "LLM"
    return (
        f"{pretty_dataset_label(dataset_id)} — {kind} | "
        f"{holding_days}-Day Rebalance | {model_label} | 2025"
    )


def config_max_weight() -> float:
    return _CONFIG_CACHE["max_weight"]


def method_plot_order() -> list[str]:
    slug = _CONFIG_CACHE.get("slug") or ""
    if slug:
        return ["BL", "MVO", "EW", f"BLM_LLM__{slug}", f"RelViewBL__{slug}"]
    return ["BL", "MVO", "EW"]


# ---------- per-dataset output ----------------------------------------------


def save_holding_results(
    dataset_root: Path, holding_days: int, result: dict[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    hp_dir = dataset_root / f"holding_{holding_days}"
    hp_dir.mkdir(parents=True, exist_ok=True)
    daily = result["daily"]
    weights = result["weights"]
    diagnostics = result["diagnostics"]
    daily_nav = chained_nav(daily) if not daily.empty else daily
    save_frame(daily_nav, hp_dir / "daily_nav")
    save_frame(weights, hp_dir / "weights")
    if not diagnostics.empty:
        save_frame(diagnostics, hp_dir / "llm_diagnostics")

    methods = list({c[: -len("_Return")] for c in daily.columns if c.endswith("_Return")})
    methods = [m for m in method_plot_order() if m in methods] + [m for m in methods if m not in method_plot_order()]
    metrics_rows = [method_metrics(daily_nav, m, float(config["annual_risk_free_rate"])) for m in methods]
    metrics = pd.DataFrame([row for row in metrics_rows if row])
    save_frame(metrics, hp_dir / "method_metrics")
    atomic_json(hp_dir / "summary.json", {
        "dataset": dataset_root.name,
        "holding_days": holding_days,
        "rebalances_attempted": result["rebalances_attempted"],
        "rebalances_run": int(daily_nav["Rebalance"].nunique()) if not daily_nav.empty and "Rebalance" in daily_nav.columns else 0,
        "errors": result["errors"],
    })

    model_label = _CONFIG_CACHE.get("model_label") or "LLM"
    dataset_label = pretty_dataset_label(dataset_root.name)
    title = plot_title("Cumulative NAV", dataset_root.name, holding_days)
    plot_nav(daily_nav, title, hp_dir / "nav.png", methods)
    plot_drawdown(daily_nav, plot_title("Drawdown", dataset_root.name, holding_days), hp_dir / "drawdown.png", methods)
    plot_metrics(metrics, f"{dataset_label}: Metrics ({holding_days}-Day Rebalance, {model_label})", hp_dir / "metrics.png")

    assets = result["assets"]
    if not weights.empty:
        labeled = weights.copy()
        labeled["Display_Label"] = labeled.apply(
            lambda r: f"{display_method_name(r['Method'])} | rb{r['Rebalance']} {r['Formation_End']}",
            axis=1,
        )
        plot_weight_heatmap(
            labeled,
            plot_title("Portfolio Weights", dataset_root.name, holding_days),
            hp_dir / "weights_heatmap.png",
            assets,
        )
    return {"daily": daily_nav, "metrics": metrics, "methods": methods}


# ---------- main -------------------------------------------------------------


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if args.holding_periods:
        if any(d <= 0 for d in args.holding_periods):
            raise ValueError("holding periods must be positive")
        config = {**config, "holding_periods": args.holding_periods}
    if args.datasets:
        config = {**config, "datasets": args.datasets}
    set_config_max_weight(float(config["max_weight"]))
    set_model_context(config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "config_used.json", config)

    datasets = load_manifest(config)
    dataset_ids = [str(item["id"]) for item in datasets]
    holding_periods = [int(d) for d in config["holding_periods"]]
    min_formation = int(config["min_formation_days"])
    workers = max(1, int(args.workers))
    retry_calls = max(0, int(args.retry_calls))

    total_rebalances = 0
    total_calls = 0
    repeats = int(config["repeats"])
    max_pairs = int(config["max_pairs"])
    for dataset in datasets:
        trailing, panel_2025, _, _ = load_dataset_returns(config, dataset, output_root)
        n_assets = len(panel_2025.columns)
        for hp in holding_periods:
            rb = rebalance_indices(hp, len(panel_2025), min_formation)
            total_rebalances += len(rb)
            total_calls += len(rb) * (n_assets + max_pairs) * repeats
    print("=" * 72)
    print(f"Datasets: {dataset_ids}")
    print(f"Holding periods: {holding_periods} days")
    print(f"Formation look-back: {config.get('formation_lookback_days')} trading days")
    print(f"Min formation days: {min_formation}")
    print(f"Total rebalances: {total_rebalances}")
    print(f"Repeats per rebalance: {repeats}")
    print(f"Estimated NVIDIA NIM calls: {total_calls}")
    print("=" * 72)
    if args.dry_run:
        return

    api_key = os.getenv(str(config["api_key_env"]))
    if not api_key and not args.skip_llm:
        raise ValueError(f"environment variable {config['api_key_env']} is not set")

    model = dict(config["models"][0])
    all_errors: list[dict[str, Any]] = []

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        print(f"\n=== {dataset_id} ===", flush=True)
        trailing, panel_2025, metadata, provenance = load_dataset_returns(config, dataset, output_root)
        atomic_json(output_root / dataset_id / "data" / "provenance.json", provenance)
        dataset_root = output_root / dataset_id
        for hp in holding_periods:
            print(f"\n--- {dataset_id}: {hp}-day rebalance ---", flush=True)
            result = run_holding_period(
                hp, trailing, panel_2025, metadata, model, config, dataset_root,
                workers, retry_calls, args.force_views, args.skip_llm,
            )
            all_errors.extend([{"Dataset": dataset_id, **e} for e in result["errors"]])
            save_holding_results(dataset_root, hp, result, config)

    atomic_json(output_root / "run_errors.json", all_errors)

    summary_rows = []
    for dataset in dataset_ids:
        for hp in holding_periods:
            mp = output_root / dataset / f"holding_{hp}" / "method_metrics.csv"
            if mp.exists():
                frame = pd.read_csv(mp)
                frame.insert(0, "Dataset", dataset)
                frame.insert(1, "Holding_Days", hp)
                summary_rows.append(frame)
    if summary_rows:
        summary_dir = output_root / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)
        full = pd.concat(summary_rows, ignore_index=True)
        full.to_csv(summary_dir / "all_method_metrics.csv", index=False)
        save_frame(full, summary_dir / "all_method_metrics")

    print(f"\nSaved walk-forward backtest artifacts under {output_root}", flush=True)


if __name__ == "__main__":
    main()