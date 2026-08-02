"""Run three-dataset NVIDIA NIM portfolio comparisons at a fixed 2025 cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


DEFAULT_CONFIG = Path("experiments/nvidia_nim_2025_2026/config.json")
BASELINE_METHODS = ("BL", "MVO", "EW")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root", type=Path, default=Path("experiments/nvidia_nim_2025_2026")
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-calls", type=int, default=6)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--force-views", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Split existing summary tables and regenerate per-dataset plots without API calls",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def save_frame(frame: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    frame.to_parquet(base.with_suffix(".parquet"), index=False)


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "source_experiment_root", "dataset_manifest", "datasets", "formation_start",
        "cutoff_date", "test_start", "test_end", "base_url", "api_key_env", "models",
        "repeats", "minimum_successful_calls", "max_pairs", "temperature", "calibration",
        "abstention_threshold", "tau", "risk_aversion", "market_risk_aversion",
        "max_weight", "transaction_cost_bps", "annual_risk_free_rate",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing required fields: {missing}")
    if pd.Timestamp(config["cutoff_date"]) >= pd.Timestamp(config["test_start"]):
        raise ValueError("cutoff_date must precede test_start")
    if int(config["repeats"]) < 2:
        raise ValueError("repeats must be at least two to estimate LLM view variance")
    if int(config["minimum_successful_calls"]) > int(config["repeats"]):
        raise ValueError("minimum_successful_calls cannot exceed repeats")
    slugs = [str(item["slug"]) for item in config["models"]]
    if len(slugs) != len(set(slugs)):
        raise ValueError("model slugs must be unique")


def load_manifest_datasets(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = read_json(Path(str(config["dataset_manifest"])))
    by_id = {str(item["id"]): item for item in manifest["datasets"]}
    selected = [str(item) for item in config["datasets"]]
    missing = [item for item in selected if item not in by_id]
    if missing:
        raise ValueError(f"dataset manifest is missing: {missing}")
    if len(selected) != 3:
        raise ValueError("this experiment requires exactly three datasets")
    return [by_id[item] for item in selected]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(
    config: Mapping[str, Any], dataset: Mapping[str, Any], output_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dataset_id = str(dataset["id"])
    source = Path(str(config["source_experiment_root"])) / dataset_id
    formation_path = source / "data" / "formation_returns.csv"
    realized_path = source / "data" / "realized_returns.csv"
    metadata_path = source / "metadata.csv"
    for path in (formation_path, realized_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"missing source input: {path}")

    formation = pd.read_csv(formation_path, parse_dates=["Date"]).set_index("Date")
    realized = pd.read_csv(realized_path, parse_dates=["Date"]).set_index("Date")
    metadata = pd.read_csv(metadata_path)
    assets = [str(item["ticker"]) for item in dataset["assets"]]
    if list(formation.columns) != assets or list(realized.columns) != assets:
        raise ValueError(f"{dataset_id}: asset columns do not match the manifest")

    formation_start = pd.Timestamp(config["formation_start"])
    cutoff = pd.Timestamp(config["cutoff_date"])
    test_start = pd.Timestamp(config["test_start"])
    test_end = pd.Timestamp(config["test_end"])
    if formation.empty or formation.index.min() < formation_start or formation.index.max() > cutoff:
        raise ValueError(f"{dataset_id}: formation data are outside the configured 2025 window")
    if realized.empty or realized.index.min() < test_start or realized.index.max() > test_end:
        raise ValueError(f"{dataset_id}: realized data are outside the configured 2026 window")

    target_data = output_root / dataset_id / "data"
    target_data.mkdir(parents=True, exist_ok=True)
    for path in (formation_path, realized_path, metadata_path, source / "universe.json"):
        if path.exists():
            shutil.copy2(path, target_data / path.name)
    provenance = {
        "dataset": dataset_id,
        "formation_source": str(formation_path),
        "formation_sha256": sha256_file(formation_path),
        "realized_source": str(realized_path),
        "realized_sha256": sha256_file(realized_path),
        "formation_rows": len(formation),
        "actual_formation_start": str(formation.index.min().date()),
        "actual_formation_end": str(formation.index.max().date()),
        "realized_rows": len(realized),
        "actual_test_start": str(realized.index.min().date()),
        "actual_test_end": str(realized.index.max().date()),
    }
    atomic_json(target_data / "provenance.json", provenance)
    return formation, realized, metadata, provenance


def model_methods(model: Mapping[str, Any]) -> tuple[str, str]:
    slug = str(model["slug"])
    return f"BLM_LLM__{slug}", f"RelViewBL__{slug}"


def method_inventory(config: Mapping[str, Any], statuses: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"Method": "BL", "Family": "Black-Litterman", "Uses_LLM": False,
         "Requested_Model": "", "API_Model": "", "Model_Status": "not_applicable"},
        {"Method": "MVO", "Family": "Mean-Variance Optimization", "Uses_LLM": False,
         "Requested_Model": "", "API_Model": "", "Model_Status": "not_applicable"},
        {"Method": "EW", "Family": "Equal Weight", "Uses_LLM": False,
         "Requested_Model": "", "API_Model": "", "Model_Status": "not_applicable"},
    ]
    status_by_slug = {str(item["slug"]): item for item in statuses}
    for model in config["models"]:
        absolute_method, relative_method = model_methods(model)
        status = status_by_slug[str(model["slug"])]
        common = {
            "Uses_LLM": True,
            "Requested_Model": model["requested_model"],
            "API_Model": model["api_model"],
            "Model_Status": status["status"],
        }
        rows.extend([
            {"Method": absolute_method, "Family": "BLM-LLM", **common},
            {"Method": relative_method, "Family": "RelViewBL", **common},
        ])
    return pd.DataFrame(rows)


def check_models(config: Mapping[str, Any], skip_llm: bool) -> list[dict[str, Any]]:
    if skip_llm:
        return [
            {**dict(item), "status": "skipped", "http_status": None,
             "detail": "LLM collection disabled by --skip-llm"}
            for item in config["models"]
        ]
    api_key = os.getenv(str(config["api_key_env"]))
    if not api_key:
        raise ValueError(f"environment variable {config['api_key_env']} is not set")
    from openai import OpenAI

    client = OpenAI(
        base_url=str(config["base_url"]), api_key=api_key, timeout=60, max_retries=0
    )
    statuses: list[dict[str, Any]] = []
    for model in config["models"]:
        try:
            completion = client.chat.completions.create(
                model=str(model["api_model"]),
                messages=[
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": "Return {\"status\":\"ok\"}."},
                ],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("provider returned no answer content")
            statuses.append({**dict(model), "status": "available", "http_status": 200,
                             "detail": "NVIDIA NIM preflight succeeded"})
        except Exception as error:
            statuses.append({
                **dict(model),
                "status": "unavailable",
                "http_status": getattr(error, "status_code", None),
                "detail": str(error)[:1000],
            })
    return statuses


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


def cache_is_complete(
    absolute_path: Path, relative_path: Path, assets: list[str], config: Mapping[str, Any]
) -> bool:
    if not absolute_path.exists() or not relative_path.exists():
        return False
    try:
        absolute = read_json(absolute_path)
        relative = read_json(relative_path)
        minimum = int(config["minimum_successful_calls"])
        return (
            all(len(absolute.get(asset, {}).get("expected_return", [])) >= minimum for asset in assets)
            and len(relative.get("views", [])) == int(config["max_pairs"])
            and all(
                item.get("status") == "ok"
                and int(item.get("successful_repeats", 0)) >= minimum
                for item in relative["views"]
            )
        )
    except Exception:
        return False


def collect_model_views(
    dataset_root: Path,
    formation: pd.DataFrame,
    metadata_frame: pd.DataFrame,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    workers: int,
    retry_calls: int,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    api_key = str(os.environ[str(config["api_key_env"])])
    assets = formation.columns.astype(str).tolist()
    response_root = dataset_root / "responses" / str(model["slug"])
    absolute_path = response_root / "absolute_views.json"
    relative_path = response_root / "relative_views.json"
    if not force and cache_is_complete(absolute_path, relative_path, assets, config):
        return read_json(absolute_path), read_json(relative_path)["views"]
    if force:
        absolute_checkpoint = None
        relative_checkpoint = None
    else:
        absolute_checkpoint = absolute_path
        relative_checkpoint = relative_path
    metadata = _metadata_lookup(metadata_frame)
    horizon_days = len(pd.read_csv(dataset_root / "data" / "realized_returns.csv"))
    absolute = collect_absolute_views(
        formation, str(model["api_model"]), str(config["base_url"]), api_key,
        metadata, {}, int(config["repeats"]), horizon_days, float(config["temperature"]),
        "disabled", workers, retry_calls, False, "generic", absolute_checkpoint,
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
        retry_calls, False, "calibrated", relative_checkpoint,
    )
    payload = {
        "requested_model": model["requested_model"],
        "model": model["api_model"],
        "base_url": config["base_url"],
        "cutoff_date": config["cutoff_date"],
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
    diagnostics = {
        "calibration_method": relview.calibrator.fitted_method,
        "accepted_view_count": len(relview.matrices.accepted_views),
        "rejected_view_count": len(relview.matrices.rejected_views),
        "accepted_views": relview.matrices.accepted_views,
        "rejected_views": relview.matrices.rejected_views,
        "raw_cycle_count": relview.matrices.raw_cycle_count,
        "consistency_rmse": relview.matrices.consistency_rmse,
        "latent_scores": relview.matrices.latent_scores.to_dict(),
    }
    return absolute_portfolio, relview.weights.reindex(assets).to_numpy(dtype=float), diagnostics


def extended_metrics(
    daily: pd.DataFrame, metrics: dict[str, float | int], risk_free_rate: float
) -> dict[str, float | int]:
    values = daily["Portfolio_Return"].to_numpy(dtype=float)
    daily_rf = risk_free_rate / 252.0
    downside = np.minimum(values - daily_rf, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(252.0))
    annualized_excess = float((np.mean(values) - daily_rf) * 252.0)
    var_95 = float(np.quantile(values, 0.05))
    tail = values[values <= var_95]
    result = dict(metrics)
    result.update({
        "final_nav": float(1.0 + float(metrics["cumulative_return"])),
        "mean_daily_return": float(np.mean(values)),
        "best_day": float(np.max(values)),
        "worst_day": float(np.min(values)),
        "positive_day_ratio": float(np.mean(values > 0.0)),
        "annualized_downside_deviation": downside_deviation,
        "sortino": annualized_excess / downside_deviation if downside_deviation > 0 else 0.0,
        "calmar": (
            float(metrics["annualized_return"]) / abs(float(metrics["max_drawdown"]))
            if float(metrics["max_drawdown"]) < 0 else 0.0
        ),
        "daily_var_95": var_95,
        "daily_cvar_95": float(np.mean(tail)) if len(tail) else var_95,
    })
    return result


def evaluate_method(
    dataset_id: str,
    method: str,
    weights: np.ndarray,
    realized: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assets = realized.columns.astype(str).tolist()
    daily, base_metrics = evaluate_realized_portfolio(
        realized,
        pd.Series(weights, index=assets),
        annual_risk_free_rate=float(config["annual_risk_free_rate"]),
        turnover=1.0,
        transaction_cost_bps=float(config["transaction_cost_bps"]),
    )
    daily = daily.rename(columns={"Portfolio_Return": "Daily_Return"})
    daily["NAV"] = 1.0 + daily["Cumulative_Return"]
    daily.insert(0, "Method", method)
    daily.insert(0, "Dataset", dataset_id)
    weights_frame = pd.DataFrame({
        "Dataset": dataset_id, "Method": method, "Asset": assets, "Weight": weights,
    })
    metric_values = extended_metrics(
        daily.rename(columns={"Daily_Return": "Portfolio_Return"}),
        base_metrics,
        float(config["annual_risk_free_rate"]),
    )
    return daily, weights_frame, {"Dataset": dataset_id, "Method": method,
                                  "Status": "success", **metric_values}


def plot_nav(daily: pd.DataFrame, title: str, output: Path) -> None:
    if daily.empty:
        return
    fig, axis = plt.subplots(figsize=(12, 7))
    style_by_method: dict[str, dict[str, Any]] = {
        "BL": {
            "color": "tab:blue", "linestyle": (0, (4, 2)),
            "linewidth": 2.2, "zorder": 10,
        },
        "MVO": {"color": "tab:orange", "linewidth": 1.8, "zorder": 3},
        "EW": {"color": "tab:green", "linewidth": 1.8, "zorder": 3},
    }
    groups = {method: group for method, group in daily.groupby("Method", sort=False)}
    # Draw BL last. If it exactly overlaps another method, its dashed segments
    # remain visible while the underlying solid line is still identifiable.
    method_order = [method for method in groups if method != "BL"]
    if "BL" in groups:
        method_order.append("BL")
    nav_by_method: dict[str, np.ndarray] = {}
    for method in method_order:
        group = groups[method]
        ordered = group.sort_values("Date")
        nav_by_method[method] = ordered["NAV"].to_numpy(dtype=float)
        if method.startswith("BLM_LLM__"):
            style = {"color": "tab:red", "linewidth": 2.0, "zorder": 4}
        elif method.startswith("RelViewBL__"):
            style = {"color": "tab:purple", "linewidth": 2.0, "zorder": 4}
        else:
            style = style_by_method.get(method, {})
        axis.plot(
            pd.to_datetime(ordered["Date"]), ordered["NAV"], label=method,
            linewidth=style.get("linewidth", 1.8),
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            zorder=style.get("zorder", 3),
        )
    overlaps: list[str] = []
    methods = list(nav_by_method)
    for index, left in enumerate(methods):
        for right in methods[index + 1:]:
            if len(nav_by_method[left]) == len(nav_by_method[right]) and np.allclose(
                nav_by_method[left], nav_by_method[right], rtol=1e-10, atol=1e-12
            ):
                overlaps.append(f"{left} = {right}")
    axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel("NAV (initial capital = 1)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    if overlaps:
        axis.text(
            0.01, 0.01, "Overlapping NAV: " + "; ".join(overlaps),
            transform=axis.transAxes, fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
        )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_method_artifacts(
    dataset_root: Path, method: str, daily: pd.DataFrame,
    weights: pd.DataFrame, metrics: Mapping[str, Any],
) -> None:
    root = dataset_root / "results" / "by_method" / method
    save_frame(daily, root / "daily_nav")
    save_frame(weights, root / "weights")
    atomic_json(root / "metrics.json", dict(metrics))


def save_per_dataset_summaries(
    output_root: Path,
    dataset_ids: list[str],
    daily: pd.DataFrame,
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Write self-contained result tables and a NAV plot for every dataset."""
    for dataset_id in dataset_ids:
        dataset_root = output_root / dataset_id
        dataset_daily = daily[daily["Dataset"] == dataset_id].copy()
        dataset_weights = weights[weights["Dataset"] == dataset_id].copy()
        dataset_metrics = metrics[metrics["Dataset"] == dataset_id].copy()
        if dataset_daily.empty or dataset_weights.empty or dataset_metrics.empty:
            raise ValueError(f"existing summaries are incomplete for {dataset_id}")
        save_frame(dataset_daily, dataset_root / "results" / "daily_returns_nav")
        save_frame(dataset_weights, dataset_root / "results" / "weights")
        save_frame(dataset_metrics, dataset_root / "results" / "method_metrics")
        plot_nav(
            dataset_daily,
            f"{dataset_id}: 2026 portfolio NAV",
            dataset_root / "plots" / "nav_2026.png",
        )


def postprocess_existing(output_root: Path, dataset_ids: list[str]) -> None:
    summary_root = output_root / "summary"
    daily = pd.read_csv(summary_root / "daily_returns_nav_long.csv")
    weights = pd.read_csv(summary_root / "weights_long.csv")
    metrics = pd.read_csv(summary_root / "method_metrics.csv")
    save_per_dataset_summaries(output_root, dataset_ids, daily, weights, metrics)
    print(f"Regenerated per-dataset metrics, returns, weights, and plots under {output_root}")


def aggregate_successful(
    daily: pd.DataFrame, dataset_count: int, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for method, group in daily.groupby("Method"):
        if group["Dataset"].nunique() != dataset_count:
            continue
        averaged = group.groupby("Date", as_index=True)["Daily_Return"].mean().to_frame()
        synthetic = averaged.rename(columns={"Daily_Return": "Portfolio_Return"})
        evaluated, base = evaluate_realized_portfolio(
            synthetic, {"Portfolio_Return": 1.0},
            annual_risk_free_rate=float(config["annual_risk_free_rate"]),
        )
        evaluated = evaluated.rename(columns={"Portfolio_Return": "Daily_Return"})
        evaluated["NAV"] = 1.0 + evaluated["Cumulative_Return"]
        evaluated.insert(0, "Method", method)
        evaluated.insert(0, "Dataset", "equal_dataset_aggregate")
        parts.append(evaluated)
        metrics = extended_metrics(
            evaluated.rename(columns={"Daily_Return": "Portfolio_Return"}),
            base,
            float(config["annual_risk_free_rate"]),
        )
        metric_rows.append({"Dataset": "equal_dataset_aggregate", "Method": method,
                            "Status": "success", **metrics})
    return (
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
        pd.DataFrame(metric_rows),
    )


def write_report(
    output_root: Path, statuses: list[dict[str, Any]], metrics: pd.DataFrame,
    aggregate_metrics: pd.DataFrame, provenance: list[dict[str, Any]],
) -> None:
    lines = [
        "# NVIDIA NIM portfolio experiment results", "", "## Data windows", "",
        "Views and portfolio weights use only 2025 observations. Fixed weights are evaluated "
        "on the available 2026 sessions; no 2026 return enters a view or optimizer input.", "",
    ]
    for item in provenance:
        lines.append(
            f"- `{item['dataset']}`: {item['formation_rows']} formation rows "
            f"({item['actual_formation_start']} to {item['actual_formation_end']}), "
            f"{item['realized_rows']} test rows ({item['actual_test_start']} to "
            f"{item['actual_test_end']})."
        )
    lines.extend(["", "## NVIDIA NIM model status", "",
                  "| Requested model | API model | Status | HTTP |", "|---|---|---|---:|"])
    for item in statuses:
        lines.append(
            f"| `{item['requested_model']}` | `{item['api_model']}` | "
            f"{item['status']} | {item.get('http_status') or ''} |"
        )
    lines.extend(["", "Unavailable models were not substituted.", "", "## Results", ""])
    successful = metrics[metrics["Status"] == "success"].copy()
    if not successful.empty:
        table = successful[["Dataset", "Method", "cumulative_return", "sharpe", "max_drawdown"]]
        lines.extend(["Per-dataset metrics are saved in full under `summary/`.", "", "```text",
                      table.to_string(index=False), "```", ""])
    if not aggregate_metrics.empty:
        table = aggregate_metrics[["Method", "cumulative_return", "sharpe", "max_drawdown"]]
        lines.extend(["### Equal-dataset aggregate", "", "```text",
                      table.to_string(index=False), "```", ""])
    lines.extend([
        "## Interpretation guardrails", "",
        "The LLM calls were made after the 2026 test window, with recognizable asset names. "
        "Therefore the LLM methods can be affected by training-cutoff or provider-side temporal "
        "leakage even though the numeric pipeline is point-in-time. Results are research outputs, "
        "not evidence of deployable live performance.", "",
    ])
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retry_calls < 0:
        raise ValueError("workers must be positive and retry-calls non-negative")
    config = read_json(args.config)
    validate_config(config)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = load_manifest_datasets(config)
    dataset_ids = [str(item["id"]) for item in datasets]
    if args.postprocess_only:
        postprocess_existing(output_root, dataset_ids)
        return
    statuses = check_models(config, args.skip_llm)
    atomic_json(output_root / "model_status.json", {
        "base_url": config["base_url"], "models": statuses,
    })
    inventory = method_inventory(config, statuses)
    save_frame(inventory, output_root / "summary" / "method_inventory")

    available_slugs = {
        str(item["slug"]) for item in statuses if item["status"] == "available"
    }
    all_daily: list[pd.DataFrame] = []
    all_weights: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    run_errors: list[dict[str, Any]] = []

    for dataset in datasets:
        dataset_id = str(dataset["id"])
        print(f"\n=== {dataset_id} ===", flush=True)
        formation, realized, metadata, provenance = load_dataset(config, dataset, output_root)
        provenance_rows.append(provenance)
        dataset_root = output_root / dataset_id
        weights_by_method, prior, covariance = baseline_weights(formation, config)

        for model in config["models"]:
            absolute_method, relative_method = model_methods(model)
            if str(model["slug"]) not in available_slugs:
                continue
            try:
                absolute, views = collect_model_views(
                    dataset_root, formation, metadata, model, config,
                    args.workers, args.retry_calls, args.force_views,
                )
                absolute_portfolio, relative_portfolio, diagnostics = llm_weights(
                    formation, absolute, views, prior, covariance, config
                )
                weights_by_method[absolute_method] = absolute_portfolio
                weights_by_method[relative_method] = relative_portfolio
                atomic_json(
                    dataset_root / "results" / f"relview_diagnostics__{model['slug']}.json",
                    diagnostics,
                )
            except Exception as error:
                detail = str(error)[:2000]
                print(f"{model['api_model']} failed for {dataset_id}: {detail}", flush=True)
                run_errors.append({"Dataset": dataset_id, "Model": model["api_model"],
                                   "Status": "failed", "Detail": detail})

        for method, weights in weights_by_method.items():
            daily, weights_frame, metrics = evaluate_method(
                dataset_id, method, weights, realized, config
            )
            all_daily.append(daily)
            all_weights.append(weights_frame)
            metric_rows.append(metrics)
            save_method_artifacts(dataset_root, method, daily, weights_frame, metrics)

        completed = set(weights_by_method)
        for row in inventory.to_dict(orient="records"):
            method = str(row["Method"])
            if method not in completed:
                metric_rows.append({
                    "Dataset": dataset_id,
                    "Method": method,
                    "Status": row["Model_Status"] if row["Model_Status"] != "available" else "failed",
                })
        dataset_daily = pd.concat(
            [item for item in all_daily if item["Dataset"].iloc[0] == dataset_id],
            ignore_index=True,
        )
        plot_nav(dataset_daily, f"{dataset_id}: 2026 portfolio NAV",
                 dataset_root / "plots" / "nav_2026.png")

    daily_all = pd.concat(all_daily, ignore_index=True)
    weights_all = pd.concat(all_weights, ignore_index=True)
    metrics_all = pd.DataFrame(metric_rows)
    summary_root = output_root / "summary"
    save_frame(daily_all, summary_root / "daily_returns_nav_long")
    save_frame(weights_all, summary_root / "weights_long")
    save_frame(metrics_all, summary_root / "method_metrics")
    save_per_dataset_summaries(
        output_root, dataset_ids, daily_all, weights_all, metrics_all
    )
    daily_wide = daily_all.pivot(index="Date", columns=["Dataset", "Method"], values="NAV")
    daily_wide.columns = [f"{dataset}__{method}" for dataset, method in daily_wide.columns]
    save_frame(daily_wide.reset_index(), summary_root / "nav_wide")

    aggregate_daily, aggregate_metrics = aggregate_successful(
        daily_all, len(datasets), config
    )
    if not aggregate_daily.empty:
        save_frame(aggregate_daily, summary_root / "equal_dataset_daily_nav")
        save_frame(aggregate_metrics, summary_root / "equal_dataset_metrics")
        plot_nav(aggregate_daily, "Equal-dataset aggregate: 2026 NAV",
                 output_root / "plots" / "aggregate_nav_2026.png")
    atomic_json(output_root / "run_errors.json", run_errors)
    catalog = {
        "experiment_id": config.get("experiment_id"),
        "config": config,
        "security": "The NVIDIA API key is read from the environment and is not persisted.",
        "dataset_count": len(datasets),
        "datasets": [item["id"] for item in datasets],
        "method_count_requested": len(inventory),
        "successful_method_dataset_rows": int((metrics_all["Status"] == "success").sum()),
        "model_status_file": "model_status.json",
        "tables": {
            "method_inventory": "summary/method_inventory.{csv,parquet}",
            "daily_returns_and_nav": "summary/daily_returns_nav_long.{csv,parquet}",
            "weights": "summary/weights_long.{csv,parquet}",
            "metrics": "summary/method_metrics.{csv,parquet}",
            "aggregate_daily": "summary/equal_dataset_daily_nav.{csv,parquet}",
            "aggregate_metrics": "summary/equal_dataset_metrics.{csv,parquet}",
        },
        "metric_definitions": {
            "cumulative_return": "Final compounded return after the one-time entry cost",
            "annualized_return": "Geometrically annualized return using 252 trading days",
            "annualized_volatility": "Sample daily volatility times sqrt(252)",
            "sharpe": "Annualized mean excess return divided by daily volatility",
            "max_drawdown": "Minimum peak-to-trough NAV decline",
            "sortino": "Annualized mean excess return divided by annualized downside deviation",
            "calmar": "Annualized return divided by absolute maximum drawdown",
            "daily_var_95": "Fifth percentile of daily net returns",
            "daily_cvar_95": "Mean daily net return at or below daily_var_95",
            "positive_day_ratio": "Fraction of test sessions with a positive net return",
            "turnover": "One-way absolute entry turnover; fixed at 1 for a fully invested new portfolio",
        },
    }
    atomic_json(summary_root / "data_catalog.json", catalog)
    write_report(output_root, statuses, metrics_all, aggregate_metrics, provenance_rows)
    print(f"\nSaved experiment artifacts under {output_root}", flush=True)


if __name__ == "__main__":
    main()
