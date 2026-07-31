"""Collect sparse pairwise LLM views for RelView-BL.

Example:
    py collect_relative_views.py --returns yfinance/returns_2024-06-01_2024-06-30.csv \
        --universe universe.json --repeats 30 \
        --output responses_relative/deepseek-v4-flash_2024-06.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from tqdm import tqdm

from env_utils import load_env_file
from prompt_ensemble import (
    ENSEMBLE_NAME,
    collect_repeated_calls,
    diversified_system_prompt,
    prompt_sha256,
)
from relview_bl import select_candidate_pairs


OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
REQUEST_TIMEOUT_SECONDS = 45.0


class ProviderUsageLimitError(RuntimeError):
    """Provider quota/rate limit with a machine-readable retry delay."""

    def __init__(self, message: str, retry_after_seconds: int = 60):
        super().__init__(message)
        self.retry_after_seconds = max(60, int(retry_after_seconds))


class ProviderRegionOptInError(RuntimeError):
    """Provider requires an account-level data-region opt-in before calls can continue."""


def provider_limit_from_errors(errors: list[str]) -> ProviderUsageLimitError | None:
    limited = [
        error for error in errors
        if "429" in error or "RateLimitError" in error or "GoUsageLimitError" in error
    ]
    if not limited:
        return None
    message = limited[0]
    hours = re.search(r"Resets in\s+(\d+)hr", message, flags=re.IGNORECASE)
    minutes = re.search(r"(?:\d+hr\s+)?(\d+)min", message, flags=re.IGNORECASE)
    seconds = 60
    if hours or minutes:
        seconds = (
            3600 * (int(hours.group(1)) if hours else 0)
            + 60 * (int(minutes.group(1)) if minutes else 0)
            + 60
        )
    return ProviderUsageLimitError(message, seconds)


def provider_region_from_errors(errors: list[str]) -> ProviderRegionOptInError | None:
    matches = [
        error for error in errors
        if "RegionError" in error
        or "requires explicit opt in" in error
        or ("403" in error and "hosted in China" in error)
    ]
    return ProviderRegionOptInError(matches[0]) if matches else None


def atomic_checkpoint_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def relative_checkpoint_item_complete(
    item: Mapping[str, Any],
    repeats: int,
    prompt_ensemble: bool,
    prompt_mode: str,
) -> bool:
    return (
        item.get("status") == "ok"
        and int(item.get("successful_repeats", -1)) == repeats
        and item.get("prompt_mode") == prompt_mode
        and item.get("prompt_ensemble")
        == (ENSEMBLE_NAME if prompt_ensemble else "single_prompt")
        and (
            not prompt_ensemble
            or len(item.get("system_prompt_sha256", [])) == repeats
        )
    )


def _load_returns(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "Date" in frame.columns:
        frame = frame.set_index("Date")
    return frame.select_dtypes(include=[np.number]).dropna(axis=1, how="any")


def _load_json_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_universe(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value: Any = json.load(handle)
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


def _metadata_lookup(metadata: pd.DataFrame | None) -> dict[str, dict[str, str]]:
    if metadata is None:
        return {}
    symbol_column = next((item for item in ("Symbol", "symbol", "ticker") if item in metadata.columns), None)
    if symbol_column is None:
        raise ValueError("metadata needs a Symbol, symbol, or ticker column")
    result: dict[str, dict[str, str]] = {}
    for _, row in metadata.iterrows():
        symbol = str(row[symbol_column])
        result[symbol] = {
            str(column): str(row[column])
            for column in metadata.columns
            if column != symbol_column and pd.notna(row[column])
        }
    return result


def make_pairwise_prompt(
    asset_a: str,
    asset_b: str,
    returns: pd.DataFrame,
    metadata: Mapping[str, Mapping[str, str]],
    context: Mapping[str, Any],
    horizon_days: int,
    prompt_mode: str = "calibrated",
) -> tuple[str, str]:
    if prompt_mode == "calibrated":
        system = (
            "You generate strictly point-in-time pairwise equity views. Compare only the two supplied assets "
            "using only the supplied information. Return JSON with preferred_asset, probability, and evidence. "
            "probability is the probability (0.50 to 1.00) that preferred_asset outperforms the other asset. "
            "If evidence is weak, keep probability near 0.50. Do not predict an absolute return."
        )
    elif prompt_mode in {"decisive_v1", "decisive_v2", "decisive_v3"}:
        fixed_rule = (
            " Apply this fixed decision rule before any stylistic lens: compare relative cumulative "
            "return and mean daily return (50% weight), downside-adjusted consistency (25%), relative "
            "sector/market behavior (15%), and company/industry context (10%). If the weighted evidence "
            "is tied, prefer the asset with the higher cumulative return; if still tied, use the ticker "
            "that is lexicographically first."
            if prompt_mode in {"decisive_v2", "decisive_v3"} else ""
        )
        system = (
            "You generate a forced, strictly point-in-time pairwise ranking using only the supplied "
            "information. You must select one of the two assets as more likely to outperform over the "
            "requested horizon. Return JSON with preferred_asset, probability, and evidence. The probability "
            "must be in [0.55, 0.95]: use 0.55-0.59 for a weak but measurable edge, 0.60-0.69 for a moderate "
            "edge, 0.70-0.79 for a strong edge, and 0.80-0.95 only for unusually convergent evidence. Never "
            "return 0.50-0.5499. Do not invent evidence or use future information. Do not predict an absolute "
            "return. This is a confidence-forcing experimental prompt; the number is a ranking score and must "
            "not be presented as an externally calibrated real-world probability. Evidence must be a JSON "
            "array of at most two short strings, each no more than 12 words."
            + fixed_rule
        )
    else:
        raise ValueError(
            "prompt_mode must be calibrated, decisive_v1, decisive_v2, or decisive_v3"
        )
    context_a = context.get(asset_a, {})
    context_b = context.get(asset_b, {})
    percentage_context = all(
        isinstance(item, Mapping)
        and "sector_returns" in item
        and "market_returns" in item
        for item in (context_a, context_b)
    )
    return_scale = 100.0 if percentage_context else 1.0
    payload = {
        "horizon_days": horizon_days,
        "asset_a": {
            "ticker": asset_a,
            "metadata": dict(metadata.get(asset_a, {})),
            "point_in_time_context": context_a,
            "daily_returns": (return_scale * returns[asset_a].astype(float)).tolist(),
        },
        "asset_b": {
            "ticker": asset_b,
            "metadata": dict(metadata.get(asset_b, {})),
            "point_in_time_context": context_b,
            "daily_returns": (return_scale * returns[asset_b].astype(float)).tolist(),
        },
        "return_units": "percentage points" if percentage_context else "decimal return",
    }
    return system, json.dumps(payload, ensure_ascii=False)


def parse_pairwise_response(
    content: str,
    asset_a: str,
    asset_b: str,
    minimum_probability: float = 0.5,
    maximum_probability: float = 1.0,
) -> dict[str, Any]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    value = json.loads(text)
    preferred = str(value["preferred_asset"])
    probability = float(value["probability"])
    if preferred not in (asset_a, asset_b):
        raise ValueError(f"preferred_asset must be {asset_a} or {asset_b}")
    if not minimum_probability <= probability <= maximum_probability:
        raise ValueError(
            f"probability must be in [{minimum_probability}, {maximum_probability}]"
        )
    evidence = value.get("evidence", [])
    if isinstance(evidence, str):
        evidence = [evidence]
    return {"preferred_asset": preferred, "probability": probability, "evidence": list(evidence)}


def aggregate_repeated_predictions(
    predictions: list[dict[str, Any]],
    asset_a: str,
    asset_b: str,
    probability_estimator: str = "mean",
) -> dict[str, Any]:
    """Estimate pair probability from repeated independent LLM calls.

    ``votes`` uses the empirical selection frequency. ``mean`` averages the
    confidence reported by the model after orienting every call as P(A > B).
    """

    if probability_estimator not in {"votes", "mean"}:
        raise ValueError("probability_estimator must be votes or mean")
    if not predictions:
        raise ValueError("at least one successful prediction is required")

    vote_samples_a = np.asarray(
        [float(item["preferred_asset"] == asset_a) for item in predictions], dtype=float
    )
    reported_samples_a = np.asarray([
        float(item["probability"])
        if item["preferred_asset"] == asset_a
        else 1.0 - float(item["probability"])
        for item in predictions
    ], dtype=float)
    vote_probability_a = float(np.mean(vote_samples_a))
    mean_reported_probability_a = float(np.mean(reported_samples_a))
    if probability_estimator == "votes":
        probability_a = vote_probability_a
        uncertainty_samples_a = vote_samples_a
    else:
        probability_a = mean_reported_probability_a
        uncertainty_samples_a = reported_samples_a

    preferred = asset_a if probability_a >= 0.5 else asset_b
    preferred_samples = (
        uncertainty_samples_a if preferred == asset_a else 1.0 - uncertainty_samples_a
    )
    evidence = list(dict.fromkeys(
        str(piece)
        for item in predictions
        for piece in item.get("evidence", [])
        if str(piece).strip()
    ))
    return {
        "asset_a": asset_a,
        "asset_b": asset_b,
        "preferred_asset": preferred,
        "probability": max(probability_a, 1.0 - probability_a),
        "probability_a": probability_a,
        "probability_samples_a": uncertainty_samples_a.tolist(),
        "probability_samples": preferred_samples.tolist(),
        "probability_estimator": probability_estimator,
        "vote_probability_a": vote_probability_a,
        "mean_reported_probability_a": mean_reported_probability_a,
        "votes": {
            asset_a: int(np.sum(vote_samples_a)),
            asset_b: int(len(vote_samples_a) - np.sum(vote_samples_a)),
        },
        "reported_probability_samples_a": reported_samples_a.tolist(),
        "evidence": evidence,
        "successful_repeats": len(predictions),
        "prompt_variant_ids": [
            int(item["prompt_variant_id"])
            for item in predictions
            if "prompt_variant_id" in item
        ],
        "system_prompt_sha256": [
            str(item["system_prompt_sha256"])
            for item in predictions
            if "system_prompt_sha256" in item
        ],
    }


def completion_request_options(
    model: str,
    system: str,
    user: str,
    temperature: float,
    thinking: str = "disabled",
) -> dict[str, Any]:
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("thinking must be enabled or disabled")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
        # DeepSeek V4 defaults to thinking mode, so this must be explicit.
        "extra_body": {"thinking": {"type": thinking}},
    }


def collect_pairwise_views(
    returns: pd.DataFrame,
    pairs: list[tuple[str, str]],
    model: str,
    base_url: str | None,
    api_key: str,
    metadata: Mapping[str, Mapping[str, str]],
    context: Mapping[str, Any],
    repeats: int,
    horizon_days: int,
    temperature: float,
    probability_estimator: str = "mean",
    thinking: str = "disabled",
    workers: int = 1,
    retry_calls: int = 15,
    prompt_ensemble: bool = False,
    prompt_mode: str = "calibrated",
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package to collect LLM views") from error

    client = (
        OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
        if base_url
        else OpenAI(
            api_key=api_key,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    )
    checkpoint_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint_views = (
                checkpoint.get("views", []) if isinstance(checkpoint, dict) else checkpoint
            )
            checkpoint_by_pair = {
                (str(item["asset_a"]), str(item["asset_b"])): item
                for item in checkpoint_views
                if isinstance(item, dict) and "asset_a" in item and "asset_b" in item
            }
        except Exception:
            checkpoint_by_pair = {}
    views: list[dict[str, Any]] = []
    for asset_a, asset_b in tqdm(pairs, desc="Pairwise LLM views"):
        existing = checkpoint_by_pair.get((asset_a, asset_b))
        if existing and relative_checkpoint_item_complete(
            existing, repeats, prompt_ensemble, prompt_mode
        ):
            views.append(existing)
            continue
        base_system, user = make_pairwise_prompt(
            asset_a, asset_b, returns, metadata, context, horizon_days, prompt_mode
        )
        def call_once(call_id: int) -> dict[str, Any]:
            reference_date = (
                context.get(asset_a, {}).get("reference_date", "undated")
                if isinstance(context.get(asset_a, {}), Mapping) else "undated"
            )
            system = (
                diversified_system_prompt(
                    base_system,
                    call_id,
                    f"relative:{asset_a}:{asset_b}:{reference_date}",
                ) if prompt_ensemble else base_system
            )
            completion = client.chat.completions.create(**completion_request_options(
                model, system, user, temperature, thinking
            ))
            parsed = parse_pairwise_response(
                completion.choices[0].message.content,
                asset_a,
                asset_b,
                0.55 if prompt_mode.startswith("decisive_") else 0.5,
                0.95 if prompt_mode.startswith("decisive_") else 1.0,
            )
            if prompt_ensemble:
                parsed["prompt_variant_id"] = call_id
                parsed["system_prompt_sha256"] = prompt_sha256(system)
            return parsed

        calls, errors, attempts = collect_repeated_calls(
            call_once, repeats, workers, retry_calls
        )
        predictions = [item for _, item in calls]
        if not predictions and errors and any("401" in error or "AuthError" in error for error in errors):
            raise RuntimeError("OpenCode Go authentication failed; no output file was written")
        if not predictions:
            item = {
                "asset_a": asset_a, "asset_b": asset_b,
                "successful_repeats": 0, "attempted_calls": attempts,
                "errors": errors, "status": "failed",
                "temperature": temperature,
                "prompt_ensemble": ENSEMBLE_NAME if prompt_ensemble else "single_prompt",
                "prompt_mode": prompt_mode,
            }
        else:
            aggregated = aggregate_repeated_predictions(
                predictions, asset_a, asset_b, probability_estimator
            )
            item = {
                **aggregated,
                "horizon_days": horizon_days,
                "attempted_calls": attempts,
                "errors": errors,
                "status": "ok",
                "temperature": temperature,
                "prompt_ensemble": ENSEMBLE_NAME if prompt_ensemble else "single_prompt",
                "prompt_mode": prompt_mode,
            }
        views.append(item)
        if checkpoint_path:
            atomic_checkpoint_json(checkpoint_path, {"views": views})
        if len(predictions) < repeats:
            provider_region = provider_region_from_errors(errors)
            if provider_region:
                raise provider_region
            provider_limit = provider_limit_from_errors(errors)
            if provider_limit:
                raise provider_limit
            raise ProviderUsageLimitError(
                f"Pair {asset_a}/{asset_b} produced {len(predictions)}/{repeats} "
                "successful calls; retrying from its checkpoint",
                60,
            )
    return views


def load_explicit_pairs(path: Path, available_assets: set[str]) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".csv":
        value: Any = pd.read_csv(path).to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle) if path.suffix.lower() == ".json" else None
        if isinstance(value, dict):
            value = value.get("pairs", value.get("views", []))
    if not isinstance(value, list):
        raise ValueError("explicit pairs must be a CSV or a JSON object containing a pairs list")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            asset_a, asset_b = str(item["asset_a"]), str(item["asset_b"])
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            asset_a, asset_b = str(item[0]), str(item[1])
        else:
            raise ValueError(f"invalid explicit pair: {item}")
        if asset_a == asset_b or asset_a not in available_assets or asset_b not in available_assets:
            raise ValueError(f"pair ({asset_a}, {asset_b}) is invalid for the return universe")
        pairs.append((asset_a, asset_b))
    return list(dict.fromkeys(pairs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect sparse relative LLM views for RelView-BL")
    parser.add_argument("--returns", type=Path, required=True, help="CSV of point-in-time daily returns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Provider model identifier")
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument("--api-key-env", default="OPENCODE_GO_API_KEY")
    parser.add_argument(
        "--thinking",
        choices=["disabled", "enabled"],
        default="disabled",
        help="DeepSeek thinking mode (disabled by default)",
    )
    parser.add_argument("--metadata", type=Path, help="Optional company metadata CSV")
    parser.add_argument("--context", type=Path, help="Optional point-in-time news/earnings JSON keyed by ticker")
    parser.add_argument("--market-caps", type=Path, help="Optional ticker-to-market-cap JSON")
    parser.add_argument("--universe", type=Path, help="Optional JSON/CSV/TXT asset universe")
    parser.add_argument("--pairs", type=Path, help="Optional curated pair CSV/JSON (competitors or shared events)")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--min-abs-correlation", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=30, help="Independent calls per pair")
    parser.add_argument(
        "--probability-estimator",
        choices=["mean", "votes"],
        default="mean",
        help="mean: average oriented call probabilities; votes: selection frequency",
    )
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent calls within each pair")
    parser.add_argument("--retry-calls", type=int, default=15, help="Extra attempts used to replace invalid responses")
    parser.add_argument("--prompt-ensemble", action="store_true", help="Use a unique system prompt for every call")
    parser.add_argument(
        "--prompt-mode", choices=["calibrated", "decisive_v1", "decisive_v2", "decisive_v3"], default="calibrated",
        help="decisive modes force a ranking score of at least 0.55 and are not probability-calibrated",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"environment variable {args.api_key_env} is not set")
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to estimate disagreement")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.retry_calls < 0:
        raise ValueError("--retry-calls cannot be negative")
    returns = _load_returns(args.returns)
    if args.universe:
        universe = _load_universe(args.universe)
        missing = [asset for asset in universe if asset not in returns.columns]
        if missing:
            raise ValueError(f"universe assets missing from returns: {missing}")
        returns = returns[universe]
    metadata_frame = pd.read_csv(args.metadata) if args.metadata else None
    metadata = _metadata_lookup(metadata_frame)
    context = _load_json_mapping(args.context)
    market_caps = _load_json_mapping(args.market_caps)
    if args.pairs:
        pairs = load_explicit_pairs(args.pairs, set(returns.columns))[:args.max_pairs]
    else:
        pairs = select_candidate_pairs(
            returns,
            metadata_frame,
            market_caps,
            max_pairs=args.max_pairs,
            min_abs_correlation=args.min_abs_correlation,
        )
    views = collect_pairwise_views(
        returns,
        pairs,
        args.model,
        args.base_url,
        api_key,
        metadata,
        context,
        args.repeats,
        args.horizon_days,
        args.temperature,
        args.probability_estimator,
        args.thinking,
        args.workers,
        args.retry_calls,
        args.prompt_ensemble,
        args.prompt_mode,
    )
    payload = {
        "model": args.model,
        "source_returns": str(args.returns),
        "horizon_days": args.horizon_days,
        "repeats": args.repeats,
        "probability_estimator": args.probability_estimator,
        "thinking": args.thinking,
        "temperature": args.temperature,
        "prompt_ensemble": ENSEMBLE_NAME if args.prompt_ensemble else "single_prompt",
        "prompt_mode": args.prompt_mode,
        "pairs_requested": len(pairs),
        "views": views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved {sum(item.get('status') == 'ok' for item in views)} views to {args.output}")


if __name__ == "__main__":
    main()
