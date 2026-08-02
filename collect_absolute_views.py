"""Collect absolute-return views for the original LLM-BLM with DeepSeek.

Each asset is queried repeatedly. The sample mean becomes Q and the sample
variance becomes the corresponding diagonal element of Omega, matching the
original LLM-BLM formulation while using the same provider as RelView-BL.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm import tqdm

from collect_relative_views import (
    DEFAULT_MODEL,
    OPENCODE_GO_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    ProviderUsageLimitError,
    atomic_checkpoint_json,
    provider_limit_from_errors,
    provider_region_from_errors,
    thinking_body_supported,
    _load_json_mapping,
    _load_returns,
    _load_universe,
    _metadata_lookup,
    completion_request_options,
)
from env_utils import load_env_file
from prompt_ensemble import (
    ENSEMBLE_NAME,
    collect_repeated_calls,
    diversified_system_prompt,
    prompt_sha256,
)


def make_absolute_prompt(
    ticker: str,
    returns: list[float],
    metadata: Mapping[str, Any],
    context: Any,
    horizon_days: int,
    prompt_mode: str = "generic",
) -> tuple[str, str]:
    if prompt_mode == "paper_v2":
        if not isinstance(context, Mapping):
            raise ValueError("paper_v2 context must be a mapping")
        required = {"reference_date", "sector_returns", "market_returns"}
        missing = sorted(required - set(context))
        if missing:
            raise ValueError(f"paper_v2 context is missing: {missing}")
        company_name = metadata.get("Security", metadata.get("Name", ticker))
        sector = metadata.get("GICS Sector", metadata.get("Sector", "Unknown"))
        sub_industry = metadata.get(
            "GICS Sub-Industry", metadata.get("Sub-Industry", "Unknown")
        )
        system = (
            f"You are providing analysis on {context['reference_date']}. Predict the average daily "
            "return for the next two weeks based on the information provided about a stock's past "
            "performance. You will receive the stock's daily returns from the past two weeks, its "
            "GICS sector, the sector's daily returns, the S&P 500's daily returns, and company "
            "information. Analyze the time-series data, sector and market performance, and company "
            "sector and sub-industry. Predict the average daily return for the next two weeks. "
            "Return JSON with exactly one numeric field named expected_return and no commentary. "
            "The input returns and expected_return are percentage values: for example, -0.36 means "
            "-0.36%, not -36%."
        )
        user = json.dumps({
            "Daily Returns": [100.0 * float(value) for value in returns],
            "Company Sector": sector,
            "Sector Returns": [float(value) for value in context["sector_returns"]],
            "Market Returns": [float(value) for value in context["market_returns"]],
            "Company Information": {
                "Ticker": ticker,
                "Company Name": company_name,
                "GICS Sector": sector,
                "GICS Sub-Industry": sub_industry,
            },
        }, ensure_ascii=False)
        return system, user
    if prompt_mode != "generic":
        raise ValueError("prompt_mode must be generic or paper_v2")
    system = (
        "You generate a strictly point-in-time absolute equity return view using only supplied information. "
        "Predict the asset's average daily arithmetic return over the requested future horizon. Return JSON "
        "with exactly one numeric field named expected_return. Express it as a decimal daily return: for "
        "example, 0.002 means +0.2% per day and -0.001 means -0.1% per day. Do not output a percentage, "
        "annualized return, explanation, markdown, or reasoning."
    )
    user = json.dumps({
        "ticker": ticker,
        "horizon_days": horizon_days,
        "metadata": dict(metadata),
        "point_in_time_context": context,
        "historical_daily_returns": returns,
    }, ensure_ascii=False)
    return system, user


def parse_absolute_response(content: str, percentage_units: bool = False) -> float:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    value = float(json.loads(text)["expected_return"])
    if not np.isfinite(value):
        raise ValueError("expected_return must be finite")
    if percentage_units:
        return value / 100.0
    if abs(value) > 0.25:
        raise ValueError(
            "expected_return magnitude exceeds 25% per day; the model likely returned percentage units"
        )
    return value


def absolute_checkpoint_item_complete(
    item: Mapping[str, Any],
    repeats: int,
    model: str,
    thinking: str,
    temperature: float,
    prompt_ensemble: bool,
    prompt_mode: str,
) -> bool:
    return (
        len(item.get("expected_return", [])) == repeats
        and int(item.get("successful_repeats", -1)) == repeats
        and item.get("model") == model
        and item.get("thinking") == thinking
        and float(item.get("temperature", -1)) == float(temperature)
        and item.get("prompt_ensemble")
        == (ENSEMBLE_NAME if prompt_ensemble else "single_prompt")
        and item.get("prompt_mode") == prompt_mode
        and (
            not prompt_ensemble
            or len(item.get("system_prompt_sha256", [])) == repeats
        )
    )


def collect_absolute_views(
    returns,
    model: str,
    base_url: str,
    api_key: str,
    metadata: Mapping[str, Mapping[str, str]],
    context: Mapping[str, Any],
    repeats: int = 30,
    horizon_days: int = 10,
    temperature: float = 0.3,
    thinking: str = "disabled",
    workers: int = 1,
    retry_calls: int = 15,
    prompt_ensemble: bool = False,
    prompt_mode: str = "generic",
    checkpoint_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package to collect LLM views") from error

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )
    checkpoint: dict[str, Any] = {}
    if checkpoint_path and checkpoint_path.exists():
        try:
            value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint = value if isinstance(value, dict) else {}
        except Exception:
            checkpoint = {}
    result: dict[str, dict[str, Any]] = {}
    for ticker in tqdm(returns.columns.astype(str), desc="Absolute LLM views"):
        existing = checkpoint.get(ticker, {})
        if isinstance(existing, Mapping) and absolute_checkpoint_item_complete(
            existing, repeats, model, thinking, temperature,
            prompt_ensemble, prompt_mode,
        ):
            result[ticker] = dict(existing)
            continue
        base_system, user = make_absolute_prompt(
            ticker,
            returns[ticker].astype(float).tolist(),
            metadata.get(ticker, {}),
            context.get(ticker, {}),
            horizon_days,
            prompt_mode,
        )
        def call_once(call_id: int) -> dict[str, Any]:
            reference_date = (
                context.get(ticker, {}).get("reference_date", "undated")
                if isinstance(context.get(ticker, {}), Mapping) else "undated"
            )
            system = (
                diversified_system_prompt(
                    base_system, call_id, f"absolute:{ticker}:{reference_date}"
                )
                if prompt_ensemble else base_system
            )
            completion = client.chat.completions.create(**completion_request_options(
                model, system, user, temperature, thinking,
                thinking_body_supported(model, base_url),
                4096 if "gpt-oss" in model.lower() else 1024,
            ))
            return {
                "value": parse_absolute_response(
                    completion.choices[0].message.content,
                    percentage_units=prompt_mode == "paper_v2",
                ),
                "system_prompt_sha256": prompt_sha256(system),
            }

        calls, errors, attempts = collect_repeated_calls(
            call_once, repeats, workers, retry_calls
        )
        samples = [item["value"] for _, item in calls]
        variant_ids = [call_id for call_id, _ in calls]
        prompt_hashes = [item["system_prompt_sha256"] for _, item in calls]
        if not samples and errors and any("401" in error or "AuthError" in error for error in errors):
            raise RuntimeError("OpenCode Go authentication failed; no output file was written")
        result[ticker] = {
            "ticker": ticker,
            "pct_change": returns[ticker].astype(float).tolist(),
            "expected_return": samples,
            "successful_repeats": len(samples),
            "attempted_calls": attempts,
            "errors": errors,
            "horizon_days": horizon_days,
            "model": model,
            "thinking": thinking,
            "temperature": temperature,
            "prompt_ensemble": ENSEMBLE_NAME if prompt_ensemble else "single_prompt",
            "prompt_mode": prompt_mode,
            "stored_return_units": "decimal_daily_return",
            "model_output_units": "percentage_daily_return" if prompt_mode == "paper_v2" else "decimal_daily_return",
            "prompt_variant_ids": variant_ids if prompt_ensemble else [],
            "system_prompt_sha256": prompt_hashes if prompt_ensemble else [],
            **dict(metadata.get(ticker, {})),
        }
        if checkpoint_path:
            atomic_checkpoint_json(checkpoint_path, result)
        if len(samples) < repeats:
            provider_region = provider_region_from_errors(errors)
            if provider_region:
                raise provider_region
            provider_limit = provider_limit_from_errors(errors)
            if provider_limit:
                raise provider_limit
            raise ProviderUsageLimitError(
                f"Asset {ticker} produced {len(samples)}/{repeats} successful calls; "
                "retrying from its checkpoint",
                60,
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect absolute DeepSeek views for LLM-BLM")
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url", default=os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL)
    )
    parser.add_argument("--api-key-env", default="OPENCODE_GO_API_KEY")
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent calls within each asset")
    parser.add_argument("--retry-calls", type=int, default=15, help="Extra attempts used to replace invalid responses")
    parser.add_argument(
        "--wait-on-rate-limit",
        action="store_true",
        help="On a provider 429/usage limit, sleep for the reported delay and resume from the checkpoint",
    )
    parser.add_argument("--prompt-ensemble", action="store_true", help="Use a unique system prompt for every call")
    parser.add_argument(
        "--prompt-mode", choices=["generic", "paper_v2"], default="generic",
        help="paper_v2 reproduces the prompt inputs and percentage units described in arXiv v2",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 to estimate view variance")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.retry_calls < 0:
        raise ValueError("--retry-calls cannot be negative")
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"environment variable {args.api_key_env} is not set")
    returns = _load_returns(args.returns)
    universe = _load_universe(args.universe)
    missing = [ticker for ticker in universe if ticker not in returns.columns]
    if missing:
        raise ValueError(f"universe assets missing from returns: {missing}")
    returns = returns[universe]
    metadata_frame = None
    if args.metadata:
        import pandas as pd
        metadata_frame = pd.read_csv(args.metadata)
    metadata = _metadata_lookup(metadata_frame)
    context = _load_json_mapping(args.context)
    while True:
        try:
            result = collect_absolute_views(
                returns,
                args.model,
                args.base_url,
                api_key,
                metadata,
                context,
                args.repeats,
                args.horizon_days,
                args.temperature,
                args.thinking,
                args.workers,
                args.retry_calls,
                args.prompt_ensemble,
                args.prompt_mode,
                args.output,
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
    successful = sum(bool(item["expected_return"]) for item in result.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved absolute views for {successful}/{len(result)} assets to {args.output}")


if __name__ == "__main__":
    main()
