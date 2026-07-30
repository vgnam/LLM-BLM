"""Collect absolute-return views for the original LLM-BLM with DeepSeek.

Each asset is queried repeatedly. The sample mean becomes Q and the sample
variance becomes the corresponding diagonal element of Omega, matching the
original LLM-BLM formulation while using the same provider as RelView-BL.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm import tqdm

from collect_relative_views import (
    DEFAULT_MODEL,
    OPENCODE_GO_BASE_URL,
    _load_json_mapping,
    _load_returns,
    _load_universe,
    _metadata_lookup,
    completion_request_options,
)
from env_utils import load_env_file


def make_absolute_prompt(
    ticker: str,
    returns: list[float],
    metadata: Mapping[str, Any],
    context: Any,
    horizon_days: int,
) -> tuple[str, str]:
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


def parse_absolute_response(content: str) -> float:
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
    if abs(value) > 0.25:
        raise ValueError(
            "expected_return magnitude exceeds 25% per day; the model likely returned percentage units"
        )
    return value


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
) -> dict[str, dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the openai package to collect LLM views") from error

    client = OpenAI(base_url=base_url, api_key=api_key)
    result: dict[str, dict[str, Any]] = {}
    for ticker in tqdm(returns.columns.astype(str), desc="Absolute LLM views"):
        system, user = make_absolute_prompt(
            ticker,
            returns[ticker].astype(float).tolist(),
            metadata.get(ticker, {}),
            context.get(ticker, {}),
            horizon_days,
        )
        samples: list[float] = []
        errors: list[str] = []
        def call_once() -> float:
            completion = client.chat.completions.create(**completion_request_options(
                model, system, user, temperature, thinking
            ))
            return parse_absolute_response(completion.choices[0].message.content)

        attempts = 0
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        while len(samples) < repeats and attempts < repeats + retry_calls:
            batch_size = min(
                workers,
                repeats - len(samples),
                repeats + retry_calls - attempts,
            )
            attempts += batch_size
            futures: Any = (
                [executor.submit(call_once) for _ in range(batch_size)]
                if executor else [None] * batch_size
            )
            for future in (as_completed(futures) if executor else futures):
                try:
                    samples.append(future.result() if executor else call_once())
                except Exception as error:
                    errors.append(str(error))
        if executor:
            executor.shutdown(wait=True)
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
            **dict(metadata.get(ticker, {})),
        }
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
    )
    successful = sum(bool(item["expected_return"]) for item in result.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"Saved absolute views for {successful}/{len(result)} assets to {args.output}")


if __name__ == "__main__":
    main()
