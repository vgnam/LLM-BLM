"""Resume-safe monthly collection for same-model Absolute and RelView views."""

from __future__ import annotations

import argparse
import calendar
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from collect_absolute_views import collect_absolute_views
from collect_relative_views import (
    DEFAULT_MODEL,
    OPENCODE_GO_BASE_URL,
    _load_json_mapping,
    _load_returns,
    _load_universe,
    _metadata_lookup,
    collect_pairwise_views,
)
from env_utils import load_env_file
from relview_bl import select_candidate_pairs


def monthly_periods(start_month: str, end_month: str) -> list[tuple[str, str, str]]:
    starts = pd.date_range(f"{start_month}-01", f"{end_month}-01", freq="MS")
    result = []
    for start in starts:
        end_day = calendar.monthrange(start.year, start.month)[1]
        result.append((start.strftime("%Y-%m"), start.strftime("%Y-%m-%d"), f"{start:%Y-%m}-{end_day:02d}"))
    return result


def valid_absolute(path: Path, universe: list[str], minimum: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return all(
            ticker in value and len(value[ticker].get("expected_return", [])) >= minimum
            for ticker in universe
        )
    except Exception:
        return False


def valid_relative(path: Path, pair_count: int, minimum: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        valid = [
            item for item in value.get("views", [])
            if item.get("status") == "ok" and int(item.get("successful_repeats", 0)) >= minimum
        ]
        return len(valid) >= pair_count
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect monthly Absolute and RelView DeepSeek views")
    parser.add_argument("--start-month", default="2024-06")
    parser.add_argument("--end-month", default="2025-05")
    parser.add_argument("--returns-dir", type=Path, default=Path("yfinance"))
    parser.add_argument("--universe", type=Path, default=Path("universe.json"))
    parser.add_argument("--market-caps", type=Path, default=Path("market_caps.json"))
    parser.add_argument("--absolute-dir", type=Path, default=Path("responses"))
    parser.add_argument("--relative-dir", type=Path, default=Path("responses_relative"))
    parser.add_argument("--methods", nargs="+", choices=["absolute", "relative"], default=["absolute", "relative"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.getenv("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL))
    parser.add_argument("--api-key-env", default="OPENCODE_GO_API_KEY")
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--min-successful-calls", type=int, default=20)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--retry-calls", type=int, default=15)
    parser.add_argument("--horizon-days", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-pairs", type=int, default=45)
    parser.add_argument("--probability-estimator", choices=["mean", "votes"], default="mean")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_env_file()
    args = parse_args()
    if not 1 <= args.min_successful_calls <= args.repeats:
        raise ValueError("min-successful-calls must be in [1, repeats]")
    universe = _load_universe(args.universe)
    periods = monthly_periods(args.start_month, args.end_month)
    tasks: list[tuple[str, str, str, Path]] = []
    for month, start, end in periods:
        returns_path = args.returns_dir / f"returns_{start}_{end}.csv"
        if not returns_path.exists():
            raise FileNotFoundError(f"missing formation returns: {returns_path}")
        absolute_path = args.absolute_dir / f"{args.model}_{start}_{end}.json"
        relative_path = args.relative_dir / f"{args.model}_{month}.json"
        if "absolute" in args.methods and (args.force or not valid_absolute(
            absolute_path, universe, args.min_successful_calls
        )):
            tasks.append(("absolute", month, f"{start}_{end}", returns_path))
        if "relative" in args.methods and (args.force or not valid_relative(
            relative_path, args.max_pairs, args.min_successful_calls
        )):
            tasks.append(("relative", month, f"{start}_{end}", returns_path))

    estimated_calls = sum(
        len(universe) * args.repeats if method == "absolute" else args.max_pairs * args.repeats
        for method, _, _, _ in tasks
    )
    print(f"Pending tasks: {len(tasks)}; estimated API calls: {estimated_calls}")
    for method, month, _, _ in tasks:
        print(f"  {month}: {method}")
    if args.dry_run or not tasks:
        return

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"environment variable {args.api_key_env} is not set")
    metadata_frame = pd.read_csv(args.metadata) if args.metadata else None
    metadata = _metadata_lookup(metadata_frame)
    context: dict[str, Any] = _load_json_mapping(args.context)
    market_caps = _load_json_mapping(args.market_caps)
    args.absolute_dir.mkdir(parents=True, exist_ok=True)
    args.relative_dir.mkdir(parents=True, exist_ok=True)

    for task_number, (method, month, date_range, returns_path) in enumerate(tasks, start=1):
        start, end = date_range.split("_")
        returns = _load_returns(returns_path)
        missing = [ticker for ticker in universe if ticker not in returns.columns]
        if missing:
            raise ValueError(f"{returns_path} is missing universe assets: {missing}")
        returns = returns[universe]
        print(f"[{task_number}/{len(tasks)}] Collecting {method} views for {month}")
        if method == "absolute":
            output = args.absolute_dir / f"{args.model}_{start}_{end}.json"
            value = collect_absolute_views(
                returns, args.model, args.base_url, api_key, metadata, context,
                args.repeats, args.horizon_days, args.temperature, args.thinking,
                args.workers, args.retry_calls,
            )
            incomplete = [
                ticker for ticker in universe
                if len(value[ticker].get("expected_return", [])) < args.min_successful_calls
            ]
            if incomplete:
                raise RuntimeError(
                    f"absolute collection below min-successful-calls for {incomplete}; output not written"
                )
        else:
            output = args.relative_dir / f"{args.model}_{month}.json"
            pairs = select_candidate_pairs(
                returns, metadata_frame, market_caps, max_pairs=args.max_pairs
            )
            views = collect_pairwise_views(
                returns, pairs, args.model, args.base_url, api_key, metadata, context,
                args.repeats, args.horizon_days, args.temperature,
                args.probability_estimator, args.thinking, args.workers,
                args.retry_calls,
            )
            value = {
                "model": args.model,
                "source_returns": str(returns_path),
                "horizon_days": args.horizon_days,
                "repeats": args.repeats,
                "probability_estimator": args.probability_estimator,
                "thinking": args.thinking,
                "pairs_requested": len(pairs),
                "views": views,
            }
            incomplete = [
                f"{item.get('asset_a')}/{item.get('asset_b')}"
                for item in views
                if item.get("status") != "ok"
                or int(item.get("successful_repeats", 0)) < args.min_successful_calls
            ]
            if incomplete:
                raise RuntimeError(
                    f"relative collection below min-successful-calls for {incomplete}; output not written"
                )
        temporary = output.with_suffix(f"{output.suffix}.tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(output)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
