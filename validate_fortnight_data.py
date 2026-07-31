"""Validate prepared fortnight data and, optionally, completed response caches/results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_paper_reproduction import sha256
from prompt_ensemble import ENSEMBLE_NAME


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--require-results", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    config = load_json(args.config)
    manifest = load_json(args.root / "data_manifest.json")
    assets = [str(item) for item in load_json(args.root / "universe.json")]
    caps = load_json(args.root / "market_caps.json")
    metadata = pd.read_csv(args.root / "metadata.csv")
    periods = pd.read_csv(args.root / "periods.csv")
    if len(assets) != len(set(assets)):
        errors.append("universe contains duplicates")
    if set(assets) != set(caps):
        errors.append("market caps do not match universe")
    if set(assets) != set(metadata["Symbol"].astype(str)):
        errors.append("metadata does not match universe")
    expected_counts = {"validation": 5, "test": 20}
    actual_counts = periods["phase"].value_counts().to_dict()
    if actual_counts != expected_counts:
        errors.append(f"period counts {actual_counts} != {expected_counts}")
    for relative, expected_hash in manifest.get("files", {}).items():
        path = args.root / relative
        if not path.exists():
            errors.append(f"missing hashed file {relative}")
        elif sha256(path) != expected_hash:
            errors.append(f"hash mismatch {relative}")

    expected_days = int(config["lookback_trading_days"])
    expected_holding = int(config["holding_trading_days"])
    for period in periods.itertuples(index=False):
        period_root = args.root / "periods" / period.period_id
        try:
            formation = pd.read_csv(period_root / "formation_returns.csv", parse_dates=["Date"])
            realized = pd.read_csv(period_root / "realized_returns.csv", parse_dates=["Date"])
            context = load_json(period_root / "context.json")
        except Exception as error:
            errors.append(f"{period.period_id} unreadable: {error}")
            continue
        if len(formation) != expected_days or len(realized) != expected_holding:
            errors.append(
                f"{period.period_id} lengths formation={len(formation)} realized={len(realized)}"
            )
        if list(formation.columns[1:]) != assets or list(realized.columns[1:]) != assets:
            errors.append(f"{period.period_id} asset order mismatch")
        if formation["Date"].max() >= realized["Date"].min():
            errors.append(f"{period.period_id} formation overlaps realized data")
        if str(formation["Date"].max().date()) != str(period.reference_date):
            errors.append(f"{period.period_id} reference date mismatch")
        if set(context) != set(assets):
            errors.append(f"{period.period_id} context universe mismatch")
        for asset in assets:
            item = context.get(asset, {})
            if (
                len(item.get("sector_returns", [])) != expected_days
                or len(item.get("market_returns", [])) != expected_days
                or item.get("reference_date") != str(period.reference_date)
            ):
                errors.append(f"{period.period_id}/{asset} context mismatch")
                break

    if args.run_root:
        repeats = int(args.repeats or config["comparison_repeats"])
        expected_ensemble = ENSEMBLE_NAME if config["prompt_ensemble"] else "single_prompt"
        for period in periods.itertuples(index=False):
            absolute_path = args.run_root / "responses_absolute" / f"{period.period_id}.json"
            relative_path = args.run_root / "responses_relative" / f"{period.period_id}.json"
            if not absolute_path.exists():
                errors.append(f"missing absolute response {period.period_id}")
            else:
                absolute = load_json(absolute_path)
                for asset in assets:
                    item = absolute.get(asset, {})
                    if (
                        len(item.get("expected_return", [])) != repeats
                        or item.get("prompt_mode") != config["absolute_prompt_mode"]
                        or item.get("prompt_ensemble") != expected_ensemble
                    ):
                        errors.append(f"invalid absolute response {period.period_id}/{asset}")
                        break
            if not relative_path.exists():
                errors.append(f"missing relative response {period.period_id}")
            else:
                relative = load_json(relative_path)
                views = relative.get("views", [])
                if (
                    len(views) != int(config["max_pairs"])
                    or relative.get("prompt_mode") != config["relative_prompt_mode"]
                    or relative.get("prompt_ensemble") != expected_ensemble
                    or any(int(item.get("successful_repeats", 0)) != repeats for item in views)
                ):
                    errors.append(f"invalid relative response {period.period_id}")
        if args.require_results:
            for name in (
                "daily_returns", "daily_returns_long", "period_metrics",
                "weights_long", "method_metrics",
            ):
                for suffix in (".csv", ".parquet"):
                    if not (args.run_root / "results" / f"{name}{suffix}").exists():
                        errors.append(f"missing result {name}{suffix}")
            if not (args.run_root / "results" / "summary.json").exists():
                errors.append("missing result summary.json")

    print(json.dumps({
        "root": str(args.root),
        "universe_count": len(assets),
        "period_counts": actual_counts,
        "errors": errors,
        "error_count": len(errors),
    }, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
