"""Prepare the point-in-time panels and 10-day windows described by the paper."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_monthly_returns import download_chart_close


DEFAULT_CONFIG = Path("experiments/paper_reproduction/config.json")
DEFAULT_ROOT = Path("experiments/paper_reproduction/paper_sp500_top50")
DEFAULT_METADATA_SOURCE = Path("responses/gemma_2024-06-01_2024-06-30.json")

SECTOR_PROXIES = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def top_market_cap_universe(caps: dict[str, Any], size: int) -> list[str]:
    valid = [
        (str(ticker), float(value))
        for ticker, value in caps.items()
        if value is not None and float(value) > 0
    ]
    return [ticker for ticker, _ in sorted(valid, key=lambda item: (-item[1], item[0]))[:size]]


def extract_metadata(
    source: dict[str, Any],
    assets: list[str],
    source_tickers: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ticker in assets:
        source_ticker = (source_tickers or {}).get(ticker, ticker)
        item = source.get(source_ticker)
        if not isinstance(item, dict):
            raise ValueError(f"metadata source is missing {ticker}")
        rows.append({
            "Symbol": ticker,
            "Security": str(item.get("Security", ticker)),
            "GICS Sector": str(item.get("GICS Sector", "Unknown")),
            "GICS Sub-Industry": str(item.get("GICS Sub-Industry", "Unknown")),
        })
    return rows


def complete_windows(
    dates: pd.DatetimeIndex,
    phase: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int,
    holding_days: int,
    consume_phase_lookback: bool,
) -> list[dict[str, Any]]:
    positions = [index for index, date in enumerate(dates) if start <= date <= end]
    if consume_phase_lookback:
        positions = positions[lookback_days:]
    windows: list[dict[str, Any]] = []
    for offset in range(0, len(positions), holding_days):
        realized_positions = positions[offset:offset + holding_days]
        if len(realized_positions) != holding_days:
            break
        realized_start = realized_positions[0]
        formation_positions = list(range(realized_start - lookback_days, realized_start))
        if len(formation_positions) != lookback_days or formation_positions[0] < 0:
            continue
        number = len(windows) + 1
        windows.append({
            "period_id": f"{phase}_{number:02d}",
            "phase": phase,
            "reference_date": dates[formation_positions[-1]].strftime("%Y-%m-%d"),
            "formation_start": dates[formation_positions[0]].strftime("%Y-%m-%d"),
            "formation_end": dates[formation_positions[-1]].strftime("%Y-%m-%d"),
            "test_start": dates[realized_positions[0]].strftime("%Y-%m-%d"),
            "test_end": dates[realized_positions[-1]].strftime("%Y-%m-%d"),
            "formation_positions": formation_positions,
            "realized_positions": realized_positions,
        })
    return windows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--market-caps", type=Path, default=Path("market_caps.json"))
    parser.add_argument("--metadata-source", type=Path, default=DEFAULT_METADATA_SOURCE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    snapshot = load_json(Path(config["universe_snapshot_file"]))
    snapshot_assets = snapshot["assets"]
    assets = [str(item["ticker"]) for item in snapshot_assets]
    if len(assets) != int(config["universe_size"]):
        raise ValueError("universe snapshot size does not match config")
    source_tickers = {str(item["ticker"]): str(item["source_ticker"]) for item in snapshot_assets}
    metadata_rows = extract_metadata(load_json(args.metadata_source), assets, source_tickers)
    caps_all = {
        str(item["ticker"]): float(item["index_weight_percent"])
        for item in snapshot_assets
    }
    sectors = sorted({row["GICS Sector"] for row in metadata_rows})
    missing_proxies = sorted(set(sectors) - set(SECTOR_PROXIES))
    if missing_proxies:
        raise ValueError(f"no sector proxy for: {missing_proxies}")

    root = args.root
    data_dir = root / "data"
    periods_dir = root / "periods"
    data_dir.mkdir(parents=True, exist_ok=True)
    (root / "universe.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")
    atomic_json(root / "market_caps.json", {ticker: float(caps_all[ticker]) for ticker in assets})
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    panel_path = data_dir / "adjusted_close.csv"
    requested = assets + [str(config["market_proxy"])] + [SECTOR_PROXIES[sector] for sector in sectors]
    requested = list(dict.fromkeys(requested))
    if panel_path.exists() and not args.overwrite:
        close = pd.read_csv(panel_path, parse_dates=["Date"]).set_index("Date")
        if any(ticker not in close.columns for ticker in requested):
            raise ValueError("cached adjusted_close.csv does not contain the configured proxies")
    else:
        close = download_chart_close(
            requested,
            pd.Timestamp(config["data_start"]) - pd.Timedelta(days=21),
            pd.Timestamp(config["data_end"]) + pd.Timedelta(days=2),
        ).reindex(columns=requested)
        close.index.name = "Date"
        close.reset_index().to_csv(panel_path, index=False)

    all_returns = close.pct_change(fill_method=None)
    stock_returns = all_returns[assets]
    stock_returns = stock_returns[
        (stock_returns.index >= pd.Timestamp(config["data_start"]))
        & (stock_returns.index <= pd.Timestamp(config["data_end"]))
    ].dropna(axis=0, how="any")
    market_returns = all_returns[str(config["market_proxy"])].reindex(stock_returns.index)
    sector_returns = pd.DataFrame({
        sector: all_returns[SECTOR_PROXIES[sector]].reindex(stock_returns.index)
        for sector in sectors
    })
    if market_returns.isna().any() or sector_returns.isna().any().any():
        raise RuntimeError("market or sector proxy has missing values on stock trading dates")
    stock_returns.index.name = "Date"
    stock_returns.reset_index().to_csv(data_dir / "stock_returns.csv", index=False)
    market_returns.rename("Market_Return").to_csv(data_dir / "market_returns.csv", header=True)
    sector_returns.to_csv(data_dir / "sector_returns.csv")

    dates = pd.DatetimeIndex(stock_returns.index)
    lookback = int(config["lookback_trading_days"])
    holding = int(config["holding_trading_days"])
    windows = complete_windows(
        dates, "validation", pd.Timestamp(config["validation_start"]),
        pd.Timestamp(config["validation_end"]), lookback, holding, True,
    ) + complete_windows(
        dates, "test", pd.Timestamp(config["test_start"]),
        pd.Timestamp(config["test_end"]), lookback, holding, False,
    )
    metadata_by_ticker = {row["Symbol"]: row for row in metadata_rows}
    period_records: list[dict[str, Any]] = []
    for window in windows:
        formation = stock_returns.iloc[window.pop("formation_positions")]
        realized = stock_returns.iloc[window.pop("realized_positions")]
        period_root = periods_dir / window["period_id"]
        period_root.mkdir(parents=True, exist_ok=True)
        formation.reset_index().to_csv(period_root / "formation_returns.csv", index=False)
        realized.reset_index().to_csv(period_root / "realized_returns.csv", index=False)
        context = {
            ticker: {
                "reference_date": window["reference_date"],
                "sector_returns": (
                    100.0 * sector_returns.loc[formation.index, metadata_by_ticker[ticker]["GICS Sector"]]
                ).astype(float).tolist(),
                "market_returns": (
                    100.0 * market_returns.loc[formation.index]
                ).astype(float).tolist(),
            }
            for ticker in assets
        }
        atomic_json(period_root / "context.json", context)
        period_records.append(window)
    pd.DataFrame(period_records).to_csv(root / "periods.csv", index=False)

    generated = [
        root / "universe.json", root / "market_caps.json", root / "metadata.csv",
        data_dir / "stock_returns.csv", data_dir / "market_returns.csv",
        data_dir / "sector_returns.csv", root / "periods.csv",
    ]
    manifest = {
        "config": config,
        "universe": assets,
        "universe_count": len(assets),
        "sectors": sectors,
        "sector_proxies": {sector: SECTOR_PROXIES[sector] for sector in sectors},
        "period_counts": pd.DataFrame(period_records)["phase"].value_counts().to_dict(),
        "files": {str(path.relative_to(root)): sha256(path) for path in generated},
        "universe_snapshot": snapshot,
        "known_deviations": [
            "nearest public top-50 reconstruction is March 2025 month-end, not exact paper date 2025-03-26",
            "sector construction is unspecified by paper; Select Sector SPDR proxies are used",
        ],
    }
    atomic_json(root / "data_manifest.json", manifest)
    print(
        f"Prepared {len(assets)} assets and {len(period_records)} periods at {root}; "
        f"counts={manifest['period_counts']}"
    )


if __name__ == "__main__":
    main()
