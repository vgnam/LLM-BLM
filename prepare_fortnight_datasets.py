"""Prepare five disjoint 15-asset datasets on the paper's fortnight calendar."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from prepare_monthly_returns import download_chart_close
from prepare_paper_reproduction import (
    SECTOR_PROXIES,
    atomic_json,
    complete_windows,
    extract_metadata,
    load_json,
    sha256,
)


DEFAULT_CONFIG = Path("experiments/fortnight_5_datasets/config.json")
DEFAULT_MANIFEST = Path("experiments/fortnight_5_datasets/datasets.json")
DEFAULT_ROOT = Path("experiments/fortnight_5_datasets")
DEFAULT_METADATA_SOURCE = Path("responses/gemma_2024-06-01_2024-06-30.json")
DEFAULT_PREVIOUS_MANIFEST = Path("experiments/post_release_2026/datasets.json")


def manifest_tickers(manifest: dict[str, Any]) -> set[str]:
    tickers: set[str] = set()
    for dataset in manifest.get("datasets", []):
        records = dataset.get("tickers", dataset.get("assets", []))
        for record in records:
            ticker = record.get("ticker") if isinstance(record, dict) else record
            tickers.add(str(ticker))
    return tickers


def prepare_dataset(
    root: Path,
    dataset: dict[str, Any],
    config: dict[str, Any],
    caps_all: dict[str, Any],
    metadata_source: dict[str, Any],
    overwrite: bool,
) -> None:
    assets = [str(item) for item in dataset["tickers"]]
    if len(assets) != len(set(assets)):
        raise ValueError(f"{dataset['id']} has duplicate tickers")
    missing_caps = [asset for asset in assets if asset not in caps_all]
    if missing_caps:
        raise ValueError(f"{dataset['id']} missing market caps: {missing_caps}")
    metadata_rows = extract_metadata(metadata_source, assets)
    sectors = sorted({row["GICS Sector"] for row in metadata_rows})
    missing_proxies = sorted(set(sectors) - set(SECTOR_PROXIES))
    if missing_proxies:
        raise ValueError(f"{dataset['id']} has sectors without proxies: {missing_proxies}")

    data_dir = root / "data"
    periods_dir = root / "periods"
    data_dir.mkdir(parents=True, exist_ok=True)
    (root / "universe.json").write_text(json.dumps(assets, indent=2), encoding="utf-8")
    atomic_json(root / "market_caps.json", {asset: float(caps_all[asset]) for asset in assets})
    with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    market_proxy = str(config["market_proxy"])
    requested = list(dict.fromkeys(
        assets + [market_proxy] + [SECTOR_PROXIES[sector] for sector in sectors]
    ))
    close_path = data_dir / "adjusted_close.csv"
    if close_path.exists() and not overwrite:
        close = pd.read_csv(close_path, parse_dates=["Date"]).set_index("Date")
        missing = [ticker for ticker in requested if ticker not in close.columns]
        if missing:
            raise ValueError(f"{close_path} is missing: {missing}")
    else:
        close = download_chart_close(
            requested,
            pd.Timestamp(config["data_start"]) - pd.Timedelta(days=21),
            pd.Timestamp(config["data_end"]) + pd.Timedelta(days=2),
        ).reindex(columns=requested)
        close.index.name = "Date"
        close.reset_index().to_csv(close_path, index=False)

    all_returns = close.pct_change(fill_method=None)
    stock_returns = all_returns[assets]
    stock_returns = stock_returns[
        (stock_returns.index >= pd.Timestamp(config["data_start"]))
        & (stock_returns.index <= pd.Timestamp(config["data_end"]))
    ].dropna(axis=0, how="any")
    market_returns = all_returns[market_proxy].reindex(stock_returns.index)
    sector_returns = pd.DataFrame({
        sector: all_returns[SECTOR_PROXIES[sector]].reindex(stock_returns.index)
        for sector in sectors
    })
    if market_returns.isna().any() or sector_returns.isna().any().any():
        raise RuntimeError(f"{dataset['id']} has missing market/sector proxy returns")
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
        formation_positions = window.pop("formation_positions")
        realized_positions = window.pop("realized_positions")
        formation = stock_returns.iloc[formation_positions]
        realized = stock_returns.iloc[realized_positions]
        period_root = periods_dir / window["period_id"]
        period_root.mkdir(parents=True, exist_ok=True)
        formation.reset_index().to_csv(period_root / "formation_returns.csv", index=False)
        realized.reset_index().to_csv(period_root / "realized_returns.csv", index=False)
        context = {
            ticker: {
                "reference_date": window["reference_date"],
                "sector_returns": (
                    100.0 * sector_returns.loc[
                        formation.index, metadata_by_ticker[ticker]["GICS Sector"]
                    ]
                ).astype(float).tolist(),
                "market_returns": (100.0 * market_returns.loc[formation.index]).astype(float).tolist(),
            }
            for ticker in assets
        }
        atomic_json(period_root / "context.json", context)
        period_records.append(dict(window))
    pd.DataFrame(period_records).to_csv(root / "periods.csv", index=False)

    files = [
        root / "universe.json", root / "market_caps.json", root / "metadata.csv",
        data_dir / "stock_returns.csv", data_dir / "market_returns.csv",
        data_dir / "sector_returns.csv", root / "periods.csv",
    ]
    atomic_json(root / "data_manifest.json", {
        "dataset": dataset,
        "config": config,
        "universe_count": len(assets),
        "period_counts": pd.DataFrame(period_records)["phase"].value_counts().to_dict(),
        "sector_proxies": {sector: SECTOR_PROXIES[sector] for sector in sectors},
        "files": {str(path.relative_to(root)): sha256(path) for path in files},
    })
    print(f"Prepared {dataset['id']}: {len(assets)} assets, {len(period_records)} periods")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--market-caps", type=Path, default=Path("market_caps.json"))
    parser.add_argument("--metadata-source", type=Path, default=DEFAULT_METADATA_SOURCE)
    parser.add_argument("--previous-manifest", type=Path, default=DEFAULT_PREVIOUS_MANIFEST)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    caps_all = load_json(args.market_caps)
    metadata_source = load_json(args.metadata_source)
    requested = set(args.datasets or [])
    selected = [
        dataset for dataset in manifest["datasets"]
        if not requested or dataset["id"] in requested
    ]
    missing = requested - {dataset["id"] for dataset in selected}
    if missing:
        raise ValueError(f"unknown datasets: {sorted(missing)}")
    previous_tickers = (
        manifest_tickers(load_json(args.previous_manifest))
        if args.previous_manifest.exists() else set()
    )
    selected_tickers = manifest_tickers({"datasets": selected})
    reused = selected_tickers.intersection(previous_tickers)
    if reused:
        raise ValueError(f"new datasets reuse previous-study assets: {sorted(reused)}")
    seen: set[str] = set()
    for dataset in selected:
        overlap = seen.intersection(dataset["tickers"])
        if overlap:
            raise ValueError(f"datasets are not disjoint: {sorted(overlap)}")
        seen.update(dataset["tickers"])
        prepare_dataset(
            args.root / dataset["id"], dataset, config, caps_all,
            metadata_source, args.overwrite,
        )


if __name__ == "__main__":
    main()
