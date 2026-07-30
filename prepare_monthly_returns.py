"""Download and split monthly return files for walk-forward experiments."""

from __future__ import annotations

import argparse
import calendar
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from evaluate_relview import load_universe


def download_chart_close(
    universe: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Download adjusted closes through Yahoo's stateless chart endpoint."""

    period1 = int(start.tz_localize("UTC").timestamp())
    period2 = int(end.tz_localize("UTC").timestamp())
    series: list[pd.Series] = []
    for ticker in universe:
        encoded = urllib.parse.quote(ticker, safe="")
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history"
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                result = payload["chart"]["result"][0]
                dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
                adjusted = result["indicators"].get("adjclose", [])
                closes = (
                    adjusted[0].get("adjclose")
                    if adjusted and adjusted[0].get("adjclose") is not None
                    else result["indicators"]["quote"][0]["close"]
                )
                series.append(pd.Series(closes, index=dates, name=ticker, dtype=float))
                last_error = None
                break
            except Exception as error:
                last_error = error
                time.sleep(1.0 + attempt)
        if last_error is not None:
            raise RuntimeError(f"failed to download {ticker}: {last_error}") from last_error
    return pd.concat(series, axis=1).sort_index()


def month_sequence(start_month: str, end_month: str) -> list[pd.Timestamp]:
    start = pd.Timestamp(f"{start_month}-01")
    end = pd.Timestamp(f"{end_month}-01")
    if end < start:
        raise ValueError("end month must not precede start month")
    return list(pd.date_range(start, end, freq="MS"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare monthly yfinance return CSVs")
    parser.add_argument("--start-month", required=True, help="YYYY-MM")
    parser.add_argument("--end-month", required=True, help="YYYY-MM, inclusive")
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("yfinance"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    universe = load_universe(args.universe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for month_start in month_sequence(args.start_month, args.end_month):
        next_month = month_start + pd.DateOffset(months=1)
        month_end = pd.Timestamp(
            year=month_start.year,
            month=month_start.month,
            day=calendar.monthrange(month_start.year, month_start.month)[1],
        )
        target = args.output_dir / f"returns_{month_start:%Y-%m-%d}_{month_end:%Y-%m-%d}.csv"
        if target.exists() and not args.overwrite:
            frame = pd.read_csv(target, nrows=1)
            if all(ticker in frame.columns for ticker in universe):
                print(f"Keeping existing {target}")
                continue

        close = download_chart_close(
            universe,
            month_start - pd.Timedelta(days=7),
            next_month + pd.Timedelta(days=1),
        ).reindex(columns=universe)
        returns = close.pct_change(fill_method=None)
        returns = returns[(returns.index >= month_start) & (returns.index < next_month)]
        returns = returns.dropna(axis=0, how="any")
        if returns.empty:
            raise RuntimeError(f"no complete returns downloaded for {month_start:%Y-%m}")
        returns.index.name = "Date"
        returns.to_csv(target)
        print(f"Saved {len(returns)} trading days to {target}")


if __name__ == "__main__":
    main()
