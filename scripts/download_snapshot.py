"""Download a dated daily-price snapshot with the project's own loader.

Run once; the CSV it writes is committed so the leaderboard is reproducible
without network access:

    python scripts/download_snapshot.py --start 2016-01-01 --end 2026-09-29
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import fetch_commodity_prices  # noqa: E402

COMMODITIES = {
    "CL=F": "Crude Oil (WTI)",
    "BZ=F": "Brent Crude",
    "NG=F": "Natural Gas",
    "GC=F": "Gold",
    "SI=F": "Silver",
    "HG=F": "Copper",
    "ZC=F": "Corn",
    "ZW=F": "Wheat",
    "ZS=F": "Soybeans",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None, help="exclusive end date")
    args = parser.parse_args()
    downloaded = dt.date.today().isoformat()
    end = args.end or downloaded

    columns = {}
    for ticker in COMMODITIES:
        prices = fetch_commodity_prices(ticker, args.start, end)
        if prices is None or prices.empty:
            raise SystemExit(f"download failed for {ticker}")
        columns[ticker] = prices.astype(float).round(4)
        print(f"{ticker}: {len(prices)} rows {prices.index[0].date()} -> {prices.index[-1].date()}")

    frame = pd.DataFrame(columns)
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    out = ROOT / "data" / f"commodity_prices_daily_{downloaded}.csv"
    frame.to_csv(out)
    meta = {
        "source": "Yahoo Finance via yfinance (src.data_loader.fetch_commodity_prices, auto_adjust=True, Close)",
        "downloaded_on": downloaded,
        "requested_start": args.start,
        "requested_end_exclusive": end,
        "first_date": str(frame.index[0].date()),
        "last_date": str(frame.index[-1].date()),
        "file": out.name,
        "tickers": COMMODITIES,
    }
    (ROOT / "data" / "snapshot.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
