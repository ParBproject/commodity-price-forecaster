"""Download USDA NASS yields to data/usda_cache.json (needs USDA_NASS_API_KEY).

Run by the ``refresh-usda`` GitHub Actions workflow, which holds the key as a
secret; the resulting JSON is committed so everything else runs offline.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.usda_loader import API_URL, CACHE_PATH, SERIES, fetch_yields  # noqa: E402


def main() -> None:
    try:
        data = fetch_yields(start_year=2010)
    except Exception as exc:  # report the exact failure and stop
        print(f"USDA download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    payload = {
        "source": "USDA NASS Quick Stats API",
        "api": API_URL,
        "downloaded_on": dt.date.today().isoformat(),
        "filters": {"agg_level_desc": "NATIONAL", "source_desc": "SURVEY", "year__GE": 2010},
        "series": SERIES,
        "records": data,
    }
    CACHE_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    for ticker, rows in data.items():
        print(ticker, len(rows), "records")
    if not all(data.values()):
        raise SystemExit("NASS returned no records for at least one series")


if __name__ == "__main__":
    main()
