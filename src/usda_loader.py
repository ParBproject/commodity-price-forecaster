"""USDA NASS Quick Stats crop-yield data as exogenous features.

The API key is only ever read from the ``USDA_NASS_API_KEY`` environment
variable (a GitHub Actions secret in this repo). Everything downstream of the
download -- the feature builder, the benchmark variant and the tests -- reads the
committed snapshot ``data/usda_cache.json`` and never touches the network.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "usda_cache.json"

# Futures ticker -> NASS short_desc of the national yield series.
SERIES = {
    "ZC=F": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    "ZW=F": "WHEAT - YIELD, MEASURED IN BU / ACRE",
    "ZS=F": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
}
AG_TICKERS = tuple(SERIES)
KEEP_FIELDS = (
    "short_desc", "year", "reference_period_desc", "Value", "source_desc",
    "agg_level_desc", "freq_desc", "load_time",
)
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
# NASS Crop Production reports (and the same-day WASDE) are published around the
# 12th of the month; the final annual yield appears in the January Annual Summary.
RELEASE_DAY = 12


def fetch_yields(start_year: int = 2010, api_key: str | None = None, timeout: int = 60) -> dict:
    """Download national yield estimates (forecasts and finals) from Quick Stats."""
    import requests

    key = api_key or os.environ.get("USDA_NASS_API_KEY")
    if not key:
        raise RuntimeError("USDA_NASS_API_KEY is not set")
    out = {}
    for ticker, desc in SERIES.items():
        params = {
            "key": key, "short_desc": desc, "agg_level_desc": "NATIONAL",
            "source_desc": "SURVEY", "year__GE": str(start_year), "format": "JSON",
        }
        resp = requests.get(API_URL, params=params, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(
                f"NASS Quick Stats HTTP {resp.status_code} for {desc!r}: {resp.text[:500]}")
        body = resp.json()
        if "data" not in body:
            raise RuntimeError(f"NASS Quick Stats error for {desc!r}: {body}")
        out[ticker] = [{k: r.get(k) for k in KEEP_FIELDS} for r in body["data"]]
    return out


def load_cache(path: str | Path = CACHE_PATH) -> dict:
    return json.loads(Path(path).read_text())


def release_date(year: int, reference_period: str) -> pd.Timestamp | None:
    """Approximate public release date of one NASS yield estimate.

    ``"YEAR - AUG FORECAST"`` -> Aug 12 of that year; ``"YEAR"`` (final) -> Jan 12
    of the following year. Other reference periods are ignored.
    """
    ref = reference_period.strip().upper()
    if ref == "YEAR":
        return pd.Timestamp(year + 1, 1, RELEASE_DAY)
    match = re.fullmatch(r"YEAR - ([A-Z]{3}) FORECAST", ref)
    if match and match.group(1) in MONTHS:
        return pd.Timestamp(year, MONTHS[match.group(1)], RELEASE_DAY)
    return None


def _value(raw) -> float | None:
    try:
        return float(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        return None


def yield_releases(records: list[dict]) -> pd.DataFrame:
    """One row per published estimate: release date, crop year, yield."""
    rows = []
    for r in records:
        value = _value(r.get("Value"))
        released = release_date(int(r["year"]), r.get("reference_period_desc", ""))
        if value is None or released is None:
            continue
        rows.append({"released": released, "year": int(r["year"]),
                     "reference_period": r["reference_period_desc"], "yield": value})
    frame = pd.DataFrame(rows).drop_duplicates(["released", "year"], keep="last")
    return frame.sort_values(["released", "year"]).reset_index(drop=True)


def yield_surprise_weekly(records: list[dict], index: pd.DatetimeIndex) -> pd.Series:
    """Weekly as-of feature: log(latest estimate / previous crop year's final yield).

    For each week only estimates released on or before that week-end are used,
    so there is no look-ahead; the value is forward-filled between reports.
    Weeks before the first usable estimate are 0.
    """
    rel = yield_releases(records)
    finals = rel[rel["reference_period"].str.upper().str.strip() == "YEAR"].set_index("year")["yield"]
    values = []
    import numpy as np

    for week_end in index:
        known = rel[rel["released"] <= week_end]
        if known.empty:
            values.append(0.0)
            continue
        latest = known.iloc[-1]
        prior = finals.get(latest["year"] - 1)
        prior_known = prior is not None and pd.Timestamp(latest["year"], 1, RELEASE_DAY) <= week_end
        values.append(float(np.log(latest["yield"] / prior)) if prior_known else 0.0)
    return pd.Series(values, index=index, name="usda_yield_surprise")
