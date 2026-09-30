"""Price-window statistics and illustrative risk inputs.

These helpers are deliberately free of plotting and dashboard imports so the
numbers can be tested without Streamlit or Plotly.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


# Illustrative geopolitical priors. They are not estimated country risk.
_GEO_RISK = {
    "CL=F": 65,
    "BZ=F": 70,
    "NG=F": 55,
    "GC=F": 30,
    "SI=F": 25,
    "ZC=F": 20,
    "ZW=F": 35,
    "ZS=F": 18,
    "HG=F": 40,
}

_PRODUCERS = {
    "CL=F": ["Saudi Arabia", "USA", "Russia", "Iraq", "UAE"],
    "BZ=F": ["Norway", "UK", "Nigeria", "Angola", "Libya"],
    "NG=F": ["USA", "Russia", "Iran", "Qatar", "Canada"],
    "GC=F": ["China", "Australia", "Russia", "USA", "Canada"],
    "SI=F": ["Mexico", "Peru", "China", "Russia", "Poland"],
    "ZC=F": ["USA", "China", "Brazil", "Argentina", "Ukraine"],
    "ZW=F": ["China", "India", "Russia", "USA", "France"],
    "ZS=F": ["USA", "Brazil", "Argentina", "China", "India"],
    "HG=F": ["Chile", "Peru", "China", "DRC", "USA"],
}


def simulate_weather_data(
    start: str,
    end: str,
    freq: str = "W",
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic weekly weather for offline demos. Does not touch the global RNG."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=freq)
    n = len(dates)
    t = np.linspace(0, 4 * np.pi, n)

    df = pd.DataFrame(index=dates)
    df["temperature_mean"] = 15 + 12 * np.sin(t) + rng.standard_normal(n) * 3
    df["precipitation_sum"] = np.abs(
        30 * (1 - np.cos(t)) + rng.standard_normal(n) * 10
    )
    df["windspeed_max"] = 20 + 10 * np.abs(np.sin(t * 2)) + rng.standard_normal(n) * 5
    df["evapotranspiration"] = np.abs(5 + 3 * np.sin(t) + rng.standard_normal(n))
    return df


def _ordered_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    ordered = pd.Series(prices, copy=True).astype(float).sort_index()
    if ordered.index.tz is not None:
        ordered = ordered.tz_convert("UTC").tz_localize(None)
        ordered.index = pd.DatetimeIndex(ordered.index)
    ordered = ordered[~ordered.index.duplicated(keep="last")]
    ordered = ordered.replace([np.inf, -np.inf], np.nan).dropna()
    if ordered.empty:
        raise ValueError("prices must contain at least one finite observation")
    return ordered


def weekly_last(prices: pd.Series) -> pd.Series:
    """Resample to Sunday weeks and carry the last observed price across gaps.

    Empty weeks stay on the calendar. Forward-fill uses only an earlier print,
    so a later price cannot fill an earlier gap.
    """
    ordered = _ordered_prices(prices)
    weekly = ordered.resample("W").last().ffill().dropna()
    if weekly.empty:
        raise ValueError("weekly series is empty")
    return weekly


def trailing_high_low(prices: pd.Series, weeks: int = 52) -> dict:
    """High and low inside the trailing window, not the full sample."""
    if isinstance(weeks, bool) or not isinstance(weeks, (int, np.integer)):
        raise TypeError("weeks must be a positive integer")
    if weeks < 1:
        raise ValueError("weeks must be at least 1")

    clean = _ordered_prices(prices)
    end = clean.index[-1]
    start = end - pd.Timedelta(weeks=int(weeks))
    window = clean.loc[clean.index >= start]
    return {
        "high": float(window.max()),
        "low": float(window.min()),
        "complete_window": bool(clean.index[0] <= start),
        "observations": int(len(window)),
    }


def weather_price_frame(
    prices: pd.Series,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Align weekly price returns with same-week weather on exact dates.

    Price levels are excluded because a shared trend produces a spurious
    correlation. Dates are never matched to a neighboring week, so a later
    weather observation cannot be attached to an earlier price.
    """
    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a DataFrame")
    ordered = _ordered_prices(prices)
    weather_index = pd.DatetimeIndex(weather.index)
    if weather_index.tz is not None:
        weather = weather.copy()
        weather.index = weather_index.tz_convert("UTC").tz_localize(None)
    common = ordered.index.intersection(pd.DatetimeIndex(weather.index)).sort_values()
    frame = pd.DataFrame(index=common)
    frame["price_return"] = ordered.reindex(common).pct_change()
    for column in weather.columns:
        frame[column] = pd.to_numeric(weather.loc[common, column], errors="coerce")
    return frame.dropna(how="any")


def price_risk_components(prices: pd.Series) -> dict:
    """Commodity-level vol, drawdown, and trend instability.

    Drawdown is the peak-to-trough loss of a wealth index that starts at 1.
    Volatility, drawdown, and trend instability all use the last 52 weekly
    observations when that much history exists.
    """
    weekly = weekly_last(prices)
    weekly_returns = weekly.pct_change().dropna()
    if weekly_returns.empty:
        raise ValueError("at least two weekly prices are required")

    recent_returns = (
        weekly_returns.iloc[-52:] if len(weekly_returns) >= 52 else weekly_returns
    )
    ann_vol = float(recent_returns.std(ddof=1) * np.sqrt(52))

    wealth = np.cumprod(
        np.concatenate([[1.0], 1.0 + recent_returns.to_numpy(dtype=float)])
    )
    peak = np.maximum.accumulate(wealth)
    max_drawdown = float(np.min(wealth / peak - 1.0))

    recent_prices = weekly.iloc[-52:] if len(weekly) >= 52 else weekly
    if len(recent_prices) >= 26:
        signal = (
            recent_prices.rolling(4).mean() > recent_prices.rolling(26).mean()
        ).dropna()
        crosses = float(signal.astype(int).diff().abs().sum(skipna=True))
        trend_instability = crosses / max(len(signal), 1)
    else:
        trend_instability = 0.5

    return {
        "ann_vol": ann_vol,
        "max_drawdown": max_drawdown,
        "trend_instability": float(trend_instability),
    }


def _illustrative_geo_score(commodity_name: str, region: str, base: float) -> float:
    """Stable region adjustment in [-15, 15]. Does not touch the global RNG."""
    digest = hashlib.sha256(f"{commodity_name}|{region}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64)
    noise = -15.0 + 30.0 * unit
    return float(np.clip(base + noise, 0, 100))


def compute_risk_scores(prices: pd.Series, commodity_name: str) -> pd.DataFrame:
    """Illustrative producer table for one commodity price series.

    Volatility, drawdown, and trend scores are properties of the commodity,
    so they are identical for every listed producer. Only the geopolitical
    column varies, and that column is a deterministic illustrative prior,
    not an estimated regional risk.
    """
    components = price_risk_components(prices)
    ticker = prices.name if getattr(prices, "name", None) else "CL=F"
    base_geo = _GEO_RISK.get(str(ticker), 50)
    producers = _PRODUCERS.get(
        str(ticker),
        ["Producer A", "Producer B", "Producer C", "Producer D", "Producer E"],
    )

    vol_score = float(np.clip(components["ann_vol"] * 200, 0, 100))
    dd_score = float(np.clip(abs(components["max_drawdown"]) * 250, 0, 100))
    trend_score = float(np.clip(components["trend_instability"] * 150, 0, 100))

    rows = []
    for region in producers:
        geo_score = _illustrative_geo_score(commodity_name, region, base_geo)
        overall = (
            0.35 * vol_score
            + 0.25 * dd_score
            + 0.20 * trend_score
            + 0.20 * geo_score
        )
        rows.append(
            {
                "Region / Producer": region,
                "Volatility Score": round(vol_score, 1),
                "Drawdown Score": round(dd_score, 1),
                "Trend Score": round(trend_score, 1),
                "Geopolitical Score": round(geo_score, 1),
                "Overall Risk Score": round(overall, 1),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("Overall Risk Score", ascending=False)
        .reset_index(drop=True)
    )
