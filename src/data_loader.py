"""Yahoo Finance prices and Open-Meteo weather. No API key for either."""

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_commodity_prices(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series | None:
    """Daily closes from Yahoo Finance. ``end`` is exclusive, as in yfinance. None on failure."""
    try:
        import yfinance as yf

        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            logger.warning("No price data returned for %s (%s to %s)", ticker, start, end)
            return None

        # Extract Close; handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"].iloc[:, 0]
        else:
            prices = data["Close"]

        prices = prices.ffill().dropna()
        prices.name = ticker
        logger.info("Loaded %d daily closes for %s", len(prices), ticker)
        return prices

    except Exception as exc:
        logger.exception("Price download failed for %s", ticker)
        st.error(f"Error fetching commodity data: {exc}")
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> pd.DataFrame | None:
    """Daily Open-Meteo history, resampled to weeks. None if the request fails.

    https://open-meteo.com/en/docs/historical-weather-api
    """
    if variables is None:
        variables = [
            "temperature_2m_mean",
            "precipitation_sum",
            "windspeed_10m_max",
            "et0_fao_evapotranspiration",
        ]

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(variables),
        "timezone": "UTC",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "daily" not in data:
            return None

        df = pd.DataFrame(data["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
        df.columns = [_clean_col_name(c) for c in df.columns]

        # Resample to weekly to match commodity prices
        df_weekly = df.resample("W").agg({
            "temperature_mean": "mean",
            "precipitation_sum": "sum",
            "windspeed_max": "mean",
            "evapotranspiration": "mean",
        })

        return df_weekly.ffill().dropna(how="all")

    except requests.exceptions.ConnectionError:
        # The app falls back gracefully; log so the failure is visible in the terminal
        logger.warning("Open-Meteo unreachable; continuing without weather data")
        return None
    except Exception as exc:
        logger.exception("Weather data fetch failed")
        st.warning(f"Weather data fetch failed: {exc}")
        return None


def _clean_col_name(col: str) -> str:
    mapping = {
        "temperature_2m_mean": "temperature_mean",
        "precipitation_sum": "precipitation_sum",
        "windspeed_10m_max": "windspeed_max",
        "et0_fao_evapotranspiration": "evapotranspiration",
    }
    return mapping.get(col, col)


def simulate_weather_data(
    start: str,
    end: str,
    freq: str = "W",
    seed: int = 42,
) -> pd.DataFrame:
    """Seeded synthetic weather. Not a stand-in for Open-Meteo in any reported result."""
    from src.market import simulate_weather_data as _simulate_weather_data

    return _simulate_weather_data(start, end, freq=freq, seed=seed)


def _ticker_history(ticker: str, start: str, end: str) -> pd.Series | None:
    """Thread-safe single-ticker download.

    ``yf.download`` keeps module-level state and is not safe to call from several
    threads: concurrent calls were observed to return the same series for different
    tickers. ``yf.Ticker(...).history`` keeps its state per object.
    """
    import yfinance as yf

    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    except Exception:
        logger.exception("Price download failed for %s", ticker)
        return None
    if hist.empty or "Close" not in hist:
        return None
    prices = hist["Close"].copy()
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index = prices.index.normalize()
    prices = prices.ffill().dropna()
    prices.name = ticker
    return prices


def fetch_many(tickers: dict[str, str], start: str, end: str, max_workers: int = 6) -> dict:
    """Fetch several tickers in parallel. Returns {name: prices} for those that loaded."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {name: pool.submit(_ticker_history, t, start, end) for name, t in tickers.items()}
        out = {name: f.result() for name, f in futures.items()}
    missing = [n for n, s in out.items() if s is None or len(s) == 0]
    if missing:
        logger.warning("No data for %s", ", ".join(missing))
    return {n: s for n, s in out.items() if n not in missing}
