"""
data_loader.py
==============
Data retrieval for:
  - Commodity prices via yfinance
  - Historical weather data via Open-Meteo (free, no API key required)
"""

import pandas as pd
import numpy as np
import requests
import streamlit as st
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Commodity Prices
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_commodity_prices(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series | None:
    """
    Fetch commodity daily closing prices from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker : str   Yahoo Finance ticker symbol (e.g. 'CL=F' for WTI crude).
    start : str    Start date 'YYYY-MM-DD'.
    end : str      End date 'YYYY-MM-DD'.

    Returns
    -------
    pd.Series with DatetimeIndex and float prices, or None on failure.
    """
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
            return None

        # Extract Close; handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Close"].iloc[:, 0]
        else:
            prices = data["Close"]

        prices = prices.ffill().dropna()
        prices.name = ticker
        return prices

    except Exception as exc:
        st.error(f"Error fetching commodity data: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Weather Data (Open-Meteo)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_weather_data(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
) -> pd.DataFrame | None:
    """
    Fetch historical daily weather data from the Open-Meteo API.
    No API key required. Free up to 10,000 calls/day.

    API documentation: https://open-meteo.com/en/docs/historical-weather-api

    Parameters
    ----------
    latitude : float   Latitude of the region centre.
    longitude : float  Longitude of the region centre.
    start_date : str   'YYYY-MM-DD'.
    end_date : str     'YYYY-MM-DD'.
    variables : list[str], optional
        Open-Meteo variable names. Defaults to temperature_2m_mean,
        precipitation_sum, windspeed_10m_max, et0_fao_evapotranspiration.

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per weather variable.
    Returns None on connection failure.
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
        # Return None silently — app handles gracefully
        return None
    except Exception as exc:
        st.warning(f"Weather data fetch failed: {exc}")
        return None


def _clean_col_name(col: str) -> str:
    """Simplify Open-Meteo column names."""
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
    """
    Generate synthetic weather data for offline testing / demo mode.

    Parameters
    ----------
    start, end : str   Date range strings.
    freq : str         Pandas frequency (default 'W' for weekly).

    Returns
    -------
    pd.DataFrame with columns: temperature_mean, precipitation_sum,
    windspeed_max, evapotranspiration.
    """
    np.random.seed(seed)
    dates = pd.date_range(start, end, freq=freq)
    n = len(dates)
    t = np.linspace(0, 4 * np.pi, n)

    df = pd.DataFrame(index=dates)
    df["temperature_mean"] = 15 + 12 * np.sin(t) + np.random.randn(n) * 3
    df["precipitation_sum"] = np.abs(30 * (1 - np.cos(t)) + np.random.randn(n) * 10)
    df["windspeed_max"] = 20 + 10 * np.abs(np.sin(t * 2)) + np.random.randn(n) * 5
    df["evapotranspiration"] = np.abs(5 + 3 * np.sin(t) + np.random.randn(n))

    return df
