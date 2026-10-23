"""Cross-commodity comparison helpers: normalised prices, return correlations, spreads.

Correlations are computed on weekly *returns*, not price levels, because two trending
price series correlate whether or not they are related.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.market import weekly_last


def align_weekly(prices: dict[str, pd.Series]) -> pd.DataFrame:
    """Week-end closes for each series, inner-joined on the dates all of them share."""
    frame = pd.concat({name: weekly_last(s) for name, s in prices.items()}, axis=1)
    return frame.dropna(how="any")


def normalized(frame: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase every column to ``base`` at the first row."""
    return frame / frame.iloc[0] * base


def weekly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pct_change().dropna(how="any")


def return_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of weekly returns (identical to ``np.corrcoef``)."""
    r = weekly_returns(frame)
    return pd.DataFrame(np.corrcoef(r.to_numpy().T), index=r.columns, columns=r.columns)


def rolling_correlation(frame: pd.DataFrame, a: str, b: str, window: int = 12) -> pd.Series:
    """Rolling ``window``-week correlation of weekly returns between columns a and b."""
    r = weekly_returns(frame)
    return r[a].rolling(window).corr(r[b]).dropna()


def spread_bands(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Price ratio A/B with its full-sample mean and ±1σ / ±2σ bands.

    The bands use the whole selected sample, so they describe the period; they are
    not a tradable signal (they use future observations).
    """
    ratio = frame[a] / frame[b]
    mu, sd = float(ratio.mean()), float(ratio.std(ddof=1))
    return pd.DataFrame(
        {
            "ratio": ratio,
            "mean": mu,
            "+1σ": mu + sd,
            "-1σ": mu - sd,
            "+2σ": mu + 2 * sd,
            "-2σ": mu - 2 * sd,
        }
    )
