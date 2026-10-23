"""Interval-coverage (calibration) diagnostics for one-step ARIMA forecasts.

A well-calibrated 95% interval should contain the realised price about 95% of the time.
Coverage is tested with an exact two-sided binomial test against the nominal level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.backtester import arima_model_fn, walk_forward_backtest
from src.benchmark import ARIMA_ORDER, DEFAULT_ORIGINS, complete_weekly_series

LEVELS = (0.80, 0.95)


def coverage_report(weekly: pd.Series, n_origins: int = DEFAULT_ORIGINS,
                    levels=LEVELS, order=ARIMA_ORDER) -> dict:
    """Coverage of one-step ARIMA intervals over the last ``n_origins`` weeks."""
    out = {}
    for level in levels:
        bt = walk_forward_backtest(
            weekly,
            arima_model_fn(order=tuple(order), alpha=1 - level, horizon=1),
            initial_train_size=len(weekly) - n_origins,
            step_size=1,
            horizon=1,
        )
        p = bt["predictions"]
        hits = int(p["hit"].sum())
        n = int(len(p))
        width = ((p["upper"] - p["lower"]) / p["actual"]).to_numpy()
        below = int((p["actual"] < p["lower"]).sum())
        out[f"{int(round(level * 100))}%"] = {
            "nominal": level,
            "n": n,
            "hits": hits,
            "coverage": hits / n,
            "misses_below": below,
            "misses_above": n - hits - below,
            "binom_p_value": float(binomtest(hits, n, level).pvalue),
            "mean_relative_width": float(np.mean(width)),
        }
    return out


def calibrate_snapshot(snapshot: pd.DataFrame, tickers: dict[str, str],
                       n_origins: int = DEFAULT_ORIGINS) -> dict:
    return {
        t: {"name": name, **coverage_report(complete_weekly_series(snapshot[t]), n_origins)}
        for t, name in tickers.items()
    }
