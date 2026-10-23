"""Walk-forward (expanding-window) backtesting for multi-step forecasts.

At each origin the model sees only the data up to that origin, forecasts ``horizon``
steps ahead, and the window then moves forward by ``step_size`` observations.
Every prediction is out of sample.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ModelFn = Callable[[pd.Series], tuple[np.ndarray, np.ndarray, np.ndarray]]


def walk_forward_backtest(
    series: pd.Series,
    model_fn: ModelFn,
    initial_train_size: int,
    step_size: int,
    horizon: int,
) -> dict:
    """Run an expanding-window walk-forward backtest.

    Parameters
    ----------
    series : pd.Series
        Price series with a sorted DatetimeIndex.
    model_fn : callable
        ``model_fn(train) -> (forecast, lower, upper)``, each of length ``horizon``.
    initial_train_size : int
        Observations in the first training window.
    step_size : int
        Observations the origin advances between refits.
    horizon : int
        Steps forecast at each origin. Origins whose horizon would run past the end of
        the series are scored on the steps that exist.

    Returns
    -------
    dict with
        ``predictions`` : one row per (origin, step) with actual, forecast, lower, upper
        ``by_origin``   : MAE / RMSE / MAPE / coverage per origin
        ``summary``     : mean MAE, mean RMSE, mean MAPE, interval hit rate, counts
    """
    s = pd.Series(series, dtype=float).dropna()
    if not s.index.is_monotonic_increasing:
        raise ValueError("series index must be sorted")
    if initial_train_size < 2 or step_size < 1 or horizon < 1:
        raise ValueError("initial_train_size >= 2, step_size >= 1, horizon >= 1 required")
    if initial_train_size >= len(s):
        raise ValueError("initial_train_size must be smaller than the series")

    rows = []
    failed = 0
    for origin in range(initial_train_size, len(s), step_size):
        train = s.iloc[:origin]
        test = s.iloc[origin : origin + horizon]
        try:
            fc, lo, hi = (np.asarray(a, dtype=float).reshape(-1) for a in model_fn(train))
        except Exception:
            logger.warning("model_fn failed at origin %s", s.index[origin - 1], exc_info=True)
            failed += 1
            continue
        if min(len(fc), len(lo), len(hi)) < len(test):
            raise ValueError("model_fn returned fewer than horizon values")
        for step, (date, actual) in enumerate(test.items(), start=1):
            rows.append(
                {
                    "origin": s.index[origin - 1],
                    "date": date,
                    "step": step,
                    "actual": float(actual),
                    "forecast": fc[step - 1],
                    "lower": lo[step - 1],
                    "upper": hi[step - 1],
                }
            )

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise ValueError("no successful forecasts")
    err = pred["forecast"] - pred["actual"]
    pred["abs_error"] = err.abs()
    pred["sq_error"] = err**2
    pred["ape"] = np.where(pred["actual"] != 0, pred["abs_error"] / pred["actual"].abs() * 100, np.nan)
    pred["hit"] = (pred["actual"] >= pred["lower"]) & (pred["actual"] <= pred["upper"])

    by_origin = pred.groupby("origin").agg(
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda x: float(np.sqrt(x.mean()))),
        MAPE=("ape", "mean"),
        coverage=("hit", "mean"),
    )
    summary = {
        "n_origins": int(by_origin.shape[0]),
        "n_failed_origins": failed,
        "n_predictions": int(len(pred)),
        "mean_MAE": float(by_origin["MAE"].mean()),
        "mean_RMSE": float(by_origin["RMSE"].mean()),
        "mean_MAPE": float(by_origin["MAPE"].mean()),
        "hit_rate": float(pred["hit"].mean()),
    }
    return {"predictions": pred, "by_origin": by_origin, "summary": summary}


def arima_model_fn(
    order: tuple[int, int, int] = (1, 1, 1), alpha: float = 0.05, horizon: int = 4
) -> ModelFn:
    """Fixed-order statsmodels ARIMA forecaster for use with ``walk_forward_backtest``."""

    def fn(train: pd.Series):
        from statsmodels.tsa.arima.model import ARIMA

        res = ARIMA(train.to_numpy(dtype=float), order=order).fit()
        f = res.get_forecast(steps=horizon)
        ci = np.asarray(f.conf_int(alpha=alpha), dtype=float)
        return np.asarray(f.predicted_mean, dtype=float), ci[:, 0], ci[:, 1]

    return fn
