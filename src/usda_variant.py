"""ARIMA(1,1,1) with a USDA yield-surprise regressor, scored on the leaderboard window."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.benchmark import (
    ARIMA_ORDER,
    DEFAULT_ORIGINS,
    arima_one_step,
    complete_weekly_series,
    score_predictions,
)
from src.usda_loader import yield_surprise_weekly

VARIANT = "ARIMA(1,1,1)+USDA yield"
BASE = "ARIMA(1,1,1)"


def arimax_one_step(train: pd.Series, exog_train: pd.Series, exog_next: float) -> float:
    """One-step ARIMA with a regressor; ``exog_next`` must be known at the origin."""
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(np.asarray(train, dtype=float),
                       exog=np.asarray(exog_train, dtype=float).reshape(-1, 1),
                       order=ARIMA_ORDER).fit()
        return float(np.asarray(fitted.forecast(steps=1, exog=[[exog_next]]))[0])


def evaluate_usda_variant(daily: pd.Series, records: list[dict], *,
                          n_origins: int = DEFAULT_ORIGINS) -> tuple[pd.DataFrame, dict]:
    weekly = complete_weekly_series(daily)
    feature = yield_surprise_weekly(records, weekly.index)
    rows = []
    for pos in range(len(weekly) - n_origins, len(weekly)):
        train = weekly.iloc[:pos]
        rows.append({
            "date": weekly.index[pos],
            "actual": float(weekly.iloc[pos]),
            "previous_actual": float(weekly.iloc[pos - 1]),
            BASE: arima_one_step(train),
            # the regressor for week t is the value known at the origin (week t-1)
            VARIANT: arimax_one_step(train, feature.iloc[:pos], float(feature.iloc[pos - 1])),
        })
    preds = pd.DataFrame(rows)
    insample = weekly.iloc[: len(weekly) - n_origins]
    return preds, score_predictions(preds, insample, [BASE, VARIANT])
