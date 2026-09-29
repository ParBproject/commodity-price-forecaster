"""Forecast validation and baseline benchmarking for commodity time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastMetrics:
    """Scale-aware forecast diagnostics."""

    mae: float
    rmse: float
    mape: float | None
    smape: float
    mase: float
    directional_accuracy: float


def _clean_series(series: pd.Series) -> pd.Series:
    clean = pd.Series(series, copy=True).astype(float).dropna()
    if len(clean) < 3:
        raise ValueError("series must contain at least three observations")
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("series must contain only finite values")
    return clean


def last_value_forecast(train: pd.Series, horizon: int = 1) -> np.ndarray:
    """Naïve forecast that repeats the most recently observed value."""
    clean = _clean_series(train)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return np.full(horizon, float(clean.iloc[-1]))


def drift_forecast(train: pd.Series, horizon: int = 1) -> np.ndarray:
    """Random-walk-with-drift forecast using only the observed training history."""
    clean = _clean_series(train)
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    slope = (float(clean.iloc[-1]) - float(clean.iloc[0])) / (len(clean) - 1)
    steps = np.arange(1, horizon + 1, dtype=float)
    return float(clean.iloc[-1]) + slope * steps


def seasonal_naive_forecast(
    train: pd.Series,
    horizon: int = 1,
    *,
    season_length: int = 52,
) -> np.ndarray:
    """Repeat values from the most recent complete seasonal cycle."""
    clean = _clean_series(train)
    if isinstance(season_length, bool) or not isinstance(
        season_length, (int, np.integer)
    ):
        raise TypeError("season_length must be a positive integer")
    if season_length < 1:
        raise ValueError("season_length must be at least 1")
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    if len(clean) < season_length:
        return last_value_forecast(clean, horizon)

    values = clean.to_numpy()
    predictions = []
    for step in range(horizon):
        seasonal_index = len(values) - season_length + (step % season_length)
        if seasonal_index < len(values):
            predictions.append(float(values[seasonal_index]))
        else:
            predictions.append(float(predictions[step - season_length]))
    return np.asarray(predictions, dtype=float)


def forecast_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    insample: np.ndarray | pd.Series,
    previous_actual: np.ndarray | pd.Series | None = None,
) -> ForecastMetrics:
    """Compute scale-dependent, percentage, MASE, and direction metrics."""
    actual_values = np.asarray(actual, dtype=float).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=float).reshape(-1)
    insample_values = np.asarray(insample, dtype=float).reshape(-1)

    if actual_values.size == 0 or predicted_values.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("actual and predicted must have matching shapes")
    if insample_values.size < 2:
        raise ValueError("insample must contain at least two observations")
    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(predicted_values).all()
        and np.isfinite(insample_values).all()
    ):
        raise ValueError("metric inputs must contain only finite values")

    errors = predicted_values - actual_values
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    nonzero = actual_values != 0
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual_values[nonzero])) * 100)
        if np.any(nonzero)
        else None
    )

    denominator = np.abs(actual_values) + np.abs(predicted_values)
    valid_smape = denominator > 0
    smape = (
        float(
            200
            * np.mean(
                np.abs(errors[valid_smape]) / denominator[valid_smape]
            )
        )
        if np.any(valid_smape)
        else 0.0
    )

    scale = float(np.mean(np.abs(np.diff(insample_values))))
    mase = float(mae / scale) if scale > 0 else np.nan

    if previous_actual is None:
        directional_accuracy = np.nan
    else:
        previous = np.asarray(previous_actual, dtype=float).reshape(-1)
        if previous.shape != actual_values.shape:
            raise ValueError("previous_actual must match actual shape")
        actual_direction = np.sign(actual_values - previous)
        predicted_direction = np.sign(predicted_values - previous)
        directional_accuracy = float(
            np.mean(actual_direction == predicted_direction)
        )

    return ForecastMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        smape=smape,
        mase=mase,
        directional_accuracy=directional_accuracy,
    )


def rolling_origin_baseline_backtest(
    series: pd.Series,
    *,
    initial_train_size: int,
    step: int = 1,
    season_length: int = 52,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate simple one-step baselines through rolling-origin validation.

    Every prediction at time t uses only observations strictly before t.
    Returns a prediction-level frame and a model-level leaderboard.
    """
    clean = _clean_series(series)
    if isinstance(initial_train_size, bool) or not isinstance(
        initial_train_size, (int, np.integer)
    ):
        raise TypeError("initial_train_size must be an integer")
    if initial_train_size < 3 or initial_train_size >= len(clean):
        raise ValueError(
            "initial_train_size must be at least 3 and smaller than series length"
        )
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
        raise TypeError("step must be a positive integer")
    if step < 1:
        raise ValueError("step must be at least 1")

    rows: list[dict] = []
    for position in range(initial_train_size, len(clean), step):
        train = clean.iloc[:position]
        actual = float(clean.iloc[position])
        previous = float(clean.iloc[position - 1])
        timestamp = clean.index[position]

        forecasts = {
            "Last Value": float(last_value_forecast(train, 1)[0]),
            "Drift": float(drift_forecast(train, 1)[0]),
            "Seasonal Naive": float(
                seasonal_naive_forecast(
                    train,
                    1,
                    season_length=season_length,
                )[0]
            ),
        }
        for model, prediction in forecasts.items():
            rows.append(
                {
                    "date": timestamp,
                    "model": model,
                    "actual": actual,
                    "predicted": prediction,
                    "previous_actual": previous,
                }
            )

    predictions = pd.DataFrame(rows)
    leaderboard_rows: list[dict] = []
    insample = clean.iloc[:initial_train_size]
    for model, group in predictions.groupby("model", sort=False):
        metrics = forecast_metrics(
            group["actual"],
            group["predicted"],
            insample=insample,
            previous_actual=group["previous_actual"],
        )
        leaderboard_rows.append(
            {
                "Model": model,
                "MAE": metrics.mae,
                "RMSE": metrics.rmse,
                "MAPE (%)": metrics.mape,
                "sMAPE (%)": metrics.smape,
                "MASE": metrics.mase,
                "Directional Accuracy": metrics.directional_accuracy,
                "Forecasts": len(group),
            }
        )

    leaderboard = (
        pd.DataFrame(leaderboard_rows)
        .sort_values(["MASE", "RMSE"], na_position="last")
        .reset_index(drop=True)
    )
    return predictions, leaderboard
