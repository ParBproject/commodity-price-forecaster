import numpy as np
import pandas as pd
import pytest

from src.validation import (
    drift_forecast,
    forecast_metrics,
    last_value_forecast,
    rolling_origin_baseline_backtest,
    seasonal_naive_forecast,
)


@pytest.fixture
def weekly_series():
    index = pd.date_range("2022-01-02", periods=90, freq="W")
    trend = np.linspace(100.0, 130.0, len(index))
    seasonal = 4.0 * np.sin(np.arange(len(index)) * 2 * np.pi / 13)
    return pd.Series(trend + seasonal, index=index)


def test_last_value_repeats_latest_observation(weekly_series):
    forecast = last_value_forecast(weekly_series.iloc[:20], horizon=3)
    assert np.allclose(forecast, weekly_series.iloc[19])


def test_drift_uses_only_training_endpoints():
    train = pd.Series([10.0, 12.0, 14.0])
    forecast = drift_forecast(train, horizon=2)
    assert np.allclose(forecast, [16.0, 18.0])


def test_seasonal_naive_uses_prior_cycle():
    train = pd.Series(np.arange(1.0, 9.0))
    forecast = seasonal_naive_forecast(train, horizon=2, season_length=4)
    assert np.allclose(forecast, [5.0, 6.0])


def test_mase_equals_one_for_naive_scale_error():
    insample = np.array([10.0, 12.0, 14.0, 16.0])
    metrics = forecast_metrics(
        actual=[18.0, 20.0],
        predicted=[16.0, 18.0],
        insample=insample,
        previous_actual=[16.0, 18.0],
    )
    assert metrics.mae == pytest.approx(2.0)
    assert metrics.mase == pytest.approx(1.0)


def test_rolling_origin_never_uses_future_data(weekly_series):
    predictions, leaderboard = rolling_origin_baseline_backtest(
        weekly_series,
        initial_train_size=52,
        step=4,
        season_length=52,
    )
    assert not predictions.empty
    assert set(leaderboard["Model"]) == {
        "Last Value",
        "Drift",
        "Seasonal Naive",
    }
    assert predictions["date"].min() == weekly_series.index[52]
    assert (leaderboard["Forecasts"] > 0).all()


def test_directional_accuracy_is_bounded(weekly_series):
    _, leaderboard = rolling_origin_baseline_backtest(
        weekly_series,
        initial_train_size=30,
        step=2,
        season_length=13,
    )
    values = leaderboard["Directional Accuracy"].dropna()
    assert ((values >= 0.0) & (values <= 1.0)).all()
