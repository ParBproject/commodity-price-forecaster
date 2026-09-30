import numpy as np
import pandas as pd
import pytest

from src.forecaster import (
    apply_price_scenario,
    chronological_split,
    compute_metrics,
    create_features,
    decomposition_strength,
    fit_best_arima,
    forecast_export_frame,
    forecast_with_fixed_arima,
    future_forecast_index,
    select_prophet_components,
)
from src.formatting import format_mape


class TestComputeMetrics:
    def test_perfect_forecast(self):
        result = compute_metrics([10, 20, 30], [10, 20, 30])
        assert result["MAE"] == 0
        assert result["RMSE"] == 0
        assert result["MAPE"] == 0

    def test_constant_offset(self):
        result = compute_metrics([10, 20], [11, 21])
        assert result["MAE"] == 1.0

    def test_all_zeros_actual(self):
        # should not raise ZeroDivisionError
        result = compute_metrics([0, 0], [1, 2])
        assert result["MAE"] == 1.5
        assert result["MAPE"] is None
        assert format_mape(result["MAPE"]) == "n/a"

    def test_single_element(self):
        result = compute_metrics([5], [5])
        assert result["MAE"] == 0.0

    def test_negative_values(self):
        result = compute_metrics([-5, -10], [-4, -9])
        assert result["MAE"] == 1.0
        # actual = [-5, -10], predicted = [-4, -9] → handles negatives

    def test_rejects_empty_inputs(self):
        with pytest.raises(ValueError, match="must not be empty"):
            compute_metrics([], [])

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_metrics([1, 2], [1])


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "n/a"),
        (float("nan"), "n/a"),
        (0, "0.0%"),
        (12.34, "12.3%"),
    ],
)
def test_format_mape_handles_missing_and_numeric_values(value, expected):
    assert format_mape(value) == expected


@pytest.mark.parametrize(
    "optional_columns,expected_columns",
    [
        ({"yearly": [0.1], "weekly": [0.2]}, ["ds", "trend", "yearly", "weekly"]),
        ({"yearly": [0.1]}, ["ds", "trend", "yearly"]),
        ({"weekly": [0.2]}, ["ds", "trend", "weekly"]),
        ({}, ["ds", "trend"]),
    ],
)
def test_select_prophet_components_handles_disabled_seasonalities(
    optional_columns, expected_columns
):
    forecast = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-04"]),
            "trend": [100.0],
            "yhat": [101.0],
            **optional_columns,
        }
    )

    result = select_prophet_components(forecast)

    assert result.columns.tolist() == expected_columns
    assert result.index.equals(forecast.index)


def test_chronological_split_keeps_the_test_block_last():
    series = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2024-01-07", periods=10, freq="W"),
    )
    train, test = chronological_split(series)
    assert len(train) == 8
    assert train.index.max() < test.index.min()
    pd.testing.assert_index_equal(
        train.index.append(test.index),
        series.index,
    )


def test_future_forecast_index_starts_after_the_last_observation():
    index = future_forecast_index(pd.Timestamp("2024-12-29"), 2)
    assert list(index) == [
        pd.Timestamp("2025-01-05"),
        pd.Timestamp("2025-01-12"),
    ]


def test_scenario_scales_the_forecast_path_only():
    forecast = pd.DataFrame(
        {"forecast": [10.0], "lower": [8.0], "upper": [12.0]},
        index=pd.date_range("2025-01-05", periods=1, freq="W"),
    )
    scaled = apply_price_scenario(forecast, 1.5)
    assert scaled["forecast"].iloc[0] == pytest.approx(15.0)
    assert scaled["lower"].iloc[0] == pytest.approx(12.0)
    assert scaled["upper"].iloc[0] == pytest.approx(18.0)
    assert forecast["forecast"].iloc[0] == pytest.approx(10.0)
    with pytest.raises(ValueError, match="positive"):
        apply_price_scenario(forecast, 0)


def test_forecast_export_includes_prophet_without_arima():
    prophet = {
        "forecast_df": pd.DataFrame(
            {"forecast": [4.0], "lower": [3.0], "upper": [5.0]},
            index=pd.to_datetime(["2025-01-05"]),
        )
    }
    table = forecast_export_frame(None, prophet, confidence=95)
    assert list(table.columns) == [
        "Prophet forecast",
        "Prophet lower 95%",
        "Prophet upper 95%",
    ]


def test_features_at_t_do_not_use_the_price_or_weather_at_t():
    index = pd.date_range("2022-01-02", periods=40, freq="W")
    series = pd.Series(np.linspace(20.0, 40.0, 40), index=index)
    weather = pd.DataFrame({"temperature_mean": np.arange(40, dtype=float)}, index=index)
    features = create_features(series, lags=[1, 2], weather_df=weather)

    last = features.iloc[-1]
    assert last["lag_1"] == pytest.approx(series.iloc[-2])
    assert last["roll_mean_4"] == pytest.approx(series.iloc[-5:-1].mean())
    assert last["return_1"] == pytest.approx(series.iloc[-2] / series.iloc[-3] - 1)
    assert last["weather_temperature_mean"] == pytest.approx(
        weather["temperature_mean"].iloc[-2]
    )

    changed = series.copy()
    changed.iloc[-1] = 9999.0
    changed_weather = weather.copy()
    changed_weather.iloc[-1, 0] = 9999.0
    changed_features = create_features(
        changed, lags=[1, 2], weather_df=changed_weather
    )
    predictors = [column for column in features.columns if column not in ("price", "log_price")]
    pd.testing.assert_series_equal(features.iloc[-1][predictors], changed_features.iloc[-1][predictors])
    pd.testing.assert_frame_equal(
        features.iloc[:-1],
        changed_features.iloc[:-1],
    )


def test_decomposition_strength_stays_inside_zero_and_one():
    class _Result:
        trend = np.array([5.0, -5.0, 5.0, -5.0])
        seasonal = np.array([-4.8, 4.8, -4.8, 4.8])
        resid = np.array([0.2, -0.2, 0.2, -0.2])
        observed = trend + seasonal + resid

    strength = decomposition_strength(_Result())
    old_ratio = _Result.seasonal.std(ddof=1) / _Result.observed.std(ddof=1)
    assert old_ratio > 1
    assert 0 <= strength["seasonal_strength"] <= 1
    assert 0 <= strength["trend_strength"] <= 1

    resid_values = np.array([0.1, -0.1, 0.1, -0.1])
    seasonal_values = np.array([1.0, -1.0, 1.0, -1.0])
    denominator = np.var(seasonal_values + resid_values, ddof=1)
    expected = 1.0 - np.var(resid_values, ddof=1) / denominator

    class _Seasonal:
        trend = np.zeros(4)
        seasonal = seasonal_values
        resid = resid_values

    assert decomposition_strength(_Seasonal())["seasonal_strength"] == pytest.approx(
        expected
    )


def test_fixed_arima_holdout_uses_the_training_model_only():
    from statsmodels.tsa.arima.model import ARIMA

    index = pd.date_range("2021-01-03", periods=80, freq="W")
    values = 50 + np.cumsum(np.random.default_rng(0).normal(0, 1, len(index)))
    series = pd.Series(values, index=index)
    result = forecast_with_fixed_arima(series, horizon=4, alpha=0.05)

    assert result is not None
    assert result["order"] == (1, 1, 1)
    assert result["train"].index.max() < result["test"].index.min()
    assert result["forecast_df"].index.min() > series.index.max()
    assert len(result["forecast_df"]) == 4

    fresh = ARIMA(result["train"], order=(1, 1, 1)).fit()
    expected = np.asarray(fresh.forecast(steps=len(result["test"])), dtype=float)
    assert np.allclose(result["test_pred"], expected, rtol=1e-6, atol=1e-6)
    scored = compute_metrics(result["test"].to_numpy(), expected)
    for key in ("MAE", "RMSE", "MAPE"):
        assert result["metrics"][key] == scored[key]


def test_auto_arima_uses_fixed_order_when_pmdarima_is_absent():
    import importlib.util

    if importlib.util.find_spec("pmdarima") is not None:
        pytest.skip("pmdarima is installed, so the fixed-order fallback is not used")

    index = pd.date_range("2021-01-03", periods=40, freq="W")
    values = 50 + np.cumsum(np.random.default_rng(1).normal(0, 1, len(index)))
    series = pd.Series(values, index=index)
    result = fit_best_arima(series, horizon=3)
    assert result is not None
    assert result["order"] == (1, 1, 1)
    assert result["selection"] == "fixed"
    assert result["forecast_df"].index.min() > series.index.max()


def test_arima_returns_none_when_history_is_too_short():
    series = pd.Series(
        np.arange(10, dtype=float),
        index=pd.date_range("2024-01-07", periods=10, freq="W"),
    )
    assert fit_best_arima(series, horizon=4) is None
    assert forecast_with_fixed_arima(series, horizon=4) is None


@pytest.mark.parametrize("missing", ["ds", "trend"])
def test_select_prophet_components_requires_core_columns(missing):
    forecast = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-04"]),
            "trend": [100.0],
        }
    ).drop(columns=missing)

    with pytest.raises(KeyError, match="missing required columns"):
        select_prophet_components(forecast)
