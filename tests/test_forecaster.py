import pandas as pd
import pytest

from src.forecaster import compute_metrics, select_prophet_components


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
