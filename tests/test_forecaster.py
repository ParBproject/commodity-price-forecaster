import numpy as np
import pytest
import warnings
from src.forecaster import compute_metrics

class TestComputeMetrics:

    @pytest.mark.parametrize(
        "actual, predicted, expected",
        [
            # 1. Perfect forecast: MAE = RMSE = MAPE = 0
            (
                    np.array([10.0, 20.0, 30.0]),
                    np.array([10.0, 20.0, 30.0]),
                    {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0},
            ),
            # 2. Constant offset: Offset of 5.0
            (
                    np.array([10.0, 20.0]),
                    np.array([15.0, 25.0]),
                    {"MAE": 5.0, "RMSE": 5.0, "MAPE": 37.5},
            ),
            # 3. Single-element arrays
            (
                    np.array([100.0]),
                    np.array([110.0]),
                    {"MAE": 10.0, "RMSE": 10.0, "MAPE": 10.0},
            ),
            # 4. Negative values: MAE=2.0, RMSE=2.0, MAPE=15.0
            (
                    np.array([-10.0, -20.0]),
                    np.array([-12.0, -18.0]),
                    {"MAE": 2.0, "RMSE": 2.0, "MAPE": 15.0},
            ),
        ],
    )
    def test_compute_metrics_parametrized(self, actual, predicted, expected):
        """Verify metrics for perfect, offset, single-element, and negative inputs."""
        metrics = compute_metrics(actual, predicted)

        # Use pytest.approx for floating point comparisons to avoid precision issues
        assert metrics["MAE"] == pytest.approx(expected["MAE"])
        assert metrics["RMSE"] == pytest.approx(expected["RMSE"])
        assert metrics["MAPE"] == pytest.approx(expected["MAPE"])


    def test_all_zeros_actual(self):
        """Verify MAPE handles division by zero (returns NaN) without crashing."""
        actual = np.zeros(3)
        predicted = np.array([1.0, 2.0, 3.0])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            metrics = compute_metrics(actual, predicted)

        assert np.isnan(metrics["MAPE"])
        assert metrics["MAE"] == pytest.approx(2.0)
        assert metrics["RMSE"] == pytest.approx(2.1602, abs=1e-4)


    def test_multidimensional_flattening(self):
        """Verify the function correctly handles multi-dimensional arrays."""
        actual = np.array([[10, 20], [30, 40]])
        predicted = np.array([[11, 21], [31, 41]])

        metrics = compute_metrics(actual, predicted)
        assert metrics["MAE"] == 1.0
        assert metrics["RMSE"] == 1.0


    def test_mismatched_shapes(self):
        """Ensure ValueError is raised if arrays cannot be broadcast or subtracted."""
        actual = np.array([1, 2, 3])
        predicted = np.array([1, 2])
        with pytest.raises(ValueError):
            compute_metrics(actual, predicted)


    def test_partial_zeros_mape(self):
        """Ensure MAPE ignores zero-valued actuals to avoid division by zero."""
        actual = np.array([0.0, 10.0])
        predicted = np.array([5.0, 12.0])
        metrics = compute_metrics(actual, predicted)

        assert metrics["MAPE"] == pytest.approx(20.0)
        assert metrics["MAE"] == pytest.approx(3.5)


    def test_contains_nan(self):
        """Verify that NaNs in the input propagate to all returned metrics."""
        actual = np.array([10.0, np.nan])
        predicted = np.array([11.0, 11.0])

        metrics = compute_metrics(actual, predicted)

        assert np.isnan(metrics["MAE"])
        assert np.isnan(metrics["RMSE"])
        assert np.isnan(metrics["MAPE"])


    def test_empty_inputs(self):
        """Verify behavior on empty arrays returns NaN for all metrics."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            metrics = compute_metrics(np.array([]), np.array([]))

        assert all(np.isnan(metrics[k]) for k in ["MAE", "RMSE", "MAPE"])