from src.forecaster import compute_metrics

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
        assert result is not None

    def test_single_element(self):
        result = compute_metrics([5], [5])
        assert result["MAE"] == 0.0

    def test_negative_values(self):
        result = compute_metrics([-5, -10], [-4, -9])
        assert result["MAE"] == 1.0
        # actual = [-5, -10], predicted = [-4, -9] → handles negatives
        