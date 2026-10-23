import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # CI installs the pinned CPU wheel, so this runs there

from src.forecaster import fit_best_arima  # noqa: E402
from src.lstm_forecaster import EPOCHS, WINDOW, fit_lstm  # noqa: E402


def _series(n=262, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-05", periods=n, freq="W-SUN")
    return pd.Series(60 * np.exp(np.cumsum(rng.normal(0.001, 0.03, n))), index=idx)


def test_same_output_shape_as_arima():
    s = _series()
    t0 = time.perf_counter()
    res = fit_lstm(s, horizon=12, alpha=0.05)
    assert time.perf_counter() - t0 < 60
    arima = fit_best_arima(s, horizon=12, alpha=0.05)
    assert set(arima) >= {"forecast_df", "metrics", "model"}
    assert set(res) >= {"forecast_df", "metrics", "model", "test", "test_pred"}
    fc = res["forecast_df"]
    assert list(fc.columns) == ["forecast", "lower", "upper"] and len(fc) == 12
    assert fc.index.equals(arima["forecast_df"].index)
    assert (fc["lower"] < fc["forecast"]).all() and (fc["forecast"] < fc["upper"]).all()
    assert (fc["upper"] - fc["lower"]).is_monotonic_increasing
    assert {"MAE", "RMSE", "MAPE"} <= set(res["metrics"])
    assert len(res["test_pred"]) == len(res["test"]) == len(s) - int(len(s) * 0.85)


def test_metrics_are_holdout_errors():
    res = fit_lstm(_series(), horizon=4)
    err = np.asarray(res["test_pred"]) - res["test"].to_numpy()
    assert res["metrics"]["MAE"] == pytest.approx(np.mean(np.abs(err)), abs=1e-3)  # compute_metrics rounds


def test_deterministic_and_documented():
    a = fit_lstm(_series(), horizon=4)["forecast_df"]
    b = fit_lstm(_series(), horizon=4)["forecast_df"]
    pd.testing.assert_frame_equal(a, b)
    import src.lstm_forecaster as m

    for word in ("52", "32 hidden", "dropout 0.2", "150", "85/15"):
        assert word in m.__doc__
    assert WINDOW == 52 and EPOCHS == 150


def test_too_short_returns_none():
    assert fit_lstm(_series(n=60), horizon=4) is None
