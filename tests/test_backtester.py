import time

import numpy as np
import pandas as pd
import pytest

from src.backtester import arima_model_fn, walk_forward_backtest


def _series(n=260, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-06", periods=n, freq="W-SUN")
    return pd.Series(80 + np.cumsum(rng.normal(0, 1.5, n)), index=idx)


def test_last_value_model_known_errors():
    s = pd.Series([1.0, 2, 3, 4, 5, 6], index=pd.date_range("2024-01-07", periods=6, freq="W"))

    def last(train):
        v = train.iloc[-1]
        return np.full(2, v), np.full(2, v - 1.5), np.full(2, v + 1.5)

    out = walk_forward_backtest(s, last, initial_train_size=2, step_size=2, horizon=2)
    p = out["predictions"]
    # origins after 2 and 4 obs; forecasts 2 -> actual 3,4 ; 4 -> actual 5,6
    assert p["forecast"].tolist() == [2, 2, 4, 4]
    assert p["abs_error"].tolist() == [1, 2, 1, 2]
    assert out["summary"]["mean_MAE"] == pytest.approx(1.5)
    assert out["summary"]["hit_rate"] == pytest.approx(0.5)
    assert out["summary"]["n_origins"] == 2


def test_no_lookahead():
    s = _series(60)
    seen = []

    def spy(train):
        seen.append(train.index[-1])
        v = train.iloc[-1]
        return np.full(3, v), np.full(3, v), np.full(3, v)

    out = walk_forward_backtest(s, spy, initial_train_size=30, step_size=5, horizon=3)
    p = out["predictions"]
    assert (p["date"] > p["origin"]).all()
    assert seen == sorted(seen) and seen[0] == s.index[29]


def test_arima_five_year_weekly_under_60s():
    s = _series(260)
    t0 = time.perf_counter()
    out = walk_forward_backtest(s, arima_model_fn(horizon=4), 104, 4, 4)
    assert time.perf_counter() - t0 < 60
    assert out["summary"]["n_failed_origins"] == 0
    assert 0 <= out["summary"]["hit_rate"] <= 1
    assert out["by_origin"].shape[0] == out["summary"]["n_origins"] == 39


def test_validates_arguments():
    with pytest.raises(ValueError):
        walk_forward_backtest(_series(10), arima_model_fn(), 10, 1, 1)
