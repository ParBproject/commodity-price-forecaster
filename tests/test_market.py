import numpy as np
import pandas as pd
import pytest

from src.market import (
    compute_risk_scores,
    price_risk_components,
    simulate_weather_data,
    trailing_high_low,
    weather_price_frame,
    weekly_last,
)


def test_trailing_high_low_ignores_an_older_spike():
    index = pd.bdate_range("2020-01-01", "2024-12-31")
    prices = pd.Series(50.0, index=index)
    prices.iloc[0] = 400.0
    prices.iloc[-10] = 80.0
    prices.iloc[-5] = 20.0

    window = trailing_high_low(prices, weeks=52)

    assert window["complete_window"] is True
    assert window["high"] == pytest.approx(80.0)
    assert window["low"] == pytest.approx(20.0)
    assert prices.max() == pytest.approx(400.0)


def test_trailing_high_low_does_not_claim_a_full_year_when_history_is_short():
    index = pd.bdate_range("2024-11-01", "2024-12-31")
    prices = pd.Series(np.arange(len(index), dtype=float), index=index)
    window = trailing_high_low(prices, weeks=52)
    assert window["complete_window"] is False
    assert window["high"] == pytest.approx(prices.max())
    assert window["low"] == pytest.approx(prices.min())


def test_weekly_last_carries_forward_and_does_not_backfill():
    prices = pd.Series(
        [10.0, 12.0, 30.0],
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-16"]),
    )
    weekly = weekly_last(prices)
    assert list(weekly.index) == [
        pd.Timestamp("2024-01-07"),
        pd.Timestamp("2024-01-14"),
        pd.Timestamp("2024-01-21"),
    ]
    assert weekly.tolist() == pytest.approx([12.0, 12.0, 30.0])

    revised = prices.copy()
    revised.iloc[-1] = 999.0
    assert weekly_last(revised).iloc[1] == pytest.approx(12.0)


def test_weather_frame_uses_returns_and_refuses_a_later_week():
    prices = pd.Series(
        [100.0, 110.0, 121.0],
        index=pd.to_datetime(["2024-01-07", "2024-01-14", "2024-01-21"]),
    )
    weather = pd.DataFrame(
        {"temperature_mean": [10.0, 20.0, 30.0, 999.0]},
        index=pd.to_datetime(
            ["2024-01-07", "2024-01-14", "2024-01-21", "2024-01-28"]
        ),
    )
    frame = weather_price_frame(prices, weather)
    assert frame.index.max() == pd.Timestamp("2024-01-21")
    assert 999.0 not in frame["temperature_mean"].to_numpy()
    assert frame["price_return"].tolist() == pytest.approx([0.1, 0.1])
    assert frame["temperature_mean"].tolist() == pytest.approx([20.0, 30.0])


def test_drawdown_includes_a_loss_at_the_start_of_the_window():
    prices = pd.Series(
        [100.0, 50.0, 50.0],
        index=pd.date_range("2024-01-07", periods=3, freq="W"),
    )
    components = price_risk_components(prices)
    assert components["max_drawdown"] == pytest.approx(-0.5)
    assert components["ann_vol"] == pytest.approx(np.sqrt(0.125) * np.sqrt(52))


def test_risk_window_ignores_an_older_crash_and_is_deterministic():
    index = pd.date_range("2020-01-05", periods=200, freq="W")
    prices = pd.Series(100.0, index=index, name="GC=F")
    prices.iloc[20] = 10.0
    components = price_risk_components(prices)
    assert components["max_drawdown"] == pytest.approx(0.0)
    assert components["trend_instability"] == pytest.approx(0.0)

    np.random.seed(123)
    first_draw = np.random.rand()
    first = compute_risk_scores(prices, "Gold")
    second = compute_risk_scores(prices, "Gold")
    observed = np.random.rand()
    np.random.seed(123)
    np.random.rand()
    expected = np.random.rand()

    pd.testing.assert_frame_equal(first, second)
    assert observed == expected
    assert first_draw != observed
    assert first["Volatility Score"].nunique() == 1
    assert first["Drawdown Score"].nunique() == 1
    assert first["Trend Score"].nunique() == 1


def test_simulated_weather_is_repeatable_without_touching_global_rng():
    np.random.seed(123)
    np.random.rand()
    first = simulate_weather_data("2024-01-01", "2024-06-01", seed=7)
    second = simulate_weather_data("2024-01-01", "2024-06-01", seed=7)
    observed = np.random.rand()
    np.random.seed(123)
    np.random.rand()
    expected = np.random.rand()

    pd.testing.assert_frame_equal(first, second)
    assert observed == expected
