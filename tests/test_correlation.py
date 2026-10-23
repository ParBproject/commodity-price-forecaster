import numpy as np
import pandas as pd
import pytest

from src.correlation import (
    align_weekly,
    normalized,
    return_correlation,
    rolling_correlation,
    spread_bands,
)


def _prices(seed, n=400):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


@pytest.fixture
def panel():
    return align_weekly({"A": _prices(1), "B": _prices(2), "C": _prices(3)})


def test_correlation_matches_np_corrcoef(panel):
    r = panel.pct_change().dropna()
    expected = np.corrcoef(r.to_numpy().T)
    np.testing.assert_allclose(return_correlation(panel).to_numpy(), expected, atol=1e-12)
    assert np.allclose(np.diag(return_correlation(panel)), 1)


def test_normalized_starts_at_100(panel):
    assert (normalized(panel).iloc[0] == 100).all()


def test_rolling_correlation_window(panel):
    roll = rolling_correlation(panel, "A", "B", window=12)
    r = panel.pct_change().dropna()
    assert roll.iloc[0] == pytest.approx(np.corrcoef(r["A"][:12], r["B"][:12])[0, 1])
    assert roll.between(-1, 1).all()


def test_spread_bands(panel):
    b = spread_bands(panel, "A", "B")
    ratio = panel["A"] / panel["B"]
    assert b["mean"].iloc[0] == pytest.approx(ratio.mean())
    assert b["+2σ"].iloc[0] - b["mean"].iloc[0] == pytest.approx(2 * ratio.std(ddof=1))


def test_fetch_many_parallel(monkeypatch):
    from src import data_loader

    calls = []

    def fake(ticker, start, end):
        calls.append(ticker)
        return None if ticker == "BAD" else _prices(len(ticker))

    monkeypatch.setattr(data_loader, "_ticker_history", fake)
    out = data_loader.fetch_many({"x": "AA", "y": "BBB", "z": "BAD"}, "2021-01-01", "2022-01-01")
    assert sorted(calls) == ["AA", "BAD", "BBB"]
    assert set(out) == {"x", "y"}


def test_fetch_many_does_not_use_shared_yf_download(monkeypatch):
    """yf.download is not thread-safe (it returned one series for every ticker)."""
    import yfinance as yf

    from src import data_loader

    def boom(*a, **k):
        raise AssertionError("yf.download must not be called from worker threads")

    class FakeTicker:
        def __init__(self, t):
            self.t = t

        def history(self, **k):
            idx = pd.date_range("2022-01-03", periods=30, freq="B", tz="America/New_York")
            return pd.DataFrame({"Close": np.arange(30.0) + len(self.t)}, index=idx)

    monkeypatch.setattr(yf, "download", boom)
    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    out = data_loader.fetch_many({"a": "X", "b": "YYY"}, "2022-01-01", "2022-03-01")
    assert out["a"].iloc[0] == 1 and out["b"].iloc[0] == 3
    assert out["a"].index.tz is None
