import numpy as np
import pandas as pd
import pytest

from src.ticker import banner_html, fetch_quote, summarize_quote

NOW = pd.Timestamp("2026-10-21 15:00", tz="America/New_York")


def test_up_and_live():
    q = summarize_quote(105.0, 100.0, NOW - pd.Timedelta(minutes=10), NOW)
    assert q["pct_change"] == pytest.approx(5.0) and q["direction"] == "up" and q["status"] == "Live"


def test_down_and_closed_when_stale():
    q = summarize_quote(95.0, 100.0, NOW - pd.Timedelta(hours=20), NOW)
    assert q["pct_change"] == pytest.approx(-5.0) and q["direction"] == "down" and q["status"] == "Market closed"


def test_banner_colours_and_label():
    quotes = {
        "Gold": summarize_quote(2.0, 1.0, NOW, NOW),
        "Corn": summarize_quote(1.0, 2.0, NOW - pd.Timedelta(days=1), NOW),
    }
    html = banner_html(quotes, {"Gold": "GC=F", "Corn": "ZC=F"})
    assert "GC=F" in html and "#10B981" in html and "+100.00%" in html
    assert "#EF4444" in html and "-50.00%" in html and "Market closed" in html


def test_previous_close_is_prior_session(monkeypatch):
    import yfinance as yf

    tz = "America/New_York"

    class FakeTicker:
        def __init__(self, t):
            pass

        def history(self, period, interval, auto_adjust):
            if interval == "5m":
                idx = pd.date_range("2026-10-21 09:30", periods=12, freq="5min", tz=tz)
                return pd.DataFrame({"Close": np.linspace(101, 102, 12)}, index=idx)
            idx = pd.DatetimeIndex(pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21"]).tz_localize(tz))
            return pd.DataFrame({"Close": [99.0, 100.0, 102.0]}, index=idx)

    monkeypatch.setattr(yf, "Ticker", FakeTicker)
    q = fetch_quote("CL=F", now=pd.Timestamp("2026-10-21 10:40", tz=tz))
    assert q["prev_close"] == 100.0  # 10-20 close, not today's bar
    assert round(q["pct_change"], 6) == 2.0 and q["status"] == "Live"
