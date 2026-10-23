"""Live price banner: latest intraday print vs previous daily close for each future.

Quotes come from Yahoo Finance via yfinance (delayed, typically 10-15 minutes for
futures). When the last print is older than ``STALE_AFTER`` the market is treated as
closed and the banner shows the last price with a "Market closed" label.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

logger = logging.getLogger(__name__)

STALE_AFTER = pd.Timedelta(minutes=30)


def summarize_quote(last: float, prev_close: float, last_time: pd.Timestamp,
                    now: pd.Timestamp) -> dict:
    """Percent change, direction and open/closed status for one quote."""
    change = (last / prev_close - 1) * 100 if prev_close else float("nan")
    stale = (now - last_time) > STALE_AFTER
    return {
        "last": float(last),
        "prev_close": float(prev_close),
        "pct_change": float(change),
        "direction": "up" if change > 0 else "down" if change < 0 else "flat",
        "status": "Market closed" if stale else "Live",
        "as_of": last_time,
    }


def fetch_quote(ticker: str, now: pd.Timestamp | None = None) -> dict | None:
    """Latest 5-minute bar and the previous session's daily close for one ticker."""
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)  # per-object state; safe across threads (unlike yf.download)
        intraday = t.history(period="5d", interval="5m", auto_adjust=False)
        daily = t.history(period="10d", interval="1d", auto_adjust=False)
    except Exception:
        logger.warning("Quote fetch failed for %s", ticker, exc_info=True)
        return None
    if intraday.empty or daily.empty:
        return None
    last_time = intraday.index[-1]
    last = float(intraday["Close"].iloc[-1])
    session = last_time.tz_convert(daily.index.tz).normalize() if daily.index.tz else last_time.normalize()
    prior = daily.loc[daily.index < session, "Close"]
    if prior.empty:
        return None
    now = now if now is not None else pd.Timestamp.now(tz=last_time.tz)
    return summarize_quote(last, float(prior.iloc[-1]), last_time, now)


def fetch_quotes(tickers: dict[str, str], max_workers: int = 8) -> dict[str, dict]:
    """{name: quote} for every ticker that returned data, fetched in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {name: pool.submit(fetch_quote, t) for name, t in tickers.items()}
        out = {name: f.result() for name, f in futures.items()}
    return {n: q for n, q in out.items() if q is not None}


def banner_html(quotes: dict[str, dict], tickers: dict[str, str]) -> str:
    """Compact HTML strip; green for up, red for down."""
    cells = []
    for name, q in quotes.items():
        colour = {"up": "#10B981", "down": "#EF4444"}.get(q["direction"], "#94A3B8")
        arrow = {"up": "▲", "down": "▼"}.get(q["direction"], "•")
        closed = ' <span style="color:#94A3B8">· Market closed</span>' if q["status"] != "Live" else ""
        cells.append(
            f'<span style="white-space:nowrap;margin-right:1.4rem"><b>{tickers[name]}</b> '
            f'{q["last"]:,.2f} <span style="color:{colour}">{arrow} {q["pct_change"]:+.2f}%</span>'
            f"{closed}</span>"
        )
    return (
        '<div style="background:#0B1120;color:#E5E7EB;border:1px solid #1F2937;'
        'border-radius:12px;padding:.6rem 1rem;font-size:.88rem;overflow-x:auto;'
        'margin-bottom:.8rem">' + "".join(cells) + "</div>"
    )
