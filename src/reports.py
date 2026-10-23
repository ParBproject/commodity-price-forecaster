"""Downloadable validation reports (Markdown summary + CSV of every out-of-sample forecast)."""

from __future__ import annotations

import pandas as pd


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [("n/a" if pd.isna(v) else floatfmt.format(v)) if isinstance(v, float) else str(v)
                 for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validation_report_markdown(*, commodity: str, ticker: str, start, end, generated_at: str,
                               settings: dict, table: pd.DataFrame,
                               backtest_summary: dict | None = None) -> str:
    """Self-contained Markdown report of what was evaluated, how, and the scores."""
    parts = [
        f"# Forecast validation report: {commodity} (`{ticker}`)",
        "",
        f"- Generated: {generated_at}",
        f"- Data: Yahoo Finance via yfinance, daily closes {start} to {end}, resampled to week-end closes",
        "- Method: rolling-origin one-step forecasts; each forecast is fit only on weeks before it",
        *[f"- {k}: {v}" for k, v in settings.items()],
        "",
        "## Baseline leaderboard (lower MASE is better)",
        "",
        _md_table(table),
    ]
    if backtest_summary:
        parts += ["", "## Walk-forward ARIMA(1,1,1) backtest", "",
                  *[f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}"
                    for k, v in backtest_summary.items()]]
    parts += ["", "Educational research output. Not financial advice.", ""]
    return "\n".join(parts)


def predictions_csv(predictions: pd.DataFrame) -> bytes:
    """CSV bytes with ISO dates, for a download button."""
    out = predictions.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8")
