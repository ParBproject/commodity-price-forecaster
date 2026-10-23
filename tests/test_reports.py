import io

import pandas as pd

from src.reports import predictions_csv, validation_report_markdown


def test_markdown_report_contains_scores_and_settings():
    table = pd.DataFrame({"Model": ["Drift", "Naive"], "MASE": [0.98, float("nan")]})
    md = validation_report_markdown(
        commodity="Gold", ticker="GC=F", start="2020-01-01", end="2024-12-31",
        generated_at="2026-10-23 17:00 UTC", settings={"Origins": 52}, table=table,
        backtest_summary={"mean_MAPE": 5.5861, "n_origins": 40},
    )
    assert md.startswith("# Forecast validation report: Gold (`GC=F`)")
    assert "| Drift | 0.9800 |" in md and "| Naive | n/a |" in md and "- Origins: 52" in md
    assert "- mean_MAPE: 5.5861" in md and "- n_origins: 40" in md


def test_predictions_csv_round_trips_with_iso_dates():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-07", "2024-01-14"]),
                       "model": ["Drift", "Drift"], "predicted": [1.5, 2.5]})
    back = pd.read_csv(io.BytesIO(predictions_csv(df)))
    assert back["date"].tolist() == ["2024-01-07", "2024-01-14"]
    assert back["predicted"].tolist() == [1.5, 2.5]
