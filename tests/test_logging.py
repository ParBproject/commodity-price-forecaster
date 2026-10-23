"""src/ reports through the logging module, never print()."""

import logging
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"


def test_no_print_calls_in_src():
    offenders = [p.name for p in SRC.glob("*.py") if "print(" in p.read_text()]
    assert offenders == []


def test_modules_define_named_loggers():
    from src import data_loader, forecaster

    assert data_loader.logger.name == "src.data_loader"
    assert forecaster.logger.name == "src.forecaster"


def test_empty_price_download_is_logged(monkeypatch, caplog):
    import yfinance as yf

    from src import data_loader

    monkeypatch.setattr(yf, "download", lambda *a, **k: pd.DataFrame())
    with caplog.at_level(logging.WARNING, logger="src.data_loader"):
        out = data_loader.fetch_commodity_prices("TEST-EMPTY=F", "2020-01-01", "2020-02-01")
    assert out is None
    assert "No price data returned for TEST-EMPTY=F" in caplog.text
