"""Offline tests against the committed USDA NASS snapshot and variant results."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.benchmark import complete_weekly_series, load_snapshot
from src.usda_loader import AG_TICKERS, load_cache, release_date, yield_releases, yield_surprise_weekly
from src.usda_variant import BASE, VARIANT, evaluate_usda_variant

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cache():
    return load_cache()


def test_snapshot_is_real_nass_pull(cache):
    assert cache["source"] == "USDA NASS Quick Stats API"
    assert cache["downloaded_on"] == "2026-10-23"
    for t in AG_TICKERS:
        rows = cache["records"][t]
        assert len(rows) >= 40
        assert all(r["agg_level_desc"] == "NATIONAL" for r in rows)
        assert all("BU / ACRE" in r["short_desc"] for r in rows)
    assert "key" not in json.dumps(cache).lower().replace("keep", "")


def test_known_published_values(cache):
    corn = yield_releases(cache["records"]["ZC=F"]).set_index(["year", "reference_period"])["yield"]
    assert corn[(2025, "YEAR")] == pytest.approx(186.5)
    assert corn[(2026, "YEAR - AUG FORECAST")] == pytest.approx(180.7)


def test_release_dates():
    assert release_date(2025, "YEAR - AUG FORECAST") == pd.Timestamp("2025-08-12")
    assert release_date(2025, "YEAR") == pd.Timestamp("2026-01-12")
    assert release_date(2025, "MARKETING YEAR") is None


def test_feature_has_no_lookahead(cache):
    idx = pd.date_range("2026-08-02", periods=4, freq="W-SUN")
    f = yield_surprise_weekly(cache["records"]["ZC=F"], idx)
    # the Aug-2026 forecast (released Aug 12) is invisible on Aug 9, visible on Aug 16
    assert f.iloc[1] == f.iloc[0]
    assert f.iloc[2] == pytest.approx(float(pd.Series([180.7 / 186.5]).apply("log")[0]))


def test_committed_variant_numbers():
    r = json.loads((ROOT / "results" / "usda_variant.json").read_text())
    assert r["average_mase"][BASE] == pytest.approx(0.844096, abs=1e-6)
    assert r["average_mase"][VARIANT] == pytest.approx(0.841346, abs=1e-6)


def test_variant_recomputes_from_snapshots(cache):
    meta = json.loads((ROOT / "data" / "snapshot.json").read_text())
    prices = load_snapshot(ROOT / "data" / meta["file"])
    committed = json.loads((ROOT / "results" / "usda_variant.json").read_text())
    _, scores = evaluate_usda_variant(prices["ZW=F"], cache["records"]["ZW=F"], n_origins=52)
    for m in (BASE, VARIANT):
        assert scores[m]["MASE"] == pytest.approx(
            committed["per_commodity"]["ZW=F"][m]["MASE"], rel=1e-6)
    assert len(complete_weekly_series(prices["ZW=F"])) > 52
