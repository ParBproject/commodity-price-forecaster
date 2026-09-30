"""Pin the committed leaderboard and check it can be recomputed from the snapshot."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.benchmark import (
    average_scores,
    default_models,
    evaluate_commodity,
    load_snapshot,
    prophet_one_step,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "leaderboard.json"
BASELINES_AND_ARIMA = ["Naive", "Drift", "Seasonal Naive", "ARIMA(1,1,1)"]


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS_PATH.read_text())


@pytest.fixture(scope="module")
def snapshot(results):
    return load_snapshot(ROOT / results["data"]["snapshot_file"])


def test_committed_leaderboard_headline_numbers(results):
    """The README and live demo quote these numbers; change them deliberately."""
    assert results["data"]["snapshot_file"] == "data/commodity_prices_daily_2026-09-30.csv"
    assert results["evaluation"]["n_origins"] == 52
    assert results["evaluation"]["window"] == {
        "training_start": "2016-01-10",
        "first_forecast_week": "2025-10-05",
        "last_forecast_week": "2026-09-27",
    }
    assert results["ranking_by_average_mase"] == [
        "Drift",
        "Naive",
        "ARIMA(1,1,1)",
        "Prophet",
        "Seasonal Naive",
    ]
    expected_mase = {
        "Drift": 2.204153,
        "Naive": 2.207064,
        "ARIMA(1,1,1)": 2.25496,
        "Prophet": 8.036689,
        "Seasonal Naive": 16.413525,
    }
    for model, value in expected_mase.items():
        assert results["average"][model]["MASE"] == pytest.approx(value, abs=1e-6)
    assert results["winner"]["model"] == "Drift"
    assert results["winner"]["pct_better_than_seasonal_naive"] == pytest.approx(86.57)


def test_committed_averages_match_per_commodity_scores(results):
    averages = average_scores(results["per_commodity"], results["models"])
    for model in results["models"]:
        for metric in ("MAE", "sMAPE", "MASE"):
            assert results["average"][model][metric] == pytest.approx(
                averages[model][metric], abs=2e-6
            )
    assert len(results["per_commodity"]) == 9


def test_baselines_and_arima_recompute_from_snapshot(results, snapshot):
    models = {
        name: model
        for name, model in default_models(include_prophet=False).items()
        if name in BASELINES_AND_ARIMA
    }
    n_origins = results["evaluation"]["n_origins"]
    for ticker, committed in results["per_commodity"].items():
        _, predictions, scores = evaluate_commodity(
            snapshot[ticker], models, n_origins=n_origins
        )
        stored = results["predictions"][ticker]
        assert [str(d.date()) for d in predictions["date"]] == stored["dates"]
        assert np.allclose(predictions["actual"], stored["actual"], atol=1e-4)
        for name in BASELINES_AND_ARIMA:
            tolerance = 1e-6 if name != "ARIMA(1,1,1)" else 1e-3
            for metric in ("MAE", "sMAPE", "MASE"):
                assert scores[name][metric] == pytest.approx(
                    committed[name][metric], rel=tolerance, abs=1e-6
                ), (ticker, name, metric)


def test_prophet_recomputes_for_one_commodity(results, snapshot):
    pytest.importorskip("prophet")
    ticker = "CL=F"
    _, _, scores = evaluate_commodity(
        snapshot[ticker],
        {"Prophet": prophet_one_step},
        n_origins=results["evaluation"]["n_origins"],
    )
    committed = results["per_commodity"][ticker]["Prophet"]
    assert scores["Prophet"]["MASE"] == pytest.approx(committed["MASE"], rel=1e-2)


def test_default_models_excludes_prophet_when_requested():
    assert "Prophet" not in default_models(include_prophet=False)


def test_readme_results_section_is_rendered_from_results_file(results):
    from src.benchmark import README_END, README_START, results_markdown

    readme = (ROOT / "README.md").read_text()
    block = readme[readme.index(README_START) : readme.index(README_END) + len(README_END)]
    assert block == results_markdown(results)
    assert "https://parbproject.github.io/commodity-price-forecaster/" in readme
