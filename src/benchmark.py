"""Rolling-origin leaderboard across commodities, baselines, ARIMA and Prophet.

Everything here is import-safe without Streamlit, Plotly or Prophet so the
numbers can be regenerated and tested in CI. Prophet is optional: when it is
not installed it is simply left out of the model set.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.market import weekly_last
from src.validation import (
    drift_forecast,
    forecast_metrics,
    last_value_forecast,
    seasonal_naive_forecast,
)

SEASON_LENGTH = 52
DEFAULT_ORIGINS = 52
ARIMA_ORDER = (1, 1, 1)

OneStepModel = Callable[[pd.Series], float]


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Read the committed daily snapshot (one column per ticker)."""
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return frame.sort_index()


def complete_weekly_series(daily: pd.Series) -> pd.Series:
    """Sunday-ended weekly closes, dropping a trailing week that is not over.

    A week is complete when the snapshot reaches its Friday close or later.
    """
    daily = daily.dropna()
    weekly = weekly_last(daily)
    last_day = pd.Timestamp(daily.index[-1]).normalize()
    if weekly.index[-1] - last_day > pd.Timedelta(days=2):
        weekly = weekly.iloc[:-1]
    return weekly


def arima_one_step(train: pd.Series, order: tuple[int, int, int] = ARIMA_ORDER) -> float:
    """One-step ARIMA forecast fit only on the training prefix."""
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(np.asarray(train, dtype=float), order=order).fit()
        return float(np.asarray(fitted.forecast(steps=1))[0])


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
    except Exception:
        return False
    return True


def prophet_one_step(train: pd.Series) -> float:
    """One-step Prophet forecast (yearly seasonality, MAP fit)."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    history = pd.DataFrame({"ds": pd.DatetimeIndex(train.index), "y": train.to_numpy()})
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=0,
    )
    model.fit(history)
    future = model.make_future_dataframe(periods=1, freq="W", include_history=False)
    return float(model.predict(future)["yhat"].iloc[0])


def default_models(include_prophet: bool | None = None) -> dict[str, OneStepModel]:
    models: dict[str, OneStepModel] = {
        "Naive": lambda train: float(last_value_forecast(train, 1)[0]),
        "Drift": lambda train: float(drift_forecast(train, 1)[0]),
        "Seasonal Naive": lambda train: float(
            seasonal_naive_forecast(train, 1, season_length=SEASON_LENGTH)[0]
        ),
        "ARIMA(1,1,1)": arima_one_step,
    }
    if include_prophet is None:
        include_prophet = prophet_available()
    if include_prophet:
        models["Prophet"] = prophet_one_step
    return models


def rolling_origin_predictions(
    series: pd.Series,
    models: dict[str, OneStepModel],
    *,
    n_origins: int = DEFAULT_ORIGINS,
) -> pd.DataFrame:
    """One-step forecasts for the last ``n_origins`` weeks.

    The forecast for week t is fit on weeks strictly before t.
    """
    if n_origins < 1 or n_origins >= len(series) - 2:
        raise ValueError("n_origins must be positive and leave training history")
    first = len(series) - n_origins
    rows = []
    for position in range(first, len(series)):
        train = series.iloc[:position]
        row = {
            "date": series.index[position],
            "actual": float(series.iloc[position]),
            "previous_actual": float(series.iloc[position - 1]),
        }
        for name, model in models.items():
            row[name] = model(train)
        rows.append(row)
    return pd.DataFrame(rows)


def score_predictions(
    predictions: pd.DataFrame,
    insample: pd.Series,
    model_names: list[str],
) -> dict[str, dict]:
    """MAE, sMAPE, MASE (+ RMSE, MAPE, direction) per model."""
    scores = {}
    for name in model_names:
        metrics = forecast_metrics(
            predictions["actual"],
            predictions[name],
            insample=insample,
            previous_actual=predictions["previous_actual"],
        )
        scores[name] = {
            "MAE": metrics.mae,
            "RMSE": metrics.rmse,
            "MAPE": metrics.mape,
            "sMAPE": metrics.smape,
            "MASE": metrics.mase,
            "DirectionalAccuracy": (
                None
                if not np.isfinite(metrics.directional_accuracy)
                else metrics.directional_accuracy
            ),
        }
    return scores


def evaluate_commodity(
    daily: pd.Series,
    models: dict[str, OneStepModel],
    *,
    n_origins: int = DEFAULT_ORIGINS,
) -> tuple[pd.Series, pd.DataFrame, dict[str, dict]]:
    weekly = complete_weekly_series(daily)
    predictions = rolling_origin_predictions(weekly, models, n_origins=n_origins)
    insample = weekly.iloc[: len(weekly) - n_origins]
    scores = score_predictions(predictions, insample, list(models))
    return weekly, predictions, scores


def average_scores(per_commodity: dict[str, dict[str, dict]], model_names: list[str]) -> dict:
    """Unweighted mean of each metric across commodities."""
    averages = {}
    for name in model_names:
        averages[name] = {}
        for metric in ("MAE", "sMAPE", "MASE"):
            values = [per_commodity[c][name][metric] for c in per_commodity]
            averages[name][metric] = float(np.mean(values))
    return averages


README_START = "<!-- results:start (generated by scripts/run_leaderboard.py) -->"
README_END = "<!-- results:end -->"
_SHORT = {"Seasonal Naive": "Seasonal naive", "ARIMA(1,1,1)": "ARIMA(1,1,1)"}


_BASELINES = ("Naive", "Drift", "Seasonal Naive")


def _verdict(winner: str, runner_up: str, lead: float, models: list[str]) -> str:
    fitted = [m for m in models if m not in _BASELINES]
    if winner in _BASELINES:
        text = f"A simple baseline won. Its lead over {runner_up} is {lead:.4f} MASE"
        if runner_up in _BASELINES:
            text += ", so the two baselines are effectively tied"
        return text + (
            f", and {' and '.join(fitted)} did not beat it on average at a "
            "one-week horizon."
        )
    return f"Its lead over {runner_up} is {lead:.4f} MASE."


def results_markdown(results: dict) -> str:
    """README "Results" body rendered only from the leaderboard JSON."""
    ranking = results["ranking_by_average_mase"]
    win = results["winner"]
    window = results["evaluation"]["window"]
    data = results["data"]
    runner_up = ranking[1]
    lead = results["average"][runner_up]["MASE"] - win["average_mase"]
    lines = [
        README_START,
        "",
        f"**Evaluation window:** {results['evaluation']['n_origins']} weekly one-step "
        f"forecasts from {window['first_forecast_week']} to {window['last_forecast_week']} "
        f"(training history from {window['training_start']}), across "
        f"{len(results['commodities'])} futures. Data: {data['source']} daily closes, "
        f"snapshot `{data['snapshot_file']}` taken {data['snapshot_date']}.",
        "",
        f"**Winner: {win['model']}**, average MASE {win['average_mase']:.3f} versus "
        f"{win['seasonal_naive_average_mase']:.3f} for seasonal naive, "
        f"{win['mase_difference_vs_seasonal_naive']:.3f} lower "
        f"({win['pct_better_than_seasonal_naive']:.2f}% better). "
        + _verdict(win["model"], runner_up, lead, results["models"]),
        "",
        "| Rank | Model | Avg MASE | Avg sMAPE (%) | Avg MAE |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, model in enumerate(ranking, 1):
        a = results["average"][model]
        name = f"**{model}**" if model == win["model"] else model
        lines.append(
            f"| {rank} | {name} | {a['MASE']:.3f} | {a['sMAPE']:.2f} | {a['MAE']:.2f} |"
        )
    lines += [
        "",
        "MASE by commodity (lower is better, best in **bold**):",
        "",
        "| Commodity | " + " | ".join(_SHORT.get(m, m) for m in ranking) + " |",
        "|---|" + "---:|" * len(ranking),
    ]
    for ticker, scores in results["per_commodity"].items():
        best = results["commodity_winners"][ticker]
        cells = [
            f"**{scores[m]['MASE']:.3f}**" if m == best else f"{scores[m]['MASE']:.3f}"
            for m in ranking
        ]
        lines.append(
            f"| {results['commodities'][ticker]} (`{ticker}`) | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "![Average MASE by model](results/leaderboard_mase.png)",
        "",
        f"![Forecast vs actual, {win['model']} and seasonal naive](results/forecast_vs_actual.png)",
        "",
        "MASE is scaled by the in-sample mean absolute weekly change before the first "
        "origin. Values above 1 mean the evaluation year moved more than the training "
        "history did (gold and silver especially), not that the models broke. Seasonal "
        "naive repeats the price from 52 weeks earlier, which is a poor guide for "
        "trending futures prices.",
        "",
        "Reproduce: `python scripts/run_leaderboard.py` (about a minute; Prophet is "
        "included when installed). The numbers are pinned by "
        "`tests/test_leaderboard.py::test_committed_leaderboard_headline_numbers` and "
        "recomputed from the snapshot by "
        "`tests/test_leaderboard.py::test_baselines_and_arima_recompute_from_snapshot`.",
        "",
        README_END,
    ]
    return "\n".join(lines)
