"""Regenerate results/leaderboard.json and the README charts from the snapshot.

    python scripts/run_leaderboard.py            # all models (Prophet if installed)
    python scripts/run_leaderboard.py --no-charts

Every number in the README "Results" section and in the live demo comes from
the JSON file this script writes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmark import (  # noqa: E402
    ARIMA_ORDER,
    DEFAULT_ORIGINS,
    SEASON_LENGTH,
    average_scores,
    README_END,
    README_START,
    default_models,
    evaluate_commodity,
    load_snapshot,
    results_markdown,
)

SNAPSHOT_META = ROOT / "data" / "snapshot.json"
RESULTS = ROOT / "results" / "leaderboard.json"
CHART_DIR = ROOT / "results"
BG = "#0B1120"
PANEL = "#111827"
TEXT = "#E5E7EB"
MUTED = "#9CA3AF"
EMERALD = "#10B981"
GREY = "#6B7280"


def _round(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def build_results(n_origins: int) -> dict:
    meta = json.loads(SNAPSHOT_META.read_text())
    snapshot = load_snapshot(ROOT / "data" / meta["file"])
    models = default_models()
    names = list(models)

    per_commodity, predictions_out, window = {}, {}, None
    for ticker, label in meta["tickers"].items():
        started = time.time()
        weekly, preds, scores = evaluate_commodity(
            snapshot[ticker], models, n_origins=n_origins
        )
        print(f"{ticker}: {time.time() - started:.1f}s", flush=True)
        per_commodity[ticker] = {
            m: {k: _round(v) for k, v in s.items()} for m, s in scores.items()
        }
        this_window = {
            "training_start": str(weekly.index[0].date()),
            "first_forecast_week": str(preds["date"].iloc[0].date()),
            "last_forecast_week": str(preds["date"].iloc[-1].date()),
        }
        if window is None:
            window = this_window
        elif window != this_window:
            raise SystemExit(f"{ticker} has a different evaluation window: {this_window}")
        predictions_out[ticker] = {
            "name": label,
            "dates": [str(d.date()) for d in preds["date"]],
            "actual": [round(float(v), 4) for v in preds["actual"]],
            "forecasts": {m: [round(float(v), 4) for v in preds[m]] for m in names},
        }

    averages = average_scores(per_commodity, names)
    averages = {m: {k: _round(v) for k, v in s.items()} for m, s in averages.items()}
    ranking = sorted(names, key=lambda m: averages[m]["MASE"])
    winner = ranking[0]
    base = averages["Seasonal Naive"]["MASE"]
    win = averages[winner]["MASE"]
    commodity_winners = {
        t: min(names, key=lambda m: per_commodity[t][m]["MASE"]) for t in per_commodity
    }

    prophet_version = None
    if "Prophet" in names:
        import prophet

        prophet_version = prophet.__version__

    return {
        "schema_version": 1,
        "generated_by": "scripts/run_leaderboard.py",
        "data": {
            "source": "Yahoo Finance via yfinance",
            "snapshot_file": f"data/{meta['file']}",
            "snapshot_date": meta["downloaded_on"],
            "first_daily_date": meta["first_date"],
            "last_daily_date": meta["last_date"],
            "frequency": "weekly (Sunday-ended last close, incomplete final week dropped)",
        },
        "evaluation": {
            "method": "rolling-origin, one-step-ahead, expanding window, refit every origin",
            "n_origins": n_origins,
            "window": window,
            "season_length": SEASON_LENGTH,
            "arima_order": list(ARIMA_ORDER),
            "prophet_version": prophet_version,
            "mase_scale": "in-sample mean absolute weekly change before the first origin",
            "average": "unweighted mean across commodities",
        },
        "models": names,
        "commodities": {t: meta["tickers"][t] for t in per_commodity},
        "per_commodity": per_commodity,
        "average": averages,
        "ranking_by_average_mase": ranking,
        "winner": {
            "model": winner,
            "average_mase": win,
            "seasonal_naive_average_mase": base,
            "mase_difference_vs_seasonal_naive": _round(base - win),
            "pct_better_than_seasonal_naive": _round((base - win) / base * 100, 2),
        },
        "commodity_winners": commodity_winners,
        "predictions": predictions_out,
    }


def _style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#1F2937")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def make_charts(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranking = results["ranking_by_average_mase"]
    winner = results["winner"]["model"]
    values = [results["average"][m]["MASE"] for m in ranking]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    _style(ax, fig)
    colors = [EMERALD if m == winner else GREY for m in ranking]
    bars = ax.barh(ranking[::-1], values[::-1], color=colors[::-1])
    for bar, value in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}", va="center", color=TEXT, fontsize=9)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("Average MASE across 9 commodities (lower is better)")
    window = results["evaluation"]["window"]
    ax.set_title(
        f"One-step weekly forecasts, {window['first_forecast_week']} to "
        f"{window['last_forecast_week']}", fontsize=10, loc="left", color=TEXT)
    ax.text(1.0, -0.62, " MASE = 1 (in-sample naive error)", color=MUTED, fontsize=7,
            va="bottom")
    ax.set_xlim(0, max(values) * 1.15)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "leaderboard_mase.png", facecolor=BG)
    plt.close(fig)

    tickers = list(results["predictions"])
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), dpi=130)
    fig.patch.set_facecolor(BG)
    import pandas as pd

    for ax, ticker in zip(axes.flat, tickers):
        block = results["predictions"][ticker]
        dates = pd.to_datetime(block["dates"])
        _style(ax, fig)
        ax.plot(dates, block["actual"], color=TEXT, lw=1.2, label="Actual")
        ax.plot(dates, block["forecasts"]["Seasonal Naive"], color=GREY, lw=1,
                ls="--", label="Seasonal naive")
        ax.plot(dates, block["forecasts"][winner], color=EMERALD, lw=1.2,
                label=winner)
        ax.set_title(block["name"], fontsize=9, loc="left", color=TEXT)
        ax.tick_params(axis="x", rotation=30)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               labelcolor=TEXT)
    fig.suptitle("Forecast vs actual over the evaluation window", color=TEXT,
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(CHART_DIR / "forecast_vs_actual.png", facecolor=BG)
    plt.close(fig)


def update_readme(results: dict) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    start, end = text.index(README_START), text.index(README_END) + len(README_END)
    readme.write_text(text[:start] + results_markdown(results) + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", type=int, default=DEFAULT_ORIGINS)
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument(
        "--readme-only", action="store_true",
        help="re-render the README Results block from the committed JSON",
    )
    args = parser.parse_args()
    if args.readme_only:
        update_readme(json.loads(RESULTS.read_text()))
        return
    results = build_results(args.origins)
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=1) + "\n")
    print(f"wrote {RESULTS.relative_to(ROOT)}")
    if not args.no_charts:
        make_charts(results)
    update_readme(results)
    w = results["winner"]
    print(json.dumps({m: results["average"][m] for m in results["ranking_by_average_mase"]}, indent=1))
    print(w)


if __name__ == "__main__":
    main()
