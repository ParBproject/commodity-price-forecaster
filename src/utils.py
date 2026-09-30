"""
utils.py
========
Chart builders and risk scoring utilities for the Commodity Price Forecaster.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

from src.market import compute_risk_scores


# ─────────────────────────────────────────────────────────────────────────────
# Forecast Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast(
    historical: pd.Series,
    arima_result: Optional[dict],
    prophet_result: Optional[dict],
    commodity_name: str,
    confidence_level: int = 95,
    history_weeks: int = 104,
) -> go.Figure:
    """
    Build an interactive Plotly chart showing historical prices plus
    ARIMA and/or Prophet forecasts with confidence bands.

    Parameters
    ----------
    historical : pd.Series       Full historical weekly price series.
    arima_result : dict or None  Output of fit_best_arima().
    prophet_result : dict or None Output of fit_prophet().
    commodity_name : str         Display name for the chart title.
    confidence_level : int       CI percentage (e.g. 95).
    history_weeks : int          How many weeks of history to display.
    """
    fig = go.Figure()

    # Historical prices (show last `history_weeks` for readability)
    hist = historical.iloc[-history_weeks:]
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Historical Price",
        line=dict(color="#636EFA", width=2),
    ))

    # ARIMA forecast
    if arima_result:
        fc_df = arima_result["forecast_df"]
        fig.add_trace(go.Scatter(
            x=fc_df.index, y=fc_df["forecast"],
            name="ARIMA Forecast",
            line=dict(color="#EF553B", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=fc_df.index.tolist() + fc_df.index.tolist()[::-1],
            y=fc_df["upper"].tolist() + fc_df["lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(239,85,59,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"ARIMA {confidence_level}% CI",
            showlegend=True,
        ))

        # Show test-period fit if available
        if "test" in arima_result and "test_pred" in arima_result:
            test_series = arima_result["test"]
            test_pred = np.asarray(arima_result["test_pred"]).flatten()
            test_idx = (test_series.index if hasattr(test_series, "index")
                        else pd.RangeIndex(len(test_pred)))
            fig.add_trace(go.Scatter(
                x=test_idx, y=test_pred,
                name="ARIMA holdout forecast",
                line=dict(color="#EF553B", width=1, dash="dot"),
                opacity=0.6,
            ))

    # Prophet forecast
    if prophet_result:
        fc_df = prophet_result["forecast_df"]
        fig.add_trace(go.Scatter(
            x=fc_df.index, y=fc_df["forecast"],
            name="Prophet Forecast",
            line=dict(color="#00CC96", width=2, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=fc_df.index.tolist() + fc_df.index.tolist()[::-1],
            y=fc_df["upper"].tolist() + fc_df["lower"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(0,204,150,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"Prophet {confidence_level}% CI",
            showlegend=True,
        ))

    # Ensemble (average of both if both present)
    if arima_result and prophet_result:
        a_fc = arima_result["forecast_df"]
        p_fc = prophet_result["forecast_df"]
        common_idx = a_fc.index.intersection(p_fc.index)
        if len(common_idx) > 0:
            ensemble = (a_fc.loc[common_idx, "forecast"] +
                        p_fc.loc[common_idx, "forecast"]) / 2
            fig.add_trace(go.Scatter(
                x=common_idx, y=ensemble.values,
                name="Ensemble (Avg)",
                line=dict(color="white", width=2.5),
            ))

    fig.update_layout(
        title=f"{commodity_name} — Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                    x=1),
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_decomposition(decomp, commodity_name: str) -> go.Figure:
    """
    4-panel Plotly figure: Observed / Trend / Seasonal / Residual.

    Works with both statsmodels seasonal_decompose and STL results.
    """
    observed = pd.Series(decomp.observed).dropna()
    trend = pd.Series(decomp.trend).dropna()
    seasonal = pd.Series(decomp.seasonal).dropna()
    residual = pd.Series(decomp.resid).dropna()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        vertical_spacing=0.06,
    )

    for i, (data, color, name) in enumerate([
        (observed, "#636EFA", "Observed"),
        (trend, "#EF553B", "Trend"),
        (seasonal, "#00CC96", "Seasonal"),
        (residual, "#AB63FA", "Residual"),
    ], start=1):
        fig.add_trace(
            go.Scatter(x=data.index, y=data.values, name=name,
                       line=dict(color=color, width=1.5),
                       mode="lines"),
            row=i, col=1,
        )

    fig.update_layout(
        title=f"{commodity_name} — Seasonal Decomposition (STL)",
        template="plotly_dark",
        height=700,
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Weather Overlay Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_weather_overlay(
    prices: pd.Series,
    weather_df: pd.DataFrame,
    commodity_name: str,
    region_name: str,
) -> go.Figure:
    """
    Dual-axis chart overlaying commodity price with temperature and precipitation.

    Parameters
    ----------
    prices : pd.Series          Weekly commodity prices.
    weather_df : pd.DataFrame   Weekly weather data.
    commodity_name : str        Chart title label.
    region_name : str           Weather region label.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f"{commodity_name} Price ($)",
            f"Temperature °C — {region_name}",
            f"Precipitation mm — {region_name}",
        ],
        vertical_spacing=0.08,
    )

    # Price
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        name="Price", line=dict(color="#636EFA", width=2),
    ), row=1, col=1)

    # Exact dates only. method="nearest" can attach a later week to this week.
    aligned_weather = weather_df.reindex(prices.index)

    # Temperature
    if "temperature_mean" in aligned_weather.columns:
        temp = aligned_weather["temperature_mean"]
        fig.add_trace(go.Scatter(
            x=temp.index, y=temp.values,
            name="Temp (°C)", line=dict(color="orange", width=1.5),
        ), row=2, col=1)

    # Precipitation
    if "precipitation_sum" in aligned_weather.columns:
        precip = aligned_weather["precipitation_sum"]
        fig.add_trace(go.Bar(
            x=precip.index, y=precip.values,
            name="Precip (mm)", marker_color="steelblue", opacity=0.7,
        ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=580,
        showlegend=True,
        title=f"{commodity_name} vs Weather — {region_name}",
    )
    return fig


def risk_heatmap(risk_table: pd.DataFrame) -> go.Figure:
    """
    Plotly heatmap of risk factor scores per producing region.

    Parameters
    ----------
    risk_table : pd.DataFrame  Output of compute_risk_scores().
    """
    score_cols = ["Volatility Score", "Drawdown Score",
                  "Trend Score", "Geopolitical Score", "Overall Risk Score"]
    present = [c for c in score_cols if c in risk_table.columns]
    matrix = risk_table.set_index("Region / Producer")[present]

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale="RdYlGn_r",
        zmin=0, zmax=100,
        text=matrix.round(1).values,
        texttemplate="%{text}",
        colorbar=dict(title="Risk Score"),
    ))
    fig.update_layout(
        title="Producer Risk Heatmap (0=Low, 100=High)",
        template="plotly_dark",
        height=380,
    )
    return fig
