"""
Commodity Price Forecaster
==========================
Interactive Streamlit dashboard for forecasting commodity prices using
ARIMA/SARIMAX + Prophet, with weather data overlays, confidence intervals,
and supplier/producer risk scoring.

⚠️ DISCLAIMER: For educational/simulation purposes only. Not financial advice.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Commodity Price Forecaster",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #F8FAFC; }
    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 15px;
        padding: .85rem 1rem;
        box-shadow: 0 4px 14px rgba(15,23,42,.035);
    }
    section[data-testid="stSidebar"] {
        background: #F1F5F9;
        border-right: 1px solid #E2E8F0;
    }
    .commodity-hero {
        padding: 1.85rem 2.1rem;
        border-radius: 22px;
        background:
            radial-gradient(circle at 88% 10%, rgba(245,158,11,.22), transparent 30%),
            linear-gradient(135deg, #0F172A 0%, #1E3A5F 60%, #0F766E 130%);
        color: white;
        box-shadow: 0 18px 48px rgba(15,23,42,.14);
        margin-bottom: 1rem;
    }
    .commodity-hero small {
        color: #FDE68A;
        letter-spacing: .14em;
        text-transform: uppercase;
        font-weight: 750;
    }
    .commodity-hero h1 {
        margin: .45rem 0 0;
        font-size: 2.2rem;
        letter-spacing: -.03em;
    }
    .commodity-hero p {
        color: #DCE7F4;
        margin: .75rem 0 0;
        max-width: 880px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Module imports ─────────────────────────────────────────────────────────────
from src.data_loader import fetch_commodity_prices, fetch_weather_data
from src.forecaster import (
    fit_best_arima,
    fit_prophet,
    compute_metrics,
    decompose_series,
)
from src.validation import rolling_origin_baseline_backtest
from src.utils import (
    plot_forecast,
    plot_decomposition,
    plot_weather_overlay,
    risk_heatmap,
    compute_risk_scores,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛢️ Commodity Forecaster")
    st.markdown("---")

    # Commodity selection
    COMMODITIES = {
        "Crude Oil (WTI)": "CL=F",
        "Natural Gas": "NG=F",
        "Gold": "GC=F",
        "Silver": "SI=F",
        "Corn": "ZC=F",
        "Wheat": "ZW=F",
        "Soybeans": "ZS=F",
        "Copper": "HG=F",
        "Brent Crude": "BZ=F",
    }

    commodity_name = st.selectbox("Select Commodity", list(COMMODITIES.keys()))
    commodity_ticker = COMMODITIES[commodity_name]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date",
                                   value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("End Date",
                                 value=pd.Timestamp("2024-12-31"))

    # Forecast settings
    st.markdown("---")
    st.subheader("🔮 Forecast Settings")

    model_choice = st.selectbox("Model", ["Auto-ARIMA", "Prophet",
                                           "Both (Ensemble)"])
    forecast_horizon = st.slider("Forecast Horizon (weeks)", 4, 52, 12)
    confidence = st.slider("Confidence Interval (%)", 80, 99, 95)

    # Weather integration
    st.markdown("---")
    st.subheader("🌤️ Weather Integration")
    weather_enabled = st.checkbox("Include Weather Data", value=True)

    WEATHER_REGIONS = {
        "US Midwest (Corn/Soybeans)": (41.5, -93.6),
        "US Gulf Coast (Oil)": (29.7, -95.4),
        "Middle East (Oil)": (24.0, 54.0),
        "Brazil (Soybeans)": (-15.8, -47.9),
        "Australia (Wheat)": (-25.3, 131.0),
        "Russia (Wheat)": (55.7, 37.6),
    }
    weather_region = st.selectbox("Weather Region", list(WEATHER_REGIONS.keys()))
    weather_lat, weather_lon = WEATHER_REGIONS[weather_region]

    # Scenario adjustment
    st.markdown("---")
    st.subheader("⚡ Scenario Adjustments")
    weather_multiplier = st.slider(
        "Extreme Weather Impact Multiplier", 0.5, 3.0, 1.0, step=0.1
    )
    supply_shock = st.slider("Supply Shock (%)", -30, 30, 0)
    demand_shock = st.slider("Demand Shock (%)", -30, 30, 0)

    run_btn = st.button("🚀 Run Forecast", type="primary",
                        use_container_width=True)
    
    st.sidebar.caption("Data cached for 1 hour")
    if st.sidebar.button("🔄 Clear Data Cache"):
       st.cache_data.clear()
       st.rerun()
# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
    <div class="commodity-hero">
      <small>Commodity Forecasting Research</small>
      <h1>{commodity_name} Research Lab</h1>
      <p>
        Statistical forecasting, rolling-origin benchmark validation, uncertainty,
        weather context, decomposition, scenario shocks, and producer-risk analytics.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "Market Overview",
    "Forecast",
    "Forecast Validation",
    "Weather Context",
    "Decomposition",
    "Risk Dashboard",
])

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data(ticker, start, end, lat, lon, include_weather):
    prices = fetch_commodity_prices(ticker, str(start), str(end))
    weather = None
    if include_weather and prices is not None:
        w_start = str(prices.index[0].date())
        w_end = str(prices.index[-1].date())
        weather = fetch_weather_data(lat, lon, w_start, w_end)
    return prices, weather


if not run_btn:
    with tabs[0]:
        st.info("👈 Select a commodity and configure settings in the sidebar, "
                "then click **Run Forecast**.")
        st.markdown("""
        ### Available Models

        | Model | Description |
        |-------|-------------|
        | **Auto-ARIMA** | Automatically selects optimal (p,d,q)(P,D,Q) via AIC/BIC minimization using `pmdarima` |
        | **Prophet** | Facebook Prophet — handles seasonality, holidays, and trend changepoints |
        | **Ensemble** | Average of ARIMA and Prophet forecasts for reduced variance |

        ### Data Sources
        - **Commodity Prices**: Yahoo Finance via `yfinance` (futures contracts)
        - **Weather Data**: Open-Meteo historical weather API (free, no key required)
        """)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Loading {commodity_name} data…"):
    prices, weather_df = load_all_data(
        commodity_ticker, start_date, end_date,
        weather_lat, weather_lon, weather_enabled,
    )

if prices is None or prices.empty:
    st.error("Could not fetch commodity data. Check your ticker and date range.")
    st.stop()

# Weekly resample for cleaner forecasting
weekly = prices.resample("W").last().dropna()
returns = weekly.pct_change().dropna()

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader(f"📋 {commodity_name} Price History")

    # KPI row
    current_price = prices.iloc[-1]
    start_price = prices.iloc[0]
    total_return = (current_price / start_price - 1) * 100
    ann_vol = returns.std() * np.sqrt(52) * 100
    max_val = prices.max()
    min_val = prices.min()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price", f"${current_price:.2f}")
    c2.metric("Total Return", f"{total_return:+.1f}%")
    c3.metric("Ann. Volatility", f"{ann_vol:.1f}%")
    c4.metric("52-week High", f"${max_val:.2f}")
    c5.metric("52-week Low", f"${min_val:.2f}")

    # Main price chart with volume
    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              row_heights=[0.75, 0.25],
                              subplot_titles=[f"{commodity_name} Price ($)",
                                             "Weekly Returns (%)"])
    fig_main.add_trace(
        go.Scatter(x=weekly.index, y=weekly.values, name="Price",
                   line=dict(color="#636EFA", width=2),
                   fill="tozeroy", fillcolor="rgba(99,110,250,0.1)"),
        row=1, col=1,
    )
    # Returns bar chart
    colors = ["green" if r >= 0 else "red" for r in returns.values]
    fig_main.add_trace(
        go.Bar(x=returns.index, y=returns.values * 100, name="Returns",
               marker_color=colors, opacity=0.7),
        row=2, col=1,
    )
    fig_main.update_layout(template="plotly_dark", height=500,
                            showlegend=False)
    st.plotly_chart(fig_main, use_container_width=True)

    # Stats table
    st.subheader("📊 Descriptive Statistics")
    stats = {
        "Statistic": ["Mean Price", "Median Price", "Std Dev", "Skewness",
                      "Kurtosis", "Min", "Max", "Current"],
        "Value": [
            f"${weekly.mean():.2f}",
            f"${weekly.median():.2f}",
            f"${weekly.std():.2f}",
            f"{weekly.skew():.3f}",
            f"{weekly.kurtosis():.3f}",
            f"${weekly.min():.2f}",
            f"${weekly.max():.2f}",
            f"${weekly.iloc[-1]:.2f}",
        ],
    }
    st.dataframe(pd.DataFrame(stats), use_container_width=True,
                 hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader(f"🔮 {forecast_horizon}-Week Price Forecast")

    # Apply scenario adjustments to weekly series
    scenario_multiplier = (
        (1 + supply_shock / 100) *
        (1 + demand_shock / 100) *
        weather_multiplier
    )
    weekly_adj = weekly * scenario_multiplier if scenario_multiplier != 1.0 else weekly

    alpha = 1 - confidence / 100
    arima_result, prophet_result = None, None

    if model_choice in ["Auto-ARIMA", "Both (Ensemble)"]:
        with st.spinner("Fitting Auto-ARIMA model…"):
            arima_result = fit_best_arima(
                weekly_adj, forecast_horizon, alpha=alpha
            )

    if model_choice in ["Prophet", "Both (Ensemble)"]:
        with st.spinner("Fitting Prophet model…"):
            prophet_result = fit_prophet(
                weekly_adj, forecast_horizon, interval_width=confidence / 100
            )

    # Metrics display
    if arima_result:
        m = arima_result["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ARIMA MAE", f"{m['MAE']:.2f}")
        c2.metric("ARIMA RMSE", f"{m['RMSE']:.2f}")
        c3.metric("ARIMA MAPE", f"{m['MAPE']:.1f}%")
        c4.metric("ARIMA AIC", f"{m.get('AIC', 0):.1f}")

    if prophet_result:
        m = prophet_result["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Prophet MAE", f"{m['MAE']:.2f}")
        c2.metric("Prophet RMSE", f"{m['RMSE']:.2f}")
        c3.metric("Prophet MAPE", f"{m['MAPE']:.1f}%")

    st.markdown("---")

    # Forecast chart
    fig_fc = plot_forecast(
        weekly_adj, arima_result, prophet_result, commodity_name,
        confidence_level=confidence,
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # Scenario annotation
    if scenario_multiplier != 1.0:
        st.info(
            f"⚡ Scenario applied: Supply {supply_shock:+}% | "
            f"Demand {demand_shock:+}% | "
            f"Weather multiplier {weather_multiplier:.1f}x  →  "
            f"Net price adjustment: **{(scenario_multiplier - 1)*100:+.1f}%**"
        )

    # Forecast table
    st.subheader("📋 Forecast Values Table")
    if arima_result:
        fc_df = arima_result["forecast_df"].copy()
        fc_df.index = fc_df.index.strftime("%Y-%m-%d")
        fc_df.columns = [f"Forecast ($)", f"Lower {confidence}% CI",
                         f"Upper {confidence}% CI"]
        fc_df = fc_df.round(2)
        st.dataframe(fc_df, use_container_width=True)
        csv_str = fc_df.reset_index().rename(columns={"index": "Date"}).to_csv(index=False)
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv_str,
            file_name=f"{commodity_name}_forecast.csv",
            mime="text/csv",
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Rolling-Origin Baseline Validation")
    st.caption(
        "Every benchmark forecast is generated using only observations available "
        "before the forecast date. These simple models establish the performance "
        "floor that ARIMA, Prophet, or an ensemble should beat."
    )

    initial_train_size = max(12, min(len(weekly) - 2, max(52, int(len(weekly) * 0.60))))
    if initial_train_size >= len(weekly):
        st.info("More weekly history is required for rolling-origin validation.")
    else:
        try:
            validation_predictions, validation_table = (
                rolling_origin_baseline_backtest(
                    weekly,
                    initial_train_size=initial_train_size,
                    step=max(1, forecast_horizon // 4),
                    season_length=52,
                )
            )

            v1, v2, v3, v4 = st.columns(4)
            best_row = validation_table.iloc[0]
            v1.metric("Best baseline", str(best_row["Model"]))
            v2.metric("Best MASE", f'{best_row["MASE"]:.3f}')
            v3.metric("Best RMSE", f'{best_row["RMSE"]:.2f}')
            v4.metric(
                "Directional accuracy",
                f'{best_row["Directional Accuracy"]:.1%}',
            )

            st.markdown("#### Baseline leaderboard")
            st.dataframe(
                validation_table.style.format(
                    {
                        "MAE": "{:.2f}",
                        "RMSE": "{:.2f}",
                        "MAPE (%)": "{:.2f}",
                        "sMAPE (%)": "{:.2f}",
                        "MASE": "{:.3f}",
                        "Directional Accuracy": "{:.1%}",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            comparison = validation_predictions.pivot(
                index="date",
                columns="model",
                values="predicted",
            )
            actual = (
                validation_predictions[["date", "actual"]]
                .drop_duplicates("date")
                .set_index("date")
            )
            comparison = actual.join(comparison)
            comparison.columns.name = None

            validation_fig = go.Figure()
            validation_fig.add_trace(
                go.Scatter(
                    x=comparison.index,
                    y=comparison["actual"],
                    name="Actual",
                    line=dict(color="#0F172A", width=3),
                )
            )
            palette = {
                "Last Value": "#0F766E",
                "Drift": "#2563EB",
                "Seasonal Naive": "#D97706",
            }
            for model in ["Last Value", "Drift", "Seasonal Naive"]:
                if model in comparison:
                    validation_fig.add_trace(
                        go.Scatter(
                            x=comparison.index,
                            y=comparison[model],
                            name=model,
                            line=dict(
                                color=palette[model],
                                width=1.6,
                                dash="dot",
                            ),
                        )
                    )
            validation_fig.update_layout(
                title="Rolling-Origin Baseline Forecasts",
                height=450,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#0F172A"),
                xaxis=dict(title="", gridcolor="#E2E8F0"),
                yaxis=dict(title="Price", gridcolor="#E2E8F0"),
                legend=dict(orientation="h", y=-0.16),
                margin=dict(l=55, r=25, t=65, b=70),
            )
            st.plotly_chart(validation_fig, use_container_width=True)

            st.info(
                "**Interpretation:** MASE below 1.0 means a forecast improves on "
                "the in-sample one-step naïve error scale. Sophisticated models "
                "should be compared with these baselines on a common evaluation design."
            )
        except ValueError as exc:
            st.warning(f"Validation unavailable: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — WEATHER OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader(f"🌤️ Weather Overlay — {weather_region}")

    if weather_df is not None and not weather_df.empty:
        fig_weather = plot_weather_overlay(weekly, weather_df, commodity_name,
                                           weather_region)
        st.plotly_chart(fig_weather, use_container_width=True)

        # Correlation with price
        st.subheader("📊 Weather-Price Correlation Analysis")
        common_idx = weekly.index.intersection(
            pd.DatetimeIndex(weather_df.index)
        )
        if len(common_idx) > 10:
            w_aligned = weather_df.reindex(common_idx).ffill()
            p_aligned = weekly.reindex(common_idx)
            corr_df = pd.DataFrame({
                "Price": p_aligned.values,
            })
            for col in weather_df.columns:
                corr_df[col] = w_aligned[col].values

            corr_matrix = corr_df.corr()
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.index.tolist(),
                colorscale="RdBu", zmin=-1, zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text:.2f}",
            ))
            fig_corr.update_layout(
                title=f"Price vs Weather Correlation — {weather_region}",
                template="plotly_dark", height=350,
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning(
            "Weather data unavailable (API may be unreachable or dates out of "
            "range). The forecast still runs without weather features."
        )
        # Simulate example weather chart
        dates = pd.date_range(start_date, end_date, freq="W")
        temp_sim = 15 + 10 * np.sin(np.linspace(0, 4 * np.pi, len(dates))) + \
                   np.random.randn(len(dates)) * 3
        precip_sim = np.abs(np.random.randn(len(dates)) * 20)

        fig_sim = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 subplot_titles=["Temperature (°C)",
                                                 "Precipitation (mm)"])
        fig_sim.add_trace(go.Scatter(x=dates, y=temp_sim, name="Temp",
                                      line=dict(color="orange")), row=1, col=1)
        fig_sim.add_trace(go.Bar(x=dates, y=precip_sim, name="Precip",
                                  marker_color="steelblue"), row=2, col=1)
        fig_sim.update_layout(template="plotly_dark", height=400,
                               title="Simulated Weather Data (example)")
        st.plotly_chart(fig_sim, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🔬 Time-Series Decomposition")
    st.caption("Seasonal-Trend Decomposition using LOESS (STL)")

    with st.spinner("Decomposing time series…"):
        decomp = decompose_series(weekly)

    if decomp is not None:
        fig_decomp = plot_decomposition(decomp, commodity_name)
        st.plotly_chart(fig_decomp, use_container_width=True)

        # Seasonality strength
        seasonal_strength = decomp.seasonal.std() / decomp.observed.std()
        trend_strength = decomp.trend.dropna().std() / decomp.observed.std()
        st.subheader("📊 Decomposition Insights")
        col1, col2, col3 = st.columns(3)
        col1.metric("Seasonal Strength",
                    f"{seasonal_strength:.3f}",
                    help="Proportion of variance explained by seasonality")
        col2.metric("Trend Strength",
                    f"{trend_strength:.3f}",
                    help="Proportion of variance explained by trend")
        col3.metric("Residual Std",
                    f"${decomp.resid.dropna().std():.2f}")
    else:
        st.warning("Decomposition requires at least 2 full seasonal periods "
                   "(104 weeks for annual seasonality).")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("⚠️ Producer & Supplier Risk Dashboard")
    st.caption(
        "Risk scores are derived from price volatility, drawdown history, "
        "and simulated geopolitical proxies. "
        "**Not a credit rating — for scenario analysis only.**"
    )

    # Compute risk scores
    risk_table = compute_risk_scores(prices, commodity_name)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        fig_heat = risk_heatmap(risk_table)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_right:
        st.subheader("Risk Score Table")
        st.dataframe(
            risk_table.style.background_gradient(
                cmap="RdYlGn_r",
                subset=["Overall Risk Score"]
            ),
            use_container_width=True,
            height=360,
        )

    # Volatility rolling chart
    st.markdown("---")
    rolling_vol = returns.rolling(12).std() * np.sqrt(52) * 100
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values,
        name="12-week Rolling Volatility",
        line=dict(color="orange", width=2),
        fill="tozeroy", fillcolor="rgba(255,165,0,0.1)",
    ))
    fig_vol.update_layout(
        title="Rolling 12-Week Annualised Volatility (%)",
        xaxis_title="Date", yaxis_title="Volatility (%)",
        template="plotly_dark", height=350,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Risk explanation
    with st.expander("ℹ️ How risk scores are calculated"):
        st.markdown("""
        | Component | Weight | Description |
        |-----------|--------|-------------|
        | **Price Volatility** | 35% | Annualised standard deviation of weekly returns |
        | **Max Drawdown** | 25% | Largest peak-to-trough loss in the period |
        | **Trend Instability** | 20% | Number of trend reversals (rolling MA crossovers) |
        | **Geopolitical Proxy** | 20% | Simulated score based on commodity type and region |

        Scores range from 0 (lowest risk) to 100 (highest risk). The overall score is a weighted average.
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer**: Forecasts and risk scores are for educational and "
    "simulation purposes only. They do not constitute financial, commodity, "
    "or investment advice. Forecast accuracy is not guaranteed. "
    "Data sourced from Yahoo Finance and Open-Meteo."
)
