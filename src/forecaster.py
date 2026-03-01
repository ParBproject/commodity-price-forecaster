"""
forecaster.py
=============
Time-series forecasting pipeline:
  - Auto-ARIMA via pmdarima (falls back to statsmodels ARIMA)
  - Prophet (Facebook/Meta) with uncertainty intervals
  - STL seasonal decomposition
  - Forecast accuracy metrics (MAE, RMSE, MAPE)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Auto-ARIMA
# ─────────────────────────────────────────────────────────────────────────────

def fit_best_arima(
    series: pd.Series,
    horizon: int = 12,
    alpha: float = 0.05,
    seasonal: bool = True,
    m: int = 52,          # Weekly seasonality period (52 weeks = 1 year)
) -> dict | None:
    """
    Fit the best ARIMA / SARIMAX model using pmdarima's auto_arima,
    then forecast `horizon` steps ahead.

    Falls back to statsmodels ARIMA(1,1,1) if pmdarima is not installed.

    Parameters
    ----------
    series : pd.Series     Weekly commodity prices with DatetimeIndex.
    horizon : int          Forecast steps (weeks).
    alpha : float          Significance level for confidence intervals (0.05 = 95%).
    seasonal : bool        Whether to include seasonal component.
    m : int                Seasonal period in weeks.

    Returns
    -------
    dict with keys:
        forecast_df : pd.DataFrame (index=future dates, columns=forecast/lower/upper)
        metrics     : dict (MAE, RMSE, MAPE, AIC)
        model       : fitted model object
        order       : (p,d,q) or (p,d,q)(P,D,Q)m tuple
    """
    train_size = int(len(series) * 0.85)
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    try:
        import pmdarima as pm
        model = pm.auto_arima(
            train,
            start_p=0, start_q=0,
            max_p=4, max_q=4,
            d=None,                       # auto-determine differencing
            seasonal=seasonal and m <= len(train) // 2,
            m=m,
            D=None,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            n_jobs=1,
        )
        order = model.order
        fitted_model = model

        # In-sample + test predictions for metrics
        test_pred = model.predict(n_periods=len(test))
        metrics = compute_metrics(test.values, test_pred)
        metrics["AIC"] = model.aic()

        # Re-fit on all data for final forecast
        model.update(test)
        fc, conf = model.predict(n_periods=horizon, return_conf_int=True,
                                 alpha=alpha)

    except ImportError:
        # Fallback: statsmodels ARIMA(1,1,1)
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

        sm_model = SM_ARIMA(train, order=(1, 1, 1)).fit(disp=False)
        order = (1, 1, 1)
        test_pred = sm_model.forecast(steps=len(test))
        metrics = compute_metrics(test.values, test_pred.values)
        metrics["AIC"] = sm_model.aic

        # Re-fit on all data
        sm_model_full = SM_ARIMA(series, order=(1, 1, 1)).fit(disp=False)
        fc_res = sm_model_full.get_forecast(steps=horizon)
        fc = fc_res.predicted_mean.values
        ci = fc_res.conf_int(alpha=alpha)
        conf = ci.values
        fitted_model = sm_model_full

    # Build forecast DataFrame with future dates
    last_date = series.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset("W"),
        periods=horizon,
        freq="W",
    )
    forecast_df = pd.DataFrame(
        {"forecast": fc, "lower": conf[:, 0], "upper": conf[:, 1]},
        index=future_dates,
    )

    return {
        "forecast_df": forecast_df,
        "metrics": metrics,
        "model": fitted_model,
        "order": order,
        "train": train,
        "test": test,
        "test_pred": test_pred if hasattr(test_pred, "__len__") else test_pred.values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────────────────────────────────────────

def fit_prophet(
    series: pd.Series,
    horizon: int = 12,
    interval_width: float = 0.95,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = True,
) -> dict | None:
    """
    Fit a Prophet model and forecast `horizon` weeks ahead.

    Parameters
    ----------
    series : pd.Series     Weekly prices with DatetimeIndex.
    horizon : int          Forecast horizon (weeks).
    interval_width : float Confidence interval width (e.g. 0.95).

    Returns
    -------
    dict with keys: forecast_df, metrics, model
    """
    try:
        from prophet import Prophet
    except ImportError:
        try:
            from fbprophet import Prophet
        except ImportError:
            return None  # Prophet not installed

    # Prophet expects columns: ds (datetime) and y (value)
    df_prophet = pd.DataFrame({
        "ds": series.index,
        "y": series.values,
    })

    train_size = int(len(df_prophet) * 0.85)
    train_df = df_prophet.iloc[:train_size]
    test_df = df_prophet.iloc[train_size:]

    model = Prophet(
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        uncertainty_samples=500,
    )
    model.fit(train_df)

    # Test predictions
    test_future = model.make_future_dataframe(
        periods=len(test_df), freq="W", include_history=False
    )
    test_forecast = model.predict(test_future)
    test_pred = test_forecast["yhat"].values
    metrics = compute_metrics(test_df["y"].values, test_pred)

    # Re-fit on all data
    model_full = Prophet(
        interval_width=interval_width,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
    )
    model_full.fit(df_prophet)
    future = model_full.make_future_dataframe(periods=horizon, freq="W",
                                               include_history=False)
    forecast = model_full.predict(future)

    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_df = forecast_df.set_index("ds").rename(columns={
        "yhat": "forecast",
        "yhat_lower": "lower",
        "yhat_upper": "upper",
    })
    # Clip negatives (prices can't be negative)
    forecast_df = forecast_df.clip(lower=0)

    return {
        "forecast_df": forecast_df,
        "metrics": metrics,
        "model": model_full,
        "components": forecast[["ds", "trend", "yearly", "weekly"]],
        "train": train_df,
        "test": test_df,
        "test_pred": test_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """
    Compute forecast accuracy metrics.

    Parameters
    ----------
    actual : np.ndarray    Ground-truth values.
    predicted : np.ndarray Forecasted values.

    Returns
    -------
    dict with MAE, RMSE, MAPE keys.
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Avoid division by zero for MAPE
    nonzero = actual != 0
    mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) /
                          actual[nonzero])) * 100

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 3)}


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal Decomposition
# ─────────────────────────────────────────────────────────────────────────────

def decompose_series(
    series: pd.Series,
    model_type: str = "additive",
    period: int = 52,
) -> object | None:
    """
    STL seasonal decomposition of a time series.

    Parameters
    ----------
    series : pd.Series   Weekly price series.
    model_type : str     'additive' or 'multiplicative'.
    period : int         Seasonal period (52 for annual weekly).

    Returns
    -------
    statsmodels DecomposeResult or None if insufficient data.
    """
    if len(series) < 2 * period:
        return None

    try:
        from statsmodels.tsa.seasonal import STL

        stl = STL(series, period=period, robust=True)
        result = stl.fit()
        return result
    except Exception:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            result = seasonal_decompose(series, model=model_type,
                                        period=period)
            return result
        except Exception:
            return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def create_features(
    series: pd.Series,
    lags: list[int] | None = None,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Engineer features for ML-based forecasting.

    Includes:
    - Lagged price values
    - Rolling statistics (mean, std)
    - Calendar features (week of year, month)
    - Optional weather variables

    Parameters
    ----------
    series : pd.Series      Weekly prices.
    lags : list[int]        Lag periods to include (default [1, 2, 4, 8, 52]).
    weather_df : pd.DataFrame, optional  Weather features aligned to same index.

    Returns
    -------
    pd.DataFrame of engineered features.
    """
    if lags is None:
        lags = [1, 2, 4, 8, 52]

    df = pd.DataFrame({"price": series})
    df["log_price"] = np.log(series.clip(lower=1e-6))

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
        df[f"return_{lag}"] = series.pct_change(lag)

    # Rolling statistics
    for window in [4, 12, 26]:
        df[f"roll_mean_{window}"] = series.rolling(window).mean()
        df[f"roll_std_{window}"] = series.rolling(window).std()

    # Calendar
    df["week_of_year"] = series.index.isocalendar().week.astype(int)
    df["month"] = series.index.month
    df["quarter"] = series.index.quarter

    # Weather features
    if weather_df is not None:
        common_idx = df.index.intersection(weather_df.index)
        for col in weather_df.columns:
            df.loc[common_idx, f"weather_{col}"] = weather_df.loc[common_idx, col]

    return df.dropna()
