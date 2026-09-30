"""
forecaster.py
=============
Time-series forecasting pipeline:
  - Auto-ARIMA via pmdarima (falls back to statsmodels ARIMA)
  - Prophet (Facebook/Meta) with uncertainty intervals
  - STL seasonal decomposition
  - Forecast accuracy metrics (MAE, RMSE, MAPE)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MIN_FORECAST_OBSERVATIONS = 30


def _prepare_price_series(series: pd.Series) -> pd.Series:
    """Sort, drop non-finite values, and store timezone-naive timestamps."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    prepared = pd.Series(series, copy=True).astype(float).sort_index()
    if prepared.index.tz is not None:
        prepared = prepared.tz_convert("UTC").tz_localize(None)
        prepared.index = pd.DatetimeIndex(prepared.index)
    prepared.index = pd.DatetimeIndex(prepared.index).normalize()
    prepared = prepared[~prepared.index.duplicated(keep="last")]
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna()
    if prepared.empty:
        raise ValueError("series must contain at least one finite observation")
    return prepared


def chronological_split(
    series: pd.Series,
    train_fraction: float = 0.85,
) -> tuple[pd.Series, pd.Series]:
    """Prefix split. The test block is always the later observations."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    train_size = int(len(series) * train_fraction)
    if train_size < 1 or train_size >= len(series):
        raise ValueError("series is too short for a chronological holdout")
    return series.iloc[:train_size], series.iloc[train_size:]


def future_forecast_index(
    last_timestamp,
    horizon: int,
    freq: str = "W",
) -> pd.DatetimeIndex:
    """Weekly dates strictly after the last observation."""
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be a positive integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    last = pd.Timestamp(last_timestamp)
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    last = last.normalize()
    offset = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(start=last + offset, periods=int(horizon), freq=freq)


def apply_price_scenario(
    forecast_df: pd.DataFrame,
    multiplier: float,
) -> pd.DataFrame:
    """Scale a forecast path. History and holdout scores stay untouched."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("scenario multiplier must be a positive finite number")
    scaled = forecast_df.copy()
    for column in ("forecast", "lower", "upper"):
        if column in scaled.columns:
            scaled[column] = scaled[column].astype(float) * float(multiplier)
    return scaled.clip(lower=0)


def forecast_export_frame(
    arima_result: dict | None,
    prophet_result: dict | None,
    confidence: int,
) -> pd.DataFrame | None:
    """Point forecasts and intervals for every model that actually ran."""
    frames: list[pd.DataFrame] = []
    if arima_result is not None and "forecast_df" in arima_result:
        arima = arima_result["forecast_df"][["forecast", "lower", "upper"]].copy()
        arima.columns = [
            "ARIMA forecast",
            f"ARIMA lower {confidence}%",
            f"ARIMA upper {confidence}%",
        ]
        frames.append(arima)
    if prophet_result is not None and "forecast_df" in prophet_result:
        prophet = prophet_result["forecast_df"][["forecast", "lower", "upper"]].copy()
        prophet.columns = [
            "Prophet forecast",
            f"Prophet lower {confidence}%",
            f"Prophet upper {confidence}%",
        ]
        frames.append(prophet)
    if not frames:
        return None

    table = frames[0] if len(frames) == 1 else frames[0].join(frames[1], how="outer")
    if {"ARIMA forecast", "Prophet forecast"}.issubset(table.columns):
        table["Ensemble forecast"] = table[
            ["ARIMA forecast", "Prophet forecast"]
        ].mean(axis=1, skipna=False)
    table.index.name = "Date"
    return table.sort_index()


def _forecast_frame(future_dates: pd.DatetimeIndex, forecast, conf) -> pd.DataFrame:
    point = np.asarray(forecast, dtype=float).reshape(-1)
    bounds = np.asarray(conf, dtype=float)
    if bounds.shape != (len(point), 2):
        raise ValueError("confidence bounds must have one lower and upper column")
    return pd.DataFrame(
        {"forecast": point, "lower": bounds[:, 0], "upper": bounds[:, 1]},
        index=future_dates,
    )


def forecast_with_fixed_arima(
    series: pd.Series,
    horizon: int = 12,
    alpha: float = 0.05,
    order: tuple[int, int, int] = (1, 1, 1),
) -> dict | None:
    """Holdout score from ARIMA fit on the training prefix, then refit for the path.

    Reported errors come from the training model only. The full-sample refit is
    used for the forward path and is not fed back into the holdout score.
    """
    prepared = _prepare_price_series(series)
    if len(prepared) < MIN_FORECAST_OBSERVATIONS:
        return None
    train, test = chronological_split(prepared)

    try:
        from statsmodels.tsa.arima.model import ARIMA as SM_ARIMA

        holdout_model = SM_ARIMA(train, order=order).fit()
        test_pred = np.asarray(holdout_model.forecast(steps=len(test)), dtype=float)
        metrics = compute_metrics(test.to_numpy(), test_pred)
        metrics["AIC"] = float(holdout_model.aic)

        full_model = SM_ARIMA(prepared, order=order).fit()
        forecast_result = full_model.get_forecast(steps=horizon)
        point = np.asarray(forecast_result.predicted_mean, dtype=float)
        bounds = np.asarray(forecast_result.conf_int(alpha=alpha), dtype=float)
    except Exception:
        return None

    return {
        "forecast_df": _forecast_frame(
            future_forecast_index(prepared.index[-1], horizon),
            point,
            bounds,
        ),
        "metrics": metrics,
        "model": full_model,
        "order": order,
        "train": train,
        "test": test,
        "test_pred": test_pred,
    }


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
    Fit an ARIMA model with pmdarima's auto_arima, then forecast `horizon`
    steps ahead. Seasonal terms are used only when the training sample is
    long enough. No exogenous regressors are passed.

    Falls back to statsmodels ARIMA(1, 1, 1) when pmdarima is missing or
    auto-selection fails. Returns None when the history is too short or both
    fits fail. The returned ``selection`` key is ``"auto"`` or ``"fixed"``.

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
    try:
        prepared = _prepare_price_series(series)
    except (TypeError, ValueError):
        return None
    if len(prepared) < MIN_FORECAST_OBSERVATIONS:
        return None
    train, test = chronological_split(prepared)

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
        # Multi-step holdout from the training origin only. update() below
        # refits for the forward path and must not change these errors.
        test_pred = np.asarray(model.predict(n_periods=len(test)), dtype=float)
        metrics = compute_metrics(test.to_numpy(), test_pred)
        metrics["AIC"] = model.aic()

        model.update(test)
        fc, conf = model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=alpha,
        )
        forecast_df = _forecast_frame(
            future_forecast_index(prepared.index[-1], horizon),
            fc,
            conf,
        )
        return {
            "forecast_df": forecast_df,
            "metrics": metrics,
            "model": model,
            "order": order,
            "selection": "auto",
            "train": train,
            "test": test,
            "test_pred": test_pred,
        }
    except Exception:
        fallback = forecast_with_fixed_arima(
            prepared,
            horizon=horizon,
            alpha=alpha,
        )
        if fallback is None:
            return None
        return {**fallback, "selection": "fixed"}


# ─────────────────────────────────────────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────────────────────────────────────────

def select_prophet_components(forecast: pd.DataFrame) -> pd.DataFrame:
    """Return Prophet components that are present in a forecast frame.

    ``trend`` is always expected, while built-in ``yearly`` and ``weekly``
    seasonalities are optional and disappear when disabled in the model.
    """
    required = ["ds", "trend"]
    missing = [column for column in required if column not in forecast.columns]
    if missing:
        raise KeyError(f"Prophet forecast missing required columns: {missing}")

    optional = [
        column for column in ("yearly", "weekly") if column in forecast.columns
    ]
    return forecast[required + optional].copy()


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

    try:
        prepared = _prepare_price_series(series)
        if len(prepared) < MIN_FORECAST_OBSERVATIONS:
            return None
        train, test = chronological_split(prepared)
        train_df = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
        test_df = pd.DataFrame({"ds": test.index, "y": test.to_numpy()})

        model = Prophet(
            interval_width=interval_width,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=False,
            uncertainty_samples=500,
        )
        model.fit(train_df)

        # Score the actual holdout timestamps. A generated weekly calendar can
        # drift off the sample when a week is missing, and positional scoring
        # would then compare the wrong weeks.
        test_forecast = model.predict(pd.DataFrame({"ds": test.index}))
        predicted = (
            test_forecast.assign(ds=pd.DatetimeIndex(test_forecast["ds"]).normalize())
            .drop_duplicates("ds")
            .set_index("ds")["yhat"]
            .reindex(test.index)
        )
        if predicted.isna().any():
            return None
        test_pred = predicted.to_numpy(dtype=float)
        metrics = compute_metrics(test.to_numpy(), test_pred)

        model_full = Prophet(
            interval_width=interval_width,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=False,
            uncertainty_samples=500,
        )
        model_full.fit(pd.DataFrame({"ds": prepared.index, "y": prepared.to_numpy()}))
        future_dates = future_forecast_index(prepared.index[-1], horizon)
        forecast = model_full.predict(pd.DataFrame({"ds": future_dates}))
        forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_df["ds"] = pd.DatetimeIndex(forecast_df["ds"]).normalize()
        forecast_df = forecast_df.set_index("ds").rename(columns={
            "yhat": "forecast",
            "yhat_lower": "lower",
            "yhat_upper": "upper",
        })
        forecast_df = forecast_df.clip(lower=0)
    except Exception:
        return None

    return {
        "forecast_df": forecast_df,
        "metrics": metrics,
        "model": model_full,
        "components": select_prophet_components(forecast),
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
    actual = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    # Avoid division by zero for MAPE
    nonzero = actual != 0
    if np.any(nonzero):
        mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) /
                              actual[nonzero])) * 100
        mape_value = round(float(mape), 3)
    else:
        mape_value = None

    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": mape_value,
    }


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


def _component_strength(component: np.ndarray, resid: np.ndarray) -> float | None:
    """Hyndman strength: max(0, 1 - Var(resid) / Var(component + resid))."""
    component_values = np.asarray(component, dtype=float).reshape(-1)
    resid_values = np.asarray(resid, dtype=float).reshape(-1)
    if component_values.shape != resid_values.shape:
        raise ValueError("component and resid must have the same shape")
    valid = np.isfinite(component_values) & np.isfinite(resid_values)
    if int(valid.sum()) < 2:
        return None
    denominator = float(np.var(component_values[valid] + resid_values[valid], ddof=1))
    if denominator <= 0:
        return None
    strength = 1.0 - float(np.var(resid_values[valid], ddof=1)) / denominator
    return float(max(0.0, strength))


def decomposition_strength(result: object) -> dict:
    """Seasonal and trend strength in [0, 1].

    This is not a ratio of standard deviations. A ratio of standard deviations
    can exceed 1 and is not the share of variance associated with a component.
    """
    return {
        "seasonal_strength": _component_strength(result.seasonal, result.resid),
        "trend_strength": _component_strength(result.trend, result.resid),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def create_features(
    series: pd.Series,
    lags: list[int] | None = None,
    weather_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Engineer features for a supervised forecast of ``price`` at time t.

    ``price`` and ``log_price`` are the observation at t. Every other column
    uses only information available before t: lags, completed returns, rolling
    statistics through t-1, calendar fields, and weather lagged by one period.

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
    history = series.shift(1)

    # Lag features and returns that were complete before t.
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
        df[f"return_{lag}"] = series.pct_change(lag).shift(1)

    # Rolling statistics of prices known before t.
    for window in [4, 12, 26]:
        df[f"roll_mean_{window}"] = history.rolling(window).mean()
        df[f"roll_std_{window}"] = history.rolling(window).std()

    # Calendar fields are known before the week's close.
    df["week_of_year"] = series.index.isocalendar().week.astype(int)
    df["month"] = series.index.month
    df["quarter"] = series.index.quarter

    # Same-week weather is contemporaneous with the weekly close, so lag it
    # by one week on the calendar rather than by row position.
    if weather_df is not None:
        lagged_weather = weather_df.copy()
        lagged_weather.index = pd.DatetimeIndex(lagged_weather.index)
        if lagged_weather.index.tz is not None:
            lagged_weather.index = lagged_weather.index.tz_convert("UTC").tz_localize(None)
        lagged_weather = lagged_weather.shift(1, freq="W")
        common_idx = df.index.intersection(lagged_weather.index)
        for col in lagged_weather.columns:
            df.loc[common_idx, f"weather_{col}"] = lagged_weather.loc[common_idx, col]

    return df.dropna()
