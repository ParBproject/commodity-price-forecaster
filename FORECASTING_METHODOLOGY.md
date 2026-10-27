# How the forecasts are scored

The question is whether ARIMA or Prophet beats a transparent baseline when both are judged only on later observations.

## Series

Daily futures closes are taken to the last print of each Sunday week. Missing weeks stay on the calendar and are filled with the previous close, so a later price is never written into an earlier week. The default seasonal period is 52 weeks.

## Models in the leaderboard

- Last value: the forecast is the latest close.
- Drift: one step along the line from the first training close to the last.
- Seasonal naive: the close from 52 weeks earlier, or the last value if the training sample is shorter than that.
- ARIMA(1, 1, 1), refit at every origin. No exogenous regressors.
- Prophet, when the package is installed. Same one-step protocol.

The dashboard's Forecast tab is a different experiment. Auto-ARIMA (or ARIMA(1, 1, 1) if `pmdarima` is missing) and Prophet are fit on the first 85% of the selected sample, scored on the rest as one multi-step path, then refit on the full sample for the line that is drawn. The LSTM uses that same split. Those errors are not comparable to the leaderboard.

Supply, demand, and weather sliders multiply the drawn path after the fit. They do not change the history or the holdout score.

## Rolling origin

At each origin the training sample is every observation strictly before that week. The model forecasts the next week, the realised close is recorded, and the origin moves forward by one week. The leaderboard uses the last 52 weeks of the snapshot (2025-10-05 through 2026-09-27), with history from 2016-01-10. The validation tab in the app scores every week after an initial training block of about 60% of the selected sample, and it only runs the three baselines.

## Metrics

- MAE and RMSE are in price units. RMSE weights large misses more.
- MAPE is the mean absolute percentage error on non-zero actuals.
- sMAPE is the symmetric version.
- MASE divides MAE by the in-sample mean absolute one-step change before the first origin. Above 1 means the evaluation window moved more, relative to that scale, than a naive step through the training history. For gold and silver in 2025–26 that scale is the quiet earlier sample, so MASE above 1 does not mean the fit failed to run.
- Directional accuracy is the share of weeks where the predicted change from the previous close has the same sign as the realised change. A flat forecast makes no call. If every forecast is flat, or the previous close is missing, the metric is left blank rather than reported as 0%.

## Weather and the risk table

Weather is Open-Meteo, joined to weekly returns on exact shared dates. Neighbouring weeks are not matched, and price levels are not correlated with weather. It is not a lagged feature of the leaderboard models.

`create_features` does lag weather by one week, and it is not used by the fitted models in the app. It is there so a supervised model built on top of it cannot see same-week weather.

The risk table's volatility, drawdown, and trend scores are computed once from the commodity's last 52 weeks and copied onto each producer row. Drawdown is the worst peak-to-trough loss of a wealth index that starts at 1. The geopolitical column starts from a fixed number per commodity and adds a stable offset from a hash of the commodity and region. It is not a country-risk estimate.

## What this is not

There is no nested hyperparameter search, no inventory or curve data, and no claim that a lower error here would make money. Yahoo Finance and Open-Meteo series can be revised or missing. The project is a study of the forecasts, not advice.
