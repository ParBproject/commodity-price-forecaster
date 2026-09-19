# Forecast Validation Methodology

## Objective

The forecasting workflow is designed to answer a decision-relevant question:

> Does a statistical forecast improve on simple, transparent alternatives when evaluated only on future observations?

Forecast complexity is not treated as evidence of forecast skill.

## Data frequency

Daily commodity futures prices are resampled to weekly observations for the main forecasting workflow.

Weekly data reduces day-to-day noise and aligns naturally with seasonal and producer-planning use cases.

## Core models

The application supports:

- Auto-ARIMA / SARIMAX;
- Prophet;
- an ensemble view of the two model families.

Forecast uncertainty intervals are displayed alongside point estimates.

## Benchmark models

Three transparent baselines are evaluated:

### Last value

```text
Forecast[t+1] = Actual[t]
```

This is the minimum benchmark for a persistent price series.

### Drift

A linear average drift from the first to the latest training observation is extrapolated one step forward.

### Seasonal naïve

The forecast repeats the observation from the prior seasonal cycle when enough history exists.

For weekly data the default seasonal cycle is 52 weeks.

## Rolling-origin validation

The baseline validation framework uses rolling origins.

For each forecast date:

1. training data contains only observations strictly before the forecast date;
2. one-step forecasts are generated;
3. the realized observation is recorded;
4. the forecast origin advances;
5. the process repeats.

No future observation is used to create an earlier forecast.

## Evaluation metrics

### MAE

Mean absolute error preserves the units of the commodity price.

### RMSE

Root mean squared error places more weight on larger forecast misses.

### MAPE

Mean absolute percentage error is included where actual prices are non-zero.

### sMAPE

Symmetric MAPE reduces some of MAPE's asymmetry and scale dependence.

### MASE

Mean absolute scaled error compares model error with the average one-step naïve error in the in-sample history.

```text
MASE < 1  → forecast error is better than the in-sample naïve scale
MASE = 1  → approximately naïve-scale error
MASE > 1  → worse than the naïve error scale
```

### Directional accuracy

The framework also evaluates whether the predicted movement from the prior actual observation has the same sign as the realized movement.

## Weather context

Weather variables come from Open-Meteo and are used as contextual analysis rather than proof of causality.

Correlation between weather variables and commodity prices should not be interpreted as a causal estimate.

## Scenario controls

Supply, demand, and weather controls are explicit scenario multipliers.

They are sensitivity-analysis assumptions, not estimated probabilities or causal structural models.

## Model limitations

- futures prices can contain roll effects and contract-specific behavior;
- relationships can change across market regimes;
- forecast intervals depend on model assumptions;
- repeated model selection on one evaluation period can overfit;
- weather relevance differs materially by commodity and producing region;
- Yahoo Finance and Open-Meteo data can be revised or unavailable;
- baseline validation does not by itself prove that a complex model has stable economic value.

## Production extensions

A production forecasting process would normally add:

- formal rolling-origin backtests for every candidate model;
- model-selection governance;
- hyperparameter search nested inside training windows;
- forecast calibration monitoring;
- feature and data drift monitoring;
- inventory, term-structure, FX, rates, and production fundamentals;
- source-level data quality controls;
- versioned model artifacts and forecasts.

This repository is an educational research project, not trading or hedging advice.
