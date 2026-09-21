# Commodity Price Forecaster

## For a data analyst application

**Supporting forecasting piece.** Useful if the posting mentions time series. Lead with uncertainty and the decomposition, not with a promise of the price. The dashboard screenshots are the surface to open.

<p align="center"><img src="assets/screenshots/01_overview.png" alt="Commodity market overview" width="100%"></p>
<p align="center"><img src="assets/screenshots/02_forecast.png" alt="Forecast comparison" width="100%"></p>
<p align="center"><img src="assets/screenshots/05_risk_dashboard.png" alt="Producer risk dashboard" width="100%"></p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](requirements.txt)
[![Streamlit](https://img.shields.io/badge/Streamlit-Forecasting_Dashboard-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![Tests](https://img.shields.io/badge/Tests-Pytest-0A9EDC?logo=pytest&logoColor=white)](tests/test_forecaster.py)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)](.github/workflows/python-package-conda.yml)

An interactive time-series forecasting application for energy, metals, and agricultural commodities. It combines statistical forecasts, weather context, decomposition, scenario shocks, and producer-risk indicators in a decision-oriented dashboard.

## Capabilities

- Historical price retrieval for major commodity futures
- Weekly return, volatility, drawdown, and descriptive analysis
- Auto-ARIMA and Prophet forecast options
- Ensemble forecasts with uncertainty intervals
- STL trend and seasonal decomposition
- Open-Meteo weather overlays for producing regions
- Supply, demand, and weather scenario shocks
- Multi-factor regional risk scoring
- Automated tests and GitHub Actions workflow

## Dashboard Preview

### Market Overview

![Commodity market overview](assets/screenshots/01_overview.png)

### Forecast Comparison

![ARIMA and Prophet commodity forecast](assets/screenshots/02_forecast.png)

### Weather Overlay

![Weather and commodity analysis](assets/screenshots/03_weather_overlay.png)

### Decomposition

![Commodity time-series decomposition](assets/screenshots/04_decomposition.png)

### Risk Dashboard

![Commodity producer risk dashboard](assets/screenshots/05_risk_dashboard.png)

## Supported Markets

| Category | Examples |
|---|---|
| Energy | WTI crude oil, Brent crude, natural gas |
| Metals | Gold, silver, copper |
| Agriculture | Corn, wheat, soybeans |

## Run Locally

~~~bash
git clone https://github.com/ParBproject/commodity-price-forecaster.git
cd commodity-price-forecaster

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
~~~

To run the automated checks:

~~~bash
pytest
~~~

## Repository Structure

~~~text
commodity-price-forecaster/
├── app.py
├── src/
│   ├── data_loader.py
│   ├── forecaster.py
│   └── utils.py
├── notebooks/01_model_development.ipynb
├── tests/test_forecaster.py
├── assets/screenshots/
├── .github/workflows/
└── requirements.txt
~~~

## Skills Demonstrated

Time-series analysis, statistical forecasting, external API integration, weather-data enrichment, risk scoring, Streamlit, Plotly, testing, continuous integration, and clear communication of uncertainty.

## Forecasting Note

Forecasts are uncertain and sensitive to regime changes, data quality, and model assumptions. This application is an educational demonstration, not commodity-trading or investment advice.
