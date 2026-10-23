"""LSTM forecaster (PyTorch), returning the same dict shape as ``fit_best_arima``.

Design and hyperparameters
--------------------------
- **Target:** weekly log returns, standardised with the *training* mean and std only.
  Prices are trending and non-stationary; returns are not, and an LSTM fit on raw price
  levels mostly learns the level.
- **Input window:** the previous 52 weekly returns (``WINDOW = 52``), one feature.
- **Network:** 1 LSTM layer, 32 hidden units (``HIDDEN = 32``), dropout 0.2 on the
  final hidden state (``DROPOUT = 0.2``), then a linear layer to the next return.
- **Training:** Adam, learning rate 1e-3, mean-squared error, 150 full-batch epochs
  (``EPOCHS = 150``), seed 0 for repeatability; CPU only.
- **Split:** chronological 85/15 like ARIMA. The holdout score is one *multi-step*
  recursive forecast from the training origin (the same protocol as the ARIMA holdout),
  then the network is refit on the full sample for the forward path.
- **Intervals:** an LSTM gives no likelihood, so bands come from the standard deviation
  of the training-sample one-step residuals in log-return space, scaled by sqrt(h) for
  step h (random-walk accumulation) and mapped back to prices. They are approximate.

Requires ``torch``; returns ``None`` if it is not installed or the history is too short.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.forecaster import (
    _forecast_frame,
    _prepare_price_series,
    chronological_split,
    compute_metrics,
    future_forecast_index,
)

logger = logging.getLogger(__name__)

WINDOW = 52
HIDDEN = 32
DROPOUT = 0.2
EPOCHS = 150
LEARNING_RATE = 1e-3
SEED = 0


def _windows(z: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([z[i : i + window] for i in range(len(z) - window)])
    y = z[window:]
    return x[..., None].astype(np.float32), y.astype(np.float32)


class _Net:
    """Thin wrapper so torch is only imported when an LSTM is actually fit."""

    def __init__(self):
        import torch
        from torch import nn

        torch.manual_seed(SEED)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, HIDDEN, batch_first=True)
                self.drop = nn.Dropout(DROPOUT)
                self.head = nn.Linear(HIDDEN, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(self.drop(out[:, -1, :])).squeeze(-1)

        self.torch = torch
        self.model = Model()

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_Net":
        torch = self.torch
        xt, yt = torch.from_numpy(x), torch.from_numpy(y)
        opt = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.MSELoss()
        self.model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            opt.step()
        self.model.eval()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x)).numpy()

    def recursive(self, last_window: np.ndarray, steps: int) -> np.ndarray:
        buf = list(last_window.astype(np.float32))
        out = []
        for _ in range(steps):
            nxt = float(self.predict(np.asarray(buf[-WINDOW:], dtype=np.float32)[None, :, None])[0])
            out.append(nxt)
            buf.append(nxt)
        return np.asarray(out)


def _fit_on(prices: pd.Series):
    """Fit on a price series; returns (net, mean, std, residual_std_logret, last_window_z)."""
    r = np.diff(np.log(prices.to_numpy(dtype=float)))
    mu, sd = float(r.mean()), float(r.std(ddof=1)) or 1.0
    z = (r - mu) / sd
    x, y = _windows(z, WINDOW)
    net = _Net().fit(x, y)
    resid = (net.predict(x) - y) * sd
    return net, mu, sd, float(np.std(resid, ddof=1)), z[-WINDOW:]


def _path(prices_last: float, net, mu, sd, last_z, steps, resid_sd, alpha):
    zf = net.recursive(last_z, steps)
    log_ret = zf * sd + mu
    cum = np.cumsum(log_ret)
    point = prices_last * np.exp(cum)
    half = norm.ppf(1 - alpha / 2) * resid_sd * np.sqrt(np.arange(1, steps + 1))
    return point, prices_last * np.exp(cum - half), prices_last * np.exp(cum + half)


def fit_lstm(series: pd.Series, horizon: int = 12, alpha: float = 0.05) -> dict | None:
    """LSTM forecast with the same output keys as ``fit_best_arima``."""
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.info("torch is not installed; skipping the LSTM forecast")
        return None
    try:
        prepared = _prepare_price_series(series)
    except (TypeError, ValueError):
        return None
    if (prepared <= 0).any():
        logger.warning("LSTM needs positive prices (it models log returns)")
        return None
    train, test = chronological_split(prepared)
    if len(train) < WINDOW + 20:
        return None

    net, mu, sd, resid_sd, last_z = _fit_on(train)
    test_pred, _, _ = _path(float(train.iloc[-1]), net, mu, sd, last_z, len(test), resid_sd, alpha)
    metrics = compute_metrics(test.to_numpy(), test_pred)

    net_full, mu_f, sd_f, resid_f, last_zf = _fit_on(prepared)
    point, lo, hi = _path(float(prepared.iloc[-1]), net_full, mu_f, sd_f, last_zf, horizon, resid_f, alpha)
    forecast_df = _forecast_frame(
        future_forecast_index(prepared.index[-1], horizon), point, np.column_stack([lo, hi])
    )
    return {
        "forecast_df": forecast_df,
        "metrics": metrics,
        "model": net_full.model,
        "train": train,
        "test": test,
        "test_pred": test_pred,
        "hyperparameters": {"window": WINDOW, "hidden": HIDDEN, "dropout": DROPOUT,
                            "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "seed": SEED},
    }
