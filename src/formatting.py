"""Display formatting for forecast metrics.

This module stays free of plotting dependencies so the test suite can import
it in environments that do not install Plotly.
"""

import math


def format_mape(value: float | None, digits: int = 1) -> str:
    """Format MAPE for display.

    ``compute_metrics`` returns ``None`` when every actual value is zero,
    because percentage error is undefined. Show that case as ``n/a``.
    """
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}%"


def format_directional_accuracy(value: float | None) -> str:
    """Format a 0–1 directional-accuracy rate.

    Flat forecasts never call a direction, and a missing prior observation
    leaves the rate undefined. Both are shown as ``n/a`` rather than 0%.
    """
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.1%}"
