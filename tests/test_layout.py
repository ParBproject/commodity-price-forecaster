"""Guards for the narrow-screen layout fix (issue #4)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_wraps_columns_below_1024px():
    app = (ROOT / "app.py").read_text()
    assert "@media (max-width: 1024px)" in app
    assert "flex-wrap: wrap" in app


def test_no_chart_heights_above_500px():
    for path in [ROOT / "app.py", ROOT / "src" / "utils.py"]:
        heights = [int(h) for h in re.findall(r"height=(\d+)", path.read_text())]
        assert all(h <= 500 for h in heights), (path.name, heights)
