"""House style: dark theme with the #10B981 emerald accent."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_theme_is_dark_emerald():
    # tomllib is 3.11+; CI also runs 3.10, so read the two keys directly
    text = (ROOT / ".streamlit" / "config.toml").read_text()
    assert re.search(r'^base\s*=\s*"dark"', text, re.M)
    assert re.search(r'^primaryColor\s*=\s*"#10B981"', text, re.M | re.I)


def test_static_demo_uses_emerald_accent():
    assert "--accent: #10B981" in (ROOT / "site" / "style.css").read_text()
