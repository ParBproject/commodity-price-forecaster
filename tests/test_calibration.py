import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmark import complete_weekly_series, load_snapshot
from src.calibration import coverage_report

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "results" / "calibration.json").read_text())


def test_counts_are_consistent():
    for d in RES["per_commodity"].values():
        for key in ("80%", "95%"):
            c = d[key]
            assert c["hits"] + c["misses_below"] + c["misses_above"] == c["n"] == RES["n_origins"]
            assert c["coverage"] == pytest.approx(c["hits"] / c["n"], abs=1e-6)
        assert d["95%"]["hits"] >= d["80%"]["hits"]  # wider interval never covers less


def test_committed_result_reproduces_for_wti():
    meta = json.loads((ROOT / "data" / "snapshot.json").read_text())
    snap = load_snapshot(ROOT / "data" / meta["file"])
    fresh = coverage_report(complete_weekly_series(snap["CL=F"]), RES["n_origins"])
    for key in ("80%", "95%"):
        assert fresh[key]["hits"] == RES["per_commodity"]["CL=F"][key]["hits"]
        assert fresh[key]["mean_relative_width"] == pytest.approx(
            RES["per_commodity"]["CL=F"][key]["mean_relative_width"], abs=1e-6)  # JSON is rounded to 6 dp


def test_perfect_model_on_white_noise_is_near_nominal():
    rng = np.random.default_rng(7)
    idx = pd.date_range("2015-01-04", periods=400, freq="W-SUN")
    s = pd.Series(100 + np.cumsum(rng.normal(0, 1, 400)), index=idx)
    rep = coverage_report(s, n_origins=150, order=(0, 1, 0))
    assert 0.88 <= rep["95%"]["coverage"] <= 1.0
    assert rep["95%"]["binom_p_value"] > 0.01


def test_readme_table_matches_json():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_calibration import END, START, markdown

    readme = (ROOT / "README.md").read_text()
    block = readme.split(START, 1)[1].split(END, 1)[0].strip()
    assert block == markdown(RES)
