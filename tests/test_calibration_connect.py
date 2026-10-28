# tests/test_calibration_connect.py
"""Population curve export -> load -> live warnings."""
import json, subprocess, sys
import numpy as np
import pytest

from calibration import load_params, apply_calibration, identity_params
from calibration import calibration_warnings


def make_pairs(path, n_people=5, n=3000, seed=0):
    """Raw 2D reads high in mid-range (foreshortening), like REHAB24-6."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(30, 175, n_people * n)
    raw = truth + 18 * np.sin(np.radians(truth)) + rng.normal(0, 4, truth.size)
    person = np.repeat([f"P{i}" for i in range(n_people)], n)
    view = np.where(np.arange(truth.size) % 2, "Profile", "Profile/cam18")
    np.savez(path, raw=raw, truth=truth, person=person, view=view)


def test_fit_script_lowers_error_and_saves_loadable_curve(tmp_path):
    pairs, out = tmp_path / "pairs.npz", tmp_path / "curve.json"
    make_pairs(pairs)
    r = subprocess.run([sys.executable, "scripts/fit_calibration.py", str(pairs),
                        "--views", "Profile", "Profile/cam18",
                        "--view-class", "profile", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    p = load_params(str(out))                       # passes check_params
    assert p["variant"] == "A" and p["view_class"] == "profile" and p["n_people"] == 5
    assert p["lopo_mae_calibrated"] < p["lopo_mae_raw"]
    assert apply_calibration(180.0, p) == pytest.approx(180.0, abs=2)


def test_fit_script_refuses_two_people(tmp_path):
    pairs = tmp_path / "pairs.npz"
    make_pairs(pairs, n_people=2)
    r = subprocess.run([sys.executable, "scripts/fit_calibration.py", str(pairs)],
                       capture_output=True, text=True)
    assert r.returncode != 0


def test_warnings_for_missing_identity_and_bad_setup():
    assert calibration_warnings(None, "A")
    ident = identity_params()
    assert any("identity" in w for w in calibration_warnings(ident, "A"))
    bad = {**ident, "curve_version": "pop-curve-v1", "variant": "A",
           "straight_check": {"status": "warn", "message": "Straight arm reads 93 deg"}}
    w = calibration_warnings(bad, "A")
    assert any("93" in x for x in w)
    assert any("fitted on variant A" in x for x in calibration_warnings(bad, "C"))


def test_no_warnings_for_good_calibration():
    good = {**identity_params(), "curve_version": "pop-curve-v1", "variant": "A",
            "straight_check": {"status": "ok", "message": "fine"}}
    assert calibration_warnings(good, "A") == []
