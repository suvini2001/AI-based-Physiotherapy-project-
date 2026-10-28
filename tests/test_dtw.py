# tests/test_dtw.py
"""DTW scoring follows Labchurat 2026 (Eq. 12 and Eq. 14)."""
import json
import numpy as np
import pytest

from assessment.dtw import (z_to_score, dtw_score, build_template,
                            is_calibrated, load_template)
import assessment.scorer as scorer


def curl(top=160, bot=40, n=60, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    return list((top + bot) / 2 + (top - bot) / 2 * np.cos(2 * np.pi * t)
                + rng.normal(0, noise, n))


def healthy_reps(k=30):
    rng = np.random.default_rng(42)
    return [curl(top=rng.uniform(152, 168), bot=rng.uniform(35, 50),
                 n=int(rng.uniform(50, 90)), noise=1.5, seed=i) for i in range(k)]


# ── Eq. 14 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("z, expected", [(-3, 100), (0, 100), (1, 100),
                                         (2, 50), (4, 25), (10, 10)])
def test_eq14_values(z, expected):
    assert z_to_score(z) == pytest.approx(expected)


def test_no_cliff_just_above_one_sd():
    # the old Phi(-z) gave 2.3 at z = 2 and 50 at z = 0
    assert z_to_score(1.2) == pytest.approx(83.3, abs=0.1)
    assert z_to_score(2.0) == 50.0


# ── template building ────────────────────────────────────────────
def test_build_template_fields():
    t = build_template(healthy_reps(), "elbow_flexion", "healthy_volunteers", 5)
    assert is_calibrated(t)
    assert t["recorded_by"] == "healthy_volunteers"
    assert len(t["mean_trajectory"]) == 100
    assert t["n_reps"] == 30 and t["n_people"] == 5
    assert t["sigma_dtw"] >= 1.0


def test_build_template_needs_three_reps():
    with pytest.raises(ValueError):
        build_template(healthy_reps(2), "elbow_flexion", "x", 1)


def test_healthy_reps_score_high_and_worse_reps_score_lower():
    t = build_template(healthy_reps(), "elbow_flexion", "healthy_volunteers", 5)
    new_healthy = [dtw_score(curl(158, 42, 70, 1.5, seed=100 + i), t) for i in range(10)]
    assert np.median(new_healthy) >= 90
    reduced = dtw_score(curl(145, 70), t)       # band-like reduced ROM
    half = dtw_score(curl(160, 110), t)
    assert half < reduced < np.median(new_healthy)


def test_uncalibrated_sigma_gives_neutral():
    assert dtw_score(curl(), {"mean_trajectory": curl(), "mu_dtw": 0, "sigma_dtw": 0}) == 50.0


# ── the shipped template is honest ───────────────────────────────
def test_shipped_template_is_marked_synthetic():
    t = load_template("template.json")
    if t.get("calibrated"):
        assert "synthetic" not in t["recorded_by"]
        assert t["n_people"] >= 3
    else:
        assert "synthetic" in t["recorded_by"]
        assert t["n_reps"] == 0


def test_scorer_ignores_dtw_for_uncalibrated_template(tmp_path):
    p = tmp_path / "t.json"
    json.dump({"exercise": "elbow_flexion", "recorded_by": "synthetic",
               "calibrated": False, "mean_trajectory": curl(),
               "mu_dtw": 10.0, "sigma_dtw": 5.0}, open(p, "w"))
    scorer.load(str(p))
    rep = {"min_angle": 40.0, "max_angle": 160.0, "rom": 120.0,
           "duration_s": 2.5, "_angle_series": curl(150, 80), "_landmarks": None}
    scorer.score_rep(rep, 480, 55, 640)
    assert rep["dtw_used"] is False and rep["dtw_score"] == 50.0


def test_scorer_uses_dtw_for_calibrated_template(tmp_path):
    p = tmp_path / "t.json"
    json.dump(build_template(healthy_reps(), "elbow_flexion", "healthy_volunteers", 5),
              open(p, "w"))
    scorer.load(str(p))
    rep = {"min_angle": 40.0, "max_angle": 160.0, "rom": 120.0,
           "duration_s": 2.5, "_angle_series": curl(158, 42, 70), "_landmarks": None}
    scorer.score_rep(rep, 480, 55, 640)
    assert rep["dtw_used"] is True and rep["dtw_score"] >= 90
    scorer._template = None


def test_scorer_dtw_not_used_gives_pure_rules_score(tmp_path):
    p = tmp_path / "t.json"
    json.dump({"exercise": "elbow_flexion", "recorded_by": "synthetic",
               "calibrated": False, "mean_trajectory": curl(),
               "mu_dtw": 10.0, "sigma_dtw": 5.0}, open(p, "w"))
    scorer.load(str(p))
    rep = {"min_angle": 40.0, "max_angle": 160.0, "rom": 120.0,
           "duration_s": 2.5, "_angle_series": curl(150, 80), "_landmarks": None}
    scorer.score_rep(rep, 480, 55, 640)
    # 0 faults from 2 checks -> rules_score = 100.0 -> quality_score = 100.0
    assert rep["dtw_used"] is False
    assert rep["quality_score"] == 100.0
    scorer._template = None


def test_scorer_mismatched_exercise_skips_dtw(tmp_path):
    p = tmp_path / "t.json"
    json.dump(build_template(healthy_reps(), "elbow_flexion", "healthy_volunteers", 5),
              open(p, "w"))
    scorer.load(str(p))
    # Rep is shoulder_abduction, but template is elbow_flexion
    rep = {"exercise": "shoulder_abduction", "min_angle": 20.0, "max_angle": 90.0,
           "rom": 70.0, "duration_s": 2.5, "_angle_series": curl(90, 20), "_landmarks": None}
    scorer.score_rep(rep, 480, rom_params={"kind": "raise", "target": 75.0, "exercise": "shoulder_abduction"})
    assert rep["dtw_used"] is False
    assert rep["quality_score"] == 100.0
    scorer._template = None
