# tests/test_assessment.py
"""
Unit tests for assessment/rules.py and assessment/scorer.py
"""
from assessment.rules import run_all
import assessment.scorer as scorer


def make_rep(**kwargs):
    base = {
        'rep_number': 1,
        'min_angle': 40.0,
        'max_angle': 162.0,
        'rom': 122.0,
        'duration_s': 2.5,
        '_angle_series': list(range(162, 39, -2)) + list(range(40, 163, 2)),
        '_landmarks': None,
    }
    base.update(kwargs)
    return base


def test_clean_rep_passes_all_rules():
    rep = make_rep()
    faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
    assert faults == []


def test_rom_fault_detected():
    rep = make_rep(min_angle=70)   # 70 > 55+10 = 65 -> fault
    faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
    assert any('bend' in f.lower() for f in faults)


def test_speed_fault_detected():
    rep = make_rep(duration_s=0.4)
    faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
    assert any('fast' in f.lower() for f in faults)


def test_no_landmarks_skips_compensation_checks():
    rep = make_rep(min_angle=70, _landmarks=None)
    faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
    assert len(faults) == 1


def test_patient_relative_rom_impaired_patient():
    rep = make_rep(min_angle=85)
    faults = run_all(rep, frame_height=480, calibrated_flex_thresh=80)
    assert not any('bend' in f.lower() for f in faults)


def test_scorer_fills_quality_score_and_faults():
    scorer.load('template.json')
    rep = make_rep()
    scorer.score_rep(rep, frame_height=480, calibrated_flex_thresh=55)
    assert rep['quality_score'] is not None
    assert isinstance(rep['faults'], list)
    assert 0 <= rep['quality_score'] <= 100


def test_bad_rep_lower_score_than_good_rep():
    scorer.load('template.json')
    good_rep = make_rep()
    bad_rep = make_rep(min_angle=80, duration_s=0.3)
    scorer.score_rep(good_rep, calibrated_flex_thresh=55)
    scorer.score_rep(bad_rep, calibrated_flex_thresh=55)
    assert good_rep['quality_score'] >= bad_rep['quality_score']


def test_dtw_regression_true_curl_scores_higher_than_2d_collapse():
    import numpy as np
    from assessment.dtw import load_template, dtw_score
    template = load_template('template.json')
    true_curl = list(np.linspace(160, 40, 30)) + list(np.linspace(40, 160, 30))
    collapsed = list(np.linspace(150, 5, 30)) + list(np.linspace(5, 150, 30))
    score_true = dtw_score(true_curl, template)
    score_col = dtw_score(collapsed, template)
    assert score_true > 70.0, f"True curl should score > 70, got {score_true}"
    assert score_col < score_true, f"Collapsed curl ({score_col}) should score lower than true curl ({score_true})"
