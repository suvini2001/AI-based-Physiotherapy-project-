# tests/test_posture_rules.py
"""
20 Unit and Integration Tests for Posture Compensation Rules and Scorer.
Tests:
  - Clean posture passes all 5 rules
  - NumPy array input and ambiguous truth value handling
  - Shoulder elevation: 6% shrug, jitter tolerance, single glitch robustness
  - Trunk lean: 15 deg (fault), 5 deg (ok), 12 deg (pixel correction test)
  - Shoulder asymmetry: uneven shoulders vs level
  - Missing or all-NaN landmarks: only 2 checks evaluated
  - Intermittent NaN frames handling
  - Pixel coordinates support
  - Scorer honest score computation
  - End-to-end: ElbowFSM -> scorer -> rules detects shrug during curl
  - session_log.json excludes _landmarks
"""
import os
import json
import numpy as np
import pytest

from assessment.rules import (
    run_checks,
    run_all,
    check_rom,
    check_speed,
    check_shoulder_elevation,
    check_trunk_lean,
    check_asymmetry,
    _landmark_array,
)
import assessment.scorer as scorer
from exercise_state import ElbowFSM, save_session


def make_skeleton(n_frames=30, shrug_r=0.0, shrug_l=0.0, lean_deg=0.0,
                  asym=0.0, pixel=False, width=640.0, height=480.0):
    """
    Generate synthetic landmark sequence of shape (n_frames, 33, 4).
    Default upright straight posture:
      Left shoulder (11):  (0.45, 0.35)
      Right shoulder (12): (0.55, 0.35)
      Left hip (23):       (0.46, 0.65)
      Right hip (24):      (0.54, 0.65)
    """
    lms = np.zeros((n_frames, 33, 4), dtype=float)
    lms[:, :, 3] = 0.95  # visibility

    # Lean shift
    # Upright height = 0.30 * height
    h_pix = 0.30 * height
    dx_pix = h_pix * np.tan(np.radians(lean_deg))
    dx_norm = dx_pix / width

    for i in range(n_frames):
        # For multi-frame sequences, keep initial 5 frames as baseline (0 shrug)
        # For single frame generation (n_frames <= 5), use the requested shrug directly
        if n_frames > 5:
            sr = shrug_r if i >= 5 else 0.0
            sl = shrug_l if i >= 5 else 0.0
        else:
            sr = shrug_r
            sl = shrug_l

        # Shoulders (11, 12)
        lms[i, 11, 0] = 0.45 + dx_norm
        lms[i, 11, 1] = 0.35 - sl
        lms[i, 12, 0] = 0.55 + dx_norm
        lms[i, 12, 1] = (0.35 + asym) - sr

        # Hips (23, 24)
        lms[i, 23, 0] = 0.46
        lms[i, 23, 1] = 0.65
        lms[i, 24, 0] = 0.54
        lms[i, 24, 1] = 0.65

    if pixel:
        lms[:, :, 0] *= width
        lms[:, :, 1] *= height

    return lms


def make_test_rep(**kwargs):
    rep = {
        'rep_number': 1,
        'min_angle': 40.0,
        'max_angle': 160.0,
        'rom': 120.0,
        'duration_s': 2.0,
        '_angle_series': [160.0, 100.0, 40.0, 100.0, 160.0],
        '_landmarks': make_skeleton(),
    }
    rep.update(kwargs)
    return rep


# ── Test 1: Straight posture passes all 5 checks ─────────────────
def test_straight_posture_all_checks_pass():
    rep = make_test_rep()
    faults, n_checked = run_checks(rep)
    assert n_checked == 5
    assert faults == []


# ── Test 2: NumPy array input works without ambiguous truth error ─
def test_numpy_array_input():
    rep = make_test_rep(_landmarks=np.array(make_skeleton()))
    faults, n_checked = run_checks(rep)
    assert n_checked == 5
    assert faults == []


# ── Test 3: List input works seamlessly ──────────────────────────
def test_list_input():
    rep = make_test_rep(_landmarks=make_skeleton().tolist())
    faults, n_checked = run_checks(rep)
    assert n_checked == 5
    assert faults == []


# ── Test 4: 6% shoulder shrug triggers elevation fault ───────────
def test_shoulder_elevation_shrug_6_percent():
    # 6% of frame height (0.06 > 0.03 threshold)
    rep = make_test_rep(_landmarks=make_skeleton(shrug_r=0.06))
    faults, n_checked = run_checks(rep)
    assert any("shoulder down" in f.lower() for f in faults)


# ── Test 5: Small jitter does not trigger fault ───────────────────
def test_shoulder_elevation_jitter_no_fault():
    lm = make_skeleton()
    # Add random jitter up to ±0.01 (1% << 3% threshold)
    rng = np.random.default_rng(42)
    lm[:, 11:13, 1] += rng.uniform(-0.01, 0.01, size=lm[:, 11:13, 1].shape)
    rep = make_test_rep(_landmarks=lm)
    faults, _ = run_checks(rep)
    assert not any("shoulder down" in f.lower() for f in faults)


# ── Test 6: Single glitch frame does not trigger fault (90th percentile) ─
def test_shoulder_elevation_single_glitch_frame_no_fault():
    lm = make_skeleton(n_frames=30)
    # 1 single glitch frame spikes to 0.20 (20%)
    lm[15, 12, 1] -= 0.20
    rep = make_test_rep(_landmarks=lm)
    faults, _ = run_checks(rep)
    assert not any("shoulder down" in f.lower() for f in faults)


# ── Test 7: Trunk lean 15 degrees triggers fault ──────────────────
def test_trunk_lean_15_deg_fault():
    rep = make_test_rep(_landmarks=make_skeleton(lean_deg=15.0))
    faults, _ = run_checks(rep)
    assert any("leaning" in f.lower() for f in faults)


# ── Test 8: Trunk lean 5 degrees is ok ────────────────────────────
def test_trunk_lean_5_deg_ok():
    rep = make_test_rep(_landmarks=make_skeleton(lean_deg=5.0))
    faults, _ = run_checks(rep)
    assert not any("leaning" in f.lower() for f in faults)


# ── Test 9: Trunk lean 12 degrees triggers fault (pixel fix test) ─
def test_trunk_lean_12_deg_fault_pixel_bug_fix():
    # In normalized space without aspect ratio, 12 deg appears as 9.06 deg (<10).
    # With pixel aspect ratio (640x480), it correctly evaluates to 12 deg (>10).
    rep = make_test_rep(_landmarks=make_skeleton(lean_deg=12.0, width=640, height=480))
    faults, _ = run_checks(rep, frame_width=640.0, frame_height=480.0)
    assert any("leaning" in f.lower() for f in faults)


# ── Test 10: Shoulder asymmetry triggers fault ───────────────────
def test_shoulder_asymmetry_fault():
    # 5% shoulder difference (> 4% threshold)
    rep = make_test_rep(_landmarks=make_skeleton(asym=0.05))
    faults, _ = run_checks(rep)
    assert any("shoulders level" in f.lower() for f in faults)


# ── Test 11: Level shoulders have no asymmetry fault ─────────────
def test_shoulder_asymmetry_level_ok():
    rep = make_test_rep(_landmarks=make_skeleton(asym=0.0))
    faults, _ = run_checks(rep)
    assert not any("shoulders level" in f.lower() for f in faults)


# ── Test 12: Missing landmarks judges only 2 checks ──────────────
def test_missing_landmarks_judges_two_checks():
    rep = make_test_rep(_landmarks=None)
    faults, n_checked = run_checks(rep)
    assert n_checked == 2
    assert faults == []


# ── Test 13: All NaN landmarks judges only 2 checks ──────────────
def test_all_nan_landmarks_judges_two_checks():
    nan_lm = np.full((30, 33, 4), np.nan)
    rep = make_test_rep(_landmarks=nan_lm)
    faults, n_checked = run_checks(rep)
    assert n_checked == 2
    assert faults == []


# ── Test 14: Intermittent NaN frames do not crash ────────────────
def test_intermittent_nan_frames_does_not_crash():
    lm = make_skeleton(n_frames=30)
    # Simulate camera visibility drops on frames 2, 7, 18
    lm[[2, 7, 18], :, :] = np.nan
    rep = make_test_rep(_landmarks=lm)
    faults, n_checked = run_checks(rep)
    assert n_checked == 5
    assert faults == []


# ── Test 15: Fewer than 3 valid frames are safely skipped ─────────
def test_fewer_than_three_valid_frames_skipped():
    lm = np.full((30, 33, 4), np.nan)
    # Only 2 frames are finite
    lm[0] = make_skeleton(n_frames=1)[0]
    lm[1] = make_skeleton(n_frames=1)[0]
    rep = make_test_rep(_landmarks=lm)
    faults, n_checked = run_checks(rep)
    assert n_checked == 2


# ── Test 16: Pixel coordinates are supported ─────────────────────
def test_pixel_coordinates_supported():
    pixel_lm = make_skeleton(pixel=True, width=640, height=480, shrug_r=0.06)
    rep = make_test_rep(_landmarks=pixel_lm)
    faults, n_checked = run_checks(rep, frame_height=480.0, frame_width=640.0)
    assert n_checked == 5
    assert any("shoulder down" in f.lower() for f in faults)


# ── Test 17: Scorer honest score with missing landmarks ──────────
def test_scorer_honest_score_with_missing_landmarks():
    # Only speed fault, no landmarks, no angle series -> DTW neutral = 50.0
    rep = make_test_rep(duration_s=0.5, _landmarks=None, _angle_series=None)
    scorer.score_rep(rep)
    # 2 checks evaluated, 1 fault -> rules_score = (1/2)*100 = 50%
    # Quality = 0.5 * 50 (DTW default) + 0.5 * 50 (rules) = 50.0
    assert rep['quality_score'] == 50.0
    assert len(rep['faults']) == 1


# ── Test 18: Scorer honest score with all 5 checks passed ────────
def test_scorer_honest_score_all_five_checks():
    # No angle series -> DTW neutral = 50.0
    rep = make_test_rep(_angle_series=None)
    scorer.score_rep(rep)
    # 5 checks evaluated, 0 faults -> rules_score = 100%
    # Quality = 0.5 * 50 (DTW default) + 0.5 * 100 = 75.0
    assert rep['quality_score'] == 75.0
    assert rep['faults'] == []


# ── Test 19: End-to-end: FSM -> Scorer -> Rules with curl shrug ──
def test_end_to_end_fsm_to_scorer_shrug_detection():
    captured = []
    fsm = ElbowFSM(rep_callback=lambda r: captured.append(scorer.score_rep(r)))
    fps = 30.0

    # Simulate 1 rep: 160 -> 40 -> 160 (60 frames)
    angles_down = np.linspace(160, 40, 30)
    angles_up = np.linspace(40, 160, 30)
    angles = np.concatenate([[160.0]*5, angles_down, angles_up, [160.0]*5])

    for i, a in enumerate(angles):
        # Shrug right shoulder by 0.08 near peak flexion (frames 25 to 45)
        shrug = 0.08 if 25 <= i <= 45 else 0.0
        lm = make_skeleton(n_frames=1, shrug_r=shrug)[0]
        fsm.update(float(a), i / fps, landmarks=lm)

    fsm.flush(len(angles) / fps)
    assert len(captured) == 1
    rep = captured[0]
    assert any("shoulder down" in f.lower() for f in rep['faults'])


# ── Test 20: session_log.json does not contain _landmarks ────────
def test_session_log_does_not_contain_landmarks(tmp_path):
    fsm = ElbowFSM()
    angles = np.concatenate([[160.0]*5, np.linspace(160, 40, 30), np.linspace(40, 160, 30), [160.0]*5])
    for i, a in enumerate(angles):
        lm = make_skeleton(n_frames=1)[0]
        fsm.update(float(a), i / 30.0, landmarks=lm)
    fsm.flush(len(angles) / 30.0)

    log_path = str(tmp_path / "session_log.json")
    save_session(fsm, "elbow_flexion", path=log_path)

    with open(log_path, "r") as f:
        data = json.load(f)

    assert data["total_reps"] == 1
    for r in data["reps"]:
        assert "_landmarks" not in r
        assert "_angle_series" in r
