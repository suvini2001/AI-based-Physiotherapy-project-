# tests/test_shoulder_checks.py
"""Shoulder abduction: hip-relative shoulder-hike check, front-view check, variant guard."""
import numpy as np
from assessment import rules
from assessment.rules import check_shoulder_hike, run_checks
from capture.view_check import front_view_status

SHOULDER = {"kind": "raise", "target": 75.0, "exercise": "shoulder_abduction"}


def front_pose(n=40, hike=0.0, lean_deg=0.0, low_left=0.0):
    """
    Front view, normalised coordinates, 640x480. Right arm abducts 10 -> 90 -> 10.
    hike:     how far the right shoulder lifts at the top (fraction of frame height)
    lean_deg: whole trunk tilts sideways by this much at the top
    low_left: left shoulder permanently lower (natural posture)
    """
    lm = np.full((n, 33, 4), np.nan)
    phase = np.sin(np.linspace(0, np.pi, n))            # 0 -> 1 -> 0
    for f in range(n):
        p = phase[f]
        pts = {11: [0.60, 0.30 + low_left], 12: [0.40, 0.30 - hike * p],
               23: [0.56, 0.62], 24: [0.44, 0.62]}
        a = np.radians(10 + 80 * p)                     # right arm, away from the body
        pts[14] = [0.40 - 0.13 * np.sin(a), 0.30 - hike * p + 0.17 * np.cos(a)]
        t = np.radians(lean_deg * p)                    # rotate everything about the hips
        for idx, (x, y) in pts.items():
            dx, dy = (x - 0.50) * 640, (y - 0.62) * 480
            rx = dx * np.cos(t) - dy * np.sin(t)
            ry = dx * np.sin(t) + dy * np.cos(t)
            lm[f, idx] = [0.50 + rx / 640, 0.62 + ry / 480, 0.0, 1.0]
    return {"_landmarks": lm, "max_angle": 90.0, "min_angle": 10.0, "duration_s": 2.0}


def test_clean_abduction_has_no_hike_fault():
    assert check_shoulder_hike(front_pose()) == "ok"


def test_hiked_shoulder_is_flagged():
    assert "shoulder down" in check_shoulder_hike(front_pose(hike=0.06))


def test_naturally_uneven_shoulders_are_not_flagged():
    assert check_shoulder_hike(front_pose(low_left=0.04)) == "ok"


def test_trunk_lean_is_left_to_the_lean_check():
    rep = front_pose(lean_deg=18.0)
    assert check_shoulder_hike(rep) == "ok"
    assert "leaning sideways" in rules.check_trunk_lean(rep)


def test_shoulder_session_runs_the_hike_check_again():
    """The check must be ON for shoulder exercises (it was switched off before)."""
    faults, n_checked = run_checks(front_pose(hike=0.06), rom_params=SHOULDER)
    assert any("shoulder down" in f for f in faults)
    clean_faults, clean_n = run_checks(front_pose(), rom_params=SHOULDER)
    assert clean_faults == []
    assert clean_n == n_checked == 4          # rom, speed, hike, lean


def test_elbow_session_keeps_its_old_checks():
    elbow = {"kind": "flex", "target": 55.0, "exercise": "elbow_flexion"}
    rep = front_pose(); rep["min_angle"] = 40.0
    _, n_checked = run_checks(rep, rom_params=elbow)
    assert n_checked == 5


def test_hike_check_skipped_without_landmarks_or_side_on():
    assert check_shoulder_hike({"_landmarks": None}) is None
    rep = front_pose()
    rep["_landmarks"][:, 11, 0] = 0.41            # shoulders overlap: side view
    assert check_shoulder_hike(rep) is None


# ── front-view check ─────────────────────────────────────────────
def test_front_view_ok_when_facing_camera():
    assert front_view_status(front_pose()["_landmarks"][0]) == "ok"


def test_front_view_turn_when_rotated_away():
    lm = front_pose()["_landmarks"][0]
    for idx in (11, 12, 23, 24):                  # 60 degree body turn: widths shrink to half
        lm[idx, 0] = 0.5 + (lm[idx, 0] - 0.5) * 0.5
    assert front_view_status(lm) == "turn"


def test_front_view_unknown_when_landmarks_missing():
    lm = front_pose()["_landmarks"][0]
    lm[23] = np.nan
    assert front_view_status(lm) == "unknown"
    assert front_view_status(None) == "unknown"


def test_variant_c_is_not_used_for_the_shoulder(capsys):
    from demo_cv import resolve_angle_method
    v, _, _ = resolve_angle_method("C", None, None, {}, exercise="shoulder")
    assert v == "A"
    assert "ELBOW" in capsys.readouterr().out
