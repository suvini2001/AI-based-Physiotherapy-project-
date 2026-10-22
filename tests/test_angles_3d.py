# tests/test_angles_3d.py
"""Synthetic tests for angles_3d.py (B1 body_yaw, B3 yaw-corrected angle, B4 gate).
Run: pytest -q tests/test_angles_3d.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from angles_3d import body_yaw, elbow_angle_yaw_corrected, bone_length_gate


# ── helpers ──────────────────────────────────────────────────────────────────

def blank_world(n=1):
    """(N, 33, 4) array of NaN — fill in only the joints you care about."""
    return np.full((n, 33, 4), np.nan)


def frontal_world(n=5):
    """Person facing straight at the camera: shoulder line is purely along X -> yaw ≈ 0."""
    w = blank_world(n)
    w[:, 11, :3] = [-0.25,  0.3, 0.0]   # left shoulder
    w[:, 12, :3] = [ 0.25,  0.3, 0.0]   # right shoulder
    w[:, 23, :3] = [-0.15, -0.3, 0.0]   # left hip
    w[:, 24, :3] = [ 0.15, -0.3, 0.0]   # right hip
    return w


def sideways_world(n=5):
    """Person rotated 45 degrees: shoulder line has equal X and Z components -> yaw ≈ 45°."""
    w = blank_world(n)
    d = 0.25 / np.sqrt(2)
    w[:, 11, :3] = [-d,  0.3, -d]
    w[:, 12, :3] = [ d,  0.3,  d]
    w[:, 23, :3] = [-0.15, -0.3, 0.0]
    w[:, 24, :3] = [ 0.15, -0.3, 0.0]
    return w


def straight_elbow_world(n=5, side="right"):
    """Arm fully extended -> angle should be ~180°."""
    w = frontal_world(n)
    # right arm: shoulder (12), elbow (14), wrist (16) in a straight line
    w[:, 12, :3] = [0.25,  0.3, 0.0]
    w[:, 14, :3] = [0.50,  0.3, 0.0]
    w[:, 16, :3] = [0.75,  0.3, 0.0]
    return w


def bent_elbow_world(n=5, deg=90.0, side="right"):
    """Arm bent to a known angle."""
    w = frontal_world(n)
    rad = np.radians(deg)
    # shoulder -> elbow along X, elbow -> wrist at `deg` in the XY plane
    w[:, 12, :3] = [0.0, 0.3, 0.0]
    w[:, 14, :3] = [0.3, 0.3, 0.0]
    w[:, 16, :3] = [0.3 + 0.3 * np.cos(np.pi - rad),
                    0.3 + 0.3 * np.sin(np.pi - rad), 0.0]
    return w


# ── body_yaw tests ───────────────────────────────────────────────────────────

def test_yaw_frontal_near_zero():
    w = frontal_world()
    yaw = body_yaw(w)
    assert np.all(yaw < 5.0), f"frontal yaw should be ~0, got {yaw}"


def test_yaw_sideways_near_45():
    w = sideways_world()
    yaw = body_yaw(w)
    assert np.all(np.abs(yaw - 45.0) < 3.0), f"sideways yaw should be ~45, got {yaw}"


def test_yaw_nan_when_shoulder_missing():
    w = frontal_world()
    w[:, 11, :3] = np.nan   # remove left shoulder
    yaw = body_yaw(w)
    assert np.all(np.isnan(yaw)), "missing shoulder should give NaN yaw"


def test_yaw_single_frame():
    w = frontal_world(1)[0]   # shape (33, 4)
    yaw = body_yaw(w)
    assert np.isscalar(yaw) or yaw.shape == (), "single frame should return scalar"
    assert float(yaw) < 5.0


# ── yaw-corrected angle tests ────────────────────────────────────────────────

def test_yaw_corrected_frontal_same_as_raw():
    """At yaw=0, cos(yaw)=1, so corrected angle == raw angle."""
    w = bent_elbow_world(deg=90.0)
    from angles_world import elbow_angle_3d
    raw = elbow_angle_3d(w)
    corrected = elbow_angle_yaw_corrected(w)
    np.testing.assert_allclose(raw, corrected, atol=1.0,
                               err_msg="at yaw=0 corrected should equal raw")


def test_yaw_corrected_stays_in_0_180():
    w = bent_elbow_world(deg=90.0)
    corrected = elbow_angle_yaw_corrected(w)
    assert np.all((corrected >= 0) & (corrected <= 180)), "angle must stay in [0, 180]"


def test_yaw_corrected_nan_propagates():
    w = blank_world(5)   # all NaN
    corrected = elbow_angle_yaw_corrected(w)
    assert np.all(np.isnan(corrected)), "all-NaN input should give all-NaN output"


def test_yaw_cap_limits_amplification():
    """Even at 80° yaw the correction should not exceed 1/cos(60°) = 2×."""
    w = frontal_world(5)
    # Force high yaw by tilting shoulder line heavily into Z
    w[:, 11, :3] = [-0.05, 0.3,  0.24]
    w[:, 12, :3] = [ 0.05, 0.3, -0.24]   # ~80° yaw
    w[:, 14, :3] = [ 0.30, 0.3,  0.00]
    w[:, 16, :3] = [ 0.60, 0.3,  0.00]
    from angles_world import elbow_angle_3d
    raw = elbow_angle_3d(w)
    corrected = elbow_angle_yaw_corrected(w)
    ratio = np.where(raw > 0, corrected / raw, np.nan)
    assert np.nanmax(ratio) <= 2.05, f"cap failed: max ratio = {np.nanmax(ratio):.2f}"


# ── bone_length_gate tests ───────────────────────────────────────────────────

def test_gate_all_valid_on_clean_data():
    w = bent_elbow_world(n=20)
    valid = bone_length_gate(w)
    assert valid.all(), "clean consistent data should all pass the gate"


def test_gate_rejects_landmark_jump():
    w = bent_elbow_world(n=20)
    # frame 10: wrist teleports to a very different position
    w[10, 16, :3] = [5.0, 5.0, 5.0]
    valid = bone_length_gate(w)
    assert not valid[10], "frame with landmark jump should be rejected"
    assert valid[:10].sum() >= 8, "frames before the jump should mostly pass"


def test_gate_rejects_nan_frames():
    w = bent_elbow_world(n=10)
    w[5, 14, :3] = np.nan   # elbow missing in frame 5
    valid = bone_length_gate(w)
    assert not valid[5], "NaN landmark should be rejected by gate"


def test_gate_40_percent_jump_threshold():
    """A 39 % jump should pass; a 41 % jump should fail."""
    w = bent_elbow_world(n=30)
    # calculate median forearm length first
    base = np.linalg.norm(w[0, 14, :3] - w[0, 16, :3])
    # frame 15: forearm 39 % longer (should pass)
    w[15, 16, :3] = w[15, 14, :3] + (w[0, 16, :3] - w[0, 14, :3]) * 1.39
    # frame 16: forearm 41 % longer (should fail)
    w[16, 16, :3] = w[16, 14, :3] + (w[0, 16, :3] - w[0, 14, :3]) * 1.41
    valid = bone_length_gate(w)
    assert valid[15], "39 % jump should pass (below threshold)"
    assert not valid[16], "41 % jump should fail (above threshold)"


def test_gate_5_percent_minimum_threshold():
    """A bone shorter than 5 % of median should be rejected."""
    w = bent_elbow_world(n=20)
    # collapse elbow and wrist to nearly the same point in frame 8
    w[8, 16, :3] = w[8, 14, :3] + 0.001 * (w[0, 16, :3] - w[0, 14, :3])
    valid = bone_length_gate(w)
    assert not valid[8], "bone shorter than 5 % of median should be rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])