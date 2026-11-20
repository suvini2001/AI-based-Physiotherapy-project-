"""shoulder_features: truth definition, live angle, and the streaming feature rules."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from shoulder_features import (SH_FEATURE_NAMES, IDX, ShoulderFeatureStream, elevation,  # noqa: E402
                               live_angle_2d, stream_shoulder_features, truth_elevation)

SIZE = (640, 480)


def front_abduction(n=240, side="right", lo=10.0, hi=160.0, hips=True, seed=0):
    """Person facing the camera; one arm goes out to the side lo -> hi -> lo degrees.
    Returns image landmarks (normalised, (n, 33, 4)), world landmarks and the true angle."""
    rng = np.random.default_rng(seed)
    ang = lo + (hi - lo) * 0.5 * (1 - np.cos(np.linspace(0, 2 * np.pi, n)))
    lm = np.full((n, 33, 4), np.nan); wd = np.full((n, 33, 4), np.nan)
    sh_x = {"right": 0.42, "left": 0.58}
    for f in range(n):
        pts = {11: (0.58, 0.30), 12: (0.42, 0.30), 23: (0.55, 0.60), 24: (0.45, 0.60),
               13: (0.60, 0.48), 14: (0.40, 0.48)}                    # both arms down
        a = np.radians(ang[f])
        out = -1.0 if side == "right" else 1.0                         # right arm goes to image left
        arm = (sh_x[side] + out * 0.18 * np.sin(a) * SIZE[1] / SIZE[0], 0.30 + 0.18 * np.cos(a))
        pts[14 if side == "right" else 13] = arm
        for k, (x, y) in pts.items():
            lm[f, k] = [x + rng.normal(0, 0.001), y + rng.normal(0, 0.001), 0.0, 0.95]
            wd[f, k] = [(x - 0.5) * 1.3, (y - 0.5) * 1.3, 0.0, 0.95]
    if not hips:
        lm[:, 23:25, 3] = 0.1
    return lm, wd, ang


def test_elevation_reference_angles():
    hip, msh, sh = np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([0.2, 0.0])
    assert elevation(hip, msh, sh, sh + [0.0, 1.0]) == pytest.approx(0, abs=1e-6)     # arm down
    assert elevation(hip, msh, sh, sh + [1.0, 0.0]) == pytest.approx(90, abs=1e-6)    # horizontal
    assert elevation(hip, msh, sh, sh + [0.0, -1.0]) == pytest.approx(180, abs=1e-6)  # overhead


def test_truth_elevation_uses_joint_dicts():
    J = np.zeros((1, 6, 3))
    J[0, 0], J[0, 1] = [-0.2, 0, 0], [0.2, 0, 0]          # left / right shoulder
    J[0, 2], J[0, 3] = [-0.1, 0.5, 0], [0.1, 0.5, 0]      # left / right hip (y down)
    J[0, 5] = [0.2, 0, -0.3]                              # right elbow straight forward
    g = truth_elevation(J, "right", {"left": 0, "right": 1}, {"left": 4, "right": 5},
                        {"left": 2, "right": 3})
    assert g[0] == pytest.approx(90, abs=1e-6)            # flexion 90 deg counts as 90 too


def test_live_angle_matches_the_overlay_joints():
    from vision.overlay import JOINTS
    from angles import joint_angle
    lm, _, _ = front_abduction(n=20)
    assert np.allclose(live_angle_2d(lm, "right", SIZE),
                       joint_angle(lm, JOINTS["shoulder"]["right"], SIZE), equal_nan=True)


@pytest.mark.parametrize("side", ["right", "left"])
def test_front_view_2d_elevation_tracks_the_truth(side):
    lm, wd, ang = front_abduction(side=side)
    F = stream_shoulder_features(lm, wd, side, SIZE, 30.0)
    assert F.shape == (len(lm), len(SH_FEATURE_NAMES))
    unfiltered = np.array([elevation((lm[f, 23, :2] + lm[f, 24, :2]) / 2 * SIZE,
                                     (lm[f, 11, :2] + lm[f, 12, :2]) / 2 * SIZE,
                                     lm[f, 12 if side == "right" else 11, :2] * SIZE,
                                     lm[f, 14 if side == "right" else 13, :2] * SIZE) for f in range(len(lm))])
    assert np.nanmedian(np.abs(unfiltered - ang)) < 2.0
    # the causal filter only adds a short delay
    assert np.nanmedian(np.abs(F[40:, IDX["elev_2d"]] - ang[40:])) < 6.0
    assert np.nanmedian(F[40:, IDX["el_out"]][ang[40:] > 60]) > 0      # arm out to the side


def test_rows_use_only_past_frames():
    lm, wd, _ = front_abduction(n=200, seed=3)
    a = stream_shoulder_features(lm, wd, "right", SIZE, 30.0)
    lm2 = lm.copy(); lm2[120:, 14, :2] += 0.05
    b = stream_shoulder_features(lm2, wd, "right", SIZE, 30.0)
    assert np.allclose(a[:120], b[:120], equal_nan=True)
    assert not np.allclose(a[120:], b[120:], equal_nan=True)


def test_torso_scale_with_shoulder_fallback():
    lm, wd, _ = front_abduction(n=60, hips=False)
    fs = ShoulderFeatureStream("right", SIZE, 30.0)
    for n, w in zip(lm, wd):
        fs.push(n, w)
    assert fs.scale_used == "shoulder"
    lm, wd, _ = front_abduction(n=60)
    fs = ShoulderFeatureStream("right", SIZE, 30.0)
    for n, w in zip(lm, wd):
        fs.push(n, w)
    assert fs.scale_used == "torso"


def test_planarity_is_about_one_in_the_image_plane():
    lm, wd, _ = front_abduction()
    F = stream_shoulder_features(lm, wd, "right", SIZE, 30.0)
    assert 0.85 < np.nanmedian(F[60:, IDX["plan_upper"]]) <= 1.05
