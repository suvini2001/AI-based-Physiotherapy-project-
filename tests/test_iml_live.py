"""iml_live: the live feature stream must reproduce the training features."""
import json, os, sys
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pose")]
lgb = pytest.importorskip("lightgbm")
from iml_live import IMLStream, offline_features, validate, CausalButter, stream_features  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402

KEEP = [n for n in FEATURE_NAMES if n not in
        {"bone_upper", "bone_fore", "sw_frac", "is_right", "std_angle", "d_angle"}]


def synthetic(n=400, seed=0):
    """A side-on arm doing curls: shoulder fixed, elbow below it, wrist rotating."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30
    flex = 105 + 65 * np.cos(2 * np.pi * t / 3)
    lm = np.full((n, 33, 4), 0.5); lm[..., 3] = 0.95
    wd = np.zeros((n, 33, 4)); wd[..., 3] = 0.95
    th = np.radians(180 - flex)
    for side, (s, e, w), x in (("right", (12, 14, 16), 0.45), ("left", (11, 13, 15), 0.55)):
        lm[:, s, :2] = [x, 0.30]; lm[:, e, :2] = [x, 0.50]
        lm[:, w, 0] = x + 0.18 * np.sin(th); lm[:, w, 1] = 0.50 - 0.18 * np.cos(th) * -1
        wd[:, s, :3] = [x - .5, -0.4, 0]; wd[:, e, :3] = [x - .5, -0.1, 0.02]
        wd[:, w, 0] = x - .5 + 0.25 * np.sin(th); wd[:, w, 1] = -0.1 + 0.25 * np.cos(th); wd[:, w, 2] = 0.05
    lm[:, 23, :2] = [0.46, 0.75]; lm[:, 24, :2] = [0.54, 0.75]
    lm[..., :2] += rng.normal(0, 0.002, lm[..., :2].shape)
    return lm, wd


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    lm, wd = synthetic()
    F = offline_features(lm, wd, "right", (640, 480), 30.0)
    ok = np.isfinite(F[:, 0])
    cols = [FEATURE_NAMES.index(k) for k in KEEP]
    m = lgb.LGBMRegressor(n_estimators=30, num_leaves=7, min_child_samples=5, verbosity=-1)
    m.fit(F[ok][:, cols], F[ok, 0])
    path = str(tmp_path_factory.mktemp("m") / "iml.txt")
    m.booster_.save_model(path)
    json.dump({"features": KEEP}, open(path + ".json", "w"))
    return path


def test_causal_butter_matches_lfilter():
    from scipy.signal import butter, lfilter, lfilter_zi
    x = np.sin(np.arange(200) / 7.0) * 40 + 100
    b, a = butter(2, 6.0 / 15.0)
    ref, _ = lfilter(b, a, x, zi=lfilter_zi(b, a) * x[0])
    f = CausalButter(30.0)
    assert np.allclose([f(v) for v in x], ref)


def test_live_matches_offline(model):
    lm, wd = synthetic(seed=1)
    r = validate(model, lm, wd, "right", (640, 480), 30.0)
    assert r["frames"] > 300
    assert r["mean_abs_diff"] < 2.0


def test_stream_waits_for_warmup_then_outputs_angles(model):
    lm, wd = synthetic(seed=2)
    s = IMLStream(model, "right", (640, 480), 30.0)
    out = [s.step(n, w) for n, w in zip(lm, wd)]
    assert all(v is None for v in out[: s.min_frames - 1])
    vals = [v for v in out[s.min_frames:] if v is not None]
    assert len(vals) > 300 and all(0 <= v <= 180 for v in vals)


def test_model_with_unknown_feature_is_refused(tmp_path, model):
    bad = tmp_path / "bad.txt"
    bad.write_text(open(model).read())
    json.dump({"features": ["angle_2d", "not_a_feature"]}, open(str(bad) + ".json", "w"))
    with pytest.raises(ValueError):
        IMLStream(str(bad))


def test_stream_features_use_only_past_frames():
    """Row i must not change when frames after i change: that is what 'streaming' means."""
    lm, wd = synthetic(n=200, seed=3)
    a = stream_features(lm, wd, "right", (640, 480), 30.0)
    lm2, wd2 = lm.copy(), wd.copy()
    lm2[120:, 16, :2] += 0.05                     # change the future wrist only
    b = stream_features(lm2, wd2, "right", (640, 480), 30.0)
    assert a.shape == (200, len(FEATURE_NAMES))
    assert np.allclose(a[:120], b[:120], equal_nan=True)
    assert not np.allclose(a[120:], b[120:], equal_nan=True)


def test_stream_features_match_the_live_stream(model):
    """The replay and the webcam path give the model the same rows."""
    lm, wd = synthetic(seed=4)
    rows = stream_features(lm, wd, "right", (640, 480), 30.0)
    s = IMLStream(model, "right", (640, 480), 30.0)
    for n, w in zip(lm, wd):
        s.step(n, w)
    assert np.allclose(rows[-1], s.last_features, equal_nan=True)


@pytest.mark.parametrize("scale", ["shoulder", "torso"])
def test_expanding_stream_uses_only_past_frames(scale):
    lm, wd = synthetic(n=400, seed=5)
    a = stream_features(lm, wd, "right", (640, 480), 30.0, session="expanding", scale=scale)
    lm2 = lm.copy(); lm2[330:, 16, :2] += 0.05
    b = stream_features(lm2, wd, "right", (640, 480), 30.0, session="expanding", scale=scale)
    assert np.allclose(a[:330], b[:330], equal_nan=True)


def test_expanding_keeps_bone_length_after_the_window_forgets():
    """An arm seen straight only in the first seconds keeps its true length in
    'expanding' mode, but is forgotten by the 300-frame window."""
    from iml_live import FeatureStream, IDX
    lm, wd = synthetic(n=700, seed=6)
    lm[60:, 16, :2] = lm[60:, 14, :2] + (lm[60:, 16, :2] - lm[60:, 14, :2]) * 0.5   # forearm looks short later
    rows = {s: stream_features(lm, wd, "right", (640, 480), 30.0, session=s)
            for s in ("window", "expanding")}
    i = IDX["plan_fore"]
    assert np.nanmedian(rows["expanding"][-50:, i]) < 0.7          # still knows the arm is longer
    assert np.nanmedian(rows["window"][-50:, i]) > 0.9             # window has forgotten


def test_torso_scale_falls_back_to_shoulders_without_hips():
    from iml_live import FeatureStream
    lm, wd = synthetic(n=60, seed=7)
    lm[:, 23:25, 3] = 0.1                                           # hips not visible
    fs = FeatureStream("right", (640, 480), 30.0, session="expanding", scale="torso")
    for n, w in zip(lm, wd):
        fs.push(n, w)
    assert fs.scale_used == "shoulder"
    lm2, wd2 = synthetic(n=60, seed=7)
    fs2 = FeatureStream("right", (640, 480), 30.0, session="expanding", scale="torso")
    for n, w in zip(lm2, wd2):
        fs2.push(n, w)
    assert fs2.scale_used == "torso"


def test_model_meta_selects_the_feature_stream(tmp_path, model):
    m = tmp_path / "exp.txt"
    m.write_text(open(model).read())
    json.dump({"features": KEEP, "feature_stream": {"session": "expanding", "scale": "torso"}},
              open(str(m) + ".json", "w"))
    s = IMLStream(str(m), "right", (640, 480), 30.0)
    assert (s.fs.session, s.fs.scale) == ("expanding", "torso")
    assert (IMLStream(model).fs.session, IMLStream(model).fs.scale) == ("window", "shoulder")
