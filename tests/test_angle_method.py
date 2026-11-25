"""demo_cv.resolve_angle_method: config.yaml chooses the angle method, the command line overrides it,
and a missing I_ML model falls back to variant A instead of stopping the session."""
import os, sys, types
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="module")
def resolve():
    stubs = {}
    for name in ("pose", "pose.landmarks", "audio_feedback"):     # no camera / mediapipe / speech in tests
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = []
            m.LandmarkExtractor = object; m.speak = lambda *a, **k: None
            sys.modules[name] = stubs[name] = m
    try:
        import demo_cv
    except ImportError as e:                                         # cv2 etc. missing
        pytest.skip(f"demo_cv not importable here: {e}")
    yield demo_cv.resolve_angle_method
    for name in stubs:
        sys.modules.pop(name, None)


def _model(tmp_path):
    p = tmp_path / "m.txt"; p.write_text("x"); (tmp_path / "m.txt.json").write_text("{}")
    return str(p)


def test_config_selects_ml(resolve, tmp_path):
    cfg = {"method": "ML", "iml_model": _model(tmp_path)}
    v, m, c = resolve(None, None, None, cfg)
    assert v == "ML" and m == cfg["iml_model"] and c is None


def test_command_line_overrides_config(resolve, tmp_path):
    v, _, _ = resolve("A", None, None, {"method": "ML", "iml_model": _model(tmp_path)})
    assert v == "A"


def test_missing_model_falls_back_to_a(resolve, tmp_path, capsys):
    v, _, _ = resolve(None, None, None, {"method": "ML", "iml_model": str(tmp_path / "none.txt")})
    assert v == "A" and "not found" in capsys.readouterr().out


def test_fallback_curve_from_config(resolve, tmp_path):
    cal = tmp_path / "cal.json"; cal.write_text("{}")
    _, _, c = resolve(None, None, None, {"method": "ML", "iml_model": _model(tmp_path),
                                         "fallback_calibration": str(cal)})
    assert c == str(cal)


def test_no_config_keeps_old_default(resolve):
    v, _, _ = resolve(None, None, None, None)
    assert v == "A"


def test_pipeline_step_runs_with_ml_and_falls_back_first(resolve, tmp_path):
    """The live loop: AnglePipeline.step must run with variant ML, use the curve/raw path
    for the first frames and the model once the feature stream is warm."""
    import json
    import numpy as np
    lgb = pytest.importorskip("lightgbm")
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_iml_live import synthetic, KEEP
    from iml_live import stream_features
    from features import FEATURE_NAMES
    import demo_cv
    lm, wd = synthetic(n=120)
    F = stream_features(lm, wd, "right", (640, 480), 30.0, session="expanding", scale="torso")
    ok = np.isfinite(F[:, 0]); cols = [FEATURE_NAMES.index(k) for k in KEEP]
    m = lgb.LGBMRegressor(n_estimators=10, num_leaves=7, min_child_samples=5, verbosity=-1)
    m.fit(F[ok][:, cols], F[ok, 0])
    path = str(tmp_path / "iml.txt"); m.booster_.save_model(path)
    json.dump({"features": KEEP, "feature_stream": {"session": "expanding", "scale": "torso"}},
              open(path + ".json", "w"))
    pipe = demo_cv.AnglePipeline((12, 14, 16), "ML", (640, 480), 30.0, 0.7, 5, None,
                                 side="right", iml_model=path)
    sources, angles = [], []
    for n, w in zip(lm, wd):
        _, a = pipe.step(n, w, fps=30.0)
        sources.append(pipe.last_source); angles.append(a)
    assert sources[0] == "raw" and sources[-1] == "ML"
    assert pipe.source_counts["ML"] > 60
    assert all(a is None or 0 <= a <= 180 for a in angles)


def test_shoulder_never_uses_the_elbow_models(resolve, tmp_path):
    """I_ML and the curve measure the ELBOW: a shoulder session must get variant A, no curve."""
    cal = tmp_path / "cal.json"; cal.write_text("{}")
    cfg = {"method": "ML", "iml_model": _model(tmp_path), "fallback_calibration": str(cal)}
    v, _, c = resolve(None, None, None, cfg, exercise="shoulder")
    assert v == "A" and c is None
    v, _, c = resolve("ML", None, str(cal), cfg, exercise="shoulder")
    assert v == "A" and c is None
    v, _, _ = resolve("B", None, None, cfg, exercise="shoulder")      # other variants still allowed
    assert v == "B"
