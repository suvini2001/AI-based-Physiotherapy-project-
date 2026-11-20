"""Shoulder I_ML in the live loop: model loading, warm-up, config selection, fallback,
the view check, and the shipped model/config files."""
import json
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from shoulder_features import SH_FEATURE_NAMES, stream_shoulder_features  # noqa: E402
from test_shoulder_features import SIZE, front_abduction                  # noqa: E402

SHIPPED = os.path.join(ROOT, "data", "iml_shoulder_rehab24.txt")
SHOULDER_CURVE = os.path.join(ROOT, "data", "population_curve_shoulder_front.json")


@pytest.fixture(scope="module")
def demo():
    stubs = {}
    for name in ("pose", "pose.landmarks", "audio_feedback"):     # no camera / mediapipe / speech
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = []
            m.LandmarkExtractor = object; m.speak = lambda *a, **k: None
            sys.modules[name] = stubs[name] = m
    try:
        import demo_cv
    except ImportError as e:
        pytest.skip(f"demo_cv not importable here: {e}")
    yield demo_cv
    for name in stubs:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A small shoulder model trained on synthetic front-view abduction."""
    lgb = pytest.importorskip("lightgbm")
    lm, wd, ang = front_abduction(n=240)
    F = stream_shoulder_features(lm, wd, "right", SIZE, 30.0)
    ok = np.isfinite(F[:, 0])
    m = lgb.LGBMRegressor(n_estimators=20, num_leaves=7, min_child_samples=5, verbosity=-1)
    m.fit(F[ok], ang[ok])
    path = str(tmp_path_factory.mktemp("sh") / "sh.txt")
    m.booster_.save_model(path)
    json.dump({"joint": "shoulder", "features": SH_FEATURE_NAMES}, open(path + ".json", "w"))
    return path


def _fake(tmp_path, name="m.txt"):
    p = tmp_path / name; p.write_text("x"); (tmp_path / (name + ".json")).write_text("{}")
    return str(p)


# ---------------------------------------------------------------- the stream
def test_stream_waits_for_warmup_then_outputs_angles(tiny_model):
    from shoulder_live import ShoulderIMLStream
    lm, wd, ang = front_abduction(n=240, seed=1)
    s = ShoulderIMLStream(tiny_model, "right", SIZE, 30.0)
    out = [s.step(n, w) for n, w in zip(lm, wd)]
    assert all(v is None for v in out[:29])
    vals = np.array([np.nan if v is None else v for v in out[60:]])
    assert np.isfinite(vals).mean() > 0.95
    assert np.all((vals[np.isfinite(vals)] >= 0) & (vals[np.isfinite(vals)] <= 180))
    assert np.nanmedian(np.abs(vals - ang[60:])) < 15         # learned the synthetic movement


def test_stream_returns_none_when_arm_not_seen(tiny_model):
    from shoulder_live import ShoulderIMLStream
    lm, wd, _ = front_abduction(n=80)
    lm[50:, 14, 3] = 0.1                                       # right elbow lost
    s = ShoulderIMLStream(tiny_model, "right", SIZE, 30.0)
    out = [s.step(n, w) for n, w in zip(lm, wd)]
    assert out[40] is not None and all(v is None for v in out[50:])


def test_elbow_model_is_refused(tmp_path, tiny_model):
    from shoulder_live import load_shoulder_model
    meta = json.load(open(tiny_model + ".json"))
    bad = tmp_path / "e.txt"; bad.write_text(open(tiny_model).read())
    json.dump({**meta, "joint": "elbow"}, open(str(bad) + ".json", "w"))
    with pytest.raises(ValueError, match="not a shoulder model"):
        load_shoulder_model(str(bad))
    json.dump({**meta, "features": ["angle_2d", "len_fore"]}, open(str(bad) + ".json", "w"))
    with pytest.raises(ValueError, match="len_fore"):
        load_shoulder_model(str(bad))


# ---------------------------------------------------------------- config
def test_shoulder_config_selects_shoulder_ml(demo, tmp_path):
    curve = tmp_path / "c.json"; curve.write_text(json.dumps({"joint": "shoulder"}))
    sh = {"method": "ML", "iml_model": _fake(tmp_path, "sh.txt"), "fallback_calibration": str(curve)}
    elbow = {"method": "ML", "iml_model": _fake(tmp_path, "el.txt")}
    v, m, c = demo.resolve_angle_method(None, None, None, elbow, exercise="shoulder", shoulder_cfg=sh)
    assert v == "ML" and m == sh["iml_model"] and c == str(curve)


def test_missing_shoulder_model_falls_back_to_a_with_curve(demo, tmp_path, capsys):
    curve = tmp_path / "c.json"; curve.write_text(json.dumps({"joint": "shoulder"}))
    sh = {"method": "ML", "iml_model": str(tmp_path / "none.txt"), "fallback_calibration": str(curve)}
    v, _, c = demo.resolve_angle_method(None, None, None, None, exercise="shoulder", shoulder_cfg=sh)
    assert v == "A" and c == str(curve) and "not found" in capsys.readouterr().out


def test_curves_never_cross_joints(demo, tmp_path):
    el_curve = tmp_path / "el.json"; el_curve.write_text("{}")
    sh_curve = tmp_path / "sh.json"; sh_curve.write_text(json.dumps({"joint": "shoulder"}))
    sh = {"method": "ML", "iml_model": _fake(tmp_path, "sh.txt")}
    _, _, c = demo.resolve_angle_method(None, None, str(el_curve), None, exercise="shoulder", shoulder_cfg=sh)
    assert c is None                                           # elbow curve dropped for the shoulder
    _, _, c = demo.resolve_angle_method("A", None, str(sh_curve), {"method": "A"}, exercise="elbow")
    assert c is None                                           # shoulder curve dropped for the elbow


def test_command_line_and_variant_c(demo, tmp_path):
    sh = {"method": "ML", "iml_model": _fake(tmp_path, "sh.txt")}
    assert demo.resolve_angle_method("A", None, None, None, "shoulder", sh)[0] == "A"
    assert demo.resolve_angle_method("C", None, None, None, "shoulder", sh)[0] == "A"
    assert demo.resolve_angle_method("B", None, None, None, "shoulder", sh)[0] == "B"


def test_only_the_raw_shoulder_angle_needs_a_front_view(demo):
    assert demo.needs_front_view("shoulder", "A") is True
    assert demo.needs_front_view("shoulder", "ML") is False
    assert demo.needs_front_view("elbow", "A") is False


# ---------------------------------------------------------------- live loop
def test_pipeline_uses_the_shoulder_model(demo, tiny_model):
    from vision.overlay import JOINTS
    lm, wd, _ = front_abduction(n=120, seed=2)
    pipe = demo.AnglePipeline(JOINTS["shoulder"]["right"], "ML", SIZE, 30.0, 0.7, 5, None,
                              side="right", iml_model=tiny_model, exercise="shoulder")
    assert type(pipe.iml).__name__ == "ShoulderIMLStream"
    sources, angles = [], []
    for n, w in zip(lm, wd):
        _, a = pipe.step(n, w, fps=30.0)
        sources.append(pipe.last_source); angles.append(a)
    assert sources[0] == "raw" and sources[-1] == "ML"
    assert pipe.source_counts["ML"] > 60
    assert all(a is None or 0 <= a <= 180 for a in angles)


# ---------------------------------------------------------------- shipped files
def test_shipped_config_points_at_shipped_files():
    from visibility import load_config
    sh = load_config().get("shoulder_angle")
    assert sh and sh["method"] == "ML"
    assert os.path.exists(os.path.join(ROOT, sh["iml_model"]))
    assert os.path.exists(os.path.join(ROOT, sh["fallback_calibration"]))


def test_shipped_model_loads_and_records_the_decision():
    pytest.importorskip("lightgbm")
    from shoulder_live import load_shoulder_model
    from calibration import load_params
    _, meta = load_shoulder_model(SHIPPED)
    assert meta["choice_per_view"] == {"front": "iml", "profile": "iml"}
    assert meta["rule_choice_per_view"]["front"] == "curve"    # the deviation stays on record
    assert "REHAB24-6" in meta["trained_on"]
    assert load_params(SHOULDER_CURVE)["joint"] == "shoulder"
