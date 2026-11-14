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
