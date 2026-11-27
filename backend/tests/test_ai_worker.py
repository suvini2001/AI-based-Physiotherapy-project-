"""Run in .venv-ai separately. Real model/geometry tests; controlled pose fixtures are labelled as such."""
import ast
import base64
from pathlib import Path
import sys

import pytest
np = pytest.importorskip('numpy')
cv2 = pytest.importorskip('cv2')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_worker import ROOT, decode_frame, load_research_pipeline


def test_exact_research_definition_loaded_without_desktop_side_effects():
    Pipeline, resolve = load_research_pipeline()
    assert Pipeline.__name__ == 'AnglePipeline'
    assert Pipeline.step.__code__.co_filename == str(ROOT / 'demo_cv.py')
    assert 'audio_feedback' not in sys.modules
    assert 'session_packager' not in sys.modules
    assert resolve(None, None, None, {'method': 'ML'}, 'shoulder')[0] == 'A'


def test_real_shipped_model_warms_up_and_outputs_finite_angles():
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_iml_live import synthetic
    from vision.overlay import JOINTS
    Pipeline, _ = load_research_pipeline()
    pipe = Pipeline(JOINTS['elbow']['right'], 'ML', (640, 480), 20, .7, 5,
                    iml_model=str(ROOT / 'data/iml_nolook_rehab24_exp_torso.txt'))
    norm, world = synthetic(n=120)
    outputs = [pipe.step(n, w, fps=20)[1] for n, w in zip(norm, world)]
    assert all(x is None or np.isfinite(x) and 0 <= x <= 180 for x in outputs)
    assert pipe.source_counts['ML'] > 50


def test_jpeg_bounds_and_format():
    encoded = cv2.imencode('.jpg', np.zeros((480, 640, 3), np.uint8))[1].tobytes()
    assert decode_frame(base64.b64encode(encoded)).shape == (480, 640, 3)
    for raw in (b'junk', b'x'*(256*1024+1), cv2.imencode('.jpg', np.zeros((20, 20, 3), np.uint8))[1].tobytes()):
        with pytest.raises((ValueError, OSError)):
            decode_frame(base64.b64encode(raw))


def test_real_geometry_calibration_counting_and_no_pose_gap(monkeypatch):
    # Unit fixture: synthetic landmarks exercise real shoulder geometry/FSM/scoring.
    # Not camera evidence and never written to the application database.
    import pose.landmarks
    from ai_worker import ResearchSession
    class FixtureExtractor:
        def __init__(self): self.theta = 0; self.visible = True
        def process_with_world(self, frame):
            m = np.ones((33, 4), float)
            m[:, :3] = .5
            m[11, :2] = (.35, .3); m[12, :2] = (.65, .3)
            m[23, :2] = (.4, .75); m[24, :2] = (.6, .75)
            t = np.radians(self.theta)
            m[14, :2] = (.65 + .2*np.sin(t), .3 + .2*640/480*np.cos(t))
            if not self.visible: m[:] = np.nan
            return m, m.copy()
        def close(self): pass
    monkeypatch.setattr(pose.landmarks, 'LandmarkExtractor', FixtureExtractor)
    session = ResearchSession('shoulder_abduction', 'right', 2)
    try:
        timestamp = 0
        # Multiple full movements: initial readiness, calibration, then counting.
        for angle in np.tile(np.r_[np.linspace(0, 110, 60), np.linspace(110, 0, 60)], 8):
            session.extractor.theta = float(angle)
            result = session.process_frame(None, timestamp)
            timestamp += .05
            if result['phase'] == 'COMPLETE': break
        assert result['phase'] == 'COMPLETE'
        assert result['rep_count'] == 2
        summary = session.finish()
        assert summary['calibrated'] and summary['total_reps'] == 2
        assert summary['source']['variant'] == 'A'
        assert all(0 <= r['quality_score'] <= 100 for r in summary['reps'])
    finally:
        session.close()
