"""Real subprocess tests. Opt in: RUN_AI_RUNTIME_TESTS=1; no camera, no DB."""
import base64
import os
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_adapter import Worker, AIError

pytestmark = pytest.mark.skipif(os.getenv('RUN_AI_RUNTIME_TESTS') != '1', reason='Real MediaPipe needs a compatible runtime and OS graphics access')
ROOT = Path(__file__).resolve().parents[2]
PYTHON = str(ROOT / '.venv-ai/bin/python')


@pytest.mark.parametrize('exercise,side', [('elbow_flexion', 'left'), ('shoulder_abduction', 'right')])
def test_real_workers_no_pose_no_results_and_bad_frame_recovery(exercise, side):
    worker = Worker(PYTHON)
    try:
        ready = worker.call(dict(op='start', exercise=exercise, side=side, target_reps=2))
        assert ready['source']['pipeline'] == 'research_pipeline'
        assert ready['source']['variant'] == ('ML' if exercise == 'elbow_flexion' else 'A')
        assert not {'DATABASE_URL', 'PATIENT_JWT_SECRET', 'CLINICIAN_JWT_SECRET'} & set(worker.env)
        with pytest.raises(AIError) as error:
            worker.call(dict(op='frame', jpeg=base64.b64encode(b'bad jpeg').decode(), timestamp=0))
        assert error.value.status == 422
        jpeg = base64.b64encode((ROOT / 'backend/tests/fixtures/blank.jpg').read_bytes()).decode()
        for timestamp in (0, 3.1, 18.1):
            result = worker.call(dict(op='frame', jpeg=jpeg, timestamp=timestamp))
            assert result['joint_angle'] is None
            assert result['rep_count'] == 0
            assert result['tracking'] is False
        assert result['phase'] == 'CALIBRATION_FAILED'
        summary = worker.call({'op': 'finish'})
        assert not summary['calibrated'] and summary['reps'] == []
    finally:
        worker.close()
    assert worker.process.poll() is not None


def test_worker_crash_fails_closed_and_cleans_resources():
    worker = Worker(PYTHON)
    worker.process.kill()
    worker.process.wait(timeout=5)
    with pytest.raises(AIError) as error:
        worker.call({'op': 'start', 'exercise': 'elbow_flexion', 'side': 'right', 'target_reps': 1})
    assert error.value.status == 503
    assert worker.closed
