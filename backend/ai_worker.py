"""Headless, single-session research adapter. JSON-lines over private stdin/stdout.

Research sources are read-only. Load the original AnglePipeline definition without
executing demo_cv's camera, speech or uploader imports. No algorithm is rewritten.
One OS process owns one pose extractor, FSM and scorer template.
"""
import ast
import base64
from collections import deque
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAX_FRAME_BYTES = 256 * 1024
SIZE = (640, 480)


def load_research_pipeline():
    import numpy as np
    from visibility import mask_low_visibility
    from filters import OneEuroFilter
    from angles import angle
    from angles_world import mask_world_like
    from calibration import apply_calibration
    from vision.gap_fill import fill_gaps
    source = (ROOT / 'demo_cv.py').read_text()
    tree = ast.parse(source, filename=str(ROOT / 'demo_cv.py'))
    names = {'AnglePipeline', 'resolve_angle_method'}
    constants = {'WINDOW', 'SPIKE_WINDOW'}
    selected = [node for node in tree.body if
                isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names or
                isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)]
    if len(selected) != 4:
        raise RuntimeError('Research entry points changed; adapter review required')
    namespace = dict(np=np, deque=deque, os=os, mask_low_visibility=mask_low_visibility,
                     OneEuroFilter=OneEuroFilter, angle=angle, mask_world_like=mask_world_like,
                     apply_calibration=apply_calibration, fill_gaps=fill_gaps)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / 'demo_cv.py'), 'exec'), namespace)
    return namespace['AnglePipeline'], namespace['resolve_angle_method']


def decode_frame(encoded):
    import numpy as np
    import cv2
    from PIL import Image
    raw = base64.b64decode(encoded, validate=True)
    if not 0 < len(raw) <= MAX_FRAME_BYTES:
        raise ValueError('Frame must be at most 256 KiB')
    # Inspect dimensions before native decoding (avoid decompression bombs).
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'JPEG' or image.size != SIZE:
            raise ValueError('Send a 640 x 480 JPEG frame')
        image.verify()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape != (480, 640, 3):
        raise ValueError('Invalid JPEG frame')
    return frame


class ResearchSession:
    def __init__(self, exercise, side, target_reps):
        import numpy as np
        from visibility import load_config
        from calibration import load_params
        from pose.landmarks import LandmarkExtractor
        from exercise_state import ElbowFSM, ShoulderFSM
        from vision.overlay import JOINTS
        import assessment.scorer as scorer
        self.np, self.scorer = np, scorer
        self.exercise, self.side, self.target = exercise, side, target_reps
        if exercise not in ('elbow_flexion', 'shoulder_abduction') or side not in ('left', 'right'):
            raise ValueError('Unsupported exercise or side')
        if not isinstance(target_reps, int) or not 1 <= target_reps <= 50:
            raise ValueError('Invalid target')
        kind = 'elbow' if exercise == 'elbow_flexion' else 'shoulder'
        cfg = load_config(str(ROOT / 'config.yaml'))
        Pipeline, resolve = load_research_pipeline()
        variant, model, curve = resolve(None, None, None, cfg.get('angle'), exercise=kind)
        # A configured elbow ML model must exist; do not advertise a missing model as ML.
        if kind == 'elbow' and cfg.get('angle', {}).get('method') == 'ML' and variant != 'ML':
            raise RuntimeError('Configured elbow model unavailable')
        self.pipe = Pipeline(JOINTS[kind][side], variant, SIZE, 20.0, cfg['visibility_threshold'],
                             cfg['max_gap_frames'], load_params(curve) if curve else None,
                             side=side, iml_model=model)
        self.research_exercise = ('elbow_flexion_left' if kind == 'elbow' and side == 'left' else
                                  'elbow_flexion' if kind == 'elbow' else
                                  'shoulder_abduction_right' if side == 'right' else 'shoulder_abduction')
        template = ROOT / 'templates' / (self.research_exercise + '.json')
        scorer.load(str(template), exercise=self.research_exercise)
        self.feedback = 'Stand in view of the camera. Research prototype; not clinically validated.'
        self.last_rep = None
        self.fsm = (ElbowFSM if kind == 'elbow' else ShoulderFSM)(
            rep_callback=self.on_rep, attempt_callback=self.on_incomplete)
        self.extractor = LandmarkExtractor()
        self.phase = 'READY'
        self.calibration_angles = []
        self.last_time = None
        self.frames = 0
        self.valid_frames = 0
        self.variant = variant
        self.calibrated = False
        self.closed = False
        self.last_visible_time = None
        self.source = {
            'pipeline': 'research_pipeline', 'variant': variant, 'side': side,
            'research_sha256': hashlib.sha256((ROOT / 'demo_cv.py').read_bytes()).hexdigest(),
            'model_sha256': hashlib.sha256(Path(model).read_bytes()).hexdigest() if variant == 'ML' else None,
            'template_sha256': hashlib.sha256(template.read_bytes()).hexdigest() if template.exists() else None,
            'is_simulation': False, 'clinically_validated': False,
        }

    def on_rep(self, rep):
        self.scorer.score_rep(rep, frame_height=480, frame_width=640,
                              rom_params=self.fsm.rom_check_params(), exercise=self.research_exercise)
        self.last_rep = rep
        self.feedback = '; '.join(rep.get('faults', [])) or 'Repetition counted.'

    def on_incomplete(self, attempt):
        from assessment.rules import ROM_FAULT
        self.feedback = ROM_FAULT.get(attempt.get('reason'), 'Incomplete movement; not counted.')

    def process_frame(self, frame, timestamp):
        from exercise_state import calibration_cycle_complete
        from capture.view_check import front_view_status
        if not math.isfinite(timestamp) or timestamp < 0 or self.last_time is not None and timestamp <= self.last_time:
            raise ValueError('Frame timestamps must increase')
        delta = timestamp - self.last_time if self.last_time is not None else 0.125
        self.last_time = timestamp
        if self.phase in ('COMPLETE', 'CALIBRATION_FAILED', 'TRACKING_LOST'):
            return self.snapshot(None, False, timestamp)
        norm, world = self.extractor.process_with_world(frame)
        # Never turn missing-current-pose frames into stale/imputed measurements.
        joint = self.pipe.joint
        present = bool(self.np.all(self.np.isfinite(norm[list(joint), :3])) and
                       self.np.all(norm[list(joint), 3] >= self.pipe.thr))
        clean, angle = self.pipe.step(norm, world, fps=max(3.0, min(30.0, 1 / delta)))
        self.frames += 1
        if not present or angle is None or not math.isfinite(angle):
            angle = None
            self.feedback = 'Pose not visible. Keep the selected arm and upper body in view.'
        elif self.exercise == 'shoulder_abduction' and front_view_status(clean, SIZE) != 'ok':
            angle = None
            self.feedback = 'Face the camera with shoulders and hips visible.'
        else:
            self.valid_frames += 1
            self.last_visible_time = timestamp
        if timestamp < 3:
            self.phase = 'READY'
            if angle is not None:
                self.feedback = 'Get ready; keep the selected arm visible.'
        elif self.phase in ('READY', 'CALIB'):
            self.phase = 'CALIB'
            if angle is not None:
                self.calibration_angles.append(angle)
                self.feedback = 'Calibration: move through your comfortable prescribed range and return.'
            if calibration_cycle_complete(self.calibration_angles) or timestamp >= 18:
                self.calibrated = self.fsm.calibrate_from_signal(self.calibration_angles, fps=1 / delta)
                self.phase = 'COUNT' if self.calibrated else 'CALIBRATION_FAILED'
                self.feedback = ('Calibration complete. Begin your prescribed repetitions.' if self.calibrated else
                                 'Calibration failed. Stop and restart with your arm visible; no results will be saved.')
        elif self.phase == 'COUNT':
            if angle is None and (self.last_visible_time is None or timestamp - self.last_visible_time > 1.5):
                self.phase = 'TRACKING_LOST'
                self.feedback = 'Tracking lost. Stop and restart; completed repetitions can still be saved.'
                return self.snapshot(None, False, timestamp)
            self.fsm.update(angle, timestamp, landmarks=clean if angle is not None else None)
            if self.fsm.rep_count >= self.target:
                self.phase = 'COMPLETE'
                self.feedback = 'Target reached. Review and save your research session results.'
        return self.snapshot(angle, angle is not None, timestamp)

    def snapshot(self, angle, tracked, timestamp):
        return dict(source='research_pipeline', is_simulation=False, phase=self.phase,
                    joint_angle=angle, tracking=tracked, rep_count=self.fsm.rep_count,
                    target_reps=self.target, state=self.fsm.state, feedback=self.feedback,
                    timestamp=round(timestamp, 3), angle_source=self.pipe.last_source if tracked else None,
                    calibrated=self.calibrated, last_rep=self.clean_rep(self.last_rep) if self.last_rep else None)

    @staticmethod
    def clean_rep(rep):
        keys = ('rep_number', 'max_angle', 'min_angle', 'rom', 'duration_s', 'quality_score', 'faults', 'dtw_used')
        return {key: rep[key] for key in keys if key in rep}

    def finish(self):
        # Do not flush unfinished motion into an extra completed repetition.
        reps = [self.clean_rep(rep) for rep in self.fsm.reps]
        summary = self.scorer.session_summary(self.fsm.reps, self.fsm.incomplete_attempts)
        return dict(source=self.source, exercise=self.exercise, side=self.side,
                    calibrated=self.calibrated, frames=self.frames, valid_frames=self.valid_frames,
                    completed=self.phase == 'COMPLETE', total_reps=len(reps), reps=reps,
                    mean_rom=float(self.np.mean([r['rom'] for r in reps])) if reps else None,
                    mean_quality=summary.get('mean_quality'), summary=summary,
                    angle_source_counts=self.pipe.source_counts)

    def close(self):
        if not self.closed:
            self.extractor.close()
            self.closed = True


def main():
    output = sys.stdout
    # Research functions print diagnostics; protocol stdout must contain JSON only.
    with contextlib.redirect_stdout(sys.stderr):
        if '--probe' in sys.argv:
            import numpy as np
            session = ResearchSession('elbow_flexion', 'right', 1)
            try:
                session.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), 0.0)
                result = {'available': True, 'mode': 'research_pipeline'}
            finally:
                session.close()
            output.write(json.dumps(result) + '\n'); output.flush()
            return
        session = None
        try:
            for line in sys.stdin:
                try:
                    message = json.loads(line)
                    if message['op'] == 'start' and session is None:
                        session = ResearchSession(message['exercise'], message['side'], message['target_reps'])
                        result = {'ready': True, 'source': session.source}
                    elif message['op'] == 'frame' and session:
                        result = session.process_frame(decode_frame(message['jpeg']), message['timestamp'])
                    elif message['op'] == 'finish' and session:
                        result = session.finish()
                        session.close()
                    else:
                        raise ValueError('Invalid worker operation')
                    output.write(json.dumps({'ok': True, 'result': result}, allow_nan=False) + '\n')
                except (ValueError, KeyError, OSError) as error:
                    output.write(json.dumps({'ok': False, 'error': 'invalid_frame', 'detail': str(error)}) + '\n')
                output.flush()
        finally:
            if session:
                session.close()


if __name__ == '__main__':
    main()
