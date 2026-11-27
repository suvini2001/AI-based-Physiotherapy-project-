"""Bounded private subprocess adapter. No camera, research packages or DB in API process.

Run one Uvicorn process: active sessions are deliberately local/in-memory.
Client-provided measurements are never accepted. Only worker outputs can be saved.
"""
import base64
from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
MAX_FRAME_BYTES = 256 * 1024
RUNTIME_CACHE = ROOT / '.local-ai-cache'
RUNTIME_CACHE.mkdir(mode=0o700, exist_ok=True)


class AIError(Exception):
    def __init__(self, status, detail):
        self.status, self.detail = status, detail
        super().__init__(detail)


class Worker:
    def __init__(self, python):
        self.temp = tempfile.TemporaryDirectory(prefix='physiotrack-worker-')
        env = {key: os.environ[key] for key in ('PATH', 'SYSTEMROOT', 'TMPDIR') if key in os.environ}
        env.update(HOME=self.temp.name, MPLCONFIGDIR=str(RUNTIME_CACHE), XDG_CACHE_HOME=str(RUNTIME_CACHE), PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TF_CPP_MIN_LOG_LEVEL='2')
        self.env = env
        try:
            self.process = subprocess.Popen([python, '-u', str(ROOT / 'backend/ai_worker.py')],
                                            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, bufsize=0)
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
        except Exception:
            self.temp.cleanup()
            raise
        self.closed = False

    def call(self, message):
        deadline = time.monotonic() + (90 if message.get('op') == 'start' else 20)
        pending = memoryview((json.dumps(message) + '\n').encode())
        response = bytearray()
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([], [self.process.stdin], [], remaining)[1]:
                    raise TimeoutError()
                count = os.write(self.process.stdin.fileno(), pending[:65536])
                pending = pending[count:]
            while b'\n' not in response:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError()
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk or len(response) + len(chunk) > 2 * 1024 * 1024:
                    raise RuntimeError('Invalid worker response')
                response.extend(chunk)
            reply = json.loads(response)
            if not reply.get('ok'):
                raise AIError(422, 'Invalid frame. Send a 640 x 480 JPEG no larger than 256 KiB.')
            return reply['result']
        except AIError:
            raise
        except Exception as error:
            self.close()
            raise AIError(503, 'AI worker stopped or timed out. Stop and restart the session.') from error

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.process.stdin.close()
        self.process.stdout.close()
        self.temp.cleanup()


@dataclass
class ActiveSession:
    patient_id: int
    prescription: dict
    worker: Worker
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created: float = field(default_factory=time.monotonic)
    touched: float = field(default_factory=time.monotonic)
    last_frame: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest: dict = field(default_factory=dict)
    summary: dict | None = None
    saved_id: int | None = None


class AIAdapter:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.thread = None
        self.probe_cache = None
        self.probe_time = 0
        self.probe_lock = threading.Lock()

    def status(self):
        python = os.getenv('AI_PYTHON', '')
        if not python or not Path(python).is_file():
            return dict(available=False, mode='simulation_only', reason='AI runtime is not configured. Run the separate AI setup.', clinical_uploads_enabled=False)
        with self.probe_lock:
            if self.probe_cache and time.monotonic() - self.probe_time < 30:
                return self.probe_cache
            worker = None
            try:
                worker = Worker(python)
                worker.call(dict(op='start', exercise='elbow_flexion', side='right', target_reps=1))
                result = dict(available=True, mode='research_pipeline', clinical_uploads_enabled=False,
                              research_results_enabled=True, reason='Real camera inference; research prototype, not clinically validated.')
            except Exception:
                result = dict(available=False, mode='simulation_only', clinical_uploads_enabled=False,
                              reason='AI runtime or model failed its startup check. Run the AI diagnostic command.')
            finally:
                if worker:
                    worker.close()
            self.probe_cache, self.probe_time = result, time.monotonic()
            return result

    def start(self, patient_id, prescription, side):
        if not self.status()['available']:
            raise AIError(503, self.status()['reason'])
        with self.lock:
            self.reap()
            active = [s for s in self.sessions.values() if not s.worker.closed]
            if any(s.patient_id == patient_id for s in active):
                raise AIError(409, 'An AI session is already active. Stop it or wait 60 seconds for idle cleanup.')
            if len(active) >= 2 or len(self.sessions) >= 64:
                raise AIError(429, 'AI capacity reached. Try again shortly.')
            try:
                worker = Worker(os.environ['AI_PYTHON'])
            except Exception as error:
                raise AIError(503, 'Could not start AI runtime') from error
            try:
                ready = worker.call(dict(op='start', exercise=prescription['exercise'], side=side,
                                         target_reps=prescription['target_reps']))
            except Exception:
                worker.close()
                raise
            entry = ActiveSession(patient_id, {**prescription, 'side': side}, worker)
            entry.latest = dict(phase='READY', rep_count=0, source='research_pipeline', is_simulation=False)
            self.sessions[entry.id] = entry
            if self.thread is None or not self.thread.is_alive():
                self.stop.clear()
                self.thread = threading.Thread(target=self._reaper, daemon=True)
                self.thread.start()
            return dict(session_id=entry.id, prescription=entry.prescription, source=ready['source'],
                        max_frame_bytes=MAX_FRAME_BYTES, width=640, height=480, max_fps=20)

    @contextmanager
    def owned(self, session_id, patient_id, wait=False):
        with self.lock:
            entry = self.sessions.get(session_id)
        if entry is None or entry.patient_id != patient_id:
            raise AIError(404, 'AI session not found')
        if not (entry.lock.acquire(timeout=22) if wait else entry.lock.acquire(blocking=False)):
            raise AIError(409, 'A session request is already in progress')
        try:
            if time.monotonic() - entry.created > 900 or time.monotonic() - entry.touched > 60:
                entry.worker.close()
                raise AIError(410, 'AI session expired. Start a new session.')
            entry.touched = time.monotonic()
            yield entry
        finally:
            entry.lock.release()

    def frame(self, session_id, patient_id, raw):
        with self.owned(session_id, patient_id) as entry:
            return self.frame_locked(entry, raw)

    def frame_locked(self, entry, raw):
        """Caller holds entry.lock, including while accepting the bounded HTTP body."""
        if entry.worker.closed or entry.summary is not None:
            raise AIError(409, 'Session already ended')
        now = time.monotonic()
        if now - entry.last_frame < 0.05:
            raise AIError(429, 'Maximum frame rate is 20 frames per second')
        if not raw or len(raw) > MAX_FRAME_BYTES:
            raise AIError(413, 'Frame must be at most 256 KiB')
        entry.last_frame = now
        result = entry.worker.call(dict(op='frame', jpeg=base64.b64encode(raw).decode(), timestamp=now-entry.created))
        entry.latest = result
        return result

    def discard(self, session_id, patient_id):
        with self.owned(session_id, patient_id, wait=True) as entry:
            entry.worker.close()
            with self.lock:
                self.sessions.pop(session_id, None)
        return {'discarded': True}

    def live(self, patient_id):
        with self.lock:
            entries = [s for s in self.sessions.values() if s.patient_id == patient_id and not s.worker.closed]
            if not entries:
                return {'active': False, 'measurement': None}
            entry = max(entries, key=lambda s: s.created)
            age = time.monotonic() - entry.last_frame if entry.last_frame else None
            # Never display stale angles as current measurements.
            return {'active': True, 'session_id': entry.id, 'age_seconds': age,
                    'measurement': entry.latest if age is not None and age < 3 else None}

    def reap(self):
        with self.lock:
            for key, entry in list(self.sessions.items()):
                if time.monotonic()-entry.touched <= 60 and time.monotonic()-entry.created <= 900:
                    continue
                if entry.lock.acquire(blocking=False):
                    try:
                        entry.worker.close()
                        self.sessions.pop(key, None)
                    finally:
                        entry.lock.release()

    def _reaper(self):
        while not self.stop.wait(5):
            self.reap()

    def shutdown(self):
        self.stop.set()
        with self.lock:
            entries = list(self.sessions.values())
            self.sessions.clear()
        for entry in entries:
            with entry.lock:
                entry.worker.close()


adapter = AIAdapter()
