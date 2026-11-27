"""Adapter/HTTP contract tests use an explicit worker double, never a production fallback."""
import time
import pytest
from test_web_api import client, register, headers, clinician
import ai_adapter
from ai_adapter import adapter, AIError
from models import Session as SessionModel, Rep, Fault


class FixtureWorker:
    def __init__(self, python):
        self.closed = False
        self.calls = []
    def call(self, message):
        self.calls.append(message)
        if message['op'] == 'start':
            return {'ready': True, 'source': {'pipeline': 'research_pipeline'}}
        if message['op'] == 'frame':
            return dict(phase='COUNT', rep_count=0, joint_angle=None, tracking=False,
                        source='research_pipeline', is_simulation=False, timestamp=message['timestamp'])
        return dict(source={'pipeline': 'research_pipeline', 'is_simulation': False}, calibrated=True,
                    reps=[dict(rep_number=1, max_angle=160., min_angle=50., rom=110., duration_s=4.,
                               quality_score=80., faults=['Test fault'])], total_reps=1, mean_rom=110., mean_quality=80.)
    def close(self):
        self.closed = True


@pytest.fixture
def fake_runtime(monkeypatch):
    adapter.shutdown()
    monkeypatch.setenv('AI_PYTHON', '/explicit-test-runtime')
    monkeypatch.setattr(ai_adapter, 'Worker', FixtureWorker)
    monkeypatch.setattr(adapter, 'status', lambda: {'available': True})
    yield
    adapter.shutdown()


def prescribed(client):
    account = register(client)
    result = client.post('/api/prescriptions', headers=headers(clinician(client)),
                         json={'patient_id': account['patient']['id'], 'exercise': 'elbow_flexion', 'target_reps': 3})
    assert result.status_code == 200
    return account


def start(client, account):
    response = client.post('/patient/me/ai-sessions', headers=headers(account), json={'side': 'left'})
    assert response.status_code == 201, response.text
    return response.json()['session_id']


def test_requires_auth_and_prescription(client, fake_runtime):
    assert client.post('/patient/me/ai-sessions', json={}).status_code == 401
    account = register(client)
    assert client.post('/patient/me/ai-sessions', headers=headers(account), json={}).status_code == 409
    account = prescribed(client)
    assert client.post('/patient/me/ai-sessions', headers=headers(account), json={'patient_id': 999}).status_code == 422
    assert client.post('/patient/me/ai-sessions', headers=headers(account), json={'side': 'invalid'}).status_code == 422


def test_session_ownership_all_operations(client, fake_runtime):
    owner, other = prescribed(client), prescribed(client)
    sid = start(client, owner)
    for method, suffix in [('POST', '/frame'), ('POST', '/finish'), ('DELETE', '')]:
        response = client.request(method, f'/patient/me/ai-sessions/{sid}{suffix}', headers=headers(other))
        assert response.status_code == 404
    assert client.get(f"/api/patients/{owner['patient']['id']}/ai-live", headers=headers(other)).status_code == 401
    assert not adapter.sessions[sid].worker.closed


def test_size_content_type_rate_and_server_timestamp(client, fake_runtime):
    account = prescribed(client)
    sid = start(client, account)
    url = f'/patient/me/ai-sessions/{sid}/frame'
    assert client.post(url, headers=headers(account), json={'joint_angle': 42}).status_code == 415
    jpeg_headers = {**headers(account), 'Content-Type': 'image/jpeg'}
    assert client.post(url, headers=jpeg_headers, content=b'x'*(256*1024+1)).status_code == 413
    assert client.post(url, headers=jpeg_headers, content=b'').status_code == 413
    assert client.post(url, headers=jpeg_headers, content=b'fixture').status_code == 200
    assert client.post(url, headers=jpeg_headers, content=b'fixture').status_code == 429
    call = adapter.sessions[sid].worker.calls[-1]
    assert 0 <= call['timestamp'] < 10
    assert set(call) == {'op', 'jpeg', 'timestamp'}


def test_save_is_server_owned_atomic_idempotent_and_history_visible(client, fake_runtime):
    account = prescribed(client)
    sid = start(client, account)
    url = f'/patient/me/ai-sessions/{sid}/finish'
    forged = {'patient_id': 999, 'total_reps': 99, 'isMock': True}
    first = client.post(url, headers=headers(account), json=forged)
    assert first.status_code == 200, first.text
    second = client.post(url, headers=headers(account))
    assert first.json() == second.json()
    assert first.json()['total_reps'] == 1
    with client.db_factory() as db:
        assert db.query(SessionModel).count() == 1
        assert db.query(Rep).count() == 1
        assert db.query(Fault).count() == 1
        assert db.query(SessionModel).one().patient_id == account['patient']['id']
    last = client.get('/patient/me/last-session', headers=headers(account)).json()
    assert last['total_reps'] == 1
    assert last['mean_rom'] == 110


def test_empty_uncalibrated_never_saved(client, fake_runtime, monkeypatch):
    account = prescribed(client)
    sid = start(client, account)
    worker = adapter.sessions[sid].worker
    original = worker.call
    monkeypatch.setattr(worker, 'call', lambda message: {**original(message), 'calibrated': False} if message['op'] == 'finish' else original(message))
    assert client.post(f'/patient/me/ai-sessions/{sid}/finish', headers=headers(account)).json()['saved'] is False
    with client.db_factory() as db:
        assert db.query(SessionModel).count() == 0


def test_capacity_discard_and_idle_cleanup(client, fake_runtime):
    first, second, third = prescribed(client), prescribed(client), prescribed(client)
    sid = start(client, first)
    assert client.post('/patient/me/ai-sessions', headers=headers(first), json={}).status_code == 409
    start(client, second)
    assert client.post('/patient/me/ai-sessions', headers=headers(third), json={}).status_code == 429
    assert client.delete(f'/patient/me/ai-sessions/{sid}', headers=headers(first)).json()['discarded']
    sid = start(client, third)
    worker = adapter.sessions[sid].worker
    adapter.sessions[sid].touched = time.monotonic() - 61
    adapter.reap()
    assert worker.closed and sid not in adapter.sessions


def test_live_monitor_is_patient_specific_and_expires_stale_frames(client, fake_runtime):
    a, b = prescribed(client), prescribed(client)
    sid = start(client, a)
    entry = adapter.sessions[sid]
    entry.last_frame = time.monotonic()
    entry.latest = {'rep_count': 2, 'joint_angle': 72, 'timestamp': 1}
    auth = headers(clinician(client))
    assert client.get(f"/api/patients/{a['patient']['id']}/ai-live", headers=auth).json()['measurement']['rep_count'] == 2
    assert client.get(f"/api/patients/{b['patient']['id']}/ai-live", headers=auth).json()['measurement'] is None
    entry.last_frame -= 4
    assert client.get(f"/api/patients/{a['patient']['id']}/ai-live", headers=auth).json()['measurement'] is None
