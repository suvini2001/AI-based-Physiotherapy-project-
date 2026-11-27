"""Isolated fictional data only: never connects to the configured PostgreSQL DB."""
import os
import secrets
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['DATABASE_URL'] = 'sqlite://'
os.environ['PATIENT_JWT_SECRET'] = secrets.token_urlsafe(48)
os.environ['CLINICIAN_JWT_SECRET'] = secrets.token_urlsafe(48)
os.environ['CLINICIAN_ID'] = 'DRTEST'
PASSWORD = secrets.token_urlsafe(20)
os.environ['CLINICIAN_PASSWORD_HASH'] = PasswordHasher().hash(PASSWORD)
import main
from models import Base, Patient, Session as SessionModel


@pytest.fixture
def client():
    engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    def db_override():
        with factory() as db:
            yield db
    main.app.dependency_overrides[main.get_db] = db_override
    with TestClient(main.app) as client:
        client.db_factory = factory
        yield client
    main.app.dependency_overrides.clear()
    engine.dispose()


def register(client, name='Fictional Demo Patient'):
    response = client.post('/patient/register-secure', json={'name': name, 'password': PASSWORD})
    assert response.status_code == 201, response.text
    return response.json()


def headers(account):
    return {'Authorization': 'Bearer ' + account['access_token']}


def clinician(client):
    response = client.post('/auth/login', data={'username': 'DRTEST', 'password': PASSWORD})
    assert response.status_code == 200
    return response.json()


def test_registration_login_prescription_history(client):
    account = register(client)
    patient = account['patient']
    assert not patient['has_prescription']
    assert patient['exercise'] is None
    with client.db_factory() as db:
        stored = db.get(Patient, patient['id'])
        assert stored.password_hash.startswith('$argon2id$')
        assert stored.password_hash != PASSWORD
    logged = client.post('/patient/login-secure', json={'patient_id': patient['patient_id'].lower(), 'password': PASSWORD})
    assert logged.status_code == 200
    clinician_headers = headers(clinician(client))
    for reps in (7, 3):
        assert client.post('/api/prescriptions', headers=clinician_headers, json={'patient_id': patient['id'], 'exercise': 'shoulder_abduction', 'target_reps': reps}).status_code == 200
    me = client.get('/patient/me', headers=headers(account)).json()
    assert me['target_reps'] == 3
    assert me['exercise'] == 'shoulder_abduction'
    assert me['has_prescription']
    assert 'password_hash' not in me
    assert client.get('/patient/me/last-session', headers=headers(account)).json()['date'] is None


def test_other_patient_records_cannot_be_read(client):
    first, second = register(client), register(client, 'Fictional Second Patient')
    with client.db_factory() as db:
        db.add(SessionModel(patient_id=second['patient']['id'], total_reps=17, date='2026-01-01'))
        db.commit()
    own = headers(first)
    assert client.get(f"/patient/{second['patient']['id']}/last-session", headers=own).status_code == 403
    assert client.get(f"/api/session-status/{second['patient']['id']}", headers=own).status_code == 403
    assert client.get('/patient/me/last-session', headers=own).json()['total_reps'] == 0
    assert client.get('/patient/me/last-session', headers=headers(second)).json()['total_reps'] == 17
    assert client.get(f"/api/patients/{second['patient']['id']}/sessions", headers=own).status_code == 401


@pytest.mark.parametrize('path', ['/patient/me', '/patient/me/last-session', '/patient/1/last-session', '/api/session-status/1', '/api/patients', '/api/patients/1/sessions', '/api/sessions/1/reps', '/patient/me/ai-status'])
def test_reads_require_auth(client, path):
    assert client.get(path).status_code == 401


@pytest.mark.parametrize('path', ['/api/sessions', '/api/live_frame', '/api/launch-session', '/api/stop-session', '/api/seed', '/api/prescriptions'])
def test_writes_require_auth(client, path):
    assert client.post(path, json={}).status_code == 401


@pytest.mark.parametrize('path', ['/patient/register', '/patient/login'])
def test_legacy_auth_disabled(client, path):
    assert client.post(path, json={'name': 'Anyone'}).status_code == 410


def test_simulation_and_forged_uploads_never_persist(client):
    account = register(client)
    for mock in (True, False):
        response = client.post('/api/sessions', headers=headers(account), json={'patient_id': account['patient']['id'], 'isMock': mock, 'total_reps': 10})
        assert response.status_code == 503
    with client.db_factory() as db:
        assert db.query(SessionModel).count() == 0
    assert client.get('/patient/me/ai-status', headers=headers(account)).json()['available'] is False


def test_null_hash_and_bad_password_fail(client):
    with client.db_factory() as db:
        db.add(Patient(name='Fictional Legacy Patient', patient_id='PTLEGACY'))
        db.commit()
    account = register(client)
    for patient_id in ('PTLEGACY', 'UNKNOWN', account['patient']['patient_id']):
        response = client.post('/patient/login-secure', json={'patient_id': patient_id, 'password': 'wrong'})
        assert response.status_code == 401
        assert response.json()['detail'] == 'Incorrect patient ID or password'


@pytest.mark.parametrize('claims', [
    {'role': 'patient', 'exp': datetime.now(timezone.utc) - timedelta(seconds=1)},
    {'role': 'clinician', 'exp': datetime.now(timezone.utc) + timedelta(minutes=5)},
    {'role': 'patient'},
])
def test_bad_claims_fail(client, claims):
    account = register(client)
    token = jwt.encode({'sub': str(account['patient']['id']), **claims}, main.PATIENT_JWT_SECRET, algorithm='HS256')
    assert client.get('/patient/me', headers={'Authorization': 'Bearer ' + token}).status_code == 401


def test_role_separation_and_physio_profile(client):
    patient = register(client)
    physio = clinician(client)
    assert client.get('/patient/me', headers=headers(physio)).status_code == 401
    assert client.get('/auth/me', headers=headers(patient)).status_code == 401
    assert set(client.get('/auth/me', headers=headers(physio)).json()) == {'id', 'name', 'specialty'}
    assert client.post('/auth/login', data={'username': 'DRTEST', 'password': 'incorrect'}).status_code == 401


@pytest.mark.parametrize('changes', [{'target_reps': 0}, {'target_reps': 51}, {'exercise': 'unknown'}, {'target_rom': -1}])
def test_invalid_prescriptions(client, changes):
    account = register(client)
    data = {'patient_id': account['patient']['id'], 'exercise': 'elbow_flexion', 'target_reps': 3, **changes}
    assert client.post('/api/prescriptions', headers=headers(clinician(client)), json=data).status_code == 422


def test_missing_secrets_fail_closed(client, monkeypatch):
    monkeypatch.setattr(main, 'SECRET_KEY', None)
    assert client.post('/auth/login', data={'username': 'DRTEST', 'password': PASSWORD}).status_code == 503
    monkeypatch.setattr(main, 'PATIENT_JWT_SECRET', None)
    assert client.post('/patient/register-secure', json={'name': 'Fictional', 'password': PASSWORD}).status_code == 503


def test_websocket_closed(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws/live/1'):
            pass
