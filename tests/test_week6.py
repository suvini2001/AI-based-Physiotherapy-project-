"""Tests for Week 6: needs-attention rules and audit logging."""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def _physio_token():
    r = client.post("/auth/login", data={
        "username": "DR001",
        "password": "physio2026",
    })
    return r.json()["access_token"]


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_patient_summary_requires_auth():
    r = client.get("/api/patients/1/summary")
    assert r.status_code == 401


def test_patient_summary_not_found():
    token = _physio_token()
    r = client.get(
        "/api/patients/99999/summary",
        headers=_auth(token),
    )
    assert r.status_code == 404


def test_patient_summary_returns_correct_fields():
    token = _physio_token()
    # Seed a patient first
    r = client.post(
        "/api/patients",
        json={"name": "Test Patient W6"},
        headers=_auth(token),
    )
    patient_id = r.json()["id"]

    r = client.get(
        f"/api/patients/{patient_id}/summary",
        headers=_auth(token),
    )
    assert r.status_code == 200
    data = r.json()
    assert "needs_attention" in data
    assert "adherence_pct" in data
    assert "sessions_last_30_days" in data
    assert "last_synced" in data
    assert "attention_reasons" in data


def test_no_sessions_triggers_needs_attention():
    token = _physio_token()
    r = client.post(
        "/api/patients",
        json={"name": "No Sessions Patient"},
        headers=_auth(token),
    )
    patient_id = r.json()["id"]

    r = client.get(
        f"/api/patients/{patient_id}/summary",
        headers=_auth(token),
    )
    data = r.json()
    assert data["needs_attention"] is True
    assert "no_sessions_in_30_days" in data["attention_reasons"]


def test_audit_log_records_view():
    token = _physio_token()
    r = client.post(
        "/api/patients",
        json={"name": "Audit Test Patient"},
        headers=_auth(token),
    )
    patient_id = r.json()["id"]

    # View the patient summary — should create audit entry
    client.get(
        f"/api/patients/{patient_id}/summary",
        headers=_auth(token),
    )

    # Check audit log
    r = client.get(
        f"/api/audit-log?patient_id={patient_id}",
        headers=_auth(token),
    )
    assert r.status_code == 200
    entries = r.json()
    assert len(entries) >= 1
    assert entries[0]["action"] == "view_patient"
    assert entries[0]["patient_id"] == patient_id


def test_needs_attention_list_requires_auth():
    r = client.get("/api/patients/needs-attention")
    assert r.status_code == 401


def test_needs_attention_list_returns_list():
    token = _physio_token()
    r = client.get(
        "/api/patients/needs-attention",
        headers=_auth(token),
    )
    assert r.status_code == 200
    assert isinstance(r.json(), list)