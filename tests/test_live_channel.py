"""Tests for ws-ticket endpoint and room routing."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.main import app, _ws_tickets, _rooms

client = TestClient(app)


def _physio_token():
    """Get a valid physio token for DR001."""
    r = client.post("/auth/login", data={
        "username": "DR001",
        "password": "physio2026",
    })
    return r.json()["access_token"]


def test_ws_ticket_requires_auth():
    r = client.post("/api/ws-ticket", json={"patient_id": 1})
    assert r.status_code == 401


def test_ws_ticket_patient_not_found():
    token = _physio_token()
    r = client.post(
        "/api/ws-ticket",
        json={"patient_id": 99999},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code in (403, 404)


def test_push_frame_returns_frame_age():
    import time
    r = client.post("/api/live_frame", json={
        "patient_id": 1,
        "timestamp": time.time() - 0.05,  # 50ms ago
        "joint_angle": 90.0,
        "state": "COUNT",
        "rep_count": 3,
        "fault": None,
    })
    assert r.status_code == 200
    data = r.json()
    assert "frame_age_ms" in data
    assert data["frame_age_ms"] is not None
    assert data["frame_age_ms"] > 0


def test_push_frame_no_timestamp():
    r = client.post("/api/live_frame", json={
        "patient_id": 1,
        "joint_angle": 90.0,
        "state": "COUNT",
        "rep_count": 1,
    })
    assert r.status_code == 200
    assert r.json()["frame_age_ms"] is None


def test_expired_ticket_rejected():
    from backend.main import _ws_tickets
    from datetime import datetime, timezone, timedelta
    from starlette.websockets import WebSocketDisconnect

    _ws_tickets["expired-ticket"] = {
        "patient_id": 1,
        "physio_id": "DR001",
        "expires": datetime.now(timezone.utc) - timedelta(seconds=60),
    }

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/ws/live/1?ticket=expired-ticket"
        ) as ws:
            ws.receive_text()

    assert exc_info.value.code == 4401
    assert "expired-ticket" not in _ws_tickets


def test_wrong_patient_ticket_rejected():
    from backend.main import _ws_tickets
    from datetime import datetime, timezone, timedelta
    from starlette.websockets import WebSocketDisconnect

    _ws_tickets["wrong-patient-ticket"] = {
        "patient_id": 2,
        "physio_id": "DR001",
        "expires": datetime.now(timezone.utc) + timedelta(seconds=30),
    }

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/ws/live/1?ticket=wrong-patient-ticket"
        ) as ws:
            ws.receive_text()

    assert exc_info.value.code == 4403