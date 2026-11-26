"""Tests for frame-age logging and RTT endpoint."""
import os
import csv
import time
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from fastapi.testclient import TestClient
import backend.main as main_module
import backend.live_log as ll

client = TestClient(main_module.app)


def test_push_frame_logs_frame_age(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_log.csv")
    monkeypatch.setattr(ll, "LOG_PATH", log_file)
    monkeypatch.setattr(ll, "_header_written", False)

    # Patch live_log inside main_module so the import inside push_frame uses it
    monkeypatch.setattr(main_module, "live_log", ll, raising=False)

    r = client.post("/api/live_frame", json={
        "patient_id": 1,
        "timestamp": time.time() - 0.03,
        "joint_angle": 85.0,
        "state": "COUNT",
        "rep_count": 2,
        "fault": None,
    })

    assert r.status_code == 200
    assert r.json()["frame_age_ms"] > 0
    assert os.path.exists(log_file), "log file was not created"

    rows = list(csv.DictReader(open(log_file)))
    assert len(rows) == 1
    assert rows[0]["state"] == "COUNT"
    assert float(rows[0]["frame_age_ms"]) > 0


def test_push_frame_no_client_ts_no_log(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_log2.csv")
    monkeypatch.setattr(ll, "LOG_PATH", log_file)
    monkeypatch.setattr(ll, "_header_written", False)

    r = client.post("/api/live_frame", json={
        "patient_id": 1,
        "joint_angle": 90.0,
        "state": "READY",
        "rep_count": 0,
    })

    assert r.status_code == 200
    assert not os.path.exists(log_file) or os.path.getsize(log_file) == 0


def test_rtt_endpoint(tmp_path, monkeypatch):
    log_file = str(tmp_path / "test_rtt.csv")
    monkeypatch.setattr(ll, "LOG_PATH", log_file)
    monkeypatch.setattr(ll, "_header_written", False)
    monkeypatch.setattr(main_module, "live_log", ll, raising=False)

    r = client.post("/api/live_rtt", json={
        "patient_id": 1,
        "server_ts": time.time(),
        "rtt_ms": 42.5,
    })

    assert r.status_code == 200
    assert os.path.exists(log_file), "log file was not created"

    rows = list(csv.DictReader(open(log_file)))
    assert len(rows) == 1
    assert rows[0]["state"] == "RTT"
    assert float(rows[0]["rtt_ms"]) == 42.5

def test_rtt_negative_rejected():
    r = client.post("/api/live_rtt", json={
        "patient_id": 1,
        "server_ts": time.time(),
        "rtt_ms": -5.0,
    })
    assert r.status_code == 422
