"""Tests for session_packager outbox and retry logic."""
import json
import os
import pytest
from unittest.mock import patch, MagicMock
import requests

import session_packager as sp


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

SAMPLE_SESSION = {
    "exercise": "elbow",
    "total_reps": 5,
    "date": "2026-11-24",
    "summary": {"mean_rom": 90.0},
    "reps": [],
}


def _mock_response(status_code: int, json_body: dict = None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_body or {}
    r.text = str(json_body)
    return r


# ------------------------------------------------------------------ #
# Queue helpers                                                        #
# ------------------------------------------------------------------ #

def test_enqueue_saves_to_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "QUEUE_PATH", str(tmp_path / "q.json"))
    sp._enqueue(dict(SAMPLE_SESSION))
    q = json.loads((tmp_path / "q.json").read_text())
    assert len(q) == 1
    assert "session_uuid" in q[0]


def test_enqueue_adds_uuid(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "QUEUE_PATH", str(tmp_path / "q.json"))
    session = dict(SAMPLE_SESSION)
    assert "session_uuid" not in session
    sp._enqueue(session)
    assert "session_uuid" in session


# ------------------------------------------------------------------ #
# _try_send                                                            #
# ------------------------------------------------------------------ #

def test_try_send_success():
    with patch("requests.post", return_value=_mock_response(200, {"session_id": 42})):
        result = sp._try_send({"session_uuid": "abc"})
    assert result is True


def test_try_send_500_returns_false():
    with patch("requests.post", return_value=_mock_response(500)):
        result = sp._try_send({"session_uuid": "abc"})
    assert result is False


def test_try_send_timeout_returns_false():
    with patch("requests.post", side_effect=requests.exceptions.Timeout):
        result = sp._try_send({"session_uuid": "abc"})
    assert result is False


def test_try_send_connection_error_returns_false():
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError):
        result = sp._try_send({"session_uuid": "abc"})
    assert result is False


def test_try_send_400_raises():
    with patch("requests.post", return_value=_mock_response(400)):
        with pytest.raises(ValueError):
            sp._try_send({"session_uuid": "abc"})


# ------------------------------------------------------------------ #
# flush_queue                                                          #
# ------------------------------------------------------------------ #

def test_flush_success_clears_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "QUEUE_PATH", str(tmp_path / "q.json"))
    sp._enqueue(dict(SAMPLE_SESSION))

    with patch("requests.post", return_value=_mock_response(200, {"session_id": 1})):
        sp.flush_queue()

    q = json.loads((tmp_path / "q.json").read_text())
    assert q == []


def test_flush_keeps_session_on_500(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "QUEUE_PATH", str(tmp_path / "q.json"))
    monkeypatch.setattr(sp, "_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(sp, "_INITIAL_DELAY", 0)  # no waiting in tests
    sp._enqueue(dict(SAMPLE_SESSION))

    with patch("requests.post", return_value=_mock_response(500)):
        sp.flush_queue()

    q = json.loads((tmp_path / "q.json").read_text())
    assert len(q) == 1   # still there


def test_flush_moves_400_to_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "QUEUE_PATH", str(tmp_path / "q.json"))
    rej_path = str(tmp_path / "rejected_queue.json")
    monkeypatch.setattr(sp, "_INITIAL_DELAY", 0)

    # patch open for rejected path too
    import session_packager
    original = session_packager.__file__

    sp._enqueue(dict(SAMPLE_SESSION))

    with patch("requests.post", return_value=_mock_response(400)):
        with patch.object(sp, "QUEUE_PATH", str(tmp_path / "q.json")):
            # redirect rejected path
            import builtins
            real_open = builtins.open
            def patched_open(path, *a, **kw):
                if path == "rejected_queue.json":
                    return real_open(rej_path, *a, **kw)
                return real_open(path, *a, **kw)
            with patch("builtins.open", patched_open):
                sp.flush_queue()

    q = json.loads((tmp_path / "q.json").read_text())
    assert q == []  # removed from main queue