"""Tests for live frame binary encoder/decoder."""
import pytest
from network.live_frame import encode_frame, decode_frame, FRAME_SIZE


def test_frame_is_20_bytes():
    frame = encode_frame(1.5, 90.0, "COUNT", 3)
    assert len(frame) == FRAME_SIZE


def test_round_trip_basic():
    frame = encode_frame(1.5, 90.0, "COUNT", 3)
    result = decode_frame(frame)
    assert result["state"] == "COUNT"
    assert result["rep_count"] == 3
    assert abs(result["joint_angle"] - 90.0) < 0.01
    assert result["fault"] is None


def test_round_trip_with_fault():
    frame = encode_frame(2.0, 45.5, "COUNT", 5, fault="body_sway")
    result = decode_frame(frame)
    assert result["fault"] == "body_sway"


def test_round_trip_all_states():
    for state in ["READY", "CALIB", "COUNT"]:
        frame = encode_frame(0.0, 0.0, state, 0)
        result = decode_frame(frame)
        assert result["state"] == state


def test_too_short_raises():
    with pytest.raises(ValueError, match="too short"):
        decode_frame(b"\x00" * 5)


def test_zero_rep_count():
    frame = encode_frame(0.0, 0.0, "READY", 0)
    result = decode_frame(frame)
    assert result["rep_count"] == 0


def test_max_rep_count():
    frame = encode_frame(0.0, 0.0, "COUNT", 65535)
    result = decode_frame(frame)
    assert result["rep_count"] == 65535