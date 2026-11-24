"""
network/live_frame.py
Binary encoder/decoder for live frames.
Version 1 — 20-byte fixed format (see Formats.md).
"""
import struct

FRAME_FORMAT = "<Bff BH B 7x"   # version, timestamp, angle, state, rep_count, fault + 7 reserved bytes
FRAME_SIZE = 20
FRAME_VERSION = 1

# State codes
STATE_READY = 0
STATE_CALIB = 1
STATE_COUNT = 2

# Fault codes
FAULT_NONE = 0
FAULT_ELBOW_TOO_LOW = 1
FAULT_BODY_SWAY = 2
FAULT_INCOMPLETE_ROM = 3

FAULT_MAP = {
    "elbow_too_low": FAULT_ELBOW_TOO_LOW,
    "body_sway": FAULT_BODY_SWAY,
    "incomplete_rom": FAULT_INCOMPLETE_ROM,
}

STATE_MAP = {
    "READY": STATE_READY,
    "CALIB": STATE_CALIB,
    "COUNT": STATE_COUNT,
}


def encode_frame(timestamp: float, joint_angle: float, state: str,
                  rep_count: int, fault: str | None = None) -> bytes:
    """Encode one live frame to 20 bytes."""
    state_code = STATE_MAP.get(state, STATE_READY)
    fault_code = FAULT_MAP.get(fault or "", FAULT_NONE)
    return struct.pack(
        "<BffBHB7x",
        FRAME_VERSION,
        float(timestamp),
        float(joint_angle),
        state_code,
        int(rep_count),
        fault_code,
    )


def decode_frame(data: bytes) -> dict:
    """Decode 20 bytes back to a dict."""
    if len(data) < FRAME_SIZE:
        raise ValueError(f"Frame too short: {len(data)} bytes, expected {FRAME_SIZE}")
    
    version, timestamp, joint_angle, state_code, rep_count, fault_code = struct.unpack(
        "<BffBHB7x", data[:FRAME_SIZE]
    )
    
    if version != FRAME_VERSION:
        raise ValueError(f"Unknown frame version: {version}")
    
    # Reverse maps
    state_names = {v: k for k, v in STATE_MAP.items()}
    fault_names = {v: k for k, v in FAULT_MAP.items()}
    
    return {
        "version": version,
        "timestamp": round(timestamp, 3),
        "joint_angle": round(joint_angle, 2),
        "state": state_names.get(state_code, "READY"),
        "rep_count": rep_count,
        "fault": fault_names.get(fault_code, None) if fault_code != FAULT_NONE else None,
    }