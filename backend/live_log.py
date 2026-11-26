"""
backend/live_log.py
Writes frame-age and RTT measurements to a CSV for the Week 8 experiment.
One row per frame received by the server.
"""
import csv
import os
import threading
from datetime import datetime, timezone

LOG_PATH = os.environ.get("LIVE_LOG_PATH", "live_frame_log.csv")

_lock = threading.Lock()
_header_written = False


def _ensure_header(path: str):
    global _header_written
    if _header_written:
        return
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if write_header:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "iso_time",
                "patient_id",
                "server_ts",
                "client_ts",
                "frame_age_ms",
                "rtt_ms",
                "joint_angle",
                "state",
                "rep_count",
                "fault",
            ])
    _header_written = True


def log_frame(
    patient_id,
    server_ts: float,
    client_ts,
    frame_age_ms,
    joint_angle,
    state,
    rep_count,
    fault,
    rtt_ms=None,
):
    """Append one row to the live frame log. Thread-safe."""
    with _lock:
        _ensure_header(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                patient_id,
                round(server_ts, 4),
                round(client_ts, 4) if client_ts is not None else "",
                frame_age_ms if frame_age_ms is not None else "",
                rtt_ms if rtt_ms is not None else "",
                joint_angle,
                state,
                rep_count,
                fault,
            ])


def log_rtt(patient_id, server_ts: float, rtt_ms: float):
    """
    Update the most recent row for this patient+server_ts with the RTT.
    Since rows are appended in order and RTT arrives shortly after,
    we rewrite the last matching row.
    Simple approach: append a separate rtt-only row tagged for joining.
    """
    with _lock:
        _ensure_header(LOG_PATH)
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                patient_id,
                round(server_ts, 4),
                "",        # client_ts
                "",        # frame_age_ms
                round(rtt_ms, 1),
                "",        # joint_angle
                "RTT",     # state — marks this as an RTT-only row
                "",        # rep_count
                "",        # fault
            ])