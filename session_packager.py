"""
session_packager.py
Sends completed sessions to the backend.
If the server is unreachable or returns 5xx, the session is saved
to offline_queue.json and retried with exponential back-off.
"""
import json
import os
import threading
import time
import uuid
import requests

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")
QUEUE_PATH = "offline_queue.json"

# Back-off settings
_INITIAL_DELAY = 2      # seconds before first retry
_MAX_DELAY     = 60     # cap retries at 60 seconds apart
_MAX_ATTEMPTS  = 5      # give up after this many attempts per flush


# ------------------------------------------------------------------ #
# Queue helpers                                                        #
# ------------------------------------------------------------------ #

def _load_queue():
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH) as f:
        return json.load(f)


def _save_queue(q):
    with open(QUEUE_PATH, "w") as f:
        json.dump(q, f, indent=2)


def _enqueue(payload: dict):
    """Add one session to the local queue (never deletes before server confirms)."""
    q = _load_queue()
    # stamp a uuid if not already there — needed for idempotent sync (Week 3)
    if "session_uuid" not in payload:
        payload["session_uuid"] = str(uuid.uuid4())
    q.append(payload)
    _save_queue(q)
    print(f"[packager] Queued locally ({len(q)} pending)  uuid={payload['session_uuid']}")


# ------------------------------------------------------------------ #
# Send one session (returns True on success)                          #
# ------------------------------------------------------------------ #

def _try_send(payload: dict) -> bool:
    """
    Attempt a single POST.
    Returns True  → server accepted (200/201).
    Returns False → should retry (5xx, timeout, connection error).
    Raises        → bad request (4xx) — do NOT retry, move to rejected.
    """
    try:
        r = requests.post(
            f"{API_BASE}/api/sessions",
            json=payload,
            timeout=5,
        )
        if r.status_code in (200, 201):
            print(f"[packager] Uploaded OK  id={r.json().get('session_id')}  "
                  f"uuid={payload.get('session_uuid')}")
            return True
        elif r.status_code >= 500:
            print(f"[packager] Server error {r.status_code} — will retry")
            return False
        else:
            # 400, 401, 422 etc — retrying won't help
            print(f"[packager] Rejected {r.status_code} — moving to rejected list")
            raise ValueError(f"Server rejected session: {r.status_code} {r.text}")

    except requests.exceptions.Timeout:
        print("[packager] Timeout — will retry")
        return False
    except requests.exceptions.ConnectionError:
        print("[packager] Server unreachable — will retry")
        return False


# ------------------------------------------------------------------ #
# Flush the queue with exponential back-off                           #
# ------------------------------------------------------------------ #

def flush_queue():
    """
    Try to send every queued session.
    Uses exponential back-off: 2s, 4s, 8s … up to 60s between attempts.
    Sessions that the server rejects (4xx) go to rejected_queue.json.
    Never deletes a session from the queue until the server confirms.
    """
    q = _load_queue()
    if not q:
        return

    sent     = []
    rejected = []

    for session in q:
        delay = _INITIAL_DELAY
        success = False

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                success = _try_send(session)
            except ValueError:
                # 4xx — server won't accept this, stop retrying
                rejected.append(session)
                success = False
                break

            if success:
                sent.append(session)
                break

            # failed but retryable — wait and try again
            if attempt < _MAX_ATTEMPTS:
                print(f"[packager] Waiting {delay}s before retry "
                      f"{attempt + 1}/{_MAX_ATTEMPTS} ...")
                time.sleep(delay)
                delay = min(delay * 2, _MAX_DELAY)

        if not success and session not in rejected:
            print(f"[packager] Giving up on session "
                  f"uuid={session.get('session_uuid')} after {_MAX_ATTEMPTS} attempts")

    # save rejected sessions separately so they are not lost
    if rejected:
        rej_path = "rejected_queue.json"
        existing = []
        if os.path.exists(rej_path):
            with open(rej_path) as f:
                existing = json.load(f)
        with open(rej_path, "w") as f:
            json.dump(existing + rejected, f, indent=2)
        print(f"[packager] {len(rejected)} session(s) moved to rejected_queue.json")

    # only remove sessions the server confirmed
    remaining = [s for s in q if s not in sent and s not in rejected]
    _save_queue(remaining)
    print(f"[packager] Flush done — sent={len(sent)} rejected={len(rejected)} "
          f"remaining={len(remaining)}")


# ------------------------------------------------------------------ #
# Post a session (called from demo_cv.py at end of exercise)          #
# ------------------------------------------------------------------ #

def post_session(session_dict, patient_id=1):
    payload = dict(session_dict)
    payload["patient_id"] = patient_id
    payload["session_uuid"] = str(uuid.uuid4())

    # flatten reps into the shape the backend expects
    clean_reps = []
    for r in payload.get("reps", []):
        clean_reps.append({
            "rep_number":    r.get("rep_number", 0),
            "max_angle":     r.get("max_angle", 0.0),
            "min_angle":     r.get("min_angle", 0.0),
            "rom":           r.get("rom", 0.0),
            "duration_s":    r.get("duration_s", 0.0),
            "quality_score": r.get("quality_score"),
            "faults": [{"fault_text": f} for f in r.get("faults", [])],
        })
    payload["reps"] = clean_reps

    def _send():
        try:
            success = _try_send(payload)
        except ValueError:
            # 4xx rejection — queue it so it's not lost, but mark it
            payload["_rejected"] = True
            _enqueue(payload)
            return

        if not success:
            _enqueue(payload)

    threading.Thread(target=_send, daemon=False).start()


# ------------------------------------------------------------------ #
# Live frame push (called every 3rd frame from demo_cv.py)            #
# ------------------------------------------------------------------ #

_frame_counter = 0

def push_live_frame(state, angle, rep_count, timestamp, fault=None, patient_id=None):
    global _frame_counter
    _frame_counter += 1
    if _frame_counter % 3 != 0:
        return
    payload = {
        "timestamp":   round(timestamp, 2),
        "joint_angle": angle,
        "state":       state,
        "rep_count":   rep_count,
        "fault":       fault,
        "patient_id":  patient_id,
    }
    try:
        requests.post(
            f"{API_BASE}/api/live_frame",
            json=payload,
            timeout=0.04,
        )
    except Exception:
        pass   # never block the video loop