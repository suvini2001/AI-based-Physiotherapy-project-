import json
import os
import threading
import requests

API_BASE = "http://localhost:8000"
QUEUE_PATH = "offline_queue.json"


def _load_queue():
    if not os.path.exists(QUEUE_PATH):
        return []
    with open(QUEUE_PATH) as f:
        return json.load(f)


def _save_queue(q):
    with open(QUEUE_PATH, "w") as f:
        json.dump(q, f, indent=2)


def enqueue_session(session_dict):
    q = _load_queue()
    q.append(session_dict)
    _save_queue(q)
    print(f"[packager] No server — queued locally ({len(q)} pending)")


def flush_queue():
    q = _load_queue()
    if not q:
        return
    sent = []
    for s in q:
        try:
            r = requests.post(f"{API_BASE}/api/sessions",
                              json=s, timeout=5)
            if r.status_code == 200:
                sent.append(s)
                print("[packager] Flushed 1 queued session")
        except Exception:
            break
    remaining = [s for s in q if s not in sent]
    _save_queue(remaining)


def post_session(session_dict, patient_id=1):
    payload = dict(session_dict)
    payload["patient_id"] = patient_id

    # flatten reps into the shape schemas.py expects
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
            r = requests.post(f"{API_BASE}/api/sessions",
                              json=payload, timeout=5)
            if r.status_code == 200:
                print(f"[packager] Session uploaded OK "
                      f"(id={r.json().get('session_id')})")
            else:
                print(f"[packager] Server returned {r.status_code} — queuing")
                enqueue_session(payload)
        except requests.exceptions.ConnectionError:
            print("[packager] Server unreachable — queuing locally")
            enqueue_session(payload)

    threading.Thread(target=_send, daemon=False).start()


# small helper used in demo_cv.py frame loop
_frame_counter = 0

def push_live_frame(state, angle, rep_count, timestamp, fault=None):
    global _frame_counter
    _frame_counter += 1
    if _frame_counter % 3 != 0:   # every 3rd frame ≈ 10 Hz at 30 FPS
        return
    payload = {
        "timestamp":   round(timestamp, 2),
        "joint_angle": angle,
        "state":       state,
        "rep_count":   rep_count,
        "fault":       fault,
    }
    try:
        requests.post(f"{API_BASE}/api/live_frame",
                      json=payload, timeout=0.04)
    except Exception:
        pass   # never block the video loop