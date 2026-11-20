#make_fixtures.py
import json, random, uuid, datetime as dt

#තනි exercise session එකක data හදනවා.
def session(sid, day, rom, faults=(), dur=2.2, reps=10):
    rep_list = [{"rep_number": i + 1, "max_angle": 35 + rom, "min_angle": 35.0,
                 "rom": round(rom + random.uniform(-3, 3), 1), "duration_s": dur,
                 "quality_score": 0.75, "faults": [{"code": f, "fault_text": f} for f in faults]}
                for i in range(reps)]
    return {"session_id": sid, "date": day.isoformat(), "exercise": "elbow_flexion",
            "total_reps": reps, "mean_rom": rom, "mean_quality": 0.75,
            "incomplete_attempts": 0, "reps": rep_list}

# Current session එක සහ previous sessions එකතු කරලා n8n වෙත යවන්න සම්පූර්ණ event JSON එක හදනවා.
def event(roms, last_faults=(), dur=2.2):
    start = dt.date(2026, 11, 1)
    sessions = [session(i + 1, start + dt.timedelta(days=2 * i), r) for i, r in enumerate(roms)]
    current = session(len(roms) + 1, start + dt.timedelta(days=2 * len(roms)), roms[-1], last_faults, dur)
    history = [{k: s[k] for k in ("session_id", "date", "total_reps", "mean_rom", "mean_quality")}
               | {"fault_counts": {}} for s in reversed(sessions)]
    return {"contract_version": "1.0", "event": "session.completed", "request_id": str(uuid.uuid4()),
            "patient": {"id": 1}, "prescription": {"exercise": "elbow_flexion", "target_reps": 10, "target_rom": 120.0},
            "session": current, "history": history[:10]}

random.seed(7)
fixtures = {
    "improving": event([90 + 2 * i for i in range(8)]),
    "plateau": event([100.0] * 8),
    "declining_safety": event([105, 106, 105, 107, 100, 92, 85], ["SHOULDER_HIKE"]),
    "fast_and_sloppy": event([100, 101, 102], ["TOO_FAST"], dur=0.7),
}
for name, ev in fixtures.items():
    json.dump(ev, open(f"agents/fixtures/{name}.json", "w"), indent=2)