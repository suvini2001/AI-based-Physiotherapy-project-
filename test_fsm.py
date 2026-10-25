# test_fsm.py  — full corrected version
import numpy as np
from exercise_state import ElbowFSM, ShoulderFSM

def make_elbow_signal(n_reps=5, fps=30, rep_duration_s=2.0):
    """
    Simulate n_reps of elbow flexion.
    Signal goes 160° → 40° → 160° for each rep.
    Extra 0.5s of rest at 160° at start and end so FSM sees
    clear EXTENDED state at both ends.
    """
    frames_per_rep = int(rep_duration_s * fps)
    rest_frames    = int(0.5 * fps)   # 0.5s at 160° before and after

    signal = []

    # rest at start (fully extended)
    signal.extend([160.0] * rest_frames)

    for _ in range(n_reps):
        # one rep: 160 → 40 → 160 as a cosine
        t = np.linspace(0, 2 * np.pi, frames_per_rep)
        rep = 100 - 60 * np.cos(t)   # oscillates 40 to 160
        signal.extend(rep.tolist())

    # rest at end
    signal.extend([160.0] * rest_frames)

    return np.array(signal)


def make_shoulder_signal(n_reps=4, fps=30, rep_duration_s=2.0):
    """
    Simulate n_reps of shoulder abduction: 10° → 90° → 10°
    """
    frames_per_rep = int(rep_duration_s * fps)
    rest_frames    = int(0.5 * fps)

    signal = []
    signal.extend([10.0] * rest_frames)

    for _ in range(n_reps):
        t   = np.linspace(0, 2 * np.pi, frames_per_rep)
        rep = 50 - 40 * np.cos(t)    # oscillates 10 to 90
        signal.extend(rep.tolist())

    signal.extend([10.0] * rest_frames)
    return np.array(signal)


def run_fsm(fsm, signal, fps=30):
    reps_seen = []

    def capture(rep):
        reps_seen.append(rep)

    fsm.rep_callback = capture

    for i, angle in enumerate(signal):
        ts = i / fps
        fsm.update(float(angle), ts)

    # flush any in-progress rep at end of signal
    fsm.flush(len(signal) / fps)

    return reps_seen


# ── Test 1: ElbowFSM correct count ───────────────────────────────────
print("=" * 50)
print("TEST 1 — ElbowFSM with synthetic signal")
print("=" * 50)
N = 5
signal = make_elbow_signal(n_reps=N)
fsm = ElbowFSM()
reps = run_fsm(fsm, signal)

print(f"Signal: {signal.min():.0f}° – {signal.max():.0f}°  "
      f"({len(signal)} frames)")
print(f"Expected reps : {N}")
print(f"Detected reps : {fsm.rep_count}")
print("PASS ✅" if fsm.rep_count == N else f"FAIL ❌  got {fsm.rep_count}")
for r in reps:
    print(f"  Rep {r['rep_number']}: ROM={r['rom']:.0f}°  "
          f"max={r['max_angle']:.0f}°  min={r['min_angle']:.0f}°  "
          f"dur={r['duration_s']:.1f}s")


# ── Test 2: Partial rep not counted ──────────────────────────────────
print()
print("=" * 50)
print("TEST 2 — Partial rep (reverses before FLEX_THRESH=55°)")
print("=" * 50)
# goes down to 125° only — above 55° threshold — then back up
partial = [160] * 10 + list(range(160, 124, -2)) + list(range(124, 162, 2)) + [160] * 10
fsm2 = ElbowFSM()
run_fsm(fsm2, partial)
print(f"Expected reps : 0")
print(f"Detected reps : {fsm2.rep_count}")
print("PASS ✅" if fsm2.rep_count == 0 else f"FAIL ❌  got {fsm2.rep_count}")


# ── Test 3: ShoulderFSM ───────────────────────────────────────────────
print()
print("=" * 50)
print("TEST 3 — ShoulderFSM with synthetic signal")
print("=" * 50)
N_S = 4
signal_s = make_shoulder_signal(n_reps=N_S)
fsm_s = ShoulderFSM()
reps_s = run_fsm(fsm_s, signal_s)

print(f"Signal: {signal_s.min():.0f}° – {signal_s.max():.0f}°")
print(f"Expected reps : {N_S}")
print(f"Detected reps : {fsm_s.rep_count}")
print("PASS ✅" if fsm_s.rep_count == N_S else f"FAIL ❌  got {fsm_s.rep_count}")
for r in reps_s:
    print(f"  Rep {r['rep_number']}: ROM={r['rom']:.0f}°  "
          f"max={r['max_angle']:.0f}°  dur={r['duration_s']:.1f}s")


# ── Test 4: NaN frames don't crash or drop reps ───────────────────────
print()
print("=" * 50)
print("TEST 4 — NaN frames handled correctly")
print("=" * 50)
# TEST 4 — replace the NaN injection positions
signal_nan = make_elbow_signal(n_reps=3).tolist()
# inject NaN clearly mid-rep, away from transitions
# signal is 330 frames, reps start at frame 15 (0.5s rest)
# each rep is 60 frames — inject at frame centres
for i in [45, 46, 47, 105, 165]:   # mid-rep positions
    signal_nan[i] = float("nan")
fsm_nan = ElbowFSM()
run_fsm(fsm_nan, signal_nan)
print(f"Expected reps : 3")
print(f"Detected reps : {fsm_nan.rep_count}")
print("PASS ✅" if fsm_nan.rep_count == 3 else f"FAIL ❌  got {fsm_nan.rep_count}")


# ── Test 5: Rep count matches actual peaks in signal ─────────────────
print()
print("=" * 50)
print("TEST 5 — ROM values are physically correct")
print("=" * 50)
signal_check = make_elbow_signal(n_reps=3)
fsm_check = ElbowFSM()
reps_check = run_fsm(fsm_check, signal_check)
all_rom_ok = all(80 < r["rom"] < 140 for r in reps_check)
print(f"Expected ROM per rep: between 80° and 140°")
for r in reps_check:
    print(f"  Rep {r['rep_number']}: ROM={r['rom']:.0f}°  "
          f"{'✅' if 80 < r['rom'] < 140 else '❌'}")
print("PASS ✅" if all_rom_ok else "FAIL ❌")


print()
print("All tests complete.")