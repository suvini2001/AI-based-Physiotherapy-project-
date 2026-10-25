import numpy as np
from exercise_state import ElbowFSM, ShoulderFSM


def make_elbow_signal(n_reps=5, fps=30, rep_duration_s=2.0):
    frames_per_rep = int(rep_duration_s * fps)
    rest_frames    = int(0.8 * fps)
    signal = [162.0] * rest_frames
    for _ in range(n_reps):
        down = np.linspace(162, 40, frames_per_rep // 2)
        up   = np.linspace(40, 162, frames_per_rep // 2)
        signal.extend(down.tolist())
        signal.extend(up.tolist())
        signal.extend([162.0] * rest_frames)
    return np.array(signal)


def make_shoulder_signal(n_reps=4, fps=30, rep_duration_s=2.0):
    frames_per_rep = int(rep_duration_s * fps)
    rest_frames    = int(0.8 * fps)
    signal = [8.0] * rest_frames
    for _ in range(n_reps):
        up   = np.linspace(8,  90, frames_per_rep // 2)
        down = np.linspace(90,  8, frames_per_rep // 2)
        signal.extend(up.tolist())
        signal.extend(down.tolist())
        signal.extend([8.0] * rest_frames)
    return np.array(signal)


def run_fsm(fsm, signal, fps=30):
    captured = []
    fsm.rep_callback = lambda r: captured.append(r)
    for i, a in enumerate(signal):
        fsm.update(float(a), i / fps)
    fsm.flush(len(signal) / fps)
    return captured


# TEST 1
print("=" * 50)
print("TEST 1 — ElbowFSM correct count")
print("=" * 50)
N   = 5
sig = make_elbow_signal(n_reps=N)
fsm = ElbowFSM()
reps = run_fsm(fsm, sig)
print(f"Signal: {sig.min():.0f}° – {sig.max():.0f}°  ({len(sig)} frames)")
print(f"Expected: {N}   Detected: {fsm.rep_count}")
print("PASS ✅" if fsm.rep_count == N else f"FAIL ❌  got {fsm.rep_count}")
for r in reps:
    print(f"  Rep {r['rep_number']}: ROM={r['rom']:.0f}°  "
          f"max={r['max_angle']:.0f}°  min={r['min_angle']:.0f}°  "
          f"dur={r['duration_s']:.1f}s")


# TEST 2
print()
print("=" * 50)
print("TEST 2 — Partial rep not counted")
print("=" * 50)
partial = [162]*20 + list(np.linspace(162,124,20)) + list(np.linspace(124,162,20)) + [162]*20
fsm2    = ElbowFSM()
run_fsm(fsm2, partial)
print(f"Expected: 0   Detected: {fsm2.rep_count}")
print("PASS ✅" if fsm2.rep_count == 0 else f"FAIL ❌  got {fsm2.rep_count}")


# TEST 3
print()
print("=" * 50)
print("TEST 3 — ShoulderFSM correct count")
print("=" * 50)
N_S   = 4
sig_s = make_shoulder_signal(n_reps=N_S)
fsm_s = ShoulderFSM()
reps_s = run_fsm(fsm_s, sig_s)
print(f"Signal: {sig_s.min():.0f}° – {sig_s.max():.0f}°")
print(f"Expected: {N_S}   Detected: {fsm_s.rep_count}")
print("PASS ✅" if fsm_s.rep_count == N_S else f"FAIL ❌  got {fsm_s.rep_count}")
for r in reps_s:
    print(f"  Rep {r['rep_number']}: ROM={r['rom']:.0f}°  "
          f"max={r['max_angle']:.0f}°  dur={r['duration_s']:.1f}s")


# TEST 4
print()
print("=" * 50)
print("TEST 4 — NaN frames handled correctly")
print("=" * 50)
sig_nan = make_elbow_signal(n_reps=3).tolist()
# rest=24f, rep=60f, rest=24f pattern
# rep1 mid ≈ frame 54, rep2 mid ≈ frame 138, rep3 mid ≈ frame 222
for i in [52, 53, 54, 136, 220]:
    sig_nan[i] = float("nan")
fsm_nan = ElbowFSM()
run_fsm(fsm_nan, sig_nan)
print(f"Expected: 3   Detected: {fsm_nan.rep_count}")
print("PASS ✅" if fsm_nan.rep_count == 3 else f"FAIL ❌  got {fsm_nan.rep_count}")


# TEST 5
print()
print("=" * 50)
print("TEST 5 — ROM values physically correct")
print("=" * 50)
sig_c  = make_elbow_signal(n_reps=3)
fsm_c  = ElbowFSM()
reps_c = run_fsm(fsm_c, sig_c)
ok = all(80 < r["rom"] < 145 for r in reps_c)
for r in reps_c:
    flag = "✅" if 80 < r["rom"] < 145 else "❌"
    print(f"  Rep {r['rep_number']}: ROM={r['rom']:.0f}°  {flag}")
print("PASS ✅" if ok else "FAIL ❌")


# TEST 6
print()
print("=" * 50)
print("TEST 6 — calibrate_from_signal sets correct thresholds")
print("=" * 50)
sig_shift = make_elbow_signal(n_reps=3) + 20  # min=60, max=182
fsm_cal   = ElbowFSM()
print(f"Before: FLEX={fsm_cal.FLEX_THRESH}°  EXTEND={fsm_cal.EXTEND_THRESH}°")
fsm_cal.calibrate_from_signal(sig_shift)
print(f"After:  FLEX={fsm_cal.FLEX_THRESH:.0f}°  EXTEND={fsm_cal.EXTEND_THRESH:.0f}°")
reps_cal = run_fsm(fsm_cal, sig_shift)
print(f"Expected: 3   Detected: {fsm_cal.rep_count}")
print("PASS ✅" if fsm_cal.rep_count == 3 else f"FAIL ❌  got {fsm_cal.rep_count}")

print()
print("All tests complete.")