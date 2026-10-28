# tests/test_fsm.py
"""
Unit tests for ElbowFSM and ShoulderFSM rep counting, calibration, and ROM checks.
"""
import numpy as np
from exercise_state import ElbowFSM, ShoulderFSM
from assessment.rules import check_rom
import assessment.scorer as scorer


def make_elbow_signal(n_reps=5, fps=30, rep_duration_s=2.0):
    frames_per_rep = int(rep_duration_s * fps)
    rest_frames = int(0.8 * fps)
    signal = [162.0] * rest_frames
    for _ in range(n_reps):
        down = np.linspace(162, 40, frames_per_rep // 2)
        up = np.linspace(40, 162, frames_per_rep // 2)
        signal.extend(down.tolist())
        signal.extend(up.tolist())
        signal.extend([162.0] * rest_frames)
    return np.array(signal)


def make_shoulder_signal(n_reps=4, fps=30, rep_duration_s=2.0):
    frames_per_rep = int(rep_duration_s * fps)
    rest_frames = int(0.8 * fps)
    signal = [8.0] * rest_frames
    for _ in range(n_reps):
        up = np.linspace(8, 90, frames_per_rep // 2)
        down = np.linspace(90, 8, frames_per_rep // 2)
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


def test_elbow_fsm_correct_count():
    n_reps = 5
    sig = make_elbow_signal(n_reps=n_reps)
    fsm = ElbowFSM()
    reps = run_fsm(fsm, sig)
    assert fsm.rep_count == n_reps
    assert len(reps) == n_reps


def test_partial_rep_not_counted():
    partial = [162] * 20 + list(np.linspace(162, 124, 20)) + list(np.linspace(124, 162, 20)) + [162] * 20
    fsm = ElbowFSM()
    run_fsm(fsm, partial)
    assert fsm.rep_count == 0


def test_shoulder_fsm_correct_count():
    n_reps = 4
    sig = make_shoulder_signal(n_reps=n_reps)
    fsm = ShoulderFSM()
    reps = run_fsm(fsm, sig)
    assert fsm.rep_count == n_reps
    assert len(reps) == n_reps


def test_nan_frames_handled_correctly():
    sig_nan = make_elbow_signal(n_reps=3).tolist()
    for i in [52, 53, 54, 136, 220]:
        sig_nan[i] = float("nan")
    fsm = ElbowFSM()
    run_fsm(fsm, sig_nan)
    assert fsm.rep_count == 3


def test_rom_values_physically_correct():
    sig = make_elbow_signal(n_reps=3)
    fsm = ElbowFSM()
    reps = run_fsm(fsm, sig)
    assert all(80 < r["rom"] < 145 for r in reps)


def test_calibrate_from_signal_sets_correct_thresholds():
    sig_shift = make_elbow_signal(n_reps=3) + 20
    fsm = ElbowFSM()
    fsm.calibrate_from_signal(sig_shift)
    reps = run_fsm(fsm, sig_shift)
    assert fsm.rep_count == 3
    assert fsm.FLEX_THRESH > 55.0


def test_two_dips_without_full_extension():
    two_dips = (
        [162] * 20 +
        list(np.linspace(162, 40, 30)) +
        list(np.linspace(40, 130, 20)) +
        list(np.linspace(130, 40, 20)) +
        list(np.linspace(40, 162, 30)) +
        [162] * 20
    )
    fsm = ElbowFSM()
    reps = run_fsm(fsm, two_dips)
    assert fsm.rep_count == 2
    assert len(reps) == 2


def test_flexing_starts_within_two_frames():
    signal = [162.0, 162.0, 158.0, 154.0]
    fsm = ElbowFSM()
    states = []
    for i, a in enumerate(signal):
        state, _ = fsm.update(float(a), i / 30.0)
        states.append(state)
    flexing_frame = next((i for i, s in enumerate(states) if s == "FLEXING"), None)
    assert flexing_frame is not None
    assert flexing_frame <= 2


def test_rom_accuracy_within_tolerance():
    signal = (
        [162.0] * 20 +
        list(np.linspace(162, 40, 30)) +
        [40.0] * 5 +
        list(np.linspace(40, 162, 30)) +
        [162.0] * 20
    )
    fsm = ElbowFSM()
    reps = run_fsm(fsm, signal)
    assert len(reps) >= 1
    true_rom = 162.0 - 40.0
    measured = reps[0]["rom"]
    assert abs(measured - true_rom) <= 5.0


def test_shoulder_fsm_calibration_and_scoring_rom_check():
    """Test ShoulderFSM calibrate, rom_check_params, and scorer execution without crash."""
    fsm = ShoulderFSM()
    params = fsm.rom_check_params()
    assert params["kind"] == "raise"
    assert params["target"] == fsm.RAISE_THRESH  # default 75.0

    # 1. Rep with low max_angle (e.g. raised only to 50 deg when target is 75 deg; 50 < 75 - 10 = 65)
    low_rep = {
        "rep_number": 1,
        "max_angle": 50.0,
        "min_angle": 10.0,
        "rom": 40.0,
        "duration_s": 2.0,
        "_angle_series": [10.0, 30.0, 50.0, 30.0, 10.0],
        "_landmarks": None
    }
    fault = check_rom(low_rep, params)
    assert fault == "Raise your arm a little higher"

    # Score rep should not crash and should record the fault
    scorer.score_rep(low_rep, rom_params=params)
    assert "Raise your arm a little higher" in low_rep["faults"]

    # 2. Rep with good max_angle (raised to 88 deg)
    good_rep = {
        "rep_number": 2,
        "max_angle": 88.0,
        "min_angle": 10.0,
        "rom": 78.0,
        "duration_s": 2.0,
        "_angle_series": [10.0, 50.0, 88.0, 50.0, 10.0],
        "_landmarks": None
    }
    good_fault = check_rom(good_rep, params)
    assert good_fault == "ok"
    scorer.score_rep(good_rep, rom_params=params)
    assert not any("raise" in f.lower() for f in good_rep["faults"])


def test_extending_to_flexing_reversal_flags_rep_completed():
    fsm = ElbowFSM()
    signal = [162.0] * 5 + list(np.linspace(162, 40, 20)) + [60.0, 90.0, 130.0, 120.0]
    completed = []
    for i, a in enumerate(signal):
        state, rep_done = fsm.update(float(a), i / 30.0)
        if rep_done:
            completed.append((i, state))
    assert len(completed) == 1
    assert completed[0][1] == "FLEXING"
    assert fsm.rep_count == 1

