# exercise_state.py
"""
4-phase finite state machine for rehabilitation exercise rep counting.
Works frame-by-frame with the angle output from demo_cv.py AnglePipeline.

Person A owns this file.
The only output Person B needs is the rep_callback(rep_dict) call.
"""

import json
import time
import numpy as np


# Fewest angle samples the CALIB phase needs. Was 30, which a 5 s window
# cannot reach on a slow camera (about 6 FPS gives ~31 frames, minus dropouts).
MIN_CALIB_SAMPLES = 15

# A rep must come close to BOTH ends of the patient's own range (fully bent and
# fully straight) to count. "Close" is 15% of that range, and never less than
# RANGE_TOLERANCE degrees (the same value as rules.ROM_TOLERANCE).
RANGE_TOLERANCE = 10.0
RANGE_FRACTION  = 0.15


def _rep_extremes(fsm):
    """
    Lowest and highest angle of the current movement, including where the arm
    was just BEFORE the movement was detected. Detection is always one frame
    late, and on a slow camera one frame can be 10-20 degrees of movement.
    """
    angles = np.array(fsm._rep_angles)
    lo, hi = float(np.nanmin(angles)), float(np.nanmax(angles))
    start = fsm._start_extreme
    if start is not None and np.isfinite(start):
        lo, hi = min(lo, start), max(hi, start)
    return lo, hi


def calibration_cycle_complete(angles, min_excursion=30.0):
    """
    True once the CALIB samples contain one full out-and-back movement:
    the arm moved at least `min_excursion` degrees away from where it started
    and has come back most of the way. demo_cv.py ends the CALIB phase on this,
    so calibration can no longer stop half-way through the first bend.
    """
    a = np.asarray(angles, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < MIN_CALIB_SAMPLES:
        return False
    start = float(np.median(a[:3]))
    dev = np.abs(a - start)
    peak = float(dev.max())
    return peak >= min_excursion and float(dev[-1]) <= 0.25 * peak


def _frame_landmarks(landmarks):
    """
    Ensure landmarks are converted to a copy of float array of shape (33, 4).
    If landmarks is None, return a (33, 4) array filled with NaN.
    """
    if landmarks is None:
        return np.full((33, 4), np.nan)
    return np.array(landmarks, dtype=float, copy=True)


def _attempt_summary(fsm, timestamp_s):
    """
    Describe a movement that turned back before reaching the threshold.
    Returns None for noise (too small or too short to be a real attempt).
    """
    if not fsm._rep_angles or fsm._rep_start_time is None:
        return None
    lo, hi = _rep_extremes(fsm)
    rom = hi - lo
    duration = timestamp_s - fsm._rep_start_time
    if rom < fsm.MIN_ROM or duration < getattr(fsm, '_min_rep_s', 0.5):
        return None
    return {
        "exercise":   fsm.EXERCISE,
        "reason":     "raise" if "shoulder" in fsm.EXERCISE else "flex",
        "max_angle":  round(hi, 1),
        "min_angle":  round(lo, 1),
        "rom":        round(rom, 1),
        "duration_s": round(duration, 2),
        "timestamp":  round(timestamp_s, 2),
    }


def _returned_without_reaching(fsm, angle_deg):
    """
    True when the arm has come at least 75% of the way back from the furthest
    point of the current movement. Catches slow returns that never exceed
    MIN_DELTA per frame, which would otherwise leave the FSM stuck mid-rep.
    """
    angles = np.array(fsm._rep_angles)
    lo, hi = float(np.nanmin(angles)), float(np.nanmax(angles))
    rom = hi - lo
    if rom < fsm.MIN_ROM:
        return False
    start = float(angles[0])
    back = (angle_deg - lo) if start - lo >= hi - start else (hi - angle_deg)
    return back >= 0.75 * rom


def _emit_attempt(fsm):
    """Report the pending incomplete attempt once the arm is back at the start."""
    attempt, fsm._pending_attempt = fsm._pending_attempt, None
    fsm.incomplete_attempts.append(attempt)
    if fsm.attempt_callback:
        fsm.attempt_callback(attempt)


# ── Elbow Flexion FSM ────────────────────────────────────────────────
class ElbowFSM:
    """
    States:  EXTENDED → FLEXING → FLEXED → EXTENDING → EXTENDED
    
    Thresholds have ±10 deg tolerance because our system MAE is ~10 deg.
    Based on:
      - van den Hoorn et al. JSES 2025: MCID for shoulder 3-12 deg
        (our elbow threshold of 145 deg is well above any MCID)
      - Aguilar-Ortega et al. Sensors 2023: ROM>=60 deg for valid flexion
    """

    EXERCISE      = "elbow_flexion"
    EXTEND_THRESH = 145.0   # angle above this = arm extended
    FLEX_THRESH   =  55.0   # angle below this = arm fully flexed
    MIN_DELTA     =   2.0   # deg/frame minimum to confirm movement direction
    ROM_TARGET    = None    # deepest flexion the patient has shown; None = not known yet
    START_TARGET  = None    # fullest extension the patient has shown

    def __init__(self, rep_callback=None, attempt_callback=None):
        """
        rep_callback: function(rep_dict) called when a rep completes.
                      rep_dict has keys:
                        rep_number, max_angle, min_angle, rom,
                        duration_s, timestamp
                      Person B wires their assess_rep() function here.
        """
        self.state        = "EXTENDED"
        self.prev_angle   = None
        self.rep_count    = 0
        self.rep_callback = rep_callback

        self._rep_angles     = []
        self._rep_landmarks  = []
        self._rep_start_time = None
        self.reps            = []        # full history, for session_log.json
        self._min_rep_s      = 0.5

        # movements that turned back before FLEX_THRESH (not counted as reps)
        self.attempt_callback    = attempt_callback
        self.incomplete_attempts = []
        self._pending_attempt    = None

        # where the arm was before the current movement started (see _rep_extremes)
        self._idle_extreme  = None
        self._start_extreme = None

    def calibrate_from_signal(self, angle_signal, fps=30.0):
        """
        Tune thresholds from an initial calibration window.

        Signals with fewer than MIN_CALIB_SAMPLES finite samples or less than
        30 degrees of robust range keep the fixed defaults.
        Returns True if the thresholds were updated, False if defaults were kept.
        """
        finite = np.asarray(angle_signal, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < MIN_CALIB_SAMPLES:
            return False

        sig_min = float(np.percentile(finite, 5))
        sig_max = float(np.percentile(finite, 95))
        sig_range = sig_max - sig_min
        if sig_range < 30:
            return False

        self._set_range(sig_min, sig_max)
        self._min_rep_s = 0.5
        print(f"FSM calibrated: FLEX={self.FLEX_THRESH:.0f}°  "
              f"EXTEND={self.EXTEND_THRESH:.0f}°  "
              f"range={sig_range:.0f}°")
        return True

    def _set_range(self, lo, hi):
        """Derive every threshold from the patient's own range (lo = fully bent, hi = straight)."""
        rng = hi - lo
        self.FLEX_THRESH   = lo + rng * 0.30
        self.EXTEND_THRESH = hi - rng * 0.25
        self.MIN_ROM       = rng * 0.20
        self.ROM_TARGET    = lo
        self.START_TARGET  = hi

    def _learn_from(self, lo, hi):
        """
        Widen the known range when a rep goes further than anything seen so far.
        This repairs a calibration that only caught part of a movement: the
        goal becomes the patient's best rep of the session.
        """
        if self.ROM_TARGET is None:      # not calibrated: keep the fixed defaults
            return
        # only a clearly larger movement moves the goal; small differences are noise
        tol = self.range_tolerance()
        new_lo = lo if lo < self.ROM_TARGET - tol else self.ROM_TARGET
        new_hi = hi if hi > self.START_TARGET + tol else self.START_TARGET
        if (new_lo, new_hi) != (self.ROM_TARGET, self.START_TARGET):
            self._set_range(new_lo, new_hi)

    def range_tolerance(self):
        """How far short of either end of the range a rep may stop and still count."""
        if self.ROM_TARGET is None:
            return RANGE_TOLERANCE
        return max(RANGE_TOLERANCE, RANGE_FRACTION * (self.START_TARGET - self.ROM_TARGET))

    def _range_fault(self, lo, hi):
        """None if the rep covered the full range, else which end it missed."""
        if self.ROM_TARGET is None:
            return None
        tol = self.range_tolerance()
        if lo > self.ROM_TARGET + tol:
            return "flex"       # did not bend far enough
        if hi < self.START_TARGET - tol:
            return "extend"     # did not straighten before bending again
        return None

    def update(self, angle_deg, timestamp_s, landmarks=None):
        """
        Feed one frame. Returns (current_state, rep_just_completed).
        angle_deg:   float or None (NaN frame → skip)
        timestamp_s: float seconds since session start
        landmarks:   (33, 4) landmark array or None
        """
        if angle_deg is None or not np.isfinite(angle_deg):
            return self.state, False

        frame_lm = _frame_landmarks(landmarks)
        direction = (angle_deg - self.prev_angle
                     if self.prev_angle is not None else 0.0)
        self._rep_angles.append(angle_deg)
        self._rep_landmarks.append(frame_lm)
        rep_completed = False

        # straightest the arm has been since it last came up from a bend
        if self.state in ("EXTENDED", "EXTENDING"):
            seen = [a for a in (self._idle_extreme, self.prev_angle, angle_deg)
                    if a is not None]
            self._idle_extreme = max(seen)

        if self.state == "EXTENDED":
            # a shallow attempt is reported once the arm is most of the way back
            p = self._pending_attempt
            if p and angle_deg >= p["max_angle"] - 0.25 * p["rom"]:
                _emit_attempt(self)
            if direction < -self.MIN_DELTA:
                self.state = "FLEXING"
                self._rep_angles     = [angle_deg]
                self._rep_landmarks  = [frame_lm]
                self._rep_start_time = timestamp_s
                self._start_extreme, self._idle_extreme = self._idle_extreme, None

        elif self.state == "FLEXING":
            if angle_deg < self.FLEX_THRESH:
                self.state = "FLEXED"
                self._pending_attempt = None   # the bend carried on: not shallow
            elif _returned_without_reaching(self, angle_deg):
                # drifted back to the start without a full flex — shallow attempt
                self._pending_attempt = (_attempt_summary(self, timestamp_s)
                                         or self._pending_attempt)
                self.state           = "EXTENDED"
                self._rep_angles     = []
                self._rep_landmarks  = []
                self._rep_start_time = None
                if self._pending_attempt:
                    _emit_attempt(self)
            elif direction > self.MIN_DELTA:
                # reversed before full flex — partial rep, reset
                self._pending_attempt = (_attempt_summary(self, timestamp_s)
                                         or self._pending_attempt)
                self.state           = "EXTENDED"
                self._rep_angles     = []
                self._rep_landmarks  = []
                self._rep_start_time = None

        elif self.state == "FLEXED":
            if direction > self.MIN_DELTA:
                self.state = "EXTENDING"

        elif self.state == "EXTENDING":
            if angle_deg > self.EXTEND_THRESH:
                self.state    = "EXTENDED"
                rep_completed = True
                self._finish_rep(timestamp_s)
            elif direction < -self.MIN_DELTA:
                before = self.rep_count
                self._finish_rep(timestamp_s)
                if self.rep_count > before:
                    rep_completed = True
                self.state           = "FLEXING"
                self._rep_angles     = [angle_deg]
                self._rep_landmarks  = [frame_lm]
                self._rep_start_time = timestamp_s
                self._start_extreme, self._idle_extreme = self._idle_extreme, None

        self.prev_angle = angle_deg
        return self.state, rep_completed

    MIN_ROM = 30.0   # add this as a class constant at the top of each class

    def _finish_rep(self, timestamp_s):
        if not self._rep_angles:
            return
        angles = np.array(self._rep_angles)
        rom = float(np.nanmax(angles) - np.nanmin(angles))

        # ignore spurious micro-reps (noise or signal artefacts)
        duration = (timestamp_s - self._rep_start_time
            if self._rep_start_time else 0.0)
        min_rep_s = getattr(self, '_min_rep_s', 0.5)

        if rom < self.MIN_ROM or duration < min_rep_s:
            self._rep_angles     = []
            self._rep_landmarks  = []
            self._rep_start_time = None
            return

        # a rep only counts if it covers the patient's full range at both ends
        lo, hi = _rep_extremes(self)
        rom = hi - lo
        self._learn_from(lo, hi)
        # "straight" is judged where the rep STARTED: that is the moment the
        # patient chose to bend again, and it is the only top fully recorded
        top = self._start_extreme if self._start_extreme is not None else hi
        reason = self._range_fault(lo, top)
        if reason:
            if reason == "extend":
                hi, rom = top, top - lo
            self._pending_attempt = {
                "exercise":   self.EXERCISE,
                "reason":     reason,
                "max_angle":  round(hi, 1),
                "min_angle":  round(lo, 1),
                "rom":        round(rom, 1),
                "duration_s": round(duration, 2),
                "timestamp":  round(timestamp_s, 2),
            }
            self._rep_angles     = []
            self._rep_landmarks  = []
            self._rep_start_time = None
            _emit_attempt(self)
            return

        rep = {
            "exercise":      self.EXERCISE,
            "rep_number":    self.rep_count + 1,
            "max_angle":     round(hi, 1),
            "min_angle":     round(lo, 1),
            "rom":           round(rom, 1),
            "duration_s":    round(duration, 2),   # use already-computed duration
            "timestamp":     round(timestamp_s, 2),
            "quality_score": None,
            "faults":        [],
            "_angle_series": [round(float(a), 1) for a in self._rep_angles],
            "_landmarks":    np.stack(self._rep_landmarks),   # shape (frames, 33, 4)
        }
        self.rep_count += 1
        self.reps.append(rep)
        self._rep_angles     = []
        self._rep_landmarks  = []
        self._rep_start_time = None

        if self.rep_callback:
            self.rep_callback(rep)

    def flush(self, timestamp_s):
        """
        Call at end of session or clip to complete any in-progress rep.
        A rep that reached EXTENDING counts even if the arm
        did not quite return to full extension (e.g. clip ended early).
        """
        if self._pending_attempt:
            _emit_attempt(self)
        if self.state == "EXTENDING" and self._rep_angles:
            self._finish_rep(timestamp_s)
            self.state = "EXTENDED"

    def get_last_rep(self):
        return self.reps[-1] if self.reps else None

    def rom_check_params(self):
        if self.ROM_TARGET is None:
            target = self.FLEX_THRESH
        else:
            # rules.py adds its own fixed 10 degrees, so shift the target to make
            # its limit equal the one used here (ROM_TARGET + range_tolerance)
            target = self.ROM_TARGET + self.range_tolerance() - RANGE_TOLERANCE
        return {"kind": "flex", "target": target, "exercise": self.EXERCISE}

    def summary(self):
        if not self.reps:
            return {}
        roms = [r["rom"] for r in self.reps]
        durs = [r["duration_s"] for r in self.reps]
        return {
            "total_reps": self.rep_count,
            "mean_rom":   round(float(np.mean(roms)), 1),
            "min_rom":    round(float(np.min(roms)), 1),
            "max_rom":    round(float(np.max(roms)), 1),
            "mean_dur":   round(float(np.mean(durs)), 2),
        }


# ── Shoulder Abduction FSM ───────────────────────────────────────────
class ShoulderFSM:
    """
    States:  RESTING → RAISING → RAISED → LOWERING → RESTING
    
    Joints:  hip(23) - shoulder(11) - elbow(13)  for LEFT side
             hip(24) - shoulder(12) - elbow(14)  for RIGHT side
    
    Threshold justification (van den Hoorn et al. JSES Int. 2025):
      - MCID for shoulder abduction: 3-12 deg
      - RAISE_THRESH=75 deg is well above any MCID — clinically unambiguous
      - Smallest detectable difference: 4.4-7.9 deg
      - Our ±10 deg tolerance is within the SDD range
    """

    EXERCISE     = "shoulder_abduction"
    RAISE_THRESH = 75.0   # arm raised above this = abduction rep in progress
    REST_THRESH  = 30.0   # arm back below this   = rep complete
    MIN_DELTA    =  2.0
    ROM_TARGET   = None   # highest raise the patient has shown; None = not known yet
    START_TARGET = None   # lowest resting angle the patient has shown

    def __init__(self, rep_callback=None, attempt_callback=None):
        self.state        = "RESTING"
        self.prev_angle   = None
        self.rep_count    = 0
        self.rep_callback = rep_callback
        self._min_rep_s = 0.5

        self._rep_angles     = []
        self._rep_landmarks  = []
        self._rep_start_time = None
        self.reps            = []

        # movements that turned back before RAISE_THRESH (not counted as reps)
        self.attempt_callback    = attempt_callback
        self.incomplete_attempts = []
        self._pending_attempt    = None

        # where the arm was before the current movement started (see _rep_extremes)
        self._idle_extreme  = None
        self._start_extreme = None

    def calibrate_from_signal(self, angle_signal, fps=30.0):
        """Auto-tune thresholds from observed signal range."""
        finite = np.asarray(angle_signal, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < MIN_CALIB_SAMPLES:
            return False
        sig_min   = float(np.percentile(finite, 5))
        sig_max   = float(np.percentile(finite, 95))
        sig_range = sig_max - sig_min
        if sig_range < 30:
            return False
        self._set_range(sig_min, sig_max)
        self._min_rep_s   = 0.5
        print(f"ShoulderFSM calibrated: RAISE={self.RAISE_THRESH:.0f}°  "
              f"REST={self.REST_THRESH:.0f}°  range={sig_range:.0f}°")
        return True

    def _set_range(self, lo, hi):
        """Derive every threshold from the patient's own range (lo = resting, hi = fully raised)."""
        rng = hi - lo
        self.RAISE_THRESH = lo + rng * 0.30
        self.REST_THRESH  = lo + rng * 0.05
        self.MIN_ROM      = rng * 0.20
        self.ROM_TARGET   = hi
        self.START_TARGET = lo

    def _learn_from(self, lo, hi):
        """Widen the known range when a rep goes further than anything seen so far."""
        if self.ROM_TARGET is None:      # not calibrated: keep the fixed defaults
            return
        # only a clearly larger movement moves the goal; small differences are noise
        tol = self.range_tolerance()
        new_lo = lo if lo < self.START_TARGET - tol else self.START_TARGET
        new_hi = hi if hi > self.ROM_TARGET + tol else self.ROM_TARGET
        if (new_lo, new_hi) != (self.START_TARGET, self.ROM_TARGET):
            self._set_range(new_lo, new_hi)

    def range_tolerance(self):
        """How far short of either end of the range a rep may stop and still count."""
        if self.ROM_TARGET is None:
            return RANGE_TOLERANCE
        return max(RANGE_TOLERANCE, RANGE_FRACTION * (self.ROM_TARGET - self.START_TARGET))

    def _range_fault(self, lo, hi):
        """None if the rep covered the full range, else which end it missed."""
        if self.ROM_TARGET is None:
            return None
        tol = self.range_tolerance()
        if hi < self.ROM_TARGET - tol:
            return "raise"      # did not raise high enough
        if lo > self.START_TARGET + tol:
            return "lower"      # did not start from a fully lowered arm
        return None
        
    def update(self, angle_deg, timestamp_s, landmarks=None):
        if angle_deg is None or not np.isfinite(angle_deg):
            return self.state, False

        frame_lm = _frame_landmarks(landmarks)
        direction = (angle_deg - self.prev_angle
                     if self.prev_angle is not None else 0.0)
        if self.state != "RESTING":
            self._rep_angles.append(angle_deg)
            self._rep_landmarks.append(frame_lm)
        rep_completed = False

        # lowest the arm has been since it last came down from a raise
        if self.state in ("RESTING", "LOWERING"):
            seen = [a for a in (self._idle_extreme, self.prev_angle, angle_deg)
                    if a is not None]
            self._idle_extreme = min(seen)

        if self.state == "RESTING":
            # a shallow attempt is reported once the arm is most of the way back
            p = self._pending_attempt
            if p and angle_deg <= p["min_angle"] + 0.25 * p["rom"]:
                _emit_attempt(self)
            if direction > self.MIN_DELTA:
                self.state           = "RAISING"
                self._rep_angles     = [angle_deg]
                self._rep_landmarks  = [frame_lm]
                self._rep_start_time = timestamp_s
                self._start_extreme, self._idle_extreme = self._idle_extreme, None

        elif self.state == "RAISING":
            if angle_deg > self.RAISE_THRESH:
                self.state = "RAISED"
                self._pending_attempt = None   # the raise carried on: not shallow
            elif _returned_without_reaching(self, angle_deg):
                # drifted back to rest without a full raise — shallow attempt
                self._pending_attempt = (_attempt_summary(self, timestamp_s)
                                         or self._pending_attempt)
                self.state           = "RESTING"
                self._rep_angles     = []
                self._rep_landmarks  = []
                self._rep_start_time = None
                if self._pending_attempt:
                    _emit_attempt(self)
            elif direction < -self.MIN_DELTA:
                self._pending_attempt = (_attempt_summary(self, timestamp_s)
                                         or self._pending_attempt)
                self.state           = "RESTING"
                self._rep_angles     = []
                self._rep_landmarks  = []
                self._rep_start_time = None

        elif self.state == "RAISED":
            if direction < -self.MIN_DELTA:
                self.state = "LOWERING"

        elif self.state == "LOWERING":
            if angle_deg < self.REST_THRESH:
                self.state    = "RESTING"
                rep_completed = True
                self._finish_rep(timestamp_s)

        self.prev_angle = angle_deg
        return self.state, rep_completed

    MIN_ROM = 20.0   # add this as a class constant at the top of each class

    def _finish_rep(self, timestamp_s):
        if not self._rep_angles:
            return
        angles   = np.array(self._rep_angles)
        rom      = float(np.nanmax(angles) - np.nanmin(angles))
        duration = (timestamp_s - self._rep_start_time
                    if self._rep_start_time else 0.0)
        min_rep_s = getattr(self, '_min_rep_s', 0.5)

        if rom < self.MIN_ROM or duration < min_rep_s:
            self._rep_angles     = []
            self._rep_landmarks  = []
            self._rep_start_time = None
            return

        # a rep only counts if it covers the patient's full range at both ends
        lo, hi = _rep_extremes(self)
        rom = hi - lo
        self._learn_from(lo, hi)
        # "lowered" is judged where the rep STARTED (see ElbowFSM._finish_rep)
        bottom = self._start_extreme if self._start_extreme is not None else lo
        reason = self._range_fault(bottom, hi)
        if reason:
            if reason == "lower":
                lo, rom = bottom, hi - bottom
            self._pending_attempt = {
                "exercise":   self.EXERCISE,
                "reason":     reason,
                "max_angle":  round(hi, 1),
                "min_angle":  round(lo, 1),
                "rom":        round(rom, 1),
                "duration_s": round(duration, 2),
                "timestamp":  round(timestamp_s, 2),
            }
            self._rep_angles     = []
            self._rep_landmarks  = []
            self._rep_start_time = None
            _emit_attempt(self)
            return

        rep = {
            "exercise":      self.EXERCISE,
            "rep_number":    self.rep_count + 1,
            "max_angle":     round(hi, 1),
            "min_angle":     round(lo, 1),
            "rom":           round(rom, 1),
            "duration_s":    round(duration, 2),
            "timestamp":     round(timestamp_s, 2),
            "quality_score": None,
            "faults":        [],
            "_angle_series": [round(float(a), 1) for a in self._rep_angles],
            "_landmarks":    np.stack(self._rep_landmarks),   # shape (frames, 33, 4)
        }
        self.rep_count += 1
        self.reps.append(rep)
        self._rep_angles     = []
        self._rep_landmarks  = []
        self._rep_start_time = None
        if self.rep_callback:
            self.rep_callback(rep)

    def flush(self, timestamp_s):
        """
        Call at end of session or clip to complete any in-progress rep.
        A rep that reached LOWERING counts even if the arm
        did not quite return to the resting position (e.g. clip ended early).
        """
        if self._pending_attempt:
            _emit_attempt(self)
        if self.state == "LOWERING" and self._rep_angles:
            self._finish_rep(timestamp_s)
            self.state = "RESTING"

    def get_last_rep(self):
        return self.reps[-1] if self.reps else None

    def rom_check_params(self):
        if self.ROM_TARGET is None:
            target = self.RAISE_THRESH
        else:
            # rules.py subtracts its own fixed 10 degrees (see ElbowFSM.rom_check_params)
            target = self.ROM_TARGET - self.range_tolerance() + RANGE_TOLERANCE
        return {"kind": "raise", "target": target, "exercise": self.EXERCISE}

    def summary(self):
        if not self.reps:
            return {}
        roms = [r["rom"] for r in self.reps]
        durs = [r["duration_s"] for r in self.reps]
        return {
            "total_reps": self.rep_count,
            "mean_rom":   round(float(np.mean(roms)), 1),
            "min_rom":    round(float(np.min(roms)), 1),
            "max_rom":    round(float(np.max(roms)), 1),
            "mean_dur":   round(float(np.mean(durs)), 2),
        }


# ── Session save ─────────────────────────────────────────────────────
def save_session(fsm, exercise_name, path="session_log.json"):
    """Write the full session to JSON. Call at end of demo_cv.py."""
    import datetime

    def clean_rep(r):
        # keep _angle_series for DTW calibration
        # strip _landmarks (large numpy array, not JSON serialisable)
        return {k: v for k, v in r.items() if k != '_landmarks'}
    
    session = {
        "date":       datetime.datetime.now().isoformat(),
        "exercise":   exercise_name,
        "total_reps": fsm.rep_count,
        "reps":       [clean_rep(r) for r in fsm.reps],
        "incomplete_attempts": list(getattr(fsm, "incomplete_attempts", [])),
        "summary":    fsm.summary(),
    }
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    print(f"Session saved → {path}  ({fsm.rep_count} reps)")
    return session