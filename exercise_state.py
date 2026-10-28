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


def _frame_landmarks(landmarks):
    """
    Ensure landmarks are converted to a copy of float array of shape (33, 4).
    If landmarks is None, return a (33, 4) array filled with NaN.
    """
    if landmarks is None:
        return np.full((33, 4), np.nan)
    return np.array(landmarks, dtype=float, copy=True)


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

    def __init__(self, rep_callback=None):
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

    def calibrate_from_signal(self, angle_signal, fps=30.0):
        """
        Tune thresholds from an initial calibration window.

        Signals with fewer than 30 finite samples or less than 30 degrees of
        robust range keep the fixed defaults.
        """
        finite = np.asarray(angle_signal, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < 30:
            return

        sig_min = float(np.percentile(finite, 5))
        sig_max = float(np.percentile(finite, 95))
        sig_range = sig_max - sig_min
        if sig_range < 30:
            return

        self.FLEX_THRESH = sig_min + sig_range * 0.30
        self.EXTEND_THRESH = sig_max - sig_range * 0.25
        self.MIN_ROM = sig_range * 0.20
        self._min_rep_s = 0.5
        print(f"FSM calibrated: FLEX={self.FLEX_THRESH:.0f}°  "
              f"EXTEND={self.EXTEND_THRESH:.0f}°  "
              f"range={sig_range:.0f}°")

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

        if self.state == "EXTENDED":
            if direction < -self.MIN_DELTA:
                self.state = "FLEXING"
                self._rep_angles     = [angle_deg]
                self._rep_landmarks  = [frame_lm]
                self._rep_start_time = timestamp_s

        elif self.state == "FLEXING":
            if angle_deg < self.FLEX_THRESH:
                self.state = "FLEXED"
            elif direction > self.MIN_DELTA:
                # reversed before full flex — partial rep, reset
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
                self._finish_rep(timestamp_s)
                self.state           = "FLEXING"
                self._rep_angles     = [angle_deg]
                self._rep_landmarks  = [frame_lm]
                self._rep_start_time = timestamp_s

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
        rep = {
            "exercise":      self.EXERCISE,
            "rep_number":    self.rep_count + 1,
            "max_angle":     round(float(np.nanmax(angles)), 1),
            "min_angle":     round(float(np.nanmin(angles)), 1),
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
        if self.state == "EXTENDING" and self._rep_angles:
            self._finish_rep(timestamp_s)
            self.state = "EXTENDED"

    def get_last_rep(self):
        return self.reps[-1] if self.reps else None

    def rom_check_params(self):
        return {"kind": "flex", "target": self.FLEX_THRESH, "exercise": self.EXERCISE}

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

    def __init__(self, rep_callback=None):
        self.state        = "RESTING"
        self.prev_angle   = None
        self.rep_count    = 0
        self.rep_callback = rep_callback
        self._min_rep_s = 0.5

        self._rep_angles     = []
        self._rep_landmarks  = []
        self._rep_start_time = None
        self.reps            = []

    def calibrate_from_signal(self, angle_signal, fps=30.0):
        """Auto-tune thresholds from observed signal range."""
        finite = np.asarray(angle_signal, dtype=float)
        finite = finite[np.isfinite(finite)]
        if len(finite) < 30:
            return
        sig_min   = float(np.percentile(finite, 5))
        sig_max   = float(np.percentile(finite, 95))
        sig_range = sig_max - sig_min
        if sig_range < 30:
            return
        self.RAISE_THRESH = sig_min + sig_range * 0.30
        self.REST_THRESH  = sig_min + sig_range * 0.05
        self.MIN_ROM      = sig_range * 0.20
        self._min_rep_s   = 0.5
        print(f"ShoulderFSM calibrated: RAISE={self.RAISE_THRESH:.0f}°  "
              f"REST={self.REST_THRESH:.0f}°  range={sig_range:.0f}°")
        
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

        if self.state == "RESTING":
            if direction > self.MIN_DELTA:
                self.state           = "RAISING"
                self._rep_angles     = [angle_deg]
                self._rep_landmarks  = [frame_lm]
                self._rep_start_time = timestamp_s

        elif self.state == "RAISING":
            if angle_deg > self.RAISE_THRESH:
                self.state = "RAISED"
            elif direction < -self.MIN_DELTA:
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

        rep = {
            "exercise":      self.EXERCISE,
            "rep_number":    self.rep_count + 1,
            "max_angle":     round(float(np.nanmax(angles)), 1),
            "min_angle":     round(float(np.nanmin(angles)), 1),
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
        if self.state == "LOWERING" and self._rep_angles:
            self._finish_rep(timestamp_s)
            self.state = "RESTING"

    def get_last_rep(self):
        return self.reps[-1] if self.reps else None

    def rom_check_params(self):
        return {"kind": "raise", "target": self.RAISE_THRESH, "exercise": self.EXERCISE}

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
        "summary":    fsm.summary(),
    }
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    print(f"Session saved → {path}  ({fsm.rep_count} reps)")
    return session