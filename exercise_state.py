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
        self._rep_start_time = None
        self.reps            = []        # full history, for session_log.json

    def update(self, angle_deg, timestamp_s):
        """
        Feed one frame. Returns (current_state, rep_just_completed).
        angle_deg:   float or None (NaN frame → skip)
        timestamp_s: float seconds since session start
        """
        if angle_deg is None or not np.isfinite(angle_deg):
            return self.state, False

        direction = (angle_deg - self.prev_angle
                     if self.prev_angle is not None else 0.0)
        self._rep_angles.append(angle_deg)
        rep_completed = False

        if self.state == "EXTENDED":
            if direction < -self.MIN_DELTA:
                self.state = "FLEXING"
                self._rep_angles     = [angle_deg]
                self._rep_start_time = timestamp_s

        elif self.state == "FLEXING":
            if angle_deg < self.FLEX_THRESH:
                self.state = "FLEXED"
            elif direction > self.MIN_DELTA:
                # reversed before full flex — partial rep, reset
                self.state           = "EXTENDED"
                self._rep_angles     = []
                self._rep_start_time = None

        elif self.state == "FLEXED":
            if direction > self.MIN_DELTA:
                self.state = "EXTENDING"

        elif self.state == "EXTENDING":
            if angle_deg > self.EXTEND_THRESH:
                self.state    = "EXTENDED"
                rep_completed = True
                self._finish_rep(timestamp_s)

        self.prev_angle = angle_deg
        return self.state, rep_completed

    MIN_ROM = 30.0   # add this as a class constant at the top of each class

    def _finish_rep(self, timestamp_s):
        if not self._rep_angles:
            return
        angles = np.array(self._rep_angles)
        rom = float(np.nanmax(angles) - np.nanmin(angles))

        # ignore spurious micro-reps (noise or signal artefacts)
        if rom < self.MIN_ROM:
            self._rep_angles     = []
            self._rep_start_time = None
            return
        rep = {
            "rep_number": self.rep_count + 1,
            "max_angle":  round(float(np.nanmax(angles)), 1),
            "min_angle":  round(float(np.nanmin(angles)), 1),
            "rom":        round(float(np.nanmax(angles) - np.nanmin(angles)), 1),
            "duration_s": round(timestamp_s - self._rep_start_time, 2)
                          if self._rep_start_time else 0.0,
            "timestamp":  round(timestamp_s, 2),
            "quality_score": None,   # Person B fills this in
            "faults":        [],     # Person B fills this in
        }
        self.rep_count += 1
        self.reps.append(rep)
        self._rep_angles     = []
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

    RAISE_THRESH = 75.0   # arm raised above this = abduction rep in progress
    REST_THRESH  = 30.0   # arm back below this   = rep complete
    MIN_DELTA    =  2.0

    def __init__(self, rep_callback=None):
        self.state        = "RESTING"
        self.prev_angle   = None
        self.rep_count    = 0
        self.rep_callback = rep_callback

        self._rep_angles     = []
        self._rep_start_time = None
        self.reps            = []

    def update(self, angle_deg, timestamp_s):
        if angle_deg is None or not np.isfinite(angle_deg):
            return self.state, False

        direction = (angle_deg - self.prev_angle
                     if self.prev_angle is not None else 0.0)
        self._rep_angles.append(angle_deg)
        rep_completed = False

        if self.state == "RESTING":
            if direction > self.MIN_DELTA:
                self.state           = "RAISING"
                self._rep_angles     = [angle_deg]
                self._rep_start_time = timestamp_s

        elif self.state == "RAISING":
            if angle_deg > self.RAISE_THRESH:
                self.state = "RAISED"
            elif direction < -self.MIN_DELTA:
                self.state           = "RESTING"
                self._rep_angles     = []
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
        angles = np.array(self._rep_angles)
        rom = float(np.nanmax(angles) - np.nanmin(angles))

        # ignore spurious micro-reps (noise or signal artefacts)
        if rom < self.MIN_ROM:
            self._rep_angles     = []
            self._rep_start_time = None
            return
        
        rep = {
            "rep_number": self.rep_count + 1,
            "max_angle":  round(float(np.nanmax(angles)), 1),
            "min_angle":  round(float(np.nanmin(angles)), 1),
            "rom":        round(float(np.nanmax(angles) - np.nanmin(angles)), 1),
            "duration_s": round(timestamp_s - self._rep_start_time, 2)
                          if self._rep_start_time else 0.0,
            "timestamp":  round(timestamp_s, 2),
            "quality_score": None,
            "faults":        [],
        }
        self.rep_count += 1
        self.reps.append(rep)
        self._rep_angles     = []
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
    session = {
        "date":       datetime.datetime.now().isoformat(),
        "exercise":   exercise_name,
        "total_reps": fsm.rep_count,
        "reps":       fsm.reps,
        "summary":    fsm.summary(),
    }
    with open(path, "w") as f:
        json.dump(session, f, indent=2)
    print(f"Session saved → {path}  ({fsm.rep_count} reps)")
    return session