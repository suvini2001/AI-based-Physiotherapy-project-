# assessment/rules.py
"""
Rule-based compensation checks for each completed rep.
Sources:
  - Pavlikov et al. Sensors 2026 (five metrics, thresholds, normalisation)
  - Work plan (ROM tolerance, duration >= 1.0s)
"""
import numpy as np

# ── fixed clinical thresholds (Pavlikov 2026) ────────────────────
SPEED_MIN_S    = 1.0    # seconds — too fast if below this
ELEV_THRESHOLD = 0.03   # shoulder rises > 3% of frame height = fault
LEAN_THRESHOLD = 10.0   # trunk leans > 10 degrees = fault
ASYM_THRESHOLD = 0.04   # shoulder height difference > 4% of frame = fault

# ROM tolerance — how many degrees above calibrated FLEX_THRESH
# before flagging incomplete flexion
ROM_TOLERANCE  = 10.0


def check_rom(rep, calibrated_flex_thresh):
    """
    Patient-relative ROM check.
    Uses the FSM's auto-calibrated FLEX_THRESH so impaired patients
    are not penalised for limited ROM.
    """
    if rep['min_angle'] > calibrated_flex_thresh + ROM_TOLERANCE:
        return 'Incomplete flexion — try to bend your elbow a little further'
    return None


def check_speed(rep):
    """Rep too fast — less than 1 second."""
    if rep['duration_s'] < SPEED_MIN_S:
        return 'Moving too fast — slow down and control the movement'
    return None


def check_shoulder_elevation(rep, frame_height):
    """
    Shoulder rises during the rep (checks both shoulders).
    Landmark 11 = Left shoulder, Landmark 12 = Right shoulder.
    """
    lm = rep.get('_landmarks')
    if not lm or len(lm) < 2:
        return None
    try:
        # Detect if landmarks are pixel coordinates or normalized (0-1)
        is_pixel = lm[0][12][1] > 1.0
        scale = frame_height if is_pixel else 1.0

        # Check right shoulder (12) and left shoulder (11)
        y_start_r, y_start_l = lm[0][12][1], lm[0][11][1]
        y_peak_r = min(f[12][1] for f in lm)
        y_peak_l = min(f[11][1] for f in lm)

        delta_r = (y_start_r - y_peak_r) / scale
        delta_l = (y_start_l - y_peak_l) / scale

        max_delta = max(delta_r, delta_l)

        if max_delta > ELEV_THRESHOLD:
            return 'Keep your shoulder down'
    except (IndexError, TypeError):
        return None
    return None


def check_trunk_lean(rep, frame_height=None):
    """
    Torso leans sideways during the rep.
    Uses hip midpoint -> shoulder midpoint vector angle from vertical.
    """
    lm = rep.get('_landmarks')
    if not lm or len(lm) < 2:
        return None
    try:
        peak_frame = lm[len(lm) // 2]
        lhip = peak_frame[23][:2]
        rhip = peak_frame[24][:2]
        lsho = peak_frame[11][:2]
        rsho = peak_frame[12][:2]

        mid_hip = np.array([(lhip[0]+rhip[0])/2, (lhip[1]+rhip[1])/2])
        mid_sho = np.array([(lsho[0]+rsho[0])/2, (lsho[1]+rsho[1])/2])
        torso = mid_sho - mid_hip

        # Image Y increases downwards, so -torso[1] points upward
        lean = abs(np.degrees(np.arctan2(torso[0], -torso[1])))
        if lean > LEAN_THRESHOLD:
            return 'Keep your back straight — you are leaning sideways'
    except (IndexError, TypeError):
        return None
    return None


def check_asymmetry(rep, frame_height):
    """
    Left and right shoulders at different heights (one shrugged).
    """
    lm = rep.get('_landmarks')
    if not lm or len(lm) < 2:
        return None
    try:
        peak_frame = lm[len(lm) // 2]
        y_left  = peak_frame[11][1]
        y_right = peak_frame[12][1]

        is_pixel = y_left > 1.0
        scale = frame_height if is_pixel else 1.0

        asym = abs(y_left - y_right) / scale
        if asym > ASYM_THRESHOLD:
            return 'Keep your shoulders level'
    except (IndexError, TypeError):
        return None
    return None


def run_all(rep, frame_height=480.0, calibrated_flex_thresh=55.0):
    """
    Run all five checks. Returns list of fault strings.
    Empty list = clean rep.
    """
    faults = []
    checks = [
        lambda r: check_rom(r, calibrated_flex_thresh),
        check_speed,
        lambda r: check_shoulder_elevation(r, frame_height),
        lambda r: check_trunk_lean(r, frame_height),
        lambda r: check_asymmetry(r, frame_height),
    ]
    for fn in checks:
        try:
            result = fn(rep)
            if result:
                faults.append(result)
        except Exception:
            pass   # never crash the main loop
    return faults