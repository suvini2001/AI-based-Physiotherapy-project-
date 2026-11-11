# assessment/rules.py
"""
Rule-based compensation checks for each completed rep.
Sources:
  - Metric ideas adapted from Pavlikov et al. Sensors 2026 (five metrics, normalisation)
  - Work plan (ROM tolerance, duration >= 1.0s)
"""
import numpy as np

# ── preliminary thresholds (metric ideas adapted from Pavlikov 2026; thresholds are our own preliminary values) ──
SPEED_MIN_S    = 1.0    # seconds — too fast if below this
ELEV_THRESHOLD = 0.03   # shoulder rises > 3% of frame height = fault (Pavlikov: 0.174 x shoulder width)
LEAN_THRESHOLD = 10.0   # trunk leans > 10 degrees = fault (Pavlikov: 0.090 rad = 5.2 deg)
ASYM_THRESHOLD = 0.04   # shoulder height difference > 4% of frame = fault (Pavlikov: arm-angle diff >= 13.2 deg)

# ROM tolerance — how many degrees short of the patient's own full range
# (measured in the CALIB phase) a rep may stop before it is flagged
ROM_TOLERANCE  = 10.0

ROM_FAULT = {
    "flex":  'Incomplete flexion — try to bend your elbow a little further',
    "raise": 'Raise your arm a little higher',
    "extend": 'Incomplete extension — straighten your arm fully between reps',
    "lower": 'Lower your arm fully between reps',
}


def _landmark_array(lm):
    """
    Safely convert landmarks input to a float numpy array of shape (frames, 33, >=2).
    Returns None if lm is None, cannot be converted, or has invalid shape/no frames.
    Avoids ambiguous truth value errors with arrays.
    """
    if lm is None:
        return None
    try:
        arr = np.asarray(lm, dtype=float)
        if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] == 33 and arr.shape[2] >= 2:
            return arr
    except Exception:
        return None
    return None


PROFILE_RATIO = 0.35    # shoulder width / torso length below this = side-on to the camera


def _near_shoulder_if_profile(lm, frame_height=480.0, frame_width=640.0):
    """
    Decide whether the person is side-on to the camera during this rep.

    Front view  -> returns None (both shoulders are usable).
    Side view   -> returns 11 or 12, the shoulder nearer the camera. The far
                   shoulder is hidden behind the body, so its position is a guess
                   and must not be used.
    """
    def usable(idx):
        return int(np.count_nonzero(np.isfinite(lm[:, idx, :2]).all(axis=1))) >= 3

    left_ok, right_ok = usable(11), usable(12)
    if left_ok != right_ok:                    # one shoulder already masked out
        return 11 if left_ok else 12
    if not left_ok:                            # neither is usable
        return None

    is_pixel = bool(np.nanmedian(lm[:, [11, 12], 1]) > 1.0)
    sx = 1.0 if is_pixel else float(frame_width or 640.0)
    sy = 1.0 if is_pixel else float(frame_height or 480.0)

    width = np.nanmedian(np.abs(lm[:, 11, 0] - lm[:, 12, 0])) * sx
    torso = []
    for sho, hip in ((11, 23), (12, 24)):
        d = np.hypot((lm[:, sho, 0] - lm[:, hip, 0]) * sx,
                     (lm[:, sho, 1] - lm[:, hip, 1]) * sy)
        if np.isfinite(d).any():
            torso.append(np.nanmedian(d))
    if not torso or not np.isfinite(width) or max(torso) <= 0:
        return None
    if width / max(torso) >= PROFILE_RATIO:
        return None                            # front view

    # side view: MediaPipe z is smaller for points nearer the camera
    if lm.shape[2] >= 3:
        z_l, z_r = np.nanmedian(lm[:, 11, 2]), np.nanmedian(lm[:, 12, 2])
        if np.isfinite(z_l) and np.isfinite(z_r):
            return 11 if z_l < z_r else 12
    return 12


def check_rom(rep, rom_params):
    """
    Patient-relative ROM check.
    rom_params can be:
      - dict: {"kind": "flex", "target": float} or {"kind": "raise", "target": float}
      - float: backward-compatible calibrated_flex_thresh for elbow flexion
    Returns 'ok', fault string, or None if insufficient data.
    """
    if isinstance(rom_params, dict):
        kind = rom_params.get("kind", "flex")
        target = rom_params.get("target")
    else:
        kind = "flex"
        target = rom_params

    if target is None or not np.isfinite(target):
        return None

    if kind == "flex":          # elbow: low enough?
        min_angle = rep.get('min_angle')
        if min_angle is None or not np.isfinite(min_angle):
            return None
        if min_angle > target + ROM_TOLERANCE:
            return ROM_FAULT["flex"]
    else:                       # shoulder: high enough?
        max_angle = rep.get('max_angle')
        if max_angle is None or not np.isfinite(max_angle):
            return None
        if max_angle < target - ROM_TOLERANCE:
            return ROM_FAULT["raise"]
    return 'ok'


def check_speed(rep):
    """
    Rep too fast — less than 1 second.
    Returns 'ok', fault string, or None if insufficient data.
    """
    dur = rep.get('duration_s')
    if dur is None or not np.isfinite(dur):
        return None
    if dur < SPEED_MIN_S:
        return 'Moving too fast — slow down and control the movement'
    return 'ok'


def check_shoulder_elevation(rep, frame_height=480.0, frame_width=640.0):
    """
    Shoulder rises during the rep.
    Landmark 11 = Left shoulder, Landmark 12 = Right shoulder.
    Front view: checks both shoulders. Side view: checks only the near shoulder.
    Baseline = median of first 3 finite frames.
    Evaluates 90th percentile of shoulder rise across valid frames.
    Returns 'ok', fault string, or None if insufficient data.
    """
    lm = _landmark_array(rep.get('_landmarks'))
    if lm is None:
        return None

    near = _near_shoulder_if_profile(lm, frame_height, frame_width)
    shoulders = [near] if near is not None else [11, 12]

    rises = []
    for idx in shoulders:
        y = lm[:, idx, 1]
        finite = y[np.isfinite(y)]
        if len(finite) < 3:
            return None
        # Detect if landmarks are pixel coordinates or normalized (0-1)
        scale = float(frame_height) if np.nanmedian(finite) > 1.0 else 1.0
        base = float(np.median(finite[:3]))
        # Rise = baseline - y (image Y increases downwards)
        rises.append(float(np.percentile((base - finite) / scale, 90)))

    if max(rises) > ELEV_THRESHOLD:
        return 'Keep your shoulder down'
    return 'ok'


def check_trunk_lean(rep, frame_height=480.0, frame_width=640.0):
    """
    Torso leans during the rep.
    Front view: hip midpoint -> shoulder midpoint, detects leaning sideways.
    Side view:  near hip -> near shoulder, detects leaning forward or back.
    Angle from vertical in pixel space, 90th percentile across valid frames.
    Returns 'ok', fault string, or None if insufficient data.
    """
    lm = _landmark_array(rep.get('_landmarks'))
    if lm is None:
        return None

    near = _near_shoulder_if_profile(lm, frame_height, frame_width)
    if near is not None:
        sho = lm[:, near, :2]
        hip = lm[:, near + 12, :2]             # 11 -> 23, 12 -> 24
        message = 'Keep your back straight — you are leaning forward or back'
    else:
        sho = (lm[:, 11, :2] + lm[:, 12, :2]) / 2.0
        hip = (lm[:, 23, :2] + lm[:, 24, :2]) / 2.0
        message = 'Keep your back straight — you are leaning sideways'

    valid_mask = np.isfinite(sho).all(axis=1) & np.isfinite(hip).all(axis=1)
    if np.count_nonzero(valid_mask) < 3:
        return None
    sho, hip = sho[valid_mask], hip[valid_mask]

    is_pixel = bool(np.nanmedian(sho[:, 1]) > 1.0)
    sx = 1.0 if is_pixel else float(frame_width if frame_width else 640.0)
    sy = 1.0 if is_pixel else float(frame_height if frame_height else 480.0)

    dx = (sho[:, 0] - hip[:, 0]) * sx
    dy = (sho[:, 1] - hip[:, 1]) * sy

    # Image Y increases downwards, so -dy points upward towards shoulders
    leans = np.abs(np.degrees(np.arctan2(dx, -dy)))
    peak_lean = float(np.percentile(leans, 90))

    if peak_lean > LEAN_THRESHOLD:
        return message
    return 'ok'


def check_asymmetry(rep, frame_height=480.0, frame_width=640.0):
    """
    Left and right shoulders at different heights (one shrugged).
    Skipped in side view, where the far shoulder cannot be seen.
    Evaluates 90th percentile of shoulder height differences across valid frames.
    Returns 'ok', fault string, or None if insufficient data.
    """
    lm = _landmark_array(rep.get('_landmarks'))
    if lm is None:
        return None

    if _near_shoulder_if_profile(lm, frame_height, frame_width) is not None:
        return None                            # side view: cannot compare shoulders

    y_l = lm[:, 11, 1]
    y_r = lm[:, 12, 1]

    valid_mask = np.isfinite(y_l) & np.isfinite(y_r)
    if np.count_nonzero(valid_mask) < 3:
        return None

    y_l_v = y_l[valid_mask]
    y_r_v = y_r[valid_mask]

    is_pixel = bool(np.nanmedian(y_l_v) > 1.0 or np.nanmedian(y_r_v) > 1.0)
    scale = float(frame_height) if is_pixel else 1.0

    asyms = np.abs(y_l_v - y_r_v) / scale
    peak_asym = float(np.percentile(asyms, 90))

    if peak_asym > ASYM_THRESHOLD:
        return 'Keep your shoulders level'
    return 'ok'


def run_checks(rep, frame_height=480.0, rom_params=None, frame_width=640.0, calibrated_flex_thresh=None):
    """
    Run all five compensation checks.
    Returns:
        (faults, n_checked)
        faults: list of fault strings
        n_checked: number of checks evaluated (with sufficient data)
    """
    if rom_params is None:
        if calibrated_flex_thresh is not None:
            rom_params = calibrated_flex_thresh
        else:
            rom_params = {"kind": "flex", "target": 55.0}

    named_checks = [
        ("rom", lambda r: check_rom(r, rom_params)),
        ("speed", check_speed),
        ("shoulder_elevation", lambda r: check_shoulder_elevation(r, frame_height, frame_width)),
        ("lean", lambda r: check_trunk_lean(r, frame_height=frame_height, frame_width=frame_width)),
        ("asymmetry", lambda r: check_asymmetry(r, frame_height, frame_width)),
    ]
    faults = []
    n_checked = 0
    for name, fn in named_checks:
        try:
            res = fn(rep)
            if res is None:
                continue   # insufficient data, cannot judge
            n_checked += 1
            if res != 'ok':
                faults.append(res)
        except Exception as e:
            print(f"WARNING: rule '{name}' failed: {e}")
    return faults, n_checked


def run_all(rep, frame_height=480.0, rom_params=None, frame_width=640.0, calibrated_flex_thresh=None):
    """
    Run all five checks. Backward-compatible wrapper returning only faults list.
    Empty list = clean rep.
    """
    if rom_params is None and calibrated_flex_thresh is not None:
        rom_params = calibrated_flex_thresh
    faults, _ = run_checks(
        rep,
        frame_height=frame_height,
        rom_params=rom_params,
        frame_width=frame_width
    )
    return faults