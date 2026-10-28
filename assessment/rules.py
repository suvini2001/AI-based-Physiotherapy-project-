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
            return 'Incomplete flexion — try to bend your elbow a little further'
    else:                       # shoulder: high enough?
        max_angle = rep.get('max_angle')
        if max_angle is None or not np.isfinite(max_angle):
            return None
        if max_angle < target - ROM_TOLERANCE:
            return 'Raise your arm a little higher'
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


def check_shoulder_elevation(rep, frame_height=480.0):
    """
    Shoulder rises during the rep (checks both shoulders).
    Landmark 11 = Left shoulder, Landmark 12 = Right shoulder.
    Baseline = median of first 3 finite frames.
    Evaluates 90th percentile of shoulder rise across valid frames.
    Returns 'ok', fault string, or None if insufficient data.
    """
    lm = _landmark_array(rep.get('_landmarks'))
    if lm is None:
        return None

    y_r = lm[:, 12, 1]
    y_l = lm[:, 11, 1]

    finite_r = y_r[np.isfinite(y_r)]
    finite_l = y_l[np.isfinite(y_l)]

    if len(finite_r) < 3 or len(finite_l) < 3:
        return None

    # Detect if landmarks are pixel coordinates or normalized (0-1)
    is_pixel = bool(np.nanmedian(finite_r) > 1.0 or np.nanmedian(finite_l) > 1.0)
    scale = float(frame_height) if is_pixel else 1.0

    base_r = float(np.median(finite_r[:3]))
    base_l = float(np.median(finite_l[:3]))

    # Rise = baseline - y (image Y increases downwards)
    rise_r = (base_r - finite_r) / scale
    rise_l = (base_l - finite_l) / scale

    # 90th percentile of rise across valid frames
    p90_r = float(np.percentile(rise_r, 90))
    p90_l = float(np.percentile(rise_l, 90))
    max_rise = max(p90_r, p90_l)

    if max_rise > ELEV_THRESHOLD:
        return 'Keep your shoulder down'
    return 'ok'


def check_trunk_lean(rep, frame_height=480.0, frame_width=640.0):
    """
    Torso leans sideways during the rep.
    Uses hip midpoint -> shoulder midpoint vector angle from vertical in pixel space.
    Evaluates 90th percentile of lean angles across valid frames.
    Returns 'ok', fault string, or None if insufficient data.
    """
    lm = _landmark_array(rep.get('_landmarks'))
    if lm is None:
        return None

    lsho = lm[:, 11, :2]
    rsho = lm[:, 12, :2]
    lhip = lm[:, 23, :2]
    rhip = lm[:, 24, :2]

    valid_mask = (
        np.isfinite(lsho).all(axis=1) &
        np.isfinite(rsho).all(axis=1) &
        np.isfinite(lhip).all(axis=1) &
        np.isfinite(rhip).all(axis=1)
    )
    if np.count_nonzero(valid_mask) < 3:
        return None

    lsho_v = lsho[valid_mask]
    rsho_v = rsho[valid_mask]
    lhip_v = lhip[valid_mask]
    rhip_v = rhip[valid_mask]

    is_pixel = bool(np.nanmedian(rsho_v[:, 1]) > 1.0 or np.nanmedian(lsho_v[:, 1]) > 1.0)
    sx = 1.0 if is_pixel else float(frame_width if frame_width else 640.0)
    sy = 1.0 if is_pixel else float(frame_height if frame_height else 480.0)

    mid_hip_x = ((lhip_v[:, 0] + rhip_v[:, 0]) / 2.0) * sx
    mid_hip_y = ((lhip_v[:, 1] + rhip_v[:, 1]) / 2.0) * sy
    mid_sho_x = ((lsho_v[:, 0] + rsho_v[:, 0]) / 2.0) * sx
    mid_sho_y = ((lsho_v[:, 1] + rsho_v[:, 1]) / 2.0) * sy

    dx = mid_sho_x - mid_hip_x
    dy = mid_sho_y - mid_hip_y

    # Image Y increases downwards, so -dy points upward towards shoulders
    leans = np.abs(np.degrees(np.arctan2(dx, -dy)))
    peak_lean = float(np.percentile(leans, 90))

    if peak_lean > LEAN_THRESHOLD:
        return 'Keep your back straight — you are leaning sideways'
    return 'ok'


def check_asymmetry(rep, frame_height=480.0):
    """
    Left and right shoulders at different heights (one shrugged).
    Evaluates 90th percentile of shoulder height differences across valid frames.
    Returns 'ok', fault string, or None if insufficient data.
    """
    lm = _landmark_array(rep.get('_landmarks'))
    if lm is None:
        return None

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
        ("shoulder_elevation", lambda r: check_shoulder_elevation(r, frame_height)),
        ("lean", lambda r: check_trunk_lean(r, frame_height=frame_height, frame_width=frame_width)),
        ("asymmetry", lambda r: check_asymmetry(r, frame_height)),
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