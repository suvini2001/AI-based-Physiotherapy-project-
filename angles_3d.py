# angles_3d.py
"""B1 + B3 + B4 – body yaw, yaw-corrected angle, bone-length gate.
All functions take MediaPipe WORLD landmarks (N, 33, 4) in metres.
"""
import numpy as np
from angles import angle, ELBOW

# MediaPipe landmark indices
L_SH, R_SH   = 11, 12   # shoulders
L_HIP, R_HIP = 23, 24   # hips

YAW_CAP_DEG  = 60.0      # never divide by cos(yaw) beyond this — cos(60°) = 0.5, so max 2× amplification
BONE_JUMP    = 0.40      # flag a frame if a bone length jumps more than 40 % from median
BONE_MIN_FRAC = 0.05     # a bone shorter than 5 % of median is also flagged (arm folded into body)


# ── B1 ──────────────────────────────────────────────────────────────────────

def body_yaw(world):
    """Estimate how many degrees the person has rotated left/right away from the camera.

    Why: MediaPipe's 2D elbow angle collapses when the arm points toward the camera.
    Yaw tells us how bad the viewing angle is so we can correct or reject the frame.

    world: (N, 33, 4) world landmarks in metres  ->  (N,) yaw in degrees [0, 90].
    NaN in any hip or shoulder landmark -> NaN out.

    Method: the shoulder line (right_shoulder - left_shoulder) is projected onto
    the X-Z plane (horizontal plane).  When the person faces straight at the camera
    that line is purely along X, so yaw = 0.  When they face sideways it tilts
    toward Z, so yaw grows toward 90.
    """
    w = np.asarray(world, float)
    # handle single-frame input
    if w.ndim == 2:
        w = w[None]

    ls = w[:, L_SH, :3]   # left shoulder  (N, 3)
    rs = w[:, R_SH, :3]   # right shoulder (N, 3)

    dx = rs[:, 0] - ls[:, 0]   # X component of shoulder line
    dz = rs[:, 2] - ls[:, 2]   # Z component (depth)

    # yaw = angle between shoulder line and the camera-facing X axis
    yaw_rad = np.arctan2(np.abs(dz), np.abs(dx))
    yaw_deg = np.degrees(yaw_rad)

    # propagate NaN where either shoulder is missing
    bad = np.isnan(ls).any(axis=1) | np.isnan(rs).any(axis=1)
    yaw_deg[bad] = np.nan

    return yaw_deg if w.shape[0] > 1 else yaw_deg[0]


# ── B3 ──────────────────────────────────────────────────────────────────────

def elbow_angle_yaw_corrected(world, side="right", yaw_cap_deg=YAW_CAP_DEG):
    """Variant C: divide the 3D elbow angle by cos(yaw) to correct for body rotation.

    Why: when the person turns sideways the camera under-reads the elbow angle
    because depth information is lost in 2D projection.  Dividing by cos(yaw)
    reverses that foreshortening.  We cap at yaw_cap_deg (default 60°) because
    cos(yaw) gets small fast — beyond 60° the correction amplifies noise more
    than it fixes the angle.

    world: (N, 33, 4)  ->  (N,) corrected angle in degrees, clipped to [0, 180].
    Returns NaN where yaw is NaN or the raw angle is NaN.
    """
    w = np.asarray(world, float)
    if w.ndim == 2:
        w = w[None]

    # raw 3D angle (same as variant B, just the dot-product angle in 3D metres)
    i, j, k = ELBOW[side]
    raw = angle(w[:, i, :3], w[:, j, :3], w[:, k, :3])   # (N,)

    yaw = body_yaw(w)                                      # (N,)

    # cap yaw so we never divide by a very small cos
    yaw_capped = np.minimum(np.abs(yaw), yaw_cap_deg)
    cos_yaw = np.cos(np.radians(yaw_capped))               # never below cos(60°) = 0.5

    corrected = raw / cos_yaw
    corrected = np.clip(corrected, 0.0, 180.0)

    # keep NaN where either input was NaN
    corrected[np.isnan(raw) | np.isnan(yaw)] = np.nan

    return corrected if w.shape[0] > 1 else corrected[0]


# ── B4 ──────────────────────────────────────────────────────────────────────

def _bone_lengths(world, side="right"):
    """Returns (N, 2) array: column 0 = upper arm length, column 1 = forearm length."""
    i, j, k = ELBOW[side]
    w = np.asarray(world, float)
    if w.ndim == 2:
        w = w[None]
    upper = np.linalg.norm(w[:, i, :3] - w[:, j, :3], axis=1)   # shoulder-elbow
    lower = np.linalg.norm(w[:, j, :3] - w[:, k, :3], axis=1)   # elbow-wrist
    return np.stack([upper, lower], axis=1)


def bone_length_gate(world, side="right",
                     jump_frac=BONE_JUMP,
                     min_frac=BONE_MIN_FRAC):
    """Variant E gate: flag frames where bone lengths are implausible.

    Why: MediaPipe occasionally misplaces a landmark (e.g. wrist jumps to the
    opposite side of the body).  When this happens the bone length changes
    dramatically in one frame.  Flagging those frames and returning NaN prevents
    a single bad frame from corrupting the smoothed angle.

    world: (N, 33, 4)  ->  (N,) bool mask.  True = frame is VALID, False = gate rejects it.

    A frame is rejected if:
      - Any bone length is NaN (landmark missing)
      - Any bone deviates more than jump_frac (40 %) from the clip median
      - Any bone is shorter than min_frac (5 %) of its median (arm folded out of view)
    """
    w = np.asarray(world, float)
    if w.ndim == 2:
        w = w[None]

    lengths = _bone_lengths(w, side)   # (N, 2)

    # use nanmedian so a few bad frames don't corrupt the reference
    med = np.nanmedian(lengths, axis=0)   # (2,)

    valid = np.ones(len(w), dtype=bool)

    for col in range(2):
        bone = lengths[:, col]
        nan_mask = np.isnan(bone)
        # reject NaN frames
        valid[nan_mask] = False
        # reject frames that deviate more than jump_frac from median
        with np.errstate(invalid="ignore"):
            ratio = bone / med[col]
        valid[~nan_mask & (ratio > 1 + jump_frac)] = False
        valid[~nan_mask & (ratio < min_frac)]       = False

    return valid