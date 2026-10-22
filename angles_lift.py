"""Bone-length-constrained 2.5D lifting for out-of-plane elbow angles.

Why this exists
---------------
Variant B (`angles_world.elbow_angle_3d`) uses the MAGNITUDE of MediaPipe's
world-landmark z. That magnitude is a monocular guess and is the weakest part
of the estimate: on the REHAB24-6 benchmark it scored 21.93 deg against 16.12
for the plain 2D projection, with roughly twice the rate of errors above 20 deg.

This module uses z only for its SIGN and derives the magnitude from geometry:

    dx^2 + dy^2 + dz^2 = L^2        (L = true segment length)
            |dz| = sqrt(L^2 - dx^2 - dy^2)

dx and dy come straight from the image, where MediaPipe is accurate. L is a
bone, so it is constant for a person and can be measured. Only the front/back
decision is left to the model, which it makes far more reliably than the
distance.

Units: every length is divided by the subject's shoulder width, so the result
is scale-free and independent of how far the person stands from the camera.

Drop-in alongside `angles.py` / `angles_world.py`; same NaN-in, NaN-out
contract, same (frames, 33, 4) input shape.
"""
import numpy as np

# MediaPipe Pose indices
L_SH, R_SH = 11, 12
ARM = {           # (shoulder, elbow, wrist)
    'right': (12, 14, 16),
    'left':  (11, 13, 15),
}

__all__ = ['shoulder_width', 'robust_scale', 'bone_lengths', 'lift_segments',
           'elbow_angle_lifted', 'planarity']


# --------------------------------------------------------------------- scale
def shoulder_width(landmarks, size=(640, 480)):
    """Per-frame shoulder width in pixels. (frames, 33, 4) -> (frames,).

    Shoulders are used rather than the torso because the hips leave the frame
    whenever the subject sits close to a laptop, which makes the torso scale in
    `pose_normalise.py` NaN for the whole clip.
    """
    lm = np.asarray(landmarks, float)
    w, h = size
    dx = (lm[:, L_SH, 0] - lm[:, R_SH, 0]) * w
    dy = (lm[:, L_SH, 1] - lm[:, R_SH, 1]) * h
    sw = np.hypot(dx, dy)
    return np.where(sw > 1e-6, sw, np.nan)


def _segment(landmarks, a, b, size):
    """Per-frame (dx, dy) from joint b to joint a, in pixels."""
    lm = np.asarray(landmarks, float)
    w, h = size
    return ((lm[:, a, 0] - lm[:, b, 0]) * w,
            (lm[:, a, 1] - lm[:, b, 1]) * h)


# ---------------------------------------------------------------- bone length
def robust_scale(landmarks, size=(640, 480), per_frame=False, band=(0.7, 1.3)):
    """A stable shoulder-width scale for the session.

    The per-frame width is not usable on its own: torso rotation foreshortens
    the shoulders too, and on the project's own test clip it drops from a
    median of 114 px to 68 px, which would inflate every length measured
    against it. The session median is the default; `per_frame=True` tracks a
    subject who moves toward or away from the camera, but clipped to `band`
    times the median so rotation cannot corrupt it.
    """
    sw = shoulder_width(landmarks, size)
    med = np.nanmedian(sw)
    if not np.isfinite(med) or med <= 0:
        return np.full(len(sw), np.nan)
    if not per_frame:
        return np.full(len(sw), med)
    return np.clip(sw, band[0] * med, band[1] * med)


def bone_lengths(landmarks, side='right', size=(640, 480), pct=95.0,
                 min_frames=30):
    """Estimate (upper-arm, forearm) length in shoulder-width units.

    Projection can only ever make a segment look SHORTER than it is, never
    longer, so a high percentile of the observed length over a session is a
    sound estimate of the true length. The 95th percentile rather than the
    maximum, so one bad frame cannot set the scale.

    Length and scale are estimated separately — the percentile is taken on the
    raw pixel length and divided by the session scale afterwards. Taking the
    percentile of a per-frame ratio instead lets frames where the shoulders are
    foreshortened spike the ratio, which inflated this estimate by about 60%
    on the project's own test clip.

    Returns (L_upper, L_fore) in shoulder-width units; either may be NaN.
    """
    sh, el, wr = ARM[side]
    sc = np.nanmedian(shoulder_width(landmarks, size))
    if not np.isfinite(sc) or sc <= 0:
        return np.nan, np.nan

    out = []
    for a, b in ((sh, el), (wr, el)):          # both measured from the elbow
        dx, dy = _segment(landmarks, a, b, size)
        px = np.hypot(dx, dy)
        px = px[np.isfinite(px)]
        out.append(float(np.percentile(px, pct)) / sc
                   if len(px) >= min_frames else np.nan)
    return out[0], out[1]


def bone_lengths_from_pose(landmarks, side='right', size=(640, 480)):
    """Bone lengths from a calibration pose instead of a whole session.

    Use with the straight-arm pose that `calibration.session_check` already
    asks for: pass only those frames. Takes the median, since every frame of a
    good calibration pose should give the same answer.
    """
    sh, el, wr = ARM[side]
    sc = np.nanmedian(shoulder_width(landmarks, size))
    if not np.isfinite(sc) or sc <= 0:
        return np.nan, np.nan
    res = []
    for a, b in ((sh, el), (wr, el)):
        dx, dy = _segment(landmarks, a, b, size)
        px = np.hypot(dx, dy)
        res.append(float(np.nanmedian(px)) / sc if np.isfinite(px).any() else np.nan)
    return res[0], res[1]


# ----------------------------------------------------------------- planarity
def planarity(landmarks, side='right', size=(640, 480), L_up=None, L_fo=None):
    """How much of each segment is visible in the image plane, 0..1.

    1.0 means the arm lies in the camera plane and the 2D angle is already
    correct. Near 0 means the segment points at the camera: the lift is
    ill-conditioned there and the reading should be withheld.
    """
    if L_up is None or L_fo is None:
        L_up, L_fo = bone_lengths(landmarks, side, size)
    sh, el, wr = ARM[side]
    sc = robust_scale(landmarks, size)
    out = []
    for (a, b), L in (((sh, el), L_up), ((wr, el), L_fo)):
        dx, dy = _segment(landmarks, a, b, size)
        out.append(np.hypot(dx, dy) / sc / L)
    return np.minimum(out[0], out[1])          # the worse of the two segments


# ---------------------------------------------------------------------- lift
def lift_segments(landmarks, world, side='right', size=(640, 480),
                  L_up=None, L_fo=None, sign_window=5):
    """Lift both arm segments to 3D. Returns (u, v, ratio).

    u = elbow -> shoulder, v = elbow -> wrist, each (frames, 3) in
    shoulder-width units. ratio is the planarity of the worse segment.

    `world` supplies nothing but the sign of each segment's depth difference.
    A short rolling majority over `sign_window` frames keeps a noisy frame from
    flipping the arm through the image plane; set sign_window=0 to disable.
    """
    sh, el, wr = ARM[side]
    lm = np.asarray(landmarks, float)
    wd = np.asarray(world, float) if world is not None else None
    if L_up is None or L_fo is None:
        L_up, L_fo = bone_lengths(lm, side, size)

    sc = robust_scale(lm, size)
    vecs = []
    ratios = []
    for (a, b), L in (((sh, el), L_up), ((wr, el), L_fo)):
        dx, dy = _segment(lm, a, b, size)
        dx, dy = dx / sc, dy / sc                       # shoulder-width units
        planar = np.hypot(dx, dy)

        if not np.isfinite(L) or L <= 0:
            dz = np.full_like(planar, np.nan)
            ratio = np.full_like(planar, np.nan)
        else:
            # geometry, not guesswork: the part of the bone that is not
            # visible in the image must be pointing at or away from the camera
            dz = np.sqrt(np.clip(L * L - planar * planar, 0.0, None))
            ratio = planar / L

        # the model only decides which way
        if wd is not None:
            s = np.sign(wd[:, a, 2] - wd[:, b, 2])
            s[~np.isfinite(s) | (s == 0)] = 1.0
            if sign_window and sign_window > 1:
                s = _rolling_sign(s, sign_window)
            dz = dz * s

        vecs.append(np.stack([dx, dy, dz], axis=1))
        ratios.append(ratio)

    return vecs[0], vecs[1], np.minimum(ratios[0], ratios[1])


def _rolling_sign(s, win):
    """Majority sign over a centred window; keeps one noisy frame from flipping
    the segment through the image plane."""
    s = np.asarray(s, float)
    k = int(win) | 1                                    # force odd
    pad = k // 2
    padded = np.pad(s, pad, mode='edge')
    acc = np.zeros(len(s))
    for i in range(k):
        acc += padded[i:i + len(s)]
    out = np.sign(acc)
    out[out == 0] = 1.0
    return out


def elbow_angle_lifted(landmarks, world, side='right', size=(640, 480),
                       L_up=None, L_fo=None, min_ratio=0.30, sign_window=5):
    """Elbow angle in degrees from the bone-length-constrained 3D lift.

    Frames where the arm is too foreshortened for the lift to be conditioned
    (planarity below `min_ratio`) come back as NaN. Withholding a reading beats
    reporting one the geometry cannot support — the same decision the coverage
    gate makes in the benchmark.

    Parameters mirror `angles.elbow_angle` so the two can be swapped in place.
    """
    u, v, ratio = lift_segments(landmarks, world, side, size,
                                L_up, L_fo, sign_window)

    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cos = (u * v).sum(axis=1) / (nu * nv)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    bad = (~np.isfinite(ratio)) | (ratio < min_ratio) | (nu < 1e-9) | (nv < 1e-9)
    ang[bad] = np.nan
    return ang


# ------------------------------------------------------------------ hybrid
def elbow_angle_hybrid(landmarks, world, side='right', size=(640, 480),
                       L_up=None, L_fo=None, full_plane=0.92, min_ratio=0.30):
    """Blend the plain 2D angle with the lifted one, weighted by planarity.

    When the arm lies in the camera plane the 2D angle is already right and the
    lift adds only noise, so it is weighted out. As the arm turns away from the
    plane the lift takes over. Useful if the lift turns out to help in lateral
    views but cost accuracy in frontal ones.
    """
    from angles import elbow_angle            # same repo, 2D path

    a2d = elbow_angle(landmarks, side=side, size=size)
    a3d = elbow_angle_lifted(landmarks, world, side, size,
                             L_up, L_fo, min_ratio=min_ratio)
    r = planarity(landmarks, side, size, L_up, L_fo)

    w = np.clip((r - min_ratio) / max(full_plane - min_ratio, 1e-6), 0.0, 1.0)
    out = w * a2d + (1.0 - w) * a3d
    out = np.where(np.isfinite(a3d), out, a2d)          # fall back to 2D
    out[~np.isfinite(a2d)] = np.nan
    return out


if __name__ == '__main__':
    # smoke test on the repo's sample landmarks
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/elbow_A_landmarks.npy'
    lm = np.load(path)
    print('frames', lm.shape)
    for side in ('right', 'left'):
        L_up, L_fo = bone_lengths(lm, side)
        print(f'{side:5}  L_upper {L_up:.3f}  L_fore {L_fo:.3f}  (shoulder widths)')
        r = planarity(lm, side, L_up=L_up, L_fo=L_fo)
        print(f'       planarity  median {np.nanmedian(r):.2f}  '
              f'min {np.nanmin(r):.2f}  frames below 0.30: '
              f'{100 * np.nanmean(r < 0.30):.1f}%')
