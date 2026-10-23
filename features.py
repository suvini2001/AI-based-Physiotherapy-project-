"""Per-frame geometric features for learned joint-angle correction.

Method after Maeda et al., BMC Musculoskeletal Disorders 2026, 27:552, which
converts MediaPipe skeletal coordinates into geometric features and regresses
the true joint angle from them (R^2 up to 0.94, MAE 4-6 deg against an IMU
reference). The idea generalises the project's existing calibration curve: that
curve maps one number (the measured angle) to a corrected one, so it can only
remove a bias that depends on the angle. A feature set that also describes HOW
the arm sits relative to the camera lets the model learn the projection
distortion itself, which is what varies between viewpoints.

Every feature here is computable from MediaPipe output alone, so the same
vector can be built in the browser at run time. Nothing is derived from ground
truth, and the camera-viewpoint label is deliberately NOT a feature: the
application has no such label, and the geometry features (planarity, torso
ratio) already encode it.
"""
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
ARM = {'right': (12, 14, 16), 'left': (11, 13, 15)}   # shoulder, elbow, wrist

FEATURE_NAMES = [
    'angle_2d',        # the plain projected elbow angle
    'angle_world',     # MediaPipe world-landmark angle
    'angle_lift',      # bone-length constrained 2.5D lift
    'plan_upper',      # upper arm: visible length / true length
    'plan_fore',       # forearm: same
    'plan_min',        # the worse of the two
    'len_upper',       # observed upper-arm length, shoulder widths
    'len_fore',
    'bone_upper',      # estimated true upper-arm length for this person
    'bone_fore',
    'sw_frac',         # shoulder width / frame height — how far away they are
    'vis_sh', 'vis_el', 'vis_wr',
    'wz_sh_el',        # world-z difference, shoulder to elbow
    'wz_el_wr',
    'torso_ratio',     # shoulder width / shoulder-to-hip height — torso rotation
    'd_angle',         # frame-to-frame change in the 2D angle
    'std_angle',       # rolling standard deviation of the 2D angle
    'is_right',        # which arm
]


def _rolling_std(x, win=15):
    x = np.asarray(x, float)
    k = int(win) | 1
    pad = k // 2
    p = np.pad(x, pad, mode='edge')
    stack = np.stack([p[i:i + len(x)] for i in range(k)], axis=0)
    with np.errstate(invalid='ignore'):
        return np.nanstd(stack, axis=0)


def frame_features(landmarks, world, side, size, L_up, L_fo,
                   angle_2d, angle_world, angle_lift):
    """Build the (frames, len(FEATURE_NAMES)) feature matrix for one recording.

    The three angle series are passed in rather than recomputed so that the
    features describe exactly the signals the benchmark scores.
    """
    lm = np.asarray(landmarks, float)
    wd = np.asarray(world, float)
    w, h = size
    sh, el, wr = ARM[side]
    n = len(lm)

    # --- scale: shoulder width in pixels, session median (rotation-resistant) ---
    sx = (lm[:, L_SH, 0] - lm[:, R_SH, 0]) * w
    sy = (lm[:, L_SH, 1] - lm[:, R_SH, 1]) * h
    sw = np.hypot(sx, sy)
    sw_med = np.nanmedian(sw)
    sc = sw_med if np.isfinite(sw_med) and sw_med > 0 else np.nan

    def seg_len(a, b):
        dx = (lm[:, a, 0] - lm[:, b, 0]) * w
        dy = (lm[:, a, 1] - lm[:, b, 1]) * h
        return np.hypot(dx, dy) / sc

    len_up = seg_len(sh, el)
    len_fo = seg_len(wr, el)
    plan_up = len_up / L_up if np.isfinite(L_up) and L_up > 0 else np.full(n, np.nan)
    plan_fo = len_fo / L_fo if np.isfinite(L_fo) and L_fo > 0 else np.full(n, np.nan)

    # --- torso rotation proxy: a rotated torso has narrow shoulders for its height ---
    mid_sh_y = (lm[:, L_SH, 1] + lm[:, R_SH, 1]) / 2 * h
    mid_hip_y = (lm[:, L_HIP, 1] + lm[:, R_HIP, 1]) / 2 * h
    torso_h = np.abs(mid_hip_y - mid_sh_y)
    with np.errstate(invalid='ignore', divide='ignore'):
        torso_ratio = np.where(torso_h > 1e-6, sw / torso_h, np.nan)

    d_angle = np.concatenate([[0.0], np.diff(np.asarray(angle_2d, float))])

    cols = [
        np.asarray(angle_2d, float),
        np.asarray(angle_world, float),
        np.asarray(angle_lift, float),
        plan_up, plan_fo, np.minimum(plan_up, plan_fo),
        len_up, len_fo,
        np.full(n, L_up), np.full(n, L_fo),
        sw / h,
        lm[:, sh, 3], lm[:, el, 3], lm[:, wr, 3],
        wd[:, sh, 2] - wd[:, el, 2],
        wd[:, el, 2] - wd[:, wr, 2],
        torso_ratio,
        d_angle,
        _rolling_std(np.asarray(angle_2d, float), 15),
        np.full(n, 1.0 if side == 'right' else 0.0),
    ]
    return np.stack(cols, axis=1)


def overlap_rows(F, gt, lag):
    """Apply the same lag that benchmark_common.overlap applies, to a matrix.

    Keeps the feature rows paired with exactly the ground-truth samples the
    other variants are scored against, so the comparison stays fair.
    """
    F = np.asarray(F, float)
    gt = np.asarray(gt, float)
    n = len(gt)
    if lag >= 0:
        m = min(len(F), n - lag)
        return F[:m], gt[lag:lag + m]
    m = min(len(F) + lag, n)
    return F[-lag:-lag + m], gt[:m]
