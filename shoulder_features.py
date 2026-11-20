"""Shoulder elevation: truth definition, live angle and streaming I_ML features.

Why a separate module
---------------------
The elbow model (features.py / iml_live.py) learned how the FOREARM looks when it
points at the camera. The shoulder has different distortions: the UPPER ARM coming
toward the camera, the trunk leaning, and front vs side views. So the shoulder gets
its own features and its own model, built with the same rules that fixed the elbow:

* one implementation, frame by frame (ShoulderFeatureStream). Training data is made
  by replaying recordings through it (stream_shoulder_features), so the model is
  trained on exactly what the live system computes: no train/serve skew;
* only past frames: causal 6 Hz Butterworth, session statistics from every frame
  seen so far (expanding), lengths in torso lengths (shoulder width as fallback
  until the hips have been seen), no calibration pose;
* no look-ahead feature, so the live angle is not delayed.

Truth (both datasets, same definition)
--------------------------------------
Shoulder elevation = 3D angle between the trunk axis (mid-shoulder -> mid-hip) and
the upper arm (shoulder -> elbow). 0 deg = arm hanging down, 90 deg = horizontal,
180 deg = straight overhead. It does not care whether the arm goes out to the side
(abduction), forward (flexion) or in between (scaption), so one model can serve
front and side views. Using the two hip joints' midpoint (not one dataset's
"pelvis" joint) keeps the REHAB24-6 and Fit3D skeletons comparable.
"""
import numpy as np

from angles import angle, joint_angle
from angles_world import mask_world_like
from visibility import mask_low_visibility

L_SH, R_SH, L_EL, R_EL, L_HIP, R_HIP = 11, 12, 13, 14, 23, 24
SIDE = {   # (shoulder, elbow, same-side hip, other shoulder)
    "right": (R_SH, R_EL, R_HIP, L_SH),
    "left":  (L_SH, L_EL, L_HIP, R_SH),
}
LIVE_JOINT = {"right": (R_HIP, R_SH, R_EL), "left": (L_HIP, L_SH, L_EL)}   # vision/overlay.JOINTS

SH_FEATURE_NAMES = [
    "angle_2d",      # the angle the live system uses today: same-side hip - shoulder - elbow, 2D
    "elev_2d",       # trunk axis (mid-shoulder -> mid-hip) vs upper arm, in the image
    "elev_world",    # same, from MediaPipe world landmarks (3D)
    "elev_lift",     # upper arm lifted to 3D with the session bone length (sign of depth from world z)
    "plan_upper",    # visible upper-arm length / session true length (1 = in the image plane)
    "len_upper",     # visible upper-arm length, in scale units (torso or shoulder width)
    "torso_ratio",   # shoulder width / torso length: ~0.7 facing the camera, small side-on
    "trunk_tilt",    # trunk axis from image vertical, degrees (signed)
    "el_out",        # elbow sideways from the shoulder, away from the body = positive (scale units)
    "el_up",         # elbow above the shoulder = positive (scale units)
    "wz_sh_el",      # world z, shoulder minus elbow: upper arm toward / away from the camera
    "wz_trunk",      # world z, mid-shoulder minus mid-hip: trunk leaning toward the camera
    "vis_sh", "vis_el", "vis_hip",
]
IDX = {n: i for i, n in enumerate(SH_FEATURE_NAMES)}


# ------------------------------------------------------------------ truth
def elevation(mid_hip, mid_sh, sh, el):
    """Angle between the trunk axis (mid_sh -> mid_hip) and the upper arm (sh -> el).
    Arrays with the last axis = x, y[, z]; NaN in -> NaN out."""
    mid_hip, mid_sh, sh, el = (np.asarray(v, float) for v in (mid_hip, mid_sh, sh, el))
    # shift the trunk vector to start at the shoulder, then use angles.angle
    return angle(sh + (mid_hip - mid_sh), sh, el)


def truth_elevation(J, side, sh_idx, el_idx, hip_idx):
    """Mocap truth. J: (frames, joints, 3). sh_idx / el_idx / hip_idx: dicts side -> joint index."""
    J = np.asarray(J, float)
    mid_sh = (J[:, sh_idx["left"]] + J[:, sh_idx["right"]]) / 2
    mid_hip = (J[:, hip_idx["left"]] + J[:, hip_idx["right"]]) / 2
    return elevation(mid_hip, mid_sh, J[:, sh_idx[side]], J[:, el_idx[side]])


def live_angle_2d(landmarks, side, size):
    """What ShoulderFSM receives today (variant A): hip - shoulder - elbow in pixels."""
    return joint_angle(np.asarray(landmarks, float), LIVE_JOINT[side], size)


# ------------------------------------------------------------------ streaming
class _CausalButter:
    """One sample in, one out; identical to iml_live.CausalButter (kept local so this
    module does not need lightgbm)."""

    def __init__(self, fs, cutoff=6.0, order=2):
        from scipy.signal import butter
        self.b, self.a = butter(order, cutoff / (fs / 2.0), btype="low")
        self.z, self.last = None, None

    def __call__(self, x):
        from scipy.signal import lfilter, lfilter_zi
        missing = not np.isfinite(x)
        if missing:
            if self.last is None:
                return np.nan
            x = self.last
        if self.z is None:
            self.z = lfilter_zi(self.b, self.a) * x
        y, self.z = lfilter(self.b, self.a, [x], zi=self.z)
        self.last = x
        return np.nan if missing else float(y[0])


def _finite(h):
    v = np.asarray(h, float)
    return v[np.isfinite(v) & (v > 0)]


class ShoulderFeatureStream:
    """push(norm, world) per frame -> the SH_FEATURE_NAMES row for that frame.

    Only frames already pushed are used. Session statistics (true upper-arm length,
    scale) come from every frame since the start, so nothing is forgotten in a long
    session and no calibration pose is needed.
    """

    def __init__(self, side="right", size=(640, 480), fps=30.0, threshold=0.70,
                 min_frames=30, pct=95.0):
        if side not in SIDE:
            raise ValueError(f"side must be 'right' or 'left', not {side!r}")
        self.side, self.size, self.thr = side, tuple(size), threshold
        self.min_frames, self.pct = min_frames, pct
        self.filters = {k: _CausalButter(fps) for k in ("angle_2d", "elev_2d", "elev_world", "elev_lift")}
        self.h_up, self.h_sw, self.h_torso = [], [], []
        self.n = 0
        self.scale_used = None

    def __len__(self):
        return self.n

    def _scale(self):
        t = _finite(self.h_torso)
        if len(t) >= self.min_frames:
            return float(np.median(t)), "torso"
        s = _finite(self.h_sw)
        return (float(np.median(s)), "shoulder") if len(s) else (np.nan, None)

    def push(self, norm, world):
        self.n += 1
        m = mask_low_visibility(np.asarray(norm, float)[None], self.thr)[0]
        wd = mask_world_like(np.asarray(world, float)[None], m[None])[0]
        w, h = self.size
        sh, el, hip, osh = SIDE[self.side]
        P = m[:, :2] * np.array([w, h], float)                 # pixels
        mid_sh, mid_hip = (P[L_SH] + P[R_SH]) / 2, (P[L_HIP] + P[R_HIP]) / 2
        up_vec = P[el] - P[sh]
        trunk = mid_hip - mid_sh
        up_px, sw_px = float(np.hypot(*up_vec)), float(np.hypot(*(P[L_SH] - P[R_SH])))
        torso_px = float(np.hypot(*trunk))
        self.h_up.append(up_px); self.h_sw.append(sw_px); self.h_torso.append(torso_px)

        sc, self.scale_used = self._scale()
        ups = _finite(self.h_up)
        L_up = float(np.percentile(ups, self.pct)) if len(ups) >= self.min_frames else np.nan

        a2d = float(angle(P[hip], P[sh], P[el]))
        e2d = float(elevation(mid_hip, mid_sh, P[sh], P[el]))
        W = wd[:, :3]
        w_mid_sh, w_mid_hip = (W[L_SH] + W[R_SH]) / 2, (W[L_HIP] + W[R_HIP]) / 2
        ew = float(elevation(w_mid_hip, w_mid_sh, W[sh], W[el]))

        # lift: |dz| from the session bone length, its sign from MediaPipe world z
        el_lift = np.nan
        if np.isfinite(L_up) and np.all(np.isfinite(up_vec)) and np.all(np.isfinite(trunk)):
            dz = np.sqrt(max(L_up ** 2 - up_px ** 2, 0.0))
            s = np.sign(W[el, 2] - W[sh, 2]) if np.isfinite(W[el, 2] - W[sh, 2]) else 1.0
            u3 = np.array([up_vec[0], up_vec[1], (s or 1.0) * dz])
            t3 = np.array([trunk[0], trunk[1], 0.0])
            el_lift = float(angle(np.zeros(3) + t3, np.zeros(3), u3))

        with np.errstate(invalid="ignore", divide="ignore"):
            out_sign = np.sign(P[sh, 0] - P[osh, 0])
            row = {
                "angle_2d": self.filters["angle_2d"](a2d),
                "elev_2d": self.filters["elev_2d"](e2d),
                "elev_world": self.filters["elev_world"](ew),
                "elev_lift": self.filters["elev_lift"](el_lift),
                "plan_upper": up_px / L_up if np.isfinite(L_up) and L_up > 0 else np.nan,
                "len_upper": up_px / sc,
                "torso_ratio": sw_px / torso_px if torso_px > 1e-6 else np.nan,
                "trunk_tilt": float(np.degrees(np.arctan2(trunk[0], trunk[1]))),
                "el_out": up_vec[0] * out_sign / sc,
                "el_up": -up_vec[1] / sc,
                "wz_sh_el": W[sh, 2] - W[el, 2],
                "wz_trunk": w_mid_sh[2] - w_mid_hip[2],
                "vis_sh": m[sh, 3], "vis_el": m[el, 3], "vis_hip": m[hip, 3],
            }
        return np.array([row[k] for k in SH_FEATURE_NAMES], float)


def stream_shoulder_features(lm, world, side, size, fps, threshold=0.70):
    """Replay a recorded clip frame by frame, exactly as the webcam would deliver it.
    Returns (n_frames, len(SH_FEATURE_NAMES)); row i uses frames 0..i only."""
    fs = ShoulderFeatureStream(side, size, fps, threshold)
    return np.stack([fs.push(n, w) for n, w in zip(lm, world)]).astype(np.float32)
