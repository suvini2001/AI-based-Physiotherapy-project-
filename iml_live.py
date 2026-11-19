"""iml_live.py - I_ML (learned elbow-angle correction) for the live webcam pipeline.

The model was trained in 02_curve_vs_iml.ipynb on features built OFFLINE from a
whole recording (benchmark STEP 5 / notebook 01). Live, only the frames seen so
far exist, so this module rebuilds the SAME features over a rolling buffer of
the last `buffer` frames and keeps the last row:

  * same repo functions: mask_low_visibility, mask_world_like, bone_lengths,
    elbow_angle, elbow_angle_3d_ordered, elbow_angle_lifted, frame_features;
  * same causal 2nd-order 6 Hz Butterworth on the three angle signals, run as
    a stateful one-sample filter (identical to lfilter over the series);
  * session statistics (shoulder-width scale, bone lengths) come from the
    buffer instead of the whole recording.

The deployed model is `iml_nolook` (no std_angle / d_angle), so no centred
15-frame window is needed and the angle is not delayed. `validate()` replays a
recording both ways and reports how far the live features move the prediction.

`stream_features()` replays a recorded clip through the same FeatureStream, so
training data can be built exactly as the live system will see it (no
train/serve skew).
"""
import json
from collections import deque

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

import angles as angles_2d
from angles import elbow_angle_3d_ordered
from angles_lift import ARM, bone_lengths, elbow_angle_lifted, shoulder_width
from angles_world import mask_world_like
from visibility import mask_low_visibility
from features import frame_features, FEATURE_NAMES


class CausalButter:
    """One sample in, one sample out; same output as lfilter(b, a, series, zi=zi*x0)."""

    def __init__(self, fs, cutoff=6.0, order=2):
        self.b, self.a = butter(order, cutoff / (fs / 2.0), btype="low")
        self.z, self.last = None, None

    def __call__(self, x):
        missing = not np.isfinite(x)
        if missing:                       # training interpolated short gaps before filtering;
            if self.last is None:         # live we hold the last value to keep the state sane
                return np.nan
            x = self.last
        if self.z is None:
            self.z = lfilter_zi(self.b, self.a) * x
        y, self.z = lfilter(self.b, self.a, [x], zi=self.z)
        self.last = x
        return np.nan if missing else float(y[0])


def load_model(path):
    import lightgbm as lgb
    with open(path + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    unknown = [f for f in meta["features"] if f not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"model expects features this code does not build: {unknown}")
    return lgb.Booster(model_file=path), meta


IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


class FeatureStream:
    """The 20 training features, built ONE FRAME AT A TIME from past frames only.

    This is the single place where live features are made: the webcam path
    (IMLStream) and the training replay (stream_features) both use it, so the
    model is trained on exactly what it is later served.

    session  'window'    bone lengths and scale from the last `buffer` frames
                         (the original version; frames older than ~10 s are forgotten)
             'expanding' bone lengths and scale from EVERY frame since the session
                         started (only a few numbers per frame are kept, so it stays fast);
                         no calibration pose is needed, which suits injured patients
    scale    'shoulder'  lengths in shoulder widths (the original unit)
             'torso'     lengths in torso lengths, mid-shoulder to mid-hip, as in
                         pose_normalise.py; falls back to shoulder width until the hips
                         have been seen in `min_frames` frames (e.g. sitting close to a laptop)
    """

    def __init__(self, side="right", size=(640, 480), fps=30.0, threshold=0.70, buffer=300,
                 session="window", scale="shoulder", min_frames=30, pct=95.0):
        if session not in ("window", "expanding") or scale not in ("shoulder", "torso"):
            raise ValueError(f"unknown session={session!r} / scale={scale!r}")
        if session == "window" and scale != "shoulder":
            raise ValueError("scale='torso' needs session='expanding'")
        self.side, self.size, self.thr = side, tuple(size), threshold
        self.session, self.scale, self.min_frames, self.pct = session, scale, min_frames, pct
        self.lm, self.wd = deque(maxlen=buffer), deque(maxlen=buffer)
        self.a2, self.aw, self.al = (deque(maxlen=buffer) for _ in range(3))
        self.f2, self.fw, self.fl = CausalButter(fps), CausalButter(fps), CausalButter(fps)
        # session history: four numbers per frame, in pixels, kept for the whole session
        self.h_up, self.h_fo, self.h_sw, self.h_torso = [], [], [], []
        self.scale_used = None              # 'torso' or 'shoulder' for the last frame

    def __len__(self):
        return len(self.lm)

    def _measure(self, m):
        """upper arm, forearm, shoulder width and torso length of one frame, in pixels"""
        w, h = self.size
        sh, el, wr = ARM[self.side]
        px = lambda a, b: float(np.hypot((m[a, 0] - m[b, 0]) * w, (m[a, 1] - m[b, 1]) * h))
        mid_sh = (m[L_SH, :2] + m[R_SH, :2]) / 2
        mid_hip = (m[L_HIP, :2] + m[R_HIP, :2]) / 2
        torso = float(np.hypot((mid_sh[0] - mid_hip[0]) * w, (mid_sh[1] - mid_hip[1]) * h))
        return px(sh, el), px(wr, el), px(L_SH, R_SH), torso

    def _session_stats(self):
        """true bone lengths (pixels) = high percentile of everything seen so far; scale = median"""
        def pct(h):
            v = np.asarray(h, float); v = v[np.isfinite(v) & (v > 0)]
            return float(np.percentile(v, self.pct)) if len(v) >= self.min_frames else np.nan
        def med(h, need):
            v = np.asarray(h, float); v = v[np.isfinite(v) & (v > 0)]
            return float(np.median(v)) if len(v) >= need else np.nan
        Lup, Lfo = pct(self.h_up), pct(self.h_fo)
        sc, used = np.nan, None
        if self.scale == "torso":
            sc = med(self.h_torso, self.min_frames); used = "torso"
        if not np.isfinite(sc):
            sc = med(self.h_sw, 1); used = "shoulder"
        return Lup, Lfo, sc, used

    def push(self, norm, world):
        m = mask_low_visibility(np.asarray(norm, float)[None], self.thr)[0]
        w = mask_world_like(np.asarray(world, float)[None], m[None])[0]
        self.lm.append(m); self.wd.append(w)
        L, W = np.stack(self.lm), np.stack(self.wd)
        if self.session == "expanding":
            up, fo, sw, torso = self._measure(m)
            for hist, v in ((self.h_up, up), (self.h_fo, fo), (self.h_sw, sw), (self.h_torso, torso)):
                hist.append(v)
            Lup_px, Lfo_px, sc, self.scale_used = self._session_stats()
            # the lift and frame_features measure lengths in the buffer's median shoulder
            # width; hand them the session bone lengths in that same unit so the scale
            # cancels and only the session estimate of the true length matters
            sw_buf = float(np.nanmedian(shoulder_width(L, self.size)))
            L_up, L_fo = ((Lup_px / sw_buf, Lfo_px / sw_buf) if np.isfinite(sw_buf) and sw_buf > 0
                          else (np.nan, np.nan))
        else:
            L_up, L_fo = bone_lengths(L, self.side, self.size)
        a2 = float(angles_2d.elbow_angle(L[-1:], side=self.side, size=self.size)[0])
        aw = float(elbow_angle_3d_ordered(W[-1:], side=self.side)[0])
        al = float(elbow_angle_lifted(L, W, self.side, self.size, L_up, L_fo)[-1])
        self.a2.append(self.f2(a2)); self.aw.append(self.fw(aw)); self.al.append(self.fl(al))
        row = frame_features(L, W, self.side, self.size, L_up, L_fo,
                             np.array(self.a2), np.array(self.aw), np.array(self.al))[-1]
        if self.session == "expanding":
            # length features from the session estimates, in the chosen unit
            with np.errstate(invalid="ignore", divide="ignore"):
                row[IDX["len_upper"]], row[IDX["len_fore"]] = up / sc, fo / sc
                row[IDX["bone_upper"]], row[IDX["bone_fore"]] = Lup_px / sc, Lfo_px / sc
                row[IDX["plan_upper"]], row[IDX["plan_fore"]] = up / Lup_px, fo / Lfo_px
            row[IDX["plan_min"]] = np.fmin(row[IDX["plan_upper"]], row[IDX["plan_fore"]])
        return row


def stream_features(lm, world, side, size, fps, threshold=0.70, buffer=300,
                    session="window", scale="shoulder"):
    """Replay a recorded clip frame by frame, exactly as the webcam would deliver it.
    Returns (n_frames, 20): row i uses frames 0..i only."""
    fs = FeatureStream(side, size, fps, threshold, buffer, session=session, scale=scale)
    return np.stack([fs.push(n, w) for n, w in zip(lm, world)]).astype(np.float32)


class IMLStream:
    """norm, world (33, 4) per frame  ->  corrected elbow angle in degrees, or None."""

    def __init__(self, model_path, side="right", size=(640, 480), fps=30.0,
                 threshold=0.70, buffer=300, min_frames=30):
        self.booster, self.meta = load_model(model_path)
        self.cols = [FEATURE_NAMES.index(f) for f in self.meta["features"]]
        self.min_frames = min_frames
        mode = self.meta.get("feature_stream", {})          # models trained before Phase 4A: window
        self.fs = FeatureStream(side, size, fps, threshold, buffer,
                                session=mode.get("session", "window"),
                                scale=mode.get("scale", "shoulder"))
        self.last_features = None

    @property
    def ready(self):
        return len(self.fs) >= self.min_frames

    def step(self, norm, world):
        row = self.fs.push(norm, world)
        self.last_features = row
        if not self.ready or not np.isfinite(row[0]):     # training rows always had angle_2d
            return None
        pred = self.booster.predict(row[self.cols][None], num_threads=1)[0]
        return float(min(max(pred, 0.0), 180.0))


# ---------------------------------------------------------------- offline twin
def fill_1d(arr, max_gap=5):
    """Copied from the benchmark (STEP 5) / notebook 01."""
    arr = np.array(arr, dtype=float)
    if np.isnan(arr).all() or not np.isnan(arr).any():
        return arr
    idx, good, filled = np.arange(len(arr)), ~np.isnan(arr), arr.copy()
    in_gap, gap_start = False, 0
    for i in range(len(arr)):
        if np.isnan(arr[i]) and not in_gap:
            in_gap, gap_start = True, i
        elif not np.isnan(arr[i]) and in_gap:
            in_gap = False
            if i - gap_start <= max_gap:
                filled[gap_start:i] = np.interp(idx[gap_start:i], idx[good], arr[good])
    return filled


def offline_features(lm, world, side, size, fps, threshold=0.70):
    """The training-time features for a whole recording (notebook 01, section 3)."""
    import benchmark_common as bc

    def smooth(x):
        x = fill_1d(x, 5)
        return bc.butterworth_filter(x, fs=fps, cutoff=6.0, causal=True) if np.isfinite(x).sum() > 15 else x

    lm_m = mask_low_visibility(np.asarray(lm, float), threshold=threshold)
    wd_m = mask_world_like(np.asarray(world, float), lm_m)
    L_up, L_fo = bone_lengths(lm_m, side, size)
    pa = smooth(angles_2d.elbow_angle(lm_m, side=side, size=size))
    pb = smooth(elbow_angle_3d_ordered(wd_m, side=side))
    pe = smooth(elbow_angle_lifted(lm_m, wd_m, side, size, L_up, L_fo))
    return frame_features(lm_m, wd_m, side, size, L_up, L_fo, pa, pb, pe)


def validate(model_path, lm, world, side, size, fps, skip=30):
    """Replay one recording live-style and offline-style; compare the model's outputs."""
    live = IMLStream(model_path, side, size, fps)
    p_live = np.array([np.nan if (v := live.step(n, w)) is None else v for n, w in zip(lm, world)])
    F = offline_features(lm, world, side, size, fps)
    p_off = np.clip(live.booster.predict(F[:, live.cols]), 0, 180)
    p_off[~np.isfinite(F[:, 0])] = np.nan
    ok = np.isfinite(p_live) & np.isfinite(p_off)
    ok[:skip] = False
    d = np.abs(p_live[ok] - p_off[ok])
    return dict(frames=int(ok.sum()), mean_abs_diff=float(d.mean()) if len(d) else np.nan,
                p95_abs_diff=float(np.percentile(d, 95)) if len(d) else np.nan,
                live_coverage=float(np.isfinite(p_live[skip:]).mean()),
                p_live=p_live, p_off=p_off)
