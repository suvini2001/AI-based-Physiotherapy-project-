"""A7 - calibration screen. Countdown -> hold the arm straight for 3 s -> save data/user_calibration.json.
  python capture/calibrate_ui.py                      (2D angle, variant A)
  python capture/calibrate_ui.py --angle 3d           (variant B)
  python capture/calibrate_ui.py --with-bent          (bent arm recorded for information only)
  python capture/calibrate_ui.py --clinician-confirmed-straight   (sets the optional session offset)"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "capture", ROOT / "pose"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from angles import angle, ELBOW                                         # noqa: E402
from calibration import (identity_params, load_params, save_params, make_session_params,  # noqa: E402
                         apply_calibration)
from visibility import load_config, mask_low_visibility                 # noqa: E402
from angles_world import mask_world_like                                # noqa: E402

DEFAULTS = dict(countdown_seconds=3.0, hold_seconds=3.0, expected_straight_deg=175.0,
                straight_tolerance_deg=15.0, curve_file="data/population_curve.json",
                out_file="data/user_calibration.json")
PROMPTS = {"straight": "Hold your arm STRAIGHT",
           "bent": "Now BEND your elbow as far as is comfortable"}
VIEW_EDGES = (20.0, 50.0, 75.0)          # yaw degrees: front / half-profile / profile / side (a guess)
VARIANT_OF = {"2d": "A", "3d": "B"}


def get_cfg():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(load_config().get("calibration") or {})
    except FileNotFoundError:
        pass
    return cfg


def view_class_from_yaw(yaw):
    if yaw is None or not np.isfinite(yaw):
        return None
    for name, edge in zip(("front", "half-profile", "profile"), VIEW_EDGES):
        if yaw < edge:
            return name
    return "side"


class CalibrationSession:
    """One angle per frame in, what to show out. Each step = countdown (not recorded) then hold (recorded)."""

    def __init__(self, steps=("straight",), countdown_s=3.0, hold_s=3.0):
        self.steps, self.countdown_s, self.hold_s = list(steps), countdown_s, hold_s
        self.samples = {s: [] for s in self.steps}
        self.world_frames = {s: [] for s in self.steps}
        self.index, self.t0 = 0, None

    @property
    def done(self):
        return self.index >= len(self.steps)

    def update(self, now, angle_deg=None, world=None):
        if self.done:
            return dict(step=None, phase="done", remaining=0.0, prompt="Done")
        if self.t0 is None:
            self.t0 = now
        el, name = now - self.t0, self.steps[self.index]
        if el < self.countdown_s:
            return dict(step=name, phase="countdown", remaining=self.countdown_s - el, prompt=PROMPTS[name])
        if el < self.countdown_s + self.hold_s:
            if angle_deg is not None and np.isfinite(angle_deg):
                self.samples[name].append(float(angle_deg))
                if world is not None:
                    self.world_frames[name].append(np.asarray(world))
            return dict(step=name, phase="hold", remaining=self.countdown_s + self.hold_s - el, prompt=PROMPTS[name])
        self.index += 1
        self.t0 = None
        return self.update(now, angle_deg, world)


def finish(session, params, cfg, clinician_confirmed=False, yaw_fn=None, angle_mode="2d"):
    """yaw_fn = Person B's body_yaw (optional)."""
    yaw = None
    frames = session.world_frames.get("straight") or []
    if yaw_fn is not None and frames:
        try:
            value = float(np.nanmedian(yaw_fn(np.stack(frames))))
            yaw = value if np.isfinite(value) else None
        except Exception:
            yaw = None
    out = make_session_params(params, session.samples["straight"], cfg["expected_straight_deg"],
                              cfg["straight_tolerance_deg"], clinician_confirmed, view_class_from_yaw(yaw), yaw)
    out["angle_mode"] = angle_mode
    bent = session.samples.get("bent") or []
    if bent:                                   # information only: a bent arm is never an anchor
        out["bent_reading_info_only"] = float(apply_calibration(np.median(bent), {**params, "offset": None}))
    return out


def draw_prompt(frame, info, n_samples=0):
    import cv2
    from vision.overlay import put_big_text
    h, w = frame.shape[:2]
    put_big_text(frame, info["prompt"], (20, 70), 1.0, (255, 255, 255))
    if info["phase"] == "countdown":
        put_big_text(frame, f"{int(np.ceil(info['remaining']))}", (w // 2 - 30, h // 2 + 40), 4.0, (0, 200, 255))
    elif info["phase"] == "hold":
        put_big_text(frame, "HOLD STILL", (20, 130), 1.2, (0, 255, 0))
        cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (80, 80, 80), 2)
        cv2.rectangle(frame, (20, h - 40), (20 + int((w - 40) * min(n_samples / 60.0, 1.0)), h - 20), (0, 255, 0), -1)
    return frame


def draw_result(frame, params):
    from vision.overlay import put_big_text
    check = params["straight_check"]
    colour = {"ok": (0, 255, 0), "warn": (0, 165, 255), "fail": (0, 0, 255)}[check["status"]]
    put_big_text(frame, f"Calibration: {check['status'].upper()}", (20, 70), 1.2, colour)
    put_big_text(frame, check["message"][:48], (20, 120), 0.7, (255, 255, 255))
    put_big_text(frame, "saved", (20, 165), 0.9, (255, 255, 255))
    return frame


def frame_angle(norm, world, mode, size, side="right", thr=0.70):
    """norm, world: (33, 4) for ONE frame. Degrees or None."""
    masked = mask_low_visibility(norm[None], thr)[0]
    ids = ELBOW[side]
    if np.isnan(masked[list(ids), :3]).any():
        return None
    if mode == "3d":
        wm = mask_world_like(world[None], masked[None])[0]
        pts = [wm[i, :3] for i in ids]
    else:
        pts = [masked[i, :2] * np.array(size, float) for i in ids]
    if any(np.isnan(p).any() for p in pts):
        return None
    return float(angle(*pts))


def run(args):
    import cv2
    from capture.camera import FrameSource
    from pose.landmarks import LandmarkExtractor
    from vision.overlay import draw_skeleton
    try:
        from angles_3d import body_yaw             # Person B's function; the screen works without it
    except ImportError:
        body_yaw = None

    cfg = get_cfg()
    curve = Path(args.curve or cfg["curve_file"])
    if curve.exists():
        params = load_params(str(curve))
        fitted_on = params.get("variant")
        if fitted_on and fitted_on != VARIANT_OF[args.angle]:
            print(f"WARNING: this curve was fitted on variant {fitted_on} but you are using --angle {args.angle}.")
    else:
        params = identity_params()
        print(f"No curve at {curve}: using the identity curve (the screen works, the correction does nothing).")

    src = FrameSource(0 if args.source == "0" else args.source, args.width, args.height)
    size = src.size()
    extractor = LandmarkExtractor()
    thr = load_config()["visibility_threshold"]
    steps = ("straight", "bent") if args.with_bent else ("straight",)
    session = CalibrationSession(steps, cfg["countdown_seconds"], cfg["hold_seconds"])
    result = None

    while True:
        ok, frame = src.read_frame()
        if not ok:
            break
        norm, world = extractor.process_with_world(frame)
        ang = frame_angle(norm, world, args.angle, size, args.side, thr)
        info = session.update(time.perf_counter(), ang, world if ang is not None else None)
        if args.mirror:
            frame = cv2.flip(frame, 1)
        draw_skeleton(frame, norm, mirror=args.mirror, highlight=ELBOW[args.side])
        draw_prompt(frame, info, len(session.samples.get(info["step"], [])) if info["step"] else 0)
        cv2.imshow("calibration (press q to quit)", frame)
        if session.done:
            result = finish(session, params, cfg, args.clinician_confirmed_straight, body_yaw, args.angle)
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if result is not None:
        save_params(cfg["out_file"], result)
        load_params(cfg["out_file"])                       # reload check (acceptance gate)
        print(f"Saved {cfg['out_file']}: {result['straight_check']['status']} - {result['straight_check']['message']}")
        t_end = time.perf_counter() + 3
        while time.perf_counter() < t_end:
            ok, frame = src.read_frame()
            if not ok:
                break
            draw_result(cv2.flip(frame, 1) if args.mirror else frame, result)
            cv2.imshow("calibration (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        print("Calibration cancelled; nothing saved.")
    extractor.close()
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--angle", choices=["2d", "3d"], default="2d")
    ap.add_argument("--curve", default=None)
    ap.add_argument("--with-bent", action="store_true")
    ap.add_argument("--clinician-confirmed-straight", action="store_true")
    ap.add_argument("--mirror", action="store_true", help="display only")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    run(ap.parse_args())