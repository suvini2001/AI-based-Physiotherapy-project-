"""demo_cv.py - webcam or clip -> landmarks -> clean -> angle -> overlay.
  python demo_cv.py --exercise elbow --side right          (live webcam, variant A = old 2D angle)
  python demo_cv.py --variant B                             (3D angle from world landmarks)
  python demo_cv.py --calibration data/user_calibration.json
  python demo_cv.py --source data/elbow_A_laptop.mp4        (replay a clip)
  python demo_cv.py --no-overlay                            (FPS without drawing)
Commit 1 (Person A): the loop, variants A and B, calibration. Commit 2 (Person B): yaw message, variant C, gate."""
import argparse, sys, time
from collections import deque
import cv2
import numpy as np

sys.path.append("capture")
sys.path.append("pose")
sys.path.append("vision")
from camera import FrameSource, FpsCounter
from landmarks import LandmarkExtractor
from visibility import mask_low_visibility, load_config
from filters import OneEuroFilter
from angles import angle
from angles_world import mask_world_like
from calibration import load_params, apply_calibration
from gap_fill import fill_gaps
from vision.overlay import JOINTS, draw_angle, draw_skeleton, put_big_text
from exercise_state import ElbowFSM, ShoulderFSM, save_session


WINDOW = 15            # recent frames kept for gap filling (about 0.5 s)
SPIKE_WINDOW = 5       # median of the last 5 angles removes one-frame spikes
VARIANTS = ["A", "B", "C"]  # Person B adds "C" in the second commit


class AnglePipeline:
    """One frame in, one cleaned angle out. Separate from the window so it can be tested without a camera."""

    def __init__(self, joint, variant, size, fps, thr, max_gap, calibration=None):
        self.joint, self.variant, self.size = joint, variant, size
        self.thr, self.max_gap, self.calibration = thr, max_gap, calibration
        self.win_norm, self.win_world = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
        self.recent = deque(maxlen=SPIKE_WINDOW)
        self.smoother = OneEuroFilter(fs=fps)

    def raw_angle(self, clean_norm, clean_world):
        pts_n = [clean_norm[i] for i in self.joint]
        if any(np.isnan(p[:3]).any() for p in pts_n):
            return None
        if self.variant == "A":                               # old way: 2D pixels
            px = [np.array([p[0] * self.size[0], p[1] * self.size[1]]) for p in pts_n]
            return float(angle(*px))
        if self.variant == "B":                               # 3D world landmarks, metres
            pts_w = [clean_world[i, :3] for i in self.joint]
            if any(np.isnan(p).any() for p in pts_w):
                return None
            return float(angle(*pts_w))
        # --- Person B plugs variant C (yaw-corrected) here, in the second commit ---
        if self.variant == "C":
            from angles_3d import elbow_angle_yaw_corrected
            return float(elbow_angle_yaw_corrected(world[None], self.side)[0]) \
                if not np.isnan(clean_world).any() else None
        raise ValueError(f"unknown variant {self.variant}")
    
    def step(self, norm, world):
        """norm, world: (33, 4) for one frame. Returns (clean_norm, angle_or_None)."""
        masked = mask_low_visibility(norm[None], self.thr)[0]
        self.win_norm.append(masked)
        self.win_world.append(mask_world_like(world[None], masked[None])[0])
        clean = fill_gaps(np.stack(self.win_norm), self.max_gap)[-1]
        clean_w = fill_gaps(np.stack(self.win_world), self.max_gap)[-1]
        raw = self.raw_angle(clean, clean_w)
        # --- Person B plugs the bone-length / yaw gate here: invalid frame -> raw = None ---
        # bone-length / yaw gate (variant E)
        if raw is not None and self.variant in ("C", "E"):
            from angles_3d import bone_length_gate
            world_frame = np.stack(self.win_world)[-1]
            gate_ok = bone_length_gate(world_frame[None], self.side)[0]
            if not gate_ok:
                raw = None

        if raw is None:
            return clean, None
        self.recent.append(raw)
        ang = self.smoother(float(np.median(self.recent)))
        ang = min(max(ang, 0.0), 180.0)
        if self.calibration is not None:
            ang = float(apply_calibration(ang, self.calibration))
        return clean, ang


"""demo_cv.py - webcam or clip -> landmarks -> clean -> angle -> overlay.
  python demo_cv.py --exercise elbow --side right          (live webcam, variant A = old 2D angle)
  python demo_cv.py --variant B                             (3D angle from world landmarks)
  python demo_cv.py --calibration data/user_calibration.json
  python demo_cv.py --source data/elbow_A_laptop.mp4        (replay a clip)
  python demo_cv.py --no-overlay                            (FPS without drawing)
Commit 1 (Person A): the loop, variants A and B, calibration. Commit 2 (Person B): yaw message, variant C, gate."""
import argparse, sys, time
from collections import deque
import cv2
import numpy as np

sys.path.append("capture")
sys.path.append("pose")
sys.path.append("vision")
from camera import FrameSource, FpsCounter
from landmarks import LandmarkExtractor
from visibility import mask_low_visibility, load_config
from filters import OneEuroFilter
from angles import angle
from angles_world import mask_world_like
from calibration import load_params, apply_calibration
from gap_fill import fill_gaps
from vision.overlay import JOINTS, draw_angle, draw_skeleton, put_big_text


WINDOW = 15            # recent frames kept for gap filling (about 0.5 s)
SPIKE_WINDOW = 5       # median of the last 5 angles removes one-frame spikes
VARIANTS = ["A", "B", "C"]  # Person B adds "C" in the second commit


class AnglePipeline:
    """One frame in, one cleaned angle out. Separate from the window so it can be tested without a camera."""

    def __init__(self, joint, variant, size, fps, thr, max_gap, calibration=None):
        self.joint, self.variant, self.size = joint, variant, size
        self.thr, self.max_gap, self.calibration = thr, max_gap, calibration
        self.win_norm, self.win_world = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
        self.recent = deque(maxlen=SPIKE_WINDOW)
        self.smoother = OneEuroFilter(fs=fps)

    def raw_angle(self, clean_norm, clean_world):
        pts_n = [clean_norm[i] for i in self.joint]
        if any(np.isnan(p[:3]).any() for p in pts_n):
            return None
        if self.variant == "A":                               # old way: 2D pixels
            px = [np.array([p[0] * self.size[0], p[1] * self.size[1]]) for p in pts_n]
            return float(angle(*px))
        if self.variant == "B":                               # 3D world landmarks, metres
            pts_w = [clean_world[i, :3] for i in self.joint]
            if any(np.isnan(p).any() for p in pts_w):
                return None
            return float(angle(*pts_w))
        # --- Person B plugs variant C (yaw-corrected) here, in the second commit ---
        if self.variant == "C":
            from angles_3d import elbow_angle_yaw_corrected
            return float(elbow_angle_yaw_corrected(world[None], self.side)[0]) \
                if not np.isnan(clean_world).any() else None
        raise ValueError(f"unknown variant {self.variant}")
    
    def step(self, norm, world):
        """norm, world: (33, 4) for one frame. Returns (clean_norm, angle_or_None)."""
        masked = mask_low_visibility(norm[None], self.thr)[0]
        self.win_norm.append(masked)
        self.win_world.append(mask_world_like(world[None], masked[None])[0])
        clean = fill_gaps(np.stack(self.win_norm), self.max_gap)[-1]
        clean_w = fill_gaps(np.stack(self.win_world), self.max_gap)[-1]
        raw = self.raw_angle(clean, clean_w)
        # --- Person B plugs the bone-length / yaw gate here: invalid frame -> raw = None ---
        # bone-length / yaw gate (variant E)
        if raw is not None and self.variant in ("C", "E"):
            from angles_3d import bone_length_gate
            world_frame = np.stack(self.win_world)[-1]
            gate_ok = bone_length_gate(world_frame[None], self.side)[0]
            if not gate_ok:
                raw = None

        if raw is None:
            return clean, None
        self.recent.append(raw)
        ang = self.smoother(float(np.median(self.recent)))
        ang = min(max(ang, 0.0), 180.0)
        if self.calibration is not None:
            ang = float(apply_calibration(ang, self.calibration))
        return clean, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--exercise", choices=JOINTS, default="elbow")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--variant", choices=VARIANTS, default="A")
    ap.add_argument("--calibration", default=None, help="path to data/user_calibration.json")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--mirror", action="store_true", help="display only")
    ap.add_argument("--no-overlay", action="store_true", help="for the FPS comparison")
    args = ap.parse_args()

    cfg = load_config()
    thr, max_gap = cfg["visibility_threshold"], cfg["max_gap_frames"]
    calibration = load_params(args.calibration) if args.calibration else None

    source = int(args.source) if args.source.isdigit() else args.source
    src = FrameSource(source, args.width, args.height)
    size = src.size()
    extractor = LandmarkExtractor()
    joint = JOINTS[args.exercise][args.side]
    pipe = AnglePipeline(joint, args.variant, size, src.fps(), thr, max_gap, calibration)
    # ── FSM setup ────────────────────────────────────────────────────────
    t_start = time.perf_counter()

    def on_rep_complete(rep):
        """Called by FSM each time a rep finishes. Print to console for now."""
        print(f"  Rep {rep['rep_number']:2d} | "
            f"ROM {rep['rom']:.0f}°  "
            f"(max {rep['max_angle']:.0f}° → min {rep['min_angle']:.0f}°)  "
            f"dur {rep['duration_s']:.1f}s")

    if args.exercise == "shoulder":
        fsm = ShoulderFSM(rep_callback=on_rep_complete)
    else:
        fsm = ElbowFSM(rep_callback=on_rep_complete)
    counter = FpsCounter()
    stage = dict(read=0.0, pose=0.0, clean=0.0, draw=0.0)
    n, t_start = 0, time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = src.read_frame()
        if not ok:
            break
        t1 = time.perf_counter()
        norm, world = extractor.process_with_world(frame)   # BEFORE any flip
        t2 = time.perf_counter()
        clean, ang = pipe.step(norm, world)
        # ── FSM update ───────────────────────────────────────────────────────
        timestamp = time.perf_counter() - t_start
        state, rep_done = fsm.update(ang, timestamp)

        # ── overlay: show state and rep count ────────────────────────────────
        if not args.no_overlay:
            put_big_text(frame, f"Reps: {fsm.rep_count}", (20, 140), 0.9,
                        (0, 255, 0))
            put_big_text(frame, f"State: {state}", (20, 170), 0.7,
                        (255, 200, 0))
            if ang is not None:
                put_big_text(frame, f"Angle: {ang:.0f}°", (20, 200), 0.8,
                            (255, 255, 255))
        t3 = time.perf_counter()

        fps = counter.tick()
        if not args.no_overlay:
            if args.mirror:
                frame = cv2.flip(frame, 1)               # flip only for display
            draw_skeleton(frame, clean, mirror=args.mirror)
            draw_angle(frame, clean, joint, ang, mirror=args.mirror)
            label = f"variant {args.variant}" + (" + calibrated" if calibration else "")
            put_big_text(frame, label, (20, 100), 0.8, (255, 255, 255))
            # yaw warning (B7)
            if args.variant in ("B", "C"):
                try:
                    from angles_3d import body_yaw
                    from capture.view_check import draw_yaw_message, draw_yaw_bar
                    yaw = body_yaw(world)
                    draw_yaw_message(frame, yaw)
                    draw_yaw_bar(frame, yaw)
                except ImportError:
                    pass
        put_big_text(frame, f"FPS {fps:.0f}", (20, 60), 1.4, (255, 255, 0))
        t4 = time.perf_counter()

        cv2.imshow("demo_cv (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        for k, v in zip(stage, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            stage[k] += v
        n += 1

    # flush any in-progress rep when Q is pressed
    fsm.flush(time.perf_counter() - t_start)

    if fsm.rep_count > 0:
        save_session(fsm, args.exercise)
        print("\nSession summary:", fsm.summary())

    extractor.close()
    src.release()
    cv2.destroyAllWindows()
    if n:
        print(f"{n} frames, mean FPS {n / (time.perf_counter() - t_start):.1f}")
        print({k: f"{1000 * v / n:.1f} ms/frame" for k, v in stage.items()})


if __name__ == "__main__":
    main()