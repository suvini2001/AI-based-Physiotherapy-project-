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
import assessment.scorer as scorer
import audio_feedback
from vision.overlay import draw_feedback, draw_session_summary

from capture.camera import FrameSource, FpsCounter
from pose.landmarks import LandmarkExtractor
from visibility import mask_low_visibility, load_config
from filters import OneEuroFilter
from angles import angle
from angles_world import mask_world_like
from calibration import load_params, apply_calibration
from vision.gap_fill import fill_gaps
from vision.overlay import JOINTS, draw_angle, draw_skeleton, put_big_text
from exercise_state import ElbowFSM, ShoulderFSM, save_session


WINDOW = 15            # recent frames kept for gap filling (about 0.5 s)
SPIKE_WINDOW = 5       # median of the last 5 angles removes one-frame spikes
VARIANTS = ["A", "B", "C"]  # Person B adds "C" in the second commit


class AnglePipeline:
    """One frame in, one cleaned angle out. Separate from the window so it can be tested without a camera."""

    def __init__(self, joint, variant, size, fps, thr, max_gap, calibration=None, side="right"):
        self.joint, self.variant, self.size, self.side = joint, variant, size, side
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
            arm_pts = [11, 12, 14, 16] if self.side == "right" else [11, 12, 13, 15]
            if np.isnan(clean_world[arm_pts]).any():
                return None
            res = elbow_angle_yaw_corrected(clean_world[None], self.side)
            val = float(res[0]) if hasattr(res, "__len__") else float(res)
            return val if np.isfinite(val) else None
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
            gate_ok = bone_length_gate(world_frame[None], self.side)
            gate_val = gate_ok[0] if hasattr(gate_ok, "__len__") else gate_ok
            if not gate_val:
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
    ap.add_argument("--source",      default="0")
    ap.add_argument("--exercise",    choices=list(JOINTS.keys()), default="elbow")
    ap.add_argument("--side",        choices=["left", "right"],   default="right")
    ap.add_argument("--variant",     choices=VARIANTS,            default="A")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--width",       type=int, default=640)
    ap.add_argument("--height",      type=int, default=480)
    ap.add_argument("--mirror",      action="store_true")
    ap.add_argument("--no-overlay",  action="store_true")
    args = ap.parse_args()

    cfg          = load_config()
    thr, max_gap = cfg["visibility_threshold"], cfg["max_gap_frames"]
    calibration  = load_params(args.calibration) if args.calibration else None
    source       = int(args.source) if args.source.isdigit() else args.source
    src          = FrameSource(source, args.width, args.height)
    size         = src.size()
    extractor    = LandmarkExtractor()
    joint        = JOINTS[args.exercise][args.side]
    pipe         = AnglePipeline(joint, args.variant, size,
                                 src.fps(), thr, max_gap, calibration, side=args.side)

    # ── scorer ──────────────────────────────────────────────────────
    scorer.load('template.json')
    frame_width, frame_height = size      # src.size() → (width, height)
    rom_params = {"kind": "flex", "target": 55.0}

    # ── feedback display state ──────────────────────────────────────
    _feedback_rep   = None
    _feedback_timer = 0.0

    # ── rep callback ────────────────────────────────────────────────
    def on_rep_complete(rep):
        nonlocal _feedback_rep, _feedback_timer, rom_params
        rom_params = fsm.rom_check_params()
        scorer.score_rep(rep,
                         frame_height=frame_height,
                         rom_params=rom_params,
                         frame_width=frame_width)
        print(f"  Rep {rep['rep_number']:2d} | "
              f"ROM {rep['rom']:.0f}°  "
              f"(max {rep['max_angle']:.0f}° → min {rep['min_angle']:.0f}°)  "
              f"dur {rep['duration_s']:.1f}s  "
              f"Q={rep['quality_score']:.0f}%  "
              f"faults={rep['faults']}")
        audio_feedback.speak(f"Rep {rep['rep_number']}")
        if rep['quality_score'] < 70 and rep['faults']:
            audio_feedback.speak(
                rep['faults'][0].split('—')[-1].strip(),
                priority=True)
        elif rep['quality_score'] >= 85:
            audio_feedback.speak("Good", priority=False)
        _feedback_rep   = rep
        _feedback_timer = time.perf_counter() + 2.0

    # ── FSM ─────────────────────────────────────────────────────────
    if args.exercise == "shoulder":
        fsm = ShoulderFSM(rep_callback=on_rep_complete)
    else:
        fsm = ElbowFSM(rep_callback=on_rep_complete)
    rom_params = fsm.rom_check_params()

    # ── session timing ───────────────────────────────────────────────
    # Phase 1: READY    0  → READY_S        stand still, get in position
    # Phase 2: CALIB    READY_S → READY_S+CALIBRATION_S   move arm
    # Phase 3: COUNT    READY_S+CALIBRATION_S onwards      reps counted
    READY_S       = 3.0
    CALIBRATION_S = 5.0
    CALIB_END_S   = READY_S + CALIBRATION_S   # = 8.0

    calibration_angles = []
    phase              = "READY"   # READY → CALIB → COUNT
    _announced_ready   = False
    _announced_calib   = False
    _announced_count   = False

    counter = FpsCounter()
    stage   = dict(read=0.0, pose=0.0, clean=0.0, draw=0.0)
    n       = 0
    t_start = time.perf_counter()
    is_file = isinstance(source, str)
    state   = fsm.state

    while True:
        t0        = time.perf_counter()
        ok, frame = src.read_frame()
        if not ok:
            break

        t1          = time.perf_counter()
        norm, world = extractor.process_with_world(frame)

        t2         = time.perf_counter()
        clean, ang = pipe.step(norm, world)

        timestamp = (n / src.fps()) if is_file else (time.perf_counter() - t_start)

        # ── phase logic ─────────────────────────────────────────────
        if phase == "READY":
            # announce once
            if not _announced_ready and timestamp > 0.1:
                audio_feedback.speak(
                    "Get ready. Stand still and face the camera.",
                    priority=True)
                _announced_ready = True

            if timestamp >= READY_S:
                phase = "CALIB"

            if not args.no_overlay:
                remaining = max(0.0, READY_S - timestamp)
                put_big_text(frame,
                    f"Get ready  {remaining:.0f}s — stand still",
                    (20, 140), 0.7, (0, 200, 255))

        elif phase == "CALIB":
            # announce once
            if not _announced_calib:
                time.sleep(0.3)
                audio_feedback.speak(
                    "Move your arm slowly through its full range.",
                    priority=True)
                _announced_calib = True

            if ang is not None:
                calibration_angles.append(ang)

            cal_elapsed = timestamp - READY_S
            if cal_elapsed >= CALIBRATION_S:
                fsm.calibrate_from_signal(calibration_angles, fps=src.fps())
                rom_params = fsm.rom_check_params()
                phase      = "COUNT"
                print("Thresholds updated from calibration")

            if not args.no_overlay:
                remaining = max(0.0, CALIBRATION_S - cal_elapsed)
                put_big_text(frame,
                    f"Calibrating  {remaining:.0f}s — move full range",
                    (20, 140), 0.6, (0, 200, 255))

        elif phase == "COUNT":
            # announce once
            if not _announced_count:
                time.sleep(0.3)
                audio_feedback.speak(
                    "Good. Begin your exercise now.",
                    priority=True)
                _announced_count = True

            # FSM only runs here — not during READY or CALIB
            state, _ = fsm.update(ang, timestamp, landmarks=clean)

            if not args.no_overlay:
                put_big_text(frame, f"Reps: {fsm.rep_count}",
                             (20, 140), 0.9, (0, 255, 0))
                put_big_text(frame, f"State: {state}",
                             (20, 170), 0.7, (255, 200, 0))
                if ang is not None:
                    put_big_text(frame, f"Angle: {ang:.0f}°",
                                 (20, 200), 0.8, (255, 255, 255))

        t3  = time.perf_counter()
        fps = counter.tick()

        if not args.no_overlay:
            if args.mirror:
                frame = cv2.flip(frame, 1)
            draw_skeleton(frame, clean, mirror=args.mirror)
            draw_angle(frame, clean, joint, ang, mirror=args.mirror)

            if _feedback_rep and time.perf_counter() < _feedback_timer:
                draw_feedback(frame,
                              _feedback_rep.get('faults', []),
                              _feedback_rep.get('quality_score', 50),
                              landmarks=clean,
                              h=frame.shape[0],
                              w=frame.shape[1])

            label = (f"variant {args.variant}" +
                     (" + calibrated" if calibration else ""))
            put_big_text(frame, label, (20, 100), 0.8, (255, 255, 255))

        put_big_text(frame, f"FPS {fps:.0f}", (20, 60), 1.4, (255, 255, 0))
        t4 = time.perf_counter()

        cv2.imshow("demo_cv (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        for k, v in zip(stage, (t1-t0, t2-t1, t3-t2, t4-t3)):
            stage[k] += v
        n += 1

    # ── session end ──────────────────────────────────────────────────
    session_duration = time.perf_counter() - t_start
    fsm.flush(session_duration)

    if fsm.rep_count > 0:
        exercise_name = ("elbow_flexion" if args.exercise == "elbow"
                         else "shoulder_abduction")
        summary = scorer.session_summary(fsm.reps)
        print("\nSession summary:", fsm.summary())
        print("Quality summary:", summary)

        ok2, last_frame = src.read_frame()
        if not ok2:
            last_frame = frame
        draw_session_summary(last_frame, summary)
        cv2.imshow("demo_cv (press q to quit)", last_frame)
        cv2.waitKey(3000)

        save_session(fsm, exercise_name)
    else:
        print("\nNo reps recorded.")

    extractor.close()
    src.release()
    cv2.destroyAllWindows()
    if n:
        total = time.perf_counter() - t_start
        print(f"{n} frames, mean FPS {n/total:.1f}")
        print({k: f"{1000*v/n:.1f} ms/frame"
               for k, v in stage.items()})


if __name__ == "__main__":
    main()