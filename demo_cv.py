"""demo_cv.py - webcam or clip -> landmarks -> clean -> angle -> overlay.
  python demo_cv.py --exercise elbow --side right          (live webcam, variant A = old 2D angle)
  python demo_cv.py --variant B                             (3D angle from world landmarks)
  python demo_cv.py --calibration data/user_calibration.json
  python demo_cv.py --source data/elbow_A_laptop.mp4        (replay a clip)
  python demo_cv.py --no-overlay                            (FPS without drawing)
Commit 1 (Person A): the loop, variants A and B, calibration. Commit 2 (Person B): yaw message, variant C, gate."""
import argparse, os, sys, time
from collections import deque
import cv2
import numpy as np
import assessment.scorer as scorer
from assessment.rules import ROM_FAULT, ROM_TOLERANCE
import audio_feedback
from vision.overlay import draw_arm, draw_hud, draw_session_summary
from capture.camera import FrameSource, FpsCounter
from pose.landmarks import LandmarkExtractor
from visibility import mask_low_visibility, load_config
from filters import OneEuroFilter
from angles import angle
from angles_world import mask_world_like
from calibration import load_params, apply_calibration, calibration_warnings
from vision.gap_fill import fill_gaps
from vision.overlay import JOINTS, put_big_text
from exercise_state import (ElbowFSM, ShoulderFSM, save_session,
                            calibration_cycle_complete)

WINDOW = 15            # recent frames kept for gap filling (about 0.5 s)
SPIKE_WINDOW = 5       # median of the last 5 angles removes one-frame spikes
VARIANTS = ["A", "B", "C"]  # Person B adds "C" in the second commit
DISPLAY_W = 960        # the window is drawn at this width so text stays sharp
WINDOW_TITLE = "Physio session (press Q to finish)"


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

    def step(self, norm, world, fps=None):
        """norm, world: (33, 4) for one frame. Returns (clean_norm, angle_or_None).
        fps: the real frame rate right now (live camera). The smoothing is tuned in
        seconds, so if it is told 30 FPS while frames arrive at 7 it flattens the
        top and bottom of every rep."""
        if fps:
            self.smoother.fs = float(fps)
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
        # spike filter: 5 frames at normal speed, 3 on a slow camera (5 would be ~0.7 s)
        k = SPIKE_WINDOW if self.smoother.fs >= 20 else 3
        ang = self.smoother(float(np.median(list(self.recent)[-k:])))
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
    ap.add_argument("--session-out", default="session_log.json",
                    help="where to save the session, e.g. data/healthy/p01.json")
    args = ap.parse_args()

    cfg          = load_config()
    thr, max_gap = cfg["visibility_threshold"], cfg["max_gap_frames"]
    calibration  = load_params(args.calibration) if args.calibration else None
    for line in calibration_warnings(calibration, args.variant):
        print("WARNING:", line)
    source       = int(args.source) if args.source.isdigit() else args.source
    src          = FrameSource(source, args.width, args.height)
    size         = src.size()
    extractor    = LandmarkExtractor()
    joint        = JOINTS[args.exercise][args.side]
    pipe         = AnglePipeline(joint, args.variant, size,
                                 src.fps(), thr, max_gap, calibration, side=args.side)

    # ── scorer ──────────────────────────────────────────────────────
    exercise_name = ("elbow_flexion" if args.exercise == "elbow"
                     else "shoulder_abduction")
    template_path = os.path.join("templates", f"{exercise_name}.json")
    if not os.path.exists(template_path):
        template_path = "template.json"
    scorer.load(template_path, exercise=exercise_name)
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
                         frame_width=frame_width,
                         exercise=exercise_name)
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

    # ── shallow attempt callback (not counted as a rep) ─────────────
    def on_attempt_incomplete(attempt):
        nonlocal _feedback_rep, _feedback_timer
        msg = ROM_FAULT[attempt.get("reason") or fsm.rom_check_params()["kind"]]
        print(f"  Not counted | ROM {attempt['rom']:.0f}°  "
              f"(max {attempt['max_angle']:.0f}° → min {attempt['min_angle']:.0f}°)  "
              f"{msg}")
        audio_feedback.speak(msg.split('—')[-1].strip(), priority=True)
        _feedback_rep   = {'faults': [msg], 'quality_score': 0, 'counted': False}
        _feedback_timer = time.perf_counter() + 2.0

    # ── FSM ─────────────────────────────────────────────────────────
    if args.exercise == "shoulder":
        fsm = ShoulderFSM(rep_callback=on_rep_complete,
                          attempt_callback=on_attempt_incomplete)
    else:
        fsm = ElbowFSM(rep_callback=on_rep_complete,
                       attempt_callback=on_attempt_incomplete)
    rom_params = fsm.rom_check_params()

    # ── session timing ───────────────────────────────────────────────
    # Phase 1: READY    0  → READY_S        stand still, get in position
    # Phase 2: CALIB    until one full out-and-back movement is seen
    #                   (gives up after CALIBRATION_S seconds)
    # Phase 3: COUNT    reps counted
    READY_S       = 3.0
    CALIBRATION_S = 15.0   # longest the CALIB phase waits for one full movement

    calibration_angles = []
    calib_failed       = False
    remaining          = READY_S   # seconds left in the READY / CALIB phase
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
    fps     = 0.0          # measured frame rate, updated every frame

    while True:
        t0        = time.perf_counter()
        ok, frame = src.read_frame()
        if not ok:
            break

        t1          = time.perf_counter()
        norm, world = extractor.process_with_world(frame)

        t2         = time.perf_counter()
        # live camera: tell the smoother the measured frame rate (a clip keeps its own)
        live_fps   = min(max(fps, 3.0), 60.0) if (fps and not is_file) else None
        clean, ang = pipe.step(norm, world, fps=live_fps)

        timestamp = (n / src.fps()) if is_file else (time.perf_counter() - t_start)

        # ── phase logic ─────────────────────────────────────────────
        if phase == "READY":
            # announce once
            if not _announced_ready and timestamp > 0.1:
                audio_feedback.speak(
                    "Get ready. Stand still and face the camera.",
                    priority=True)
                _announced_ready = True

            remaining = max(0.0, READY_S - timestamp)
            if timestamp >= READY_S:
                phase = "CALIB"

        elif phase == "CALIB":
            # announce once
            if not _announced_calib:
                audio_feedback.speak(
                    "Move your arm slowly through its full range.",
                    priority=True)
                _announced_calib = True

            if ang is not None:
                calibration_angles.append(ang)

            cal_elapsed = timestamp - READY_S
            if (calibration_cycle_complete(calibration_angles)
                    or cal_elapsed >= CALIBRATION_S):
                calibrated = fsm.calibrate_from_signal(calibration_angles, fps=src.fps())
                rom_params = fsm.rom_check_params()
                phase      = "COUNT"
                if calibrated:
                    print("Thresholds updated from calibration")
                else:
                    calib_failed = True
                    print(f"WARNING: auto-calibration failed "
                          f"({len(calibration_angles)} angle samples in {CALIBRATION_S:.0f}s, "
                          f"no full bend-and-straighten seen). Using default thresholds: "
                          f"incomplete reps will NOT be flagged. Restart and try again.")
                    audio_feedback.speak("Calibration failed. Please restart.", priority=True)

            remaining = max(0.0, CALIBRATION_S - cal_elapsed)

        elif phase == "COUNT":
            # announce once
            if not _announced_count:
                audio_feedback.speak(
                    "Good. Begin your exercise now.",
                    priority=True)
                _announced_count = True

            # FSM only runs here — not during READY or CALIB
            state, _ = fsm.update(ang, timestamp, landmarks=clean)

        t3  = time.perf_counter()
        fps = counter.tick()

        if not args.no_overlay:
            if args.mirror:
                frame = cv2.flip(frame, 1)
            # enlarge first, then draw, so text and lines are sharp
            if frame.shape[1] != DISPLAY_W:
                disp_h = int(round(frame.shape[0] * DISPLAY_W / frame.shape[1]))
                frame = cv2.resize(frame, (DISPLAY_W, disp_h),
                                   interpolation=cv2.INTER_LINEAR)
            draw_arm(frame, clean, joint, ang, mirror=args.mirror)

            # the angle the arm has to reach for a rep to count
            rom_params = fsm.rom_check_params()      # the goal follows the best rep
            goal, kind = None, rom_params.get("kind", "flex")
            if phase == "COUNT":
                if kind == "flex":
                    goal = min(fsm.FLEX_THRESH, rom_params["target"] + ROM_TOLERANCE)
                else:
                    goal = max(fsm.RAISE_THRESH, rom_params["target"] - ROM_TOLERANCE)

            showing = _feedback_rep and time.perf_counter() < _feedback_timer
            draw_hud(frame, {
                "title": (f"{exercise_name.replace('_', ' ').upper()}  |  "
                          f"{args.side.upper()} ARM  |  {args.variant}"
                          + ("+cal" if calibration else "")),
                "phase":       phase,
                "remaining":   remaining,
                "phase_total": READY_S if phase == "READY" else CALIBRATION_S,
                "reps":        fsm.rep_count,
                "incomplete":  len(fsm.incomplete_attempts),
                "angle":       ang,
                "state":       state,
                "target":      goal,
                "target_kind": kind,
                "feedback":    _feedback_rep if showing else None,
                "notice":      "Calibration failed: default goal" if calib_failed else None,
                "fps":         fps,
            }, pose=clean, mirror=args.mirror)
        else:
            put_big_text(frame, f"FPS {fps:.0f}", (20, 60), 1.4, (255, 255, 0))
        t4 = time.perf_counter()

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        for k, v in zip(stage, (t1-t0, t2-t1, t3-t2, t4-t3)):
            stage[k] += v
        n += 1

    # ── session end ──────────────────────────────────────────────────
    session_duration = time.perf_counter() - t_start
    fsm.flush(session_duration)

    if fsm.rep_count > 0 or fsm.incomplete_attempts:
        # incomplete attempts count against the session (quality 0 each)
        summary = scorer.session_summary(fsm.reps, fsm.incomplete_attempts)
        print("\nSession summary:", fsm.summary())
        print("Quality summary:", summary)

        ok2, last_frame = src.read_frame()
        if not ok2:
            last_frame = frame
        if last_frame.shape[1] != DISPLAY_W:
            disp_h = int(round(last_frame.shape[0] * DISPLAY_W / last_frame.shape[1]))
            last_frame = cv2.resize(last_frame, (DISPLAY_W, disp_h))
        draw_session_summary(last_frame, summary)
        cv2.imshow(WINDOW_TITLE, last_frame)
        cv2.waitKey(5000)

        os.makedirs(os.path.dirname(args.session_out) or ".", exist_ok=True)
        save_session(fsm, exercise_name, path=args.session_out)
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