"""demo_cv.py - Milestone 1 demo: webcam or clip -> landmarks -> clean -> angle -> overlay.
Run from the main project folder:
  python demo_cv.py --exercise elbow --side right          (live webcam)
  python demo_cv.py --source data/elbow_A_laptop.mp4        (replay a clip)
  python demo_cv.py --no-overlay                            (FPS without drawing)
"""
import argparse, sys, time
from collections import deque
import cv2
import numpy as np

sys.path.append("capture")
sys.path.append("pose")
sys.path.append("vision")
from camera import FrameSource, FpsCounter          # your step 1.2
from landmarks import LandmarkExtractor             # your step 1.3
from visibility import mask_low_visibility, load_config   # B, step 1.4
from filters import OneEuroFilter                   # B, step 1.5 (live-capable)
from angles import angle                            # B, step 1.8
from gap_fill import fill_gaps                      # you, step 1.6
from overlay import JOINTS, draw_angle, draw_skeleton, put_big_text   # you, step 1.9

WINDOW = 15            # recent frames kept for gap filling (about 0.5 s)
SPIKE_WINDOW = 5       # median of the last 5 angles removes one-frame spikes (lag ~2 frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--exercise", choices=JOINTS, default="elbow")
    ap.add_argument("--side", choices=["left", "right"], default="right")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--mirror", action="store_true", help="display only")
    ap.add_argument("--no-overlay", action="store_true", help="for the FPS comparison")
    ap.add_argument("--fast", action="store_true", help="replay as fast as possible (for FPS tests)")
    args = ap.parse_args()

    cfg = load_config()
    thr, max_gap = cfg["visibility_threshold"], cfg["max_gap_frames"]

    source = int(args.source) if args.source.isdigit() else args.source
    src = FrameSource(source, args.width, args.height)
    w, h = src.size()                                   # real size, used for pixel maths
    extractor = LandmarkExtractor()
    joint = JOINTS[args.exercise][args.side]

    window = deque(maxlen=WINDOW)
    recent = deque(maxlen=SPIKE_WINDOW)
    smoother = OneEuroFilter(fs=src.fps())
    counter = FpsCounter()
    stage = dict(read=0.0, pose=0.0, clean=0.0, draw=0.0)
    n, t_start = 0, time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ok, frame = src.read_frame()
        if not ok:
            break
        t1 = time.perf_counter()
        pose = extractor.process(frame)                  # BEFORE any flip
        t2 = time.perf_counter()

        pose = mask_low_visibility(pose[None], thr)[0]   # B: drop weak points (NaN)
        window.append(pose)
        clean = fill_gaps(np.stack(window), max_gap)[-1] # you: repair short holes
        pts = [clean[i] for i in joint]
        if any(np.isnan(p[:3]).any() for p in pts):
            ang = None                                   # joint missing: show "--"
        else:
            px = [np.array([p[0] * w, p[1] * h]) for p in pts]   # pixels, like B's joint_angle
            recent.append(float(angle(*px)))
            ang = smoother(float(np.median(recent)))
            ang = min(max(ang, 0.0), 180.0)
        t3 = time.perf_counter()

        fps = counter.tick()
        if not args.no_overlay:
            if args.mirror:
                frame = cv2.flip(frame, 1)               # flip only for display
            draw_skeleton(frame, clean, mirror=args.mirror)
            draw_angle(frame, clean, joint, ang, mirror=args.mirror)
        put_big_text(frame, f"FPS {fps:.0f}", (20, 60), 1.4, (255, 255, 0))
        t4 = time.perf_counter()

        cv2.imshow("demo_cv (press q to quit)", frame)
        if src.is_live or args.fast:
            wait_ms = 1                                    # live camera: no extra wait
        else:                                              # replay at the clip's real speed
            used_ms = (time.perf_counter() - t0) * 1000
            wait_ms = max(1, int(1000 / src.fps() - used_ms))
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            break
        for k, v in zip(stage, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
            stage[k] += v
        n += 1

    extractor.close()
    src.release()
    cv2.destroyAllWindows()
    if n:
        print(f"{n} frames, mean FPS {n / (time.perf_counter() - t_start):.1f}")
        print({k: f"{1000 * v / n:.1f} ms/frame" for k, v in stage.items()})


if __name__ == "__main__":
    main()