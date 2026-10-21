# draw_check.py  (replace full file)
"""Landmark and yaw overlay check.
Usage:
  python draw_check.py data/elbow_A_laptop.mp4 data/elbow_A_landmarks.npy
  python draw_check.py data/elbow_A_laptop.mp4 data/elbow_A_landmarks.npy --world data/sample_world_landmarks.npz
"""
import sys
import argparse
import numpy as np
import cv2

sys.path.append("capture")
sys.path.append("vision")
from camera import FrameSource

ARMS  = {"left": (11, 13, 15), "right": (12, 14, 16)}
HIPS  = (23, 24)


def main(clip_path, npy_path, world_path=None):
    data = np.load(npy_path)                  # (frames, 33, 4)
    world = None
    if world_path:
        z = np.load(world_path)
        world = z["world"] if "world" in z else z[z.files[0]]

    src = FrameSource(clip_path)
    width, height = src.size()
    delay = int(1000 / src.fps())
    i = 0

    # import B's yaw function if available
    try:
        sys.path.insert(0, ".")
        from angles_3d import body_yaw
        from capture.view_check import draw_yaw_message, draw_yaw_bar
        has_yaw = True
    except ImportError:
        has_yaw = False

    while i < len(data):
        ok, frame = src.read_frame()
        if not ok:
            break
        lm = data[i]

        # draw skeleton
        for side, (s, e, w) in ARMS.items():
            pts = [(int(lm[j, 0] * width), int(lm[j, 1] * height)) for j in (s, e, w)]
            vis = min(lm[j, 3] for j in (s, e, w))
            if np.isnan(vis):
                continue
            color = (0, 255, 0) if vis >= 0.7 else (0, 165, 255)
            cv2.line(frame, pts[0], pts[1], color, 3)
            cv2.line(frame, pts[1], pts[2], color, 3)
            for p in pts:
                cv2.circle(frame, p, 8, color, -1)
        for j in HIPS:
            if not np.isnan(lm[j, 0]):
                cv2.circle(frame, (int(lm[j, 0] * width), int(lm[j, 1] * height)),
                           8, (255, 0, 0), -1)

        # draw yaw overlay if world landmarks provided
        yaw_val = None
        if has_yaw and world is not None and i < len(world):
            yaw_arr = body_yaw(world[i])   # single frame -> scalar
            yaw_val = float(yaw_arr) if np.isfinite(float(yaw_arr)) else None
            draw_yaw_message(frame, yaw_val)
            draw_yaw_bar(frame, yaw_val)

        cv2.putText(frame, f"frame {i}  green=visible  orange=low-vis",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if yaw_val is not None:
            cv2.putText(frame, f"yaw {yaw_val:.1f} deg",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        cv2.imshow("landmark + yaw check (press q to quit)", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
        i += 1

    src.release()
    cv2.destroyAllWindows()

    # summary
    for side, idx in ARMS.items():
        vis = data[:, idx, 3]
        print(f"{side} arm: shoulder/elbow/wrist visibility >= 0.7 in "
              f"{np.mean(vis >= 0.7, axis=0).round(2)} of frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("npy")
    ap.add_argument("--world", default=None)
    args = ap.parse_args()
    main(args.clip, args.npy, args.world)