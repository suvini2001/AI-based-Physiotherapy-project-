import sys
import numpy as np
import cv2
sys.path.append("capture")
from camera import FrameSource

# MediaPipe landmark numbers: shoulder, elbow, wrist, hip for each side
ARMS = {
    "left":  (11, 13, 15),
    "right": (12, 14, 16),
}
HIPS = (23, 24)


def main(clip_path, npy_path):
    data = np.load(npy_path)                 # (frames, 33, 4)
    src = FrameSource(clip_path)
    width, height = src.size()
    delay = int(1000 / src.fps())
    i = 0
    while i < len(data):
        ok, frame = src.read_frame()
        if not ok:
            break
        lm = data[i]
        # x, y are 0..1 across the picture -> multiply to get pixels
        for side, (s, e, w) in ARMS.items():
            pts = [(int(lm[j, 0] * width), int(lm[j, 1] * height)) for j in (s, e, w)]
            vis = min(lm[j, 3] for j in (s, e, w))
            if np.isnan(vis):
                continue
            color = (0, 255, 0) if vis >= 0.7 else (0, 165, 255)  # orange = low visibility
            cv2.line(frame, pts[0], pts[1], color, 3)
            cv2.line(frame, pts[1], pts[2], color, 3)
            for p in pts:
                cv2.circle(frame, p, 8, color, -1)
        for j in HIPS:
            if not np.isnan(lm[j, 0]):
                cv2.circle(frame, (int(lm[j, 0] * width), int(lm[j, 1] * height)),
                           8, (255, 0, 0), -1)
        cv2.putText(frame, f"frame {i}  green=visible  orange=low visibility",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("landmark check (press q to quit)", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
        i += 1
    src.release()
    cv2.destroyAllWindows()

    # Summary: how often each arm point is clearly visible
    for side, idx in ARMS.items():
        vis = data[:, idx, 3]
        print(f"{side} arm: shoulder/elbow/wrist visibility >= 0.7 in "
              f"{np.mean(vis >= 0.7, axis=0).round(2)} of frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])