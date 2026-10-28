"""A2 - run MediaPipe on ONE frame window of a video and save norm + world landmarks.
Usage (inside the MediaPipe 3.12 environment):
  python extract_world.py VIDEO FIRST_FRAME LAST_FRAME OUT.npz [MODEL_COMPLEXITY]
Saves: lm (N,33,4), world (N,33,4), size (w,h), fps, f0 (first frame of the window)."""
import sys
import numpy as np
import cv2
from pose.landmarks import LandmarkExtractor


def main(video, f0, f1, out, complexity=1):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ex = LandmarkExtractor(complexity)
    norm_rows, world_rows, size, i = [], [], None, 0
    while i <= f1:
        if i < f0:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            size = size or (frame.shape[1], frame.shape[0])
            norm, world = ex.process_with_world(frame)
            norm_rows.append(norm)
            world_rows.append(world)
        i += 1
    ex.close()
    cap.release()
    if not norm_rows:
        raise SystemExit("no frames read from " + video)
    np.savez(out, lm=np.stack(norm_rows), world=np.stack(world_rows),
             size=np.array(size), fps=fps, f0=f0)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 1)