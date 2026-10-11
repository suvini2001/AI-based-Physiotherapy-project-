# It applies your visibility mask, calculates the right elbow angle for every frame, and saves a CSV in the Formats.md layout.
import sys
import numpy as np
import cv2
from visibility import mask_low_visibility
from angles import elbow_angle


def main(video_path, npy_path, out_path):
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    data = np.load(npy_path)                       # (frames, 33, 4)
    clean = mask_low_visibility(data)              # threshold comes from config.yaml
    angles = elbow_angle(clean, "right", size)     # one number per frame

    with open(out_path, "w") as f:
        f.write("frame,time_s,elbow_angle,valid\n")
        for i, a in enumerate(angles):
            ok = not np.isnan(a)
            value = f"{a:.1f}" if ok else ""
            f.write(f"{i},{i / fps:.3f},{value},{int(ok)}\n")
    print(f"Saved {out_path}: {len(angles)} frames, {int(np.sum(~np.isnan(angles)))} valid, fps {fps:.2f}")
    print(f"angle range {np.nanmin(angles):.0f} to {np.nanmax(angles):.0f} degrees")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])