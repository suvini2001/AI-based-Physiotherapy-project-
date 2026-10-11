# Applies the visibility mask, calculates the right elbow angle for every frame,
# and saves a CSV in the Formats.md layout.
# Add --smooth to save a SMOOTHED version (use a different output file name).
import sys
import numpy as np
import cv2
from visibility import mask_low_visibility
from angles import elbow_angle
from filters import fill_gaps, rolling_median_filter, savitzky_golay_filter


def smooth(angles):
    """Offline smoothing: fill gaps -> median 5 (spikes) -> Savitzky-Golay (9, 3).
    Angles can't leave 0-180 degrees, so clip any overshoot."""
    y = fill_gaps(angles)
    y = rolling_median_filter(y, window_size=5)
    y = savitzky_golay_filter(y, window_length=9, polyorder=3)
    return np.clip(y, 0, 180)


def main(video_path, npy_path, out_path, use_smooth=False):
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    data = np.load(npy_path)                       # (frames, 33, 4)
    clean = mask_low_visibility(data)              # threshold comes from config.yaml
    angles = elbow_angle(clean, "right", size)     # one number per frame
    valid = ~np.isnan(angles)                      # which frames had a real measurement
    if use_smooth:
        angles = smooth(angles)

    with open(out_path, "w") as f:
        f.write("frame,time_s,elbow_angle,valid\n")
        for i, a in enumerate(angles):
            value = "" if np.isnan(a) else f"{a:.1f}"      # empty if no angle at all
            f.write(f"{i},{i / fps:.3f},{value},{int(valid[i])}\n")
    label = "smoothed" if use_smooth else "raw"
    print(f"Saved {out_path} ({label}): {len(angles)} frames, {int(valid.sum())} valid, fps {fps:.2f}")
    print(f"angle range {np.nanmin(angles):.0f} to {np.nanmax(angles):.0f} degrees")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--smooth"]
    main(args[0], args[1], args[2], use_smooth="--smooth" in sys.argv)