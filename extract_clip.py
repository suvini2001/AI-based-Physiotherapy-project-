import sys
import numpy as np
sys.path.append("capture")
sys.path.append("pose")
from camera import FrameSource
from landmarks import LandmarkExtractor


def main(clip_path, out_path):
    src = FrameSource(clip_path)          # replay a saved clip
    extractor = LandmarkExtractor()
    rows = []
    while True:
        ok, frame = src.read_frame()
        if not ok:
            break
        rows.append(extractor.process(frame))
    extractor.close()
    src.release()

    data = np.stack(rows)                  # shape: (frames, 33, 4)
    np.save(out_path, data)
    found = np.sum(~np.isnan(data[:, 0, 0]))
    print(f"Saved {out_path} with shape {data.shape}")
    print(f"Person found in {found} of {len(rows)} frames")


if __name__ == "__main__":
    clip = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "landmarks.npy"
    main(clip, out)