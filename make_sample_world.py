"""A1 - make data/sample_world_landmarks.npz from a clip (Person B reads this on Day 1).
Run from the main project folder:
  python make_sample_world.py data/elbow_A_laptop.mp4 data/sample_world_landmarks.npz"""
import sys
import numpy as np
sys.path.append("capture")
sys.path.append("pose")
from camera import FrameSource


def describe_world(world, side="right"):
    """World landmarks are in METRES: a real upper arm is roughly 0.25-0.35 m, a forearm 0.22-0.30 m."""
    sh, el, wr = (12, 14, 16) if side == "right" else (11, 13, 15)
    upper = np.linalg.norm(world[:, sh, :3] - world[:, el, :3], axis=1)
    fore = np.linalg.norm(world[:, el, :3] - world[:, wr, :3], axis=1)
    return {"upper_arm_m": float(np.nanmedian(upper)), "forearm_m": float(np.nanmedian(fore))}


def main(clip_path, out_path):
    from landmarks import LandmarkExtractor
    src = FrameSource(clip_path)
    extractor = LandmarkExtractor()
    norm_rows, world_rows = [], []
    while True:
        ok, frame = src.read_frame()
        if not ok:
            break
        norm, world = extractor.process_with_world(frame)
        norm_rows.append(norm)
        world_rows.append(world)
    fps, size = src.fps(), src.size()
    extractor.close()
    src.release()

    landmarks, world_landmarks = np.stack(norm_rows), np.stack(world_rows)
    np.savez(out_path, landmarks=landmarks, world_landmarks=world_landmarks,
             fps=fps, size=np.array(size))
    found = int(np.sum(~np.isnan(world_landmarks[:, 0, 0])))
    print(f"Saved {out_path}: {landmarks.shape} norm + {world_landmarks.shape} world, fps {fps:.1f}, size {size}")
    print(f"Person found in {found} of {len(world_landmarks)} frames")
    print("Right arm bone lengths (m):", describe_world(world_landmarks))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "data/sample_world_landmarks.npz")