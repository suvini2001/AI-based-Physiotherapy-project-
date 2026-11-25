"""Check that the LIVE I_ML features give the same angles as the TRAINING features.

Replays cached MediaPipe landmarks (data/fit3d/cache/*.npz from notebook 01, or
any npz with lm, world, size, fps) through iml_live.IMLStream frame by frame and
through the offline feature code, and compares the model's outputs.

    python scripts/validate_iml_live.py --model data/iml_nolook_rehab24.txt \
        --cache "data/fit3d/cache/*_s2_w.npz" --limit 12

Rule of thumb: mean difference < 2 deg means the live pipeline reproduces training.
"""
import argparse, glob, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "pose")]
from iml_live import validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "data", "iml_nolook_rehab24.txt"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "fit3d", "cache", "*_s2_w.npz"))
    ap.add_argument("--limit", type=int, default=12)
    a = ap.parse_args()
    files = sorted(glob.glob(a.cache))[: a.limit]
    if not files:
        sys.exit(f"no cache files match {a.cache}")
    rows = []
    for f in files:
        d = np.load(f)
        size, fps = tuple(int(v) for v in d["size"]), float(d["fps"])
        for side in ("right", "left"):
            r = validate(a.model, d["lm"], d["world"], side, size, fps)
            if r["frames"] < 50:
                continue
            rows.append(r["mean_abs_diff"])
            print(f"{os.path.basename(f):55s} {side:5s} frames {r['frames']:5d}  "
                  f"live vs offline: mean {r['mean_abs_diff']:5.2f} deg  p95 {r['p95_abs_diff']:5.2f} deg  "
                  f"live coverage {r['live_coverage']:.0%}")
    if rows:
        print(f"\nmedian over arm-recordings: {np.median(rows):.2f} deg "
              f"({'OK' if np.median(rows) < 2 else 'CHECK - live features drift from training'})")


if __name__ == "__main__":
    main()
