# ==================================================================
# STEP 7 — export calibration pairs for the live system
#
#   Paste this as a NEW cell in benchmark_3d_world.ipynb, right after
#   STEP 5 (it needs `aligned_cache` and `RESULTS` from that cell).
#
#   It saves every aligned (raw 2D angle, mocap truth angle) frame with
#   its person and view, so the curve can be fitted and re-checked in the
#   repo with scripts/fit_calibration.py, without re-running MediaPipe.
#   pred_2d here is variant A (2D pixel angle) — the same angle that
#   demo_cv.py --variant A produces live.
# ==================================================================
import os
import numpy as np

raw, truth, person, view = [], [], [], []
for item in aligned_cache:
    p2d = np.asarray(item['pred_2d'], float)
    gt = np.asarray(item['truth_3d'], float)
    n = min(len(p2d), len(gt))
    ok = np.isfinite(p2d[:n]) & np.isfinite(gt[:n])
    raw.append(p2d[:n][ok])
    truth.append(gt[:n][ok])
    person += [str(item['meta']['person'])] * int(ok.sum())
    view += [str(item['meta']['view'])] * int(ok.sum())

out = os.path.join(RESULTS, 'calib_pairs_rehab24.npz')
np.savez_compressed(out,
                    raw=np.concatenate(raw).astype(np.float32),
                    truth=np.concatenate(truth).astype(np.float32),
                    person=np.array(person), view=np.array(view))

v, c = np.unique(view, return_counts=True)
print(f'saved {out}  ({os.path.getsize(out)/1e6:.2f} MB)')
print(f'{len(person)} frames, {len(set(person))} people')
print(dict(zip(v, c)))
