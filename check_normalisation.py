# මෙය ඉහත සකස් කළ normalise.py ශ්‍රිතය Person A ගේ සැබෑ වැළමිට ව්‍යායාම වීඩියෝ දත්ත
# (data/elbow_A_landmarks.npy) මත ක්‍රියාත්මක කර, අප ගත් තාක්ෂණික තීරණ නිවැරදිදැයි විද්‍යාත්මකව ඔප්පු කිරීම සඳහා ලියන ලද පරීක්ෂණ පිටපතයි (Verification Script).


"""Reproduces the evidence for the normalisation choices.
Run from the main repo folder:  python check_normalisation.py
Needs: normalise.py, angles.py, data/elbow_A_landmarks.npy"""
import numpy as np
from pose_normalise import normalise
from angles import elbow_angle

SIZE = (640, 480)                       # video width, height
lm = np.load("data/elbow_A_landmarks.npy")          # (frames, 33, 4)
px = np.array(SIZE)                                  # to convert x, y (0-1) into pixels

print("1) Shape and visibility column")
n = normalise(lm, SIZE)
print("   same shape:", n.shape == lm.shape,
      "| visibility unchanged:", np.array_equal(n[..., 3], lm[..., 3]))
mid_hip = (n[:, 23, :2] + n[:, 24, :2]) / 2
mid_sho = (n[:, 11, :2] + n[:, 12, :2]) / 2
print("   mid-hip at origin (largest value):", np.abs(mid_hip).max().round(6),
      "| median torso length:", np.median(np.linalg.norm(mid_sho, axis=1)).round(3))

print("\n2) Does normalising change the elbow angle?")
before = elbow_angle(lm, "right", SIZE)
after = elbow_angle(n, "right", (1, 1))              # already in pixel-like units
print("   largest angle change:", np.nanmax(np.abs(before - after)).round(6), "degrees")

print("\n3) Camera distance / position test (person 30% farther away, shifted)")
b = lm.copy()
b[..., 0] = 0.5 + 0.7 * (lm[..., 0] - 0.5) + 0.1
b[..., 1] = 0.5 + 0.7 * (lm[..., 1] - 0.5) - 0.05
nb = normalise(b, SIZE)
print("   largest coordinate difference, raw:", np.abs(lm[..., :2] - b[..., :2]).max().round(3),
      "-> after normalising:", np.abs(n[..., :2] - nb[..., :2]).max().round(6))

print("\n4) Which scale reference is more stable over the clip?")
sho = (lm[:, 11, :2] + lm[:, 12, :2]) / 2
hip = (lm[:, 23, :2] + lm[:, 24, :2]) / 2
torso = np.linalg.norm((sho - hip) * px, axis=1)
shoulder_width = np.linalg.norm((lm[:, 11, :2] - lm[:, 12, :2]) * px, axis=1)
print(f"   torso length variation:   {100 * torso.std() / torso.mean():.1f}%")
print(f"   shoulder width variation: {100 * shoulder_width.std() / shoulder_width.mean():.1f}%")

print("\n5) Per-frame vs per-clip scale (hanging left wrist should stay still)")
for name, per_frame in [("per-frame torso", True), ("per-clip median torso", False)]:
    m = normalise(lm, SIZE, per_frame_scale=per_frame)
    wrist = m[:, 15, :2]
    print(f"   {name:22s} wobble std: x {wrist[:, 0].std():.4f}  y {wrist[:, 1].std():.4f} (torso units)")

print("\n6) Moving (right) wrist range in torso units")
rw = n[:, 16, :2]
print(f"   x {rw[:, 0].min():.2f} to {rw[:, 0].max():.2f}   y {rw[:, 1].min():.2f} to {rw[:, 1].max():.2f}")

print("\n7) Missing hip in one frame")
c = lm.copy()
c[10, 23] = np.nan
nc = normalise(c, SIZE)
print("   that frame becomes NaN:", bool(np.isnan(nc[10, :, :3]).all()),
      "| next frame is fine:", not np.isnan(nc[11, :, :3]).any())