"""pose_normalise.py - Normalizes pose landmarks for scale and translation invariance.

Centers landmarks relative to hip/shoulder midpoint and scales by torso
length.
"""

import numpy as np

# Key MediaPipe Landmark Indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24


def normalise(landmarks, reference_torso_length=1.0):
  """Normalizes 2D/3D pose landmarks (33, 3) or (33, 4) or batch (N, 33, C).

  Parameters:
      landmarks: np.ndarray of shape (33, C) or (N, 33, C) where C >= 3.
      reference_torso_length: Target torso length after normalization (default
        1.0).

  Returns:
      np.ndarray: Normalized landmarks with same shape as input.
  """
  lm = np.asarray(landmarks, dtype=float)
  is_single_frame = lm.ndim == 2

  if is_single_frame:
    if lm.shape[0] != 33 or lm.shape[1] < 3:
      raise ValueError(f"Expected shape (33, C>=3), got {lm.shape}")
    lm_batch = lm[np.newaxis, ...]
  elif lm.ndim == 3:
    if lm.shape[1] != 33 or lm.shape[2] < 3:
      raise ValueError(f"Expected shape (N, 33, C>=3), got {lm.shape}")
    lm_batch = lm
  else:
    raise ValueError(f"Expected 2D or 3D array, got ndim={lm.ndim}")

  out = np.copy(lm_batch)

  for i in range(len(out)):
    frame = out[i]
    coords = frame[:, :3]

    # Check for invalid / NaN or zero frame
    if np.isnan(coords).all() or np.all(coords == 0):
      continue

    l_sh, r_sh = coords[LEFT_SHOULDER], coords[RIGHT_SHOULDER]
    l_hip, r_hip = coords[LEFT_HIP], coords[RIGHT_HIP]

    # Calculate midpoints
    shoulder_mid = (l_sh + r_sh) / 2.0
    hip_mid = (l_hip + r_hip) / 2.0
    center = hip_mid if not np.isnan(hip_mid).any() else shoulder_mid

    if np.isnan(center).any():
      continue

    # 1. Translation: center at origin
    coords_centered = coords - center

    # 2. Scaling: scale by torso length
    sh_center = (
        coords_centered[LEFT_SHOULDER] + coords_centered[RIGHT_SHOULDER]
    ) / 2.0
    hip_center = (coords_centered[LEFT_HIP] + coords_centered[RIGHT_HIP]) / 2.0
    torso_len = np.linalg.norm(sh_center - hip_center)

    if np.isnan(torso_len) or torso_len < 1e-6:
      scale = 1.0
    else:
      scale = reference_torso_length / torso_len

    frame[:, :3] = coords_centered * scale

  if is_single_frame:
    return out[0]
  return out