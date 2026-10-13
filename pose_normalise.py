"""Normalise MediaPipe pose landmarks for distance and body-size invariance.
Mid-hip becomes the origin and everything is divided by torso length
(mid-shoulder to mid-hip)."""
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24


def normalise(landmarks, size=(640, 480), per_frame_scale=False):
    """(frames, 33, 4) in -> (frames, 33, 4) out.
    1. x, y -> pixels (x*width, y*height), so the picture's aspect ratio doesn't distort distances.
    2. Origin = mid-hip. 3. Divide by torso length (mid-shoulder to mid-hip).
    Torso length = median over the clip (per_frame_scale=True uses every frame instead).
    z is scaled like x. The visibility column (index 3) is left unchanged."""
    lm = np.asarray(landmarks, float)
    out = lm.copy()
    w, h = size

    # convert x, y to pixels
    out[..., 0] *= w
    out[..., 1] *= h
    out[..., 2] *= w                                   # z is in roughly the same units as x

#  origin = mid-hip
    hip = (out[:, L_HIP, :3] + out[:, R_HIP, :3]) / 2          # (frames, 3)
    
#  Mid-shoulder point
    sho = (out[:, L_SH, :3] + out[:, R_SH, :3]) / 2

#   torso length- mid sholder to mid hip
    torso = np.linalg.norm((sho - hip)[:, :2], axis=1)  
    
# 2D torso length per frame
    scale = torso if per_frame_scale else np.nanmedian(torso)
    scale = np.where(np.asarray(scale) > 0, scale, np.nan)     # avoid divide-by-zero

# All body landmarks are subtracted from the Mid-hip coordinate. 
# Then, the hip center becomes (0, 0, 0), i.e., the origin.
    out[..., :3] -= hip[:, None, :]

# Normalization by torso length.
    out[..., :3] /= np.reshape(scale, (-1, 1, 1)) if per_frame_scale else scale
    return out