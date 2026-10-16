"""A4 - variant B: the elbow angle from MediaPipe WORLD landmarks (3D, in metres)."""
import numpy as np
from angles import angle, ELBOW


def mask_world_like(world, norm_masked):
    """Drop in the world array the SAME points that visibility.mask_low_visibility dropped."""
    out = np.asarray(world, float).copy()
    out[np.isnan(np.asarray(norm_masked)[..., 0])] = np.nan
    return out


def elbow_angle_3d(world, side="right"):
    """world: (N, 33, 4) -> (N,) degrees. NaN in -> NaN out."""
    i, j, k = ELBOW[side]
    w = np.asarray(world, float)
    return angle(w[:, i, :3], w[:, j, :3], w[:, k, :3])