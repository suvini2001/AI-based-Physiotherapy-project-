import numpy as np

# MediaPipe landmark numbers: (first point, joint, last point)
ELBOW = {"right": (12, 14, 16), "left": (11, 13, 15)}   # shoulder, elbow, wrist
# KNEE = {"right": (24, 26, 28), "left": (23, 25, 27)}    # hip, knee, ankle


# how one angle is calculated
def angle(a, b, c):
    """Angle at point b, in degrees, between the lines b->a and b->c.
    Works on single points or on arrays of points (last axis = x, y [, z]).
    Returns NaN if any point is NaN or a line has zero length."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    # arrow from elbow to shoulder
    u = a - b   
    # arrow from elbow to wrist
    v = c - b

    # Calculate vector lengths
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)

    # Calculate angle between vectors using dot product formula
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (u * v).sum(axis=-1) / (nu * nv)      # 0/0 -> NaN
    # cos-->radians-->degrees
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# pplies the same angle calculation to every frame of a video.
def joint_angle(landmarks, joint, size=(640, 480)):
    """landmarks: (frames, 33, 4). joint: three landmark numbers.
    size: (width, height) of the video, so x and y become pixels.
    Returns one angle per frame, shape (frames,)."""

    # joint is three landmark numbers, for example (12, 14, 16) for the right arm.
    i, j, k = joint
    scale = np.array(size, float)

    # : =	all frames
    # i	= only landmark number i (the shoulder)
    # :2 =	only the first two of the four numbers, which are x and y


    a = landmarks[:, i, :2] * scale
    b = landmarks[:, j, :2] * scale
    c = landmarks[:, k, :2] * scale
    return angle(a, b, c)


def elbow_angle(landmarks, side="right", size=(640, 480)):
    return joint_angle(landmarks, ELBOW[side], size)


# def knee_angle(landmarks, side="right", size=(640, 480)):
#     return joint_angle(landmarks, KNEE[side], size)