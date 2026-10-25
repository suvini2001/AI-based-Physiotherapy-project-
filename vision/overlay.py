"""Step 1.9 - drawing helpers. Draw AFTER all processing, on the frame."""
import cv2
import numpy as np

# Body-only BlazePose connections (indices from the 33-landmark model)
BODY_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (11, 23), (12, 24), (23, 24),
              (23, 25), (25, 27), (24, 26), (26, 28),
              (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32)]

JOINTS = {   # exercise -> side -> (a, b, c); the angle is measured at b
    "elbow": {"left": (11, 13, 15), "right": (12, 14, 16)},
    "shoulder":  {"left": (23, 11, 13), "right": (24, 12, 14)},
}


def _px(lm, w, h, mirror):
    x = (1.0 - lm[0]) if mirror else lm[0]
    return int(x * w), int(lm[1] * h)


def draw_skeleton(frame, pose, min_vis=0.5, mirror=False, highlight=()):
    """pose is ONE frame: shape (33, 4). Skips NaN points and low visibility."""
    h, w = frame.shape[:2]
    ok = ~np.isnan(pose[:, :3]).any(axis=1) & (pose[:, 3] >= min_vis)
    for a, b in BODY_LINKS:
        if ok[a] and ok[b]:
            cv2.line(frame, _px(pose[a], w, h, mirror), _px(pose[b], w, h, mirror),
                     (0, 200, 0), 3, cv2.LINE_AA)
    for i in range(11, 33):
        if ok[i]:
            colour = (0, 165, 255) if i in highlight else (255, 255, 255)
            cv2.circle(frame, _px(pose[i], w, h, mirror), 6, colour, -1, cv2.LINE_AA)
    return frame


def put_big_text(frame, text, org, scale=2.0, colour=(255, 255, 255)):
    """Large text with a dark outline so it can be read from 2-3 m away."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 3, cv2.LINE_AA)


def draw_angle(frame, pose, joint_ids, angle, mirror=False):
    h, w = frame.shape[:2]
    b = pose[joint_ids[1]]
    if angle is None or np.isnan(angle) or np.isnan(b[:3]).any():
        put_big_text(frame, "angle: --", (30, 140), 1.6, (0, 0, 255))
        return frame
    x, y = _px(b, w, h, mirror)
    put_big_text(frame, f"{angle:.0f}", (x + 15, y), 1.6, (0, 255, 255))     # next to the joint
    put_big_text(frame, f"Angle {angle:.0f} deg", (30, 140), 1.6, (0, 255, 255))
    return frame