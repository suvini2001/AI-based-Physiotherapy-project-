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

# ADD to vision/overlay.py

import cv2

# which landmarks to highlight for each fault keyword
_FAULT_JOINTS = {
    'bend':     [14],            # elbow
    'fast':     [],              # no joint — text only
    'shoulder': [11, 12],        # both shoulders
    'back':     [11, 12, 23, 24],# shoulders + hips
    'level':    [11, 12],        # both shoulders
}

_GOOD_COLOUR = (0, 220, 0)     # green
_BAD_COLOUR  = (0,  50, 220)   # red (BGR)


def draw_feedback(frame, faults, quality_score,
                  landmarks=None, h=480, w=640):
    """
    Draw quality feedback overlay after a rep completes.

    Parameters
    ----------
    frame         : BGR frame from OpenCV
    faults        : list of fault strings from scorer
    quality_score : float 0-100
    landmarks     : (33, 4) array from MediaPipe (optional)
    h, w          : frame dimensions
    """
    colour = _GOOD_COLOUR if quality_score >= 70 else _BAD_COLOUR

    # ── highlight faulty joint ────────────────────────────────────
    if landmarks is not None and faults:
        for fault in faults[:1]:   # only first fault
            key = next((k for k in _FAULT_JOINTS
                        if k in fault.lower()), None)
            if key:
                for idx in _FAULT_JOINTS[key]:
                    try:
                        cx = int(landmarks[idx][0] * w)
                        cy = int(landmarks[idx][1] * h)
                        cv2.circle(frame, (cx, cy),
                                   20, _BAD_COLOUR, 3)
                        cv2.circle(frame, (cx, cy),
                                   8,  _BAD_COLOUR, -1)
                    except (IndexError, ValueError):
                        pass

    # ── text cue ─────────────────────────────────────────────────
    if faults:
        # show short version after the dash
        msg = faults[0].split('—')[-1].strip()
        cv2.putText(frame, msg, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    _BAD_COLOUR, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Good rep!', (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    _GOOD_COLOUR, 2, cv2.LINE_AA)

    # ── score bar ────────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 20, 75, 200, 12
    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    fill = int(bar_w * quality_score / 100)
    cv2.rectangle(frame,
                  (bar_x, bar_y),
                  (bar_x + fill, bar_y + bar_h),
                  colour, -1)
    cv2.putText(frame,
                f"{quality_score:.0f}%",
                (bar_x + bar_w + 6, bar_y + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                colour, 1, cv2.LINE_AA)


def draw_session_summary(frame, summary):
    """
    Full-screen summary overlay at end of session.
    Call after Q is pressed and session is scored.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0),
                  (frame.shape[1], frame.shape[0]),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    lines = [
        f"Session complete",
        f"",
        f"Reps:         {summary.get('n_reps', 0)}",
        f"Clean reps:   {summary.get('n_clean_reps', 0)}",
        f"Quality:      {summary.get('mean_quality', 0):.0f}%",
        f"Grade:        {summary.get('grade', '?')}",
        f"",
    ]
    top_fault = summary.get('top_fault')
    if top_fault:
        lines.append(f"Most common fault:")
        lines.append(f"  {top_fault.split('—')[-1].strip()}")
    lines.append("")
    lines.append("Press any key to exit")

    y = 100
    for line in lines:
        colour = (0, 220, 200) if line.startswith("Grade") else (220, 220, 220)
        cv2.putText(frame, line, (60, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    colour, 2, cv2.LINE_AA)
        y += 40