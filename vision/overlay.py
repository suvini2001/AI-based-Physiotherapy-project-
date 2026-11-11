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


# ══════════════════════════════════════════════════════════════════
# HUD (the live window layout used by demo_cv.py)
# ══════════════════════════════════════════════════════════════════
# OpenCV's built-in fonts are ASCII only: no "°" or "—" (they show as "??").

FONT   = cv2.FONT_HERSHEY_DUPLEX
TEXT   = (245, 245, 245)      # colours are BGR
MUTED  = (175, 175, 175)
PANEL  = (28, 22, 18)
ACCENT = (255, 200, 70)       # light blue: the tracked arm
GOOD   = (120, 215, 90)       # green
WARN   = (70, 185, 255)       # amber
BAD    = (90, 90, 240)        # red


def _panel(frame, x0, y0, x1, y1, colour=PANEL, alpha=0.62):
    """Semi-transparent filled rectangle (blends only that region, so it is cheap)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    block = np.empty_like(roi)
    block[:] = colour
    cv2.addWeighted(block, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _text(frame, text, org, scale, colour=TEXT, thick=1):
    cv2.putText(frame, str(text), (int(org[0]), int(org[1])), FONT, scale, colour,
                thick, cv2.LINE_AA)


def _text_w(text, scale, thick=1):
    return cv2.getTextSize(str(text), FONT, scale, thick)[0][0]


def _ascii(msg):
    """Short ASCII version of a fault message (the part after the dash)."""
    msg = str(msg).split('—')[-1].strip()
    msg = msg.encode('ascii', 'ignore').decode()
    return msg[:1].upper() + msg[1:]


def draw_arm(frame, pose, joint_ids, angle, mirror=False, min_vis=0.5):
    """Thin grey skeleton with the tracked arm highlighted and the angle at the joint."""
    h, w = frame.shape[:2]
    k = w / 960.0
    ok = ~np.isnan(pose[:, :3]).any(axis=1) & (pose[:, 3] >= min_vis)
    arm = {(joint_ids[0], joint_ids[1]), (joint_ids[1], joint_ids[2]),
           (joint_ids[1], joint_ids[0]), (joint_ids[2], joint_ids[1])}
    for a, b in BODY_LINKS:
        if ok[a] and ok[b] and (a, b) not in arm:
            cv2.line(frame, _px(pose[a], w, h, mirror), _px(pose[b], w, h, mirror),
                     (215, 215, 215), max(1, int(2 * k)), cv2.LINE_AA)
    for a, b in ((joint_ids[0], joint_ids[1]), (joint_ids[1], joint_ids[2])):
        if ok[a] and ok[b]:
            cv2.line(frame, _px(pose[a], w, h, mirror), _px(pose[b], w, h, mirror),
                     ACCENT, max(2, int(5 * k)), cv2.LINE_AA)
    for i in joint_ids:
        if ok[i]:
            cv2.circle(frame, _px(pose[i], w, h, mirror), max(3, int(7 * k)),
                       TEXT, -1, cv2.LINE_AA)
    b = joint_ids[1]
    if ok[b] and angle is not None and np.isfinite(angle):
        x, y = _px(pose[b], w, h, mirror)
        cv2.circle(frame, (x, y), max(6, int(15 * k)), ACCENT, max(1, int(2 * k)), cv2.LINE_AA)
        label = f"{angle:.0f}"
        _panel(frame, x + 20 * k, y - 22 * k, x + 30 * k + _text_w(label, 0.8 * k) , y + 10 * k)
        _text(frame, label, (x + 25 * k, y + 3 * k), 0.8 * k, ACCENT, max(1, int(2 * k)))
    return frame


def draw_hud(frame, hud, pose=None, mirror=False):
    """
    Draw the whole live layout. `hud` is a dict:
      title        "ELBOW FLEXION  |  RIGHT ARM"
      phase        "READY" | "CALIB" | "COUNT"
      remaining    seconds left in READY / CALIB
      phase_total  length of that phase in seconds
      reps, incomplete, angle (float or None), state (FSM state name)
      target       angle the arm must reach for a rep to count (or None)
      target_kind  "flex" (go below target) or "raise" (go above target)
      feedback     None, or {'faults': [...], 'quality_score': float, 'counted': bool}
      notice       short warning shown in the top bar (e.g. calibration failed), or None
      fps          float
    """
    h, w = frame.shape[:2]
    k = w / 960.0                       # everything scales with the window width
    pad = int(20 * k)

    # ── top bar ──────────────────────────────────────────────────────
    top_h = int(52 * k)
    _panel(frame, 0, 0, w, top_h)
    _text(frame, hud.get("title", ""), (pad, top_h * 0.66), 0.7 * k, TEXT)
    fps_txt = f"{hud.get('fps', 0):.0f} fps"
    fx = w - pad - _text_w(fps_txt, 0.5 * k)
    _text(frame, fps_txt, (fx, top_h * 0.63), 0.5 * k, MUTED)
    phase = hud.get("phase", "COUNT")
    pill, pill_col = {"READY": ("GET READY", WARN), "CALIB": ("CALIBRATING", WARN),
                      "COUNT": ("COUNTING", GOOD)}.get(phase, (phase, MUTED))
    px1 = fx - int(18 * k)
    px0 = px1 - _text_w(pill, 0.5 * k) - int(22 * k)
    cv2.rectangle(frame, (px0, int(top_h * 0.24)), (px1, int(top_h * 0.78)), pill_col,
                  max(1, int(2 * k)), cv2.LINE_AA)
    _text(frame, pill, (px0 + 11 * k, top_h * 0.63), 0.5 * k, pill_col)
    if hud.get("notice"):
        nx = px0 - int(18 * k) - _text_w(hud["notice"], 0.5 * k)
        _text(frame, hud["notice"], (nx, top_h * 0.63), 0.5 * k, WARN)

    # ── stats card (left) ────────────────────────────────────────────
    angle = hud.get("angle")
    has_angle = angle is not None and np.isfinite(angle)
    cx0, cy0 = pad, top_h + pad
    cw, ch = int(190 * k), int(236 * k)
    _panel(frame, cx0, cy0, cx0 + cw, cy0 + ch)
    tx = cx0 + int(18 * k)
    _text(frame, "REPS", (tx, cy0 + 30 * k), 0.5 * k, MUTED)
    _text(frame, hud.get("reps", 0), (tx, cy0 + 96 * k), 2.3 * k, TEXT, max(2, int(3 * k)))
    inc = hud.get("incomplete", 0)
    _text(frame, f"Incomplete  {inc}", (tx, cy0 + 130 * k), 0.55 * k, WARN if inc else MUTED)
    cv2.line(frame, (tx, int(cy0 + 148 * k)), (cx0 + cw - int(18 * k), int(cy0 + 148 * k)),
             (90, 90, 90), 1, cv2.LINE_AA)
    _text(frame, "ANGLE", (tx, cy0 + 176 * k), 0.5 * k, MUTED)
    if has_angle:
        val = f"{angle:.0f}"
        _text(frame, val, (tx, cy0 + 218 * k), 1.2 * k, ACCENT, max(1, int(2 * k)))
        _text(frame, "deg", (tx + _text_w(val, 1.2 * k, 2) + 8 * k, cy0 + 218 * k), 0.55 * k, MUTED)
    else:
        _text(frame, "--", (tx, cy0 + 218 * k), 1.2 * k, MUTED, max(1, int(2 * k)))

    # ── bottom banner height (needed by the gauge) ───────────────────
    bot_h = int(86 * k)
    by0 = h - bot_h

    # ── angle gauge (right): where the arm is, and where it has to get to ──
    gx1 = w - pad
    gx0 = gx1 - int(16 * k)
    gy0, gy1 = top_h + int(46 * k), by0 - int(40 * k)
    if gy1 - gy0 > 60 * k:
        _panel(frame, gx0 - 54 * k, gy0 - 34 * k, gx1 + 10 * k, gy1 + 30 * k)
        _text(frame, "180", (gx0 - 44 * k, gy0 - 12 * k), 0.45 * k, MUTED)
        _text(frame, "0", (gx0 - 20 * k, gy1 + 20 * k), 0.45 * k, MUTED)
        cv2.rectangle(frame, (gx0, gy0), (gx1, gy1), (70, 70, 70), -1)
        to_y = lambda a: int(gy1 - (gy1 - gy0) * min(max(a, 0.0), 180.0) / 180.0)
        target = hud.get("target")
        if target is not None and np.isfinite(target):
            ty = to_y(target)
            if hud.get("target_kind", "flex") == "flex":
                cv2.rectangle(frame, (gx0, ty), (gx1, gy1), (70, 120, 60), -1)   # goal zone
            else:
                cv2.rectangle(frame, (gx0, gy0), (gx1, ty), (70, 120, 60), -1)
            cv2.line(frame, (int(gx0 - 6 * k), ty), (int(gx1 + 4 * k), ty), GOOD,
                     max(1, int(2 * k)), cv2.LINE_AA)
            _text(frame, "goal", (gx0 - 48 * k, ty + 5 * k), 0.42 * k, GOOD)
        if has_angle:
            ay = to_y(angle)
            cv2.rectangle(frame, (int(gx0 - 4 * k), int(ay - 4 * k)),
                          (int(gx1 + 4 * k), int(ay + 4 * k)), ACCENT, -1)

    # ── bottom banner: instruction or feedback ───────────────────────
    _panel(frame, 0, by0, w, h, alpha=0.7)
    fb = hud.get("feedback")
    colour, big, small, bar = TEXT, "", "", None
    if phase == "READY":
        colour, big = WARN, "Get ready"
        small = "Stand so your whole arm is in view, then hold still"
        bar = hud.get("remaining", 0) / max(hud.get("phase_total", 1), 1e-6)
    elif phase == "CALIB":
        colour, big = WARN, "Bend and straighten fully"
        small = "Slowly, once or twice. This sets your personal goal"
        bar = hud.get("remaining", 0) / max(hud.get("phase_total", 1), 1e-6)
    if phase in ("READY", "CALIB") and not has_angle:
        small = "Arm not visible: step back until your whole arm is in the picture"

    if phase in ("READY", "CALIB"):
        pass
    elif fb:
        faults = fb.get("faults") or []
        if not fb.get("counted", True):
            colour, big, small = WARN, _ascii(faults[0]) if faults else "Not counted", "Not counted as a rep"
        elif faults:
            colour, big = BAD, _ascii(faults[0])
            small = f"Rep quality {fb.get('quality_score', 0):.0f}%" + (
                f"   (+{len(faults) - 1} more)" if len(faults) > 1 else "")
        else:
            colour, big, small = GOOD, "Good rep", f"Rep quality {fb.get('quality_score', 0):.0f}%"
    elif not has_angle:
        colour, big = WARN, "Arm not visible"
        small = "Step back and make sure the room is well lit"
    else:
        big = str(hud.get("state", "")).capitalize()
        small = "Press Q to finish"
        colour = MUTED

    cv2.rectangle(frame, (0, by0), (int(6 * k), h), colour, -1)          # status stripe
    _text(frame, big, (pad + 6 * k, by0 + 38 * k), 0.95 * k, colour, max(1, int(2 * k)))
    _text(frame, small, (pad + 6 * k, by0 + 68 * k), 0.55 * k, MUTED)
    if bar is not None:
        bw = int(220 * k)
        bx0, byy = w - pad - bw, by0 + int(40 * k)
        cv2.rectangle(frame, (bx0, byy), (bx0 + bw, byy + int(8 * k)), (80, 80, 80), -1)
        cv2.rectangle(frame, (bx0, byy), (bx0 + int(bw * (1.0 - min(max(bar, 0), 1))), byy + int(8 * k)),
                      colour, -1)
        secs = f"{hud.get('remaining', 0):.0f}s"
        _text(frame, secs, (bx0 + bw - _text_w(secs, 0.55 * k), byy - 10 * k), 0.55 * k, MUTED)

    # ── ring on the joint that the fault is about ────────────────────
    if fb and fb.get("faults") and pose is not None:
        key = next((kk for kk in _FAULT_JOINTS if kk in fb["faults"][0].lower()), None)
        for idx in _FAULT_JOINTS.get(key, []):
            p = pose[idx]
            if not np.isnan(p[:2]).any():
                cv2.circle(frame, _px(p, w, h, mirror), int(24 * k), colour,
                           max(2, int(3 * k)), cv2.LINE_AA)
    return frame


def draw_session_summary(frame, summary):
    """
    Full-screen summary card at end of session.
    Call after Q is pressed and session is scored.
    """
    h, w = frame.shape[:2]
    k = w / 960.0
    _panel(frame, 0, 0, w, h, alpha=0.82)

    cw, ch = int(560 * k), int(430 * k)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    _panel(frame, x0, y0, x0 + cw, y0 + ch, colour=(48, 40, 34), alpha=0.95)
    grade = str(summary.get('grade', '?'))
    g_col = {'A': GOOD, 'B': GOOD, 'C': WARN}.get(grade, BAD)
    cv2.rectangle(frame, (x0, y0), (x0 + cw, y0 + int(6 * k)), g_col, -1)

    tx = x0 + int(36 * k)
    _text(frame, "Session complete", (tx, y0 + 58 * k), 1.0 * k, TEXT, max(1, int(2 * k)))

    # grade badge on the right
    gx, gy = x0 + cw - int(110 * k), y0 + int(150 * k)
    cv2.circle(frame, (gx, gy), int(58 * k), g_col, max(2, int(4 * k)), cv2.LINE_AA)
    _text(frame, grade, (gx - _text_w(grade, 2.2 * k, 3) // 2, gy + 24 * k), 2.2 * k, g_col,
          max(2, int(3 * k)))
    _text(frame, "GRADE", (gx - _text_w("GRADE", 0.5 * k) // 2, gy + 88 * k), 0.5 * k, MUTED)

    n_inc = summary.get('n_incomplete', 0)
    rows = [
        ("Attempts",        f"{summary.get('n_attempts', summary.get('n_reps', 0))}", TEXT),
        ("Completed reps",  f"{summary.get('n_reps', 0)}", TEXT),
        ("Incomplete",      f"{n_inc}", WARN if n_inc else TEXT),
        ("Clean reps",      f"{summary.get('n_clean_reps', 0)}", TEXT),
        ("Session quality", f"{summary.get('mean_quality', 0):.0f}%", g_col),
    ]
    y = y0 + int(112 * k)
    for label, value, col in rows:
        _text(frame, label, (tx, y), 0.65 * k, MUTED)
        _text(frame, value, (tx + 220 * k, y), 0.75 * k, col, max(1, int(2 * k)))
        y += int(40 * k)

    top_fault = summary.get('top_fault')
    if top_fault:
        y += int(8 * k)
        _text(frame, "Most common problem", (tx, y), 0.55 * k, MUTED)
        _text(frame, _ascii(top_fault), (tx, y + 32 * k), 0.7 * k, WARN)
    _text(frame, "Closing...", (tx, y0 + ch - 22 * k), 0.5 * k, MUTED)