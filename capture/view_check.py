# capture/view_check.py
"""B1 screen side – live yaw warning overlay.

Draws a green 'Good' or red 'Please face the camera' banner on a frame.
Designed to be called once per frame inside demo_cv.py.
"""
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "vision") not in sys.path:
    sys.path.insert(0, str(ROOT / "vision"))

from overlay import put_big_text   # noqa: E402

# thresholds (degrees) — also written to config.yaml in Step 5
YAW_WARN_DEG  = 30.0   # amber warning starts here
YAW_BLOCK_DEG = 50.0   # red block: gate will reject most frames above this


def draw_yaw_message(frame, yaw_deg, warn_at=YAW_WARN_DEG, block_at=YAW_BLOCK_DEG):
    """Draw a coloured banner on frame in-place.  Returns the status string.

    yaw_deg: float or None (None = landmarks not found).
    Status: 'ok', 'warn', 'block', 'unknown'.
    """
    if yaw_deg is None or not np.isfinite(float(yaw_deg)):
        put_big_text(frame, "Align yourself in frame", (20, 170), 0.8, (200, 200, 200))
        return "unknown"

    y = float(yaw_deg)

    if y >= block_at:
        msg = f"Please face the camera  ({y:.0f} deg)"
        colour = (0, 0, 220)          # red (BGR)
        status = "block"
    elif y >= warn_at:
        msg = f"Turn slightly toward camera  ({y:.0f} deg)"
        colour = (0, 140, 255)        # amber
        status = "warn"
    else:
        msg = f"Good  ({y:.0f} deg)"
        colour = (0, 210, 0)          # green
        status = "ok"

    put_big_text(frame, msg, (20, 170), 0.75, colour)
    return status


def draw_yaw_bar(frame, yaw_deg, warn_at=YAW_WARN_DEG, block_at=YAW_BLOCK_DEG,
                 bar_x=20, bar_y=None, bar_w=200, bar_h=10):
    """Draw a small horizontal gauge showing yaw severity (optional extra visual)."""
    h = frame.shape[0]
    y0 = bar_y if bar_y is not None else h - 30
    # background bar
    cv2.rectangle(frame, (bar_x, y0), (bar_x + bar_w, y0 + bar_h), (60, 60, 60), -1)
    if yaw_deg is None or not np.isfinite(float(yaw_deg)):
        return
    fill = int(np.clip(float(yaw_deg) / 90.0 * bar_w, 0, bar_w))
    colour = (0, 210, 0) if float(yaw_deg) < warn_at else (
             (0, 140, 255) if float(yaw_deg) < block_at else (0, 0, 220))
    cv2.rectangle(frame, (bar_x, y0), (bar_x + fill, y0 + bar_h), colour, -1)
    cv2.putText(frame, "yaw", (bar_x + bar_w + 6, y0 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)