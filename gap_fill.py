"""Step 1.6 - fill short gaps in a landmark array.
(frames, 33, 4) in -> (frames, 33, 4) out. B's mask makes a dropped point NaN in all 4 numbers."""
import numpy as np


def _fill_1d(series, max_gap):
    s = series.copy()
    is_nan = np.isnan(s)
    if not is_nan.any() or (~is_nan).sum() < 2:
        return s
    idx = np.arange(len(s))
    good = ~is_nan
    straight_line = np.interp(idx, idx[good], s[good])

    change = np.diff(np.concatenate([[False], is_nan, [False]]).astype(int))
    starts = np.where(change == 1)[0]    # first NaN of each gap
    ends = np.where(change == -1)[0]     # one past the last NaN

    for a, b in zip(starts, ends):       # each gap is s[a:b]
        if a == 0 or b == len(s):        # no good frame on one side
            continue
        if (b - a) <= max_gap:
            s[a:b] = straight_line[a:b]
    return s


def fill_gaps(landmarks, max_gap=5):
    """Linearly fill gaps of <= max_gap frames. Longer gaps stay NaN."""
    out = landmarks.copy()               # never edit the caller's array
    for j in range(out.shape[1]):        # 33 landmarks
        for c in range(4):               # x, y, z AND visibility
            out[:, j, c] = _fill_1d(out[:, j, c], max_gap)
    return out