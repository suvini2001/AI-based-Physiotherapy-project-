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


import numpy as np
import pandas as pd

def fill_gaps(arr, max_gap=5, method='cubicspline'):
    """
    Fills short NaN gaps up to `max_gap` frames using cubic spline interpolation
    while leaving longer occluded segments as NaN.
    """
    s = pd.Series(arr, dtype=float)
    if method == 'cubicspline' and s.notna().sum() >= 4:
        filled = s.interpolate(method='cubicspline', limit=max_gap, limit_area='inside')
    else:
        filled = s.interpolate(method='linear', limit=max_gap, limit_area='inside')
    return np.clip(filled.to_numpy(), 0.0, 180.0)