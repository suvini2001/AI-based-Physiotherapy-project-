import numpy as np
from scipy.signal import butter, filtfilt, lfilter, lfilter_zi, savgol_filter
from scipy.ndimage import median_filter


def butterworth_filter(data, fs, cutoff=6.0, order=2, causal=False):
    """fs = real frame rate. causal=True is for live use.
    NaN-safe, and safe at low frame rates and on short clips."""
    arr = np.asarray(data, float)
    if arr.size == 0:
        return arr.copy()
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr.copy()
    filled = fill_gaps(arr.copy())

    nyq = fs / 2.0
    wn = min(cutoff / nyq, 0.99)          # ← 1. අඩු fps එකට clamp
    b, a = butter(order, wn, btype="low")

    padlen = 3 * max(len(a), len(b))
    if causal:
        zi = lfilter_zi(b, a) * filled[0]
        res = lfilter(b, a, filled, zi=zi)[0]
    elif len(filled) <= padlen:
        res = filled                       # ← 2. කෙටි clip එකට filtfilt skip
    else:
        res = filtfilt(b, a, filled)

    res = np.asarray(res, float)
    res[nan_mask] = np.nan
    return res

def savitzky_golay_filter(data, window_length=9, polyorder=3):
    """window_length must be odd and larger than polyorder."""
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)


def rolling_median_filter(data, window_size=5):
    """Removes spikes shorter than about half the window."""
    return median_filter(data, size=window_size)


class OneEuroFilter:
    """Casiez et al. 2012. Feed one value per frame.
    mincutoff: lower = less jitter when slow. beta: higher = less lag when fast."""
    def __init__(self, fs, mincutoff=1.0, beta=0.05, dcutoff=1.0):
        self.fs, self.mincutoff, self.beta, self.dcutoff = fs, mincutoff, beta, dcutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * self.fs)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) * self.fs
        ad = self._alpha(self.dcutoff)
        self.dx_prev = ad * dx + (1 - ad) * self.dx_prev
        a = self._alpha(self.mincutoff + self.beta * abs(self.dx_prev))
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev

def fill_gaps(values):
    """Fill NaN by straight-line interpolation. For testing the filters only;
    A's gap-filling function (step 1.6) is the real one."""
    values = np.asarray(values, float)
    idx = np.arange(len(values))
    good = ~np.isnan(values)
    return np.interp(idx, idx[good], values[good])