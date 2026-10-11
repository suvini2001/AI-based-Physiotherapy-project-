"""Which filter is best? Test each one on a signal where we KNOW the true answer."""
import numpy as np
from filters import (butterworth_filter, savitzky_golay_filter,
                     rolling_median_filter, OneEuroFilter)

FS = 29.5                                          # frames per second
t = np.arange(0, 10, 1 / FS)
truth = 100 + 40 * np.sin(2 * np.pi * 0.5 * t) + 15 * np.sin(2 * np.pi * 2 * t)

rng = np.random.default_rng(0)
noisy = truth + rng.normal(0, 3, len(t))           # 3 degrees of random jitter
noisy[[60, 140, 220]] += 60                        # three one-frame spikes
noisy[180:182] += 60                               # one two-frame spike


def one_euro(x, mincutoff, beta):
    f = OneEuroFilter(FS, mincutoff=mincutoff, beta=beta)
    return np.array([f(v) for v in x])


# name -> (function, can it run LIVE?)
candidates = {
    "no filter":                      (lambda x: x, True),
    "median 5 only":                  (lambda x: rolling_median_filter(x, 5), False),
    "Butterworth 6 Hz":               (lambda x: butterworth_filter(x, FS, 6), False),
    "Savitzky-Golay 9,3":             (lambda x: savitzky_golay_filter(x, 9, 3), False),
    "median 5 + Butterworth 6 Hz":    (lambda x: butterworth_filter(rolling_median_filter(x, 5), FS, 6), False),
    "median 5 + Savitzky-Golay 9,3":  (lambda x: savitzky_golay_filter(rolling_median_filter(x, 5), 9, 3), False),
    "Butterworth 6 Hz causal":        (lambda x: butterworth_filter(x, FS, 6, causal=True), True),
    "Butterworth 10 Hz causal":       (lambda x: butterworth_filter(x, FS, 10, causal=True), True),
    "One Euro (1.0, 0.05)":           (lambda x: one_euro(x, 1.0, 0.05), True),
    "One Euro (2.0, 0.10)":           (lambda x: one_euro(x, 2.0, 0.10), True),
    "One Euro (3.0, 0.30)":           (lambda x: one_euro(x, 3.0, 0.30), True),
}


def lag_in_frames(y):
    """How many frames the output is late compared with the truth."""
    def corr(k):
        return np.corrcoef(np.roll(y, -k)[15:-15], truth[15:-15])[0, 1]
    return max(range(-10, 11), key=corr)


rows = []
for name, (fn, live) in candidates.items():
    y = fn(noisy)
    rmse = np.sqrt(np.mean((y[30:] - truth[30:]) ** 2))
    lag = lag_in_frames(y)
    rows.append((name, rmse, lag, lag / FS * 1000, live))

print(f"{'filter':32s} {'error (deg)':>11s} {'lag (frames)':>12s} {'lag (ms)':>9s}  live?")
for name, rmse, lag, ms, live in sorted(rows, key=lambda r: r[1]):
    print(f"{name:32s} {rmse:11.2f} {lag:12d} {ms:9.0f}  {'yes' if live else 'no'}")