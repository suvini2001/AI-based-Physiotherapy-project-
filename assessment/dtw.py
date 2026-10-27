# assessment/dtw.py
"""
DTW-based movement quality scoring.
Source: Labchurat et al. Sensors 2026 (single-angle DTW, Z-score scoring)
        Dill & Rohr arXiv 2023 (per-joint extension, optional)
"""
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

try:
    from fastdtw import fastdtw
    _HAVE_FASTDTW = True
except ImportError:
    _HAVE_FASTDTW = False
    print("WARNING: fastdtw not installed. "
          "Run: pip install fastdtw")


def _dtw_distance(a, b):
    """DTW distance between two 1-D arrays."""
    if _HAVE_FASTDTW:
        dist, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
        return float(dist)
    # fallback: simple Euclidean distance on normalised series
    n = min(len(a), len(b))
    return float(np.sum(np.abs(a[:n] - b[:n])))


def normalise_series(angles, length=100):
    """Resample angle series to a standard length (0-100 phase)."""
    angles = np.asarray(angles, dtype=float)
    if len(angles) < 2:
        return np.zeros(length)
    x_old = np.linspace(0, 1, len(angles))
    x_new = np.linspace(0, 1, length)
    return interp1d(x_old, angles, kind='linear')(x_new)


def load_template(path='template.json'):
    """Load reference template. Returns dict."""
    with open(path) as f:
        return json.load(f)


def dtw_score(angle_series, template):
    """
    Compute 0-100 similarity score for one rep.
    100 = perfect match to template.
    Below 70 = meaningfully different from normal movement.

    Parameters
    ----------
    angle_series : list of float — angle values for this rep
    template     : dict loaded by load_template()
    """
    if not angle_series or len(angle_series) < 5:
        return 50.0   # neutral when data is insufficient

    rep_norm = normalise_series(angle_series)
    ref_norm = normalise_series(template['mean_trajectory'])

    distance = _dtw_distance(rep_norm, ref_norm)

    mu    = float(template.get('mu_dtw', 0.0))
    sigma = float(template.get('sigma_dtw', 1.0))

    if sigma < 1e-6:
        # template not yet calibrated with real data
        # fall back to a simple distance-to-score mapping
        # distance=0 → 100, distance=50 → 50, distance=100 → 0
        return float(np.clip(100.0 - distance, 0, 100))

    z     = (distance - mu) / sigma
    score = 100.0 * float(norm.cdf(-z))
    return round(float(np.clip(score, 0, 100)), 1)


def calibrate_template(angle_series_list, template_path='template.json'):
    """
    After recording N correct reps, compute mu_dtw and sigma_dtw
    so the Z-score formula is properly calibrated.

    Call this once after collecting 5-10 correct reps from a
    healthy subject or physiotherapist.
    """
    template = load_template(template_path)
    ref = normalise_series(template['mean_trajectory'])

    distances = []
    for series in angle_series_list:
        n = normalise_series(series)
        distances.append(_dtw_distance(n, ref))

    template['mu_dtw']    = float(np.mean(distances))
    template['sigma_dtw'] = float(max(np.std(distances), 1.0))
    template['n_calibration_reps'] = len(distances)

    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    print(f"Template calibrated from {len(distances)} reps: "
          f"mu={template['mu_dtw']:.1f}  "
          f"sigma={template['sigma_dtw']:.1f}")
    return template