# assessment/dtw.py
"""
DTW-based movement quality scoring.
Method: Labchurat et al., Sensors 2026, 26, 1219, Section 3.3
  - mean normal trajectory from healthy reps          (Eq. 6-9)
  - d = DTW(rep, mean trajectory)                      (Eq. 11, single angle)
  - Z = (d - mu) / sigma, mu/sigma from healthy reps   (Eq. 12)
  - score = 100 / max(Z, 1)                            (Eq. 14)
Deviation from the paper: every series is resampled to 100 points
instead of the mean length of the healthy reps.
"""
import json
import numpy as np
from scipy.interpolate import interp1d

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
    sigma = float(template.get('sigma_dtw', 0.0))
    if sigma <= 0:
        return 50.0   # no mu/sigma yet: neutral, do not guess

    z = (distance - mu) / sigma
    return round(z_to_score(z), 1)


def z_to_score(z):
    """Labchurat 2026, Eq. 14: 100 when Z <= 1, then 100/Z (Z=2 -> 50, Z=4 -> 25)."""
    return 100.0 / max(float(z), 1.0)


def is_calibrated(template):
    """True only for a template built from real recorded reps."""
    return bool(template) and template.get('calibrated') is True


def build_template(series_list, exercise, recorded_by, n_people,
                   sigma_floor=1.0):
    """
    Build a calibrated template from healthy reps (Labchurat 3.3.3-3.3.5).

    series_list : list of angle series, one per rep, from several people
    sigma_floor : lower bound on sigma so a tiny, very uniform sample
                  does not make the score over-strict
    """
    series_list = [s for s in series_list if s is not None and len(s) >= 5]
    if len(series_list) < 3:
        raise ValueError("need at least 3 reps to build a template")

    resampled = np.array([normalise_series(s) for s in series_list])
    mean_traj = resampled.mean(axis=0)
    distances = [_dtw_distance(r, mean_traj) for r in resampled]

    return {
        "exercise":           exercise,
        "recorded_by":        recorded_by,
        "calibrated":         True,
        "method":             "Labchurat2026_Eq12_Eq14",
        "n_people":           int(n_people),
        "n_reps":             len(series_list),
        "n_calibration_reps": len(series_list),
        "mean_trajectory":    mean_traj.tolist(),
        "std_trajectory":     resampled.std(axis=0).tolist(),
        "mu_dtw":             float(np.mean(distances)),
        "sigma_dtw":          float(max(np.std(distances), sigma_floor)),
        "sigma_floor":        float(sigma_floor),
    }


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