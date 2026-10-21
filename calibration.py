"""
A5 - Calibration Module:
Population correction curve, straight-arm session check, optional session offset.
It removes known systematic BIAS while preserving a person's individual range of motion (ROM).
"""

import datetime
import json
import os
import numpy as np
import pandas as pd

CURVE_VERSION = "pop-curve-v1"


def _pav(y, w):
    """Pool-adjacent-violators: closest non-decreasing sequence to y (weighted by w)."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi))
        wts.append(float(wi))
        cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wt = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wt
            c = cnt[-2] + cnt[-1]
            vals[-2:], wts[-2:], cnt[-2:] = [v], [wt], [c]
    out = []
    for v, c in zip(vals, cnt):
        out += [v] * c
    return np.array(out)


def identity_params():
    """'Do nothing' curve for baseline testing before fitting."""
    return {
        "curve_version": "identity",
        "knots_raw": [0.0, 180.0],
        "knots_corrected": [0.0, 180.0],
        "offset": None,
        "view_class": None,
        "date": None,
        "yaw_at_capture": None,
    }


def fit_curve(raw, truth, bin_width=10, min_count=30, anchor_top=True):
    """Fit monotonic non-decreasing calibration curve mapping raw -> truth angles."""
    raw, truth = np.asarray(raw, float), np.asarray(truth, float)
    ok = ~(np.isnan(raw) | np.isnan(truth))
    raw, truth = raw[ok], truth[ok]

    edges = np.arange(0, 180 + bin_width, bin_width)
    which = np.clip(np.digitize(raw, edges) - 1, 0, len(edges) - 2)

    kr, kt, kn = [], [], []
    for b in range(len(edges) - 1):
        m = which == b
        if m.sum() >= min_count:
            kr.append(float(np.median(raw[m])))
            kt.append(float(np.median(truth[m])))
            kn.append(int(m.sum()))

    if len(kr) < 3:
        raise ValueError(
            f"Only {len(kr)} bins have {min_count}+ samples; need data over a wider range."
        )

    kc = _pav(kt, kn).tolist()
    if anchor_top and kr[-1] < 180.0:
        kr.append(180.0)
        kc.append(180.0)
    kc = [min(v, 180.0) for v in kc]

    return {
        "curve_version": CURVE_VERSION,
        "knots_raw": kr,
        "knots_corrected": kc,
        "bin_width": bin_width,
        "min_count": min_count,
        "n_samples": int(len(raw)),
        "offset": None,
        "view_class": None,
        "date": None,
        "yaw_at_capture": None,
    }


def apply_calibration(angle, params):
    """Raw degrees -> corrected degrees (piecewise linear interpolation)."""
    a = np.asarray(angle, float)
    out = np.interp(a, params["knots_raw"], params["knots_corrected"])
    if params.get("offset") is not None:
        out = out + params["offset"]
    return np.clip(out, 0.0, 180.0)


def check_params(params, tol_at_180=2.0):
    kr, kc = np.asarray(params["knots_raw"], float), np.asarray(
        params["knots_corrected"], float
    )
    if len(kr) != len(kc) or len(kr) < 2:
        raise ValueError(
            "knots must be two equal-length lists with at least 2 entries"
        )
    if np.any(np.diff(kr) <= 0):
        raise ValueError("knots_raw must be strictly increasing")
    if np.any(np.diff(kc) < -1e-9):
        raise ValueError("calibration curve must never go down")

    top = float(apply_calibration(180.0, {**params, "offset": None}))
    if abs(top - 180.0) > tol_at_180:
        raise ValueError(
            f"curve maps 180 deg to {top:.1f}; expected within {tol_at_180} deg of 180"
        )
    return True


def _range(x):
    x = x[~np.isnan(x)]
    return (
        float(np.percentile(x, 95) - np.percentile(x, 5))
        if len(x)
        else np.nan
    )


def leave_one_person_out(
    records, bin_width=10, min_count=30, big_error=20.0
):
    """LOPO Cross-Validation to evaluate population calibration performance."""
    rows = []
    for p in sorted({r["person"] for r in records}):
        train = [r for r in records if r["person"] != p]
        test = [r for r in records if r["person"] == p]
        if not train:
            continue

        params = fit_curve(
            np.concatenate([r["raw"] for r in train]),
            np.concatenate([r["truth"] for r in train]),
            bin_width,
            min_count,
        )

        raw = np.concatenate([r["raw"] for r in test])
        truth = np.concatenate([r["truth"] for r in test])
        cal = apply_calibration(raw, params)

        ok = ~(np.isnan(raw) | np.isnan(truth))
        if ok.sum() < 10:
            continue

        d0, d1 = raw[ok] - truth[ok], cal[ok] - truth[ok]
        rows.append(
            dict(
                person=p,
                n=int(ok.sum()),
                mae_raw=np.abs(d0).mean(),
                mae_cal=np.abs(d1).mean(),
                gt20_raw=100 * np.mean(np.abs(d0) > big_error),
                gt20_cal=100 * np.mean(np.abs(d1) > big_error),
                range_raw=_range(raw[ok]),
                range_cal=_range(cal[ok]),
                range_truth=_range(truth[ok]),
            )
        )
    return pd.DataFrame(rows)


def fit_without_each_person(records, bin_width=10, min_count=30):
    """Fits N models excluding one person at a time (for variant predictions)."""
    out = {}
    for p in sorted({r["person"] for r in records}):
        train = [r for r in records if r["person"] != p]
        if train:
            out[p] = fit_curve(
                np.concatenate([r["raw"] for r in train]),
                np.concatenate([r["truth"] for r in train]),
                bin_width,
                min_count,
            )
    return out


def session_check(
    straight_angles, params, expected=175.0, tol=15.0, min_samples=10
):
    """Quality check during dynamic capture (e.g. initial straight arm pose)."""
    a = np.asarray(straight_angles, float)
    a = a[~np.isnan(a)]
    if len(a) < min_samples:
        return dict(
            status="fail",
            n=int(len(a)),
            raw_median=None,
            calibrated_median=None,
            message="Arm not tracked. Face the camera, show the whole arm, add light.",
        )
    raw_med = float(np.median(a))
    cal_med = float(apply_calibration(raw_med, {**params, "offset": None}))
    ok = abs(cal_med - expected) <= tol
    msg = (
        "Straight-arm reading looks right."
        if ok
        else f"Straight arm reads {cal_med:.0f} deg (expected about {expected:.0f}). Check camera position."
    )
    return dict(
        status="ok" if ok else "warn",
        n=int(len(a)),
        raw_median=raw_med,
        calibrated_median=cal_med,
        message=msg,
    )


def make_session_params(
    params,
    straight_angles,
    expected=175.0,
    tol=15.0,
    clinician_confirmed=False,
    view_class=None,
    yaw_at_capture=None,
):
    check = session_check(straight_angles, params, expected, tol)
    out = {
        **params,
        "offset": None,
        "view_class": view_class,
        "yaw_at_capture": yaw_at_capture,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "straight_check": check,
    }
    if clinician_confirmed and check["calibrated_median"] is not None:
        out["offset"] = float(expected - check["calibrated_median"])
    return out


def save_params(path, params):
    check_params({**params, "offset": None})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)


def load_params(path):
    with open(path) as f:
        params = json.load(f)
    check_params({**params, "offset": None})
    return params