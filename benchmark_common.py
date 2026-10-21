"""A3 - helpers shared by both benchmark notebooks: truth loader, alignment, metrics, results table."""
import glob
import numpy as np
import pandas as pd
from angles import angle
from filters import fill_gaps

VIEW_ORDER = ["front", "half-profile", "profile", "side"]
ROW_COLUMNS = ["variant", "video", "side", "cam", "ex", "view", "person",
               "n", "coverage", "mae", "rmse", "bias", "r", "share_gt20"]


def load_truth_array(path, names):
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=True)
    elif path.endswith(".npz"):
        z = np.load(path)
        arr = z[z.files[0]]
    else:
        arr = np.genfromtxt(path, delimiter="," if "," in open(path).readline() else None)
    arr = np.asarray(arr, float)
    nj = len(names)
    if arr.ndim == 2:
        arr = arr.reshape(len(arr), nj, -1)
    if arr.ndim == 3 and arr.shape[1] != nj and arr.shape[0] == nj:
        arr = arr.transpose(1, 0, 2)
    assert arr.ndim == 3 and arr.shape[1] == nj, f"unexpected joints shape {arr.shape}"
    return arr


def truth_elbow_3d(arr, names, arm, use_dims=3):
    """arm = (shoulder, elbow, wrist) joint NAMES. use_dims=3 true 3D angle, 2 projected 2D angle."""
    idx = [names.index(n) for n in arm]
    return angle(arr[:, idx[0], :use_dims], arr[:, idx[1], :use_dims], arr[:, idx[2], :use_dims])


def overlap(a, b, lag):
    """Pairs a[i] with b[i+lag]."""
    if lag >= 0:
        n = min(len(a), len(b) - lag)
        return a[:n], b[lag:lag + n]
    n = min(len(a) + lag, len(b))
    return a[-lag:-lag + n], b[:n]


def best_lag(pred, truth, max_lag=30):
    if np.isnan(pred).all() or np.isnan(truth).all():
        return 0, -2.0
    prep = lambda x: (lambda y: y - y.mean())(fill_gaps(np.asarray(x, float).copy()))
    a, b = prep(pred), prep(truth)
    best, best_r = 0, -2.0
    for lag in range(-max_lag, max_lag + 1):
        x, y = overlap(a, b, lag)
        if len(x) < 10 or x.std() == 0 or y.std() == 0:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if r > best_r:
            best, best_r = lag, r
    return best, best_r


def metrics(pred, truth, big_error=20.0):
    """Read share_gt20 together with coverage: a gate that deletes hard frames looks better than it is."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    ok = ~(np.isnan(pred) | np.isnan(truth))
    if ok.sum() < 10:
        return None
    x, y = pred[ok], truth[ok]
    d = x - y
    r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan
    return dict(n=int(ok.sum()), coverage=100.0 * ok.sum() / len(pred),
                mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())),
                bias=float(d.mean()), r=float(r), share_gt20=100.0 * float(np.mean(np.abs(d) > big_error)))


def make_row(variant, pred, truth, meta):
    m = metrics(pred, truth)
    if m is None:
        return None
    row = {"variant": variant, **meta, **m}
    return {c: row.get(c) for c in ROW_COLUMNS}


def person_of(name):
    """'PM_013_..' -> 'PM_013'. CHECK that one id = one person in REHAB24-6."""
    import re
    m = re.search(r"(PM_\d+)", str(name))
    return m.group(1) if m else str(name)


def save_rows(rows, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame([r for r in rows if r], columns=ROW_COLUMNS).to_csv(path, index=False)


def load_variants(pattern="results/variant_*.csv"):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _views(df):
    seen = [v for v in VIEW_ORDER if v in set(df["view"])]
    return seen + sorted(set(df["view"]) - set(seen))


def summary(df, min_coverage=50.0):
    d = df[df["coverage"] >= min_coverage]
    return d.groupby(["variant", "view"]).agg(recordings=("mae", "size"), mae=("mae", "median"),
                                              share_gt20=("share_gt20", "median"),
                                              coverage=("coverage", "median")).reset_index()


def results_table(df, min_coverage=50.0):
    """Rows = variants, columns = views, cell = 'median MAE (share >20 deg)'."""
    s = summary(df, min_coverage)
    out = pd.DataFrame(index=sorted(set(s["variant"])), columns=_views(df), dtype=object).fillna("-")
    for _, r in s.iterrows():
        out.loc[r["variant"], r["view"]] = f"{r['mae']:.1f} ({r['share_gt20']:.0f}%)"
    out.index.name = "variant"
    return out


def to_markdown(table):
    cols = list(table.columns)
    lines = ["| Variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, row in table.iterrows():
        lines.append(f"| {name} | " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def delta(df, new, old, metric="mae", min_coverage=50.0):
    """new - old per view. Negative = `new` is better."""
    s = summary(df, min_coverage).pivot(index="view", columns="variant", values=metric)
    return (s[new] - s[old]).round(2)