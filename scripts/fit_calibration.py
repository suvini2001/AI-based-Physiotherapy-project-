# scripts/fit_calibration.py
"""
Fit the population calibration curve (variant D_2D_PerView) from exported
raw/truth angle pairs, check it with leave-one-person-out, and save it
where capture/calibrate_ui.py looks for it.

The pairs file comes from STEP 7 of benchmark_3d_world.ipynb:
    calib_pairs_<dataset>.npz  with arrays  raw, truth, person, view

    python scripts/fit_calibration.py data/calib_pairs_rehab24.npz \
        --views Profile Profile/cam18 --view-class profile \
        --out data/population_curve.json

Then:
    python capture/calibrate_ui.py --side right          # straight-arm check
    python demo_cv.py --variant A --calibration data/user_calibration.json
"""
import argparse, datetime, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibration import fit_curve, apply_calibration, save_params


def load_pairs(path, views=None):
    d = np.load(path, allow_pickle=False)
    raw, truth = d["raw"].astype(float), d["truth"].astype(float)
    person, view = d["person"].astype(str), d["view"].astype(str)
    keep = np.isfinite(raw) & np.isfinite(truth)
    if views:
        keep &= np.isin(view, views)
    return raw[keep], truth[keep], person[keep], view[keep]


def extend_below(params):
    """
    Below the lowest well-supported knot, keep the same offset instead of a
    flat line. REHAB24-6 profile data has almost no truth angles under ~60 deg
    (push-ups), so a flat curve would map every flexed elbow (30-60 deg) to
    ~62 deg and squash the bottom of every curl.
    """
    kr, kc = list(params["knots_raw"]), list(params["knots_corrected"])
    if kr[0] > 0:
        off = kc[0] - kr[0]
        kr.insert(0, 0.0)
        kc.insert(0, max(0.0, off))
    return {**params, "knots_raw": kr, "knots_corrected": kc,
            "extrapolated_below": round(float(params["knots_raw"][0]), 1)}


def fit(raw, truth, min_count, extrapolate):
    params = fit_curve(raw, truth, min_count=min_count)
    return extend_below(params) if extrapolate else params


def lopo(raw, truth, person, min_count=15, extrapolate=True):
    """Per held-out person: MAE before and after a curve fitted on the others."""
    rows = []
    for p in np.unique(person):
        tr, te = person != p, person == p
        try:
            params = fit(raw[tr], truth[tr], min_count, extrapolate)
        except ValueError:
            continue
        cal = apply_calibration(raw[te], params)
        rows.append((p, int(te.sum()),
                     float(np.mean(np.abs(raw[te] - truth[te]))),
                     float(np.mean(np.abs(cal - truth[te]))),
                     float(np.mean(cal - truth[te]))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs")
    ap.add_argument("--views", nargs="*", default=None,
                    help="view labels to use, e.g. Profile Profile/cam18")
    ap.add_argument("--view-class", default=None, help="front / profile / ...")
    ap.add_argument("--variant", default="A", help="angle method the pairs came from")
    ap.add_argument("--dataset", default="REHAB24-6")
    ap.add_argument("--min-count", type=int, default=300,
                    help="frames a 10-deg bin needs before it becomes a knot")
    ap.add_argument("--no-extrapolate", action="store_true",
                    help="flat curve below the lowest knot (old behaviour)")
    ap.add_argument("--out", default="data/population_curve.json")
    a = ap.parse_args()

    raw, truth, person, view = load_pairs(a.pairs, a.views)
    n_people = len(np.unique(person))
    print(f"{len(raw)} frames from {n_people} people, views {sorted(map(str, set(view)))}")
    if n_people < 3:
        sys.exit("Need frames from at least 3 people.")

    extrapolate = not a.no_extrapolate
    rows = lopo(raw, truth, person, a.min_count, extrapolate)
    if rows:
        before = np.median([r[2] for r in rows])
        after = np.median([r[3] for r in rows])
        print("\nleave-one-person-out (median over people):")
        print(f"  MAE raw {before:.2f} deg -> calibrated {after:.2f} deg")
        for p, n, b, c, bias in rows:
            print(f"    {p:>10s}  n={n:6d}  raw {b:5.1f}  cal {c:5.1f}  bias {bias:+5.1f}")

    params = fit(raw, truth, a.min_count, extrapolate)
    lo, hi = np.percentile(truth, [5, 95])
    print(f"truth range covered (5-95%): {lo:.0f}-{hi:.0f} deg; "
          f"angles outside it are extrapolated, not measured")
    params.update({
        "truth_range_5_95": [round(float(lo), 1), round(float(hi), 1)],
        "variant": a.variant,
        "view_class": a.view_class,
        "fitted_views": sorted(map(str, set(view))),
        "dataset": a.dataset,
        "n_people": int(n_people),
        "lopo_mae_raw": round(float(before), 2) if rows else None,
        "lopo_mae_calibrated": round(float(after), 2) if rows else None,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    save_params(a.out, params)
    print(f"\nSaved {a.out} ({len(params['knots_raw'])} knots)")


if __name__ == "__main__":
    main()
