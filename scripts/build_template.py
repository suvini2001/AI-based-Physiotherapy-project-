# scripts/build_template.py
"""
Build a calibrated DTW template from HEALTHY reps (Labchurat 2026, 3.3.3-3.3.5).

One session JSON file = one person. Record each person with
demo_cv.py (side view, normal speed, full comfortable range), then:

    python scripts/build_template.py data/healthy/p01.json data/healthy/p02.json ... \
        --exercise elbow_flexion --recorded-by healthy_volunteers

The mean trajectory, mu and sigma all come from these reps, as in the paper
(subset B). Test on DIFFERENT recordings with scripts/evaluate_dtw.py.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assessment.dtw import build_template

MIN_PLAUSIBLE = {"elbow_flexion": 25.0}   # below this the 2D angle has collapsed


def load_reps(path, exercise):
    reps = json.load(open(path)).get("reps", [])
    floor = MIN_PLAUSIBLE.get(exercise, -1e9)
    good, dropped = [], 0
    for r in reps:
        s = r.get("_angle_series")
        if s and len(s) >= 5 and r.get("min_angle", 0) >= floor:
            good.append(s)
        else:
            dropped += 1
    return good, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="one session JSON per person")
    ap.add_argument("--exercise", default="elbow_flexion")
    ap.add_argument("--recorded-by", required=True,
                    help="who performed the reps, e.g. healthy_volunteers")
    ap.add_argument("--out", default="template.json")
    ap.add_argument("--min-people", type=int, default=3)
    a = ap.parse_args()

    all_series, n_people = [], 0
    for path in a.sessions:
        reps, dropped = load_reps(path, a.exercise)
        print(f"{path}: {len(reps)} reps used, {dropped} dropped")
        if reps:
            n_people += 1
            all_series += reps

    if n_people < a.min_people:
        sys.exit(f"Only {n_people} people with usable reps; need {a.min_people}.")

    t = build_template(all_series, a.exercise, a.recorded_by, n_people)
    json.dump(t, open(a.out, "w"), indent=2)
    print(f"Saved {a.out}: {t['n_reps']} reps from {n_people} people, "
          f"mu={t['mu_dtw']:.1f}, sigma={t['sigma_dtw']:.1f}")


if __name__ == "__main__":
    main()
