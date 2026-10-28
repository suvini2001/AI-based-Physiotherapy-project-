# scripts/evaluate_dtw.py
"""
Check that the DTW score separates normal from abnormal reps
(Labchurat 2026, Section 4: subsets B / N / A).

    python scripts/evaluate_dtw.py --normal data/test_normal/*.json \
        --abnormal data/test_abnormal/*.json --template template.json

    # leave-one-person-out: build the template from all OTHER people's
    # healthy files and score each held-out person
    python scripts/evaluate_dtw.py --lopo data/healthy/*.json \
        --abnormal data/test_abnormal/*.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assessment.dtw import load_template, dtw_score, build_template
from scripts.build_template import load_reps


def series(paths, exercise):
    out = []
    for p in paths:
        out += load_reps(p, exercise)[0]
    return out


def auc(pos, neg):
    """P(normal score > abnormal score); 0.5 = no separation, 1.0 = perfect."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def describe(name, scores):
    s = np.asarray(scores)
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    print(f"{name:9s} n={len(s):3d}  median={med:5.1f}  IQR=[{q1:5.1f}, {q3:5.1f}]  "
          f"score=100: {np.mean(s >= 99.95):4.0%}  score<50 (Z>2): {np.mean(s < 50):4.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", nargs="*", default=[])
    ap.add_argument("--abnormal", nargs="*", default=[])
    ap.add_argument("--lopo", nargs="*", default=[],
                    help="healthy files, one per person, for leave-one-person-out")
    ap.add_argument("--template", default="template.json")
    ap.add_argument("--exercise", default="elbow_flexion")
    a = ap.parse_args()

    normal = []
    if a.lopo:
        for i, held in enumerate(a.lopo):
            rest = [p for j, p in enumerate(a.lopo) if j != i]
            t = build_template(series(rest, a.exercise), a.exercise, "lopo", len(rest))
            normal += [dtw_score(s, t) for s in series([held], a.exercise)]
        template = build_template(series(a.lopo, a.exercise), a.exercise, "lopo", len(a.lopo))
    else:
        template = load_template(a.template)
        normal = [dtw_score(s, template) for s in series(a.normal, a.exercise)]

    abnormal = [dtw_score(s, template) for s in series(a.abnormal, a.exercise)]
    if normal:
        describe("normal", normal)
    if abnormal:
        describe("abnormal", abnormal)
    if normal and abnormal:
        print(f"AUC (normal > abnormal) = {auc(normal, abnormal):.3f}")


if __name__ == "__main__":
    main()
