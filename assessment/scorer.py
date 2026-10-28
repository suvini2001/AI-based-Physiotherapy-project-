# assessment/scorer.py
"""
Combines rules and DTW into a single quality_score per rep.
Fills quality_score and faults in the rep dict.

Usage in demo_cv.py:
    import assessment.scorer as scorer
    scorer.load('template.json')

    def on_rep(rep_dict):
        rep_dict['_angle_series'] = list(fsm._rep_angles)
        scorer.score_rep(rep_dict,
                         frame_height=480,
                         calibrated_flex_thresh=fsm.FLEX_THRESH)
"""
from assessment import rules as _rules
from assessment import dtw   as _dtw

_template = None


def load(template_path='template.json'):
    """Call once at startup before scoring any reps."""
    global _template
    try:
        _template = _dtw.load_template(template_path)
        print(f"Template loaded: {_template.get('exercise','?')}  "
              f"mu={_template.get('mu_dtw', 0):.1f}  "
              f"sigma={_template.get('sigma_dtw', 1):.1f}")
    except FileNotFoundError:
        print(f"WARNING: {template_path} not found. "
              f"DTW scoring disabled. Run scripts/make_template.py first.")
        _template = None


def score_rep(rep_dict, frame_height=480.0, calibrated_flex_thresh=55.0, frame_width=640.0):
    """
    Fills rep_dict['quality_score'] and rep_dict['faults'] in place.
    Returns the updated rep_dict.
    """
    # ── rules ────────────────────────────────────────────────────────
    faults, n_checked = _rules.run_checks(
        rep_dict,
        frame_height=frame_height,
        calibrated_flex_thresh=calibrated_flex_thresh,
        frame_width=frame_width
    )

    # ── DTW score ────────────────────────────────────────────────────
    dtw_score = 50.0  # neutral default
    if _template is not None and rep_dict.get('_angle_series'):
        dtw_score = _dtw.dtw_score(
            rep_dict['_angle_series'], _template)

    # ── combined quality score ────────────────────────────────────────
    # 0.5 * DTW + 0.5 * rules_score — from work plan
    rules_score = ((n_checked - len(faults)) / n_checked * 100.0) if n_checked > 0 else 100.0
    quality     = round(0.5 * dtw_score + 0.5 * rules_score, 1)

    rep_dict['quality_score'] = quality
    rep_dict['faults']        = faults
    return rep_dict


def session_summary(reps):
    """Compute end-of-session statistics."""
    if not reps:
        return {}
    scores     = [r['quality_score'] for r in reps
                  if r.get('quality_score') is not None]
    all_faults = [f for r in reps for f in r.get('faults', [])]
    mean_q     = round(sum(scores) / len(scores), 1) if scores else 0.0
    grade      = ('A' if mean_q >= 85 else
                  'B' if mean_q >= 70 else
                  'C' if mean_q >= 55 else 'D')
    top_fault  = (max(set(all_faults), key=all_faults.count)
                  if all_faults else None)
    return {
        'mean_quality': mean_q,
        'grade':        grade,
        'top_fault':    top_fault,
        'n_reps':       len(reps),
        'n_clean_reps': sum(1 for r in reps
                           if not r.get('faults')),
    }