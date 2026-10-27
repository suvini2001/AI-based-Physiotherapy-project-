# test_assessment.py  —  run: python test_assessment.py
"""
Unit tests for assessment/rules.py and assessment/scorer.py
No camera or dataset needed.
"""
from assessment.rules import run_all
import assessment.scorer as scorer


def make_rep(**kwargs):
    base = {
        'rep_number': 1,
        'min_angle': 40.0,
        'max_angle': 162.0,
        'rom': 122.0,
        'duration_s': 2.5,
        '_angle_series': list(range(162, 39, -2)) + list(range(40, 163, 2)),
        '_landmarks': None,
    }
    base.update(kwargs)
    return base


# ── Test 1: clean rep passes all rules ───────────────────────────
rep = make_rep()
faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
assert faults == [], f"Expected no faults, got: {faults}"
print("Test 1 PASS — clean rep has no faults")

# ── Test 2: ROM fault ─────────────────────────────────────────────
rep = make_rep(min_angle=70)   # 70 > 55+10 = 65 → fault
faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
assert any('bend' in f.lower() for f in faults), \
    f"Expected ROM fault, got: {faults}"
print("Test 2 PASS — ROM fault detected when min_angle=70")

# ── Test 3: speed fault ───────────────────────────────────────────
rep = make_rep(duration_s=0.4)
faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
assert any('fast' in f.lower() for f in faults), \
    f"Expected speed fault, got: {faults}"
print("Test 3 PASS — speed fault detected when duration=0.4s")

# ── Test 4: no landmarks skips compensation checks ───────────────
rep = make_rep(min_angle=70, _landmarks=None)
faults = run_all(rep, frame_height=480, calibrated_flex_thresh=55)
assert len(faults) == 1, \
    f"Expected 1 fault (ROM only), got: {faults}"
print("Test 4 PASS — missing landmarks skips compensation checks")

# ── Test 5: patient-relative ROM — impaired patient ──────────────
# calibrated_flex_thresh=80 means patient can only reach 80deg
# min_angle=85 is only 5deg above threshold → within tolerance (10)
rep = make_rep(min_angle=85)
faults = run_all(rep, frame_height=480, calibrated_flex_thresh=80)
assert not any('bend' in f.lower() for f in faults), \
    f"Should not flag ROM for impaired patient: {faults}"
print("Test 5 PASS — impaired patient not penalised for limited ROM")

# ── Test 6: scorer fills quality_score and faults ────────────────
scorer.load('template.json')   # loads synthetic template
rep = make_rep()
scorer.score_rep(rep, frame_height=480, calibrated_flex_thresh=55)
assert rep['quality_score'] is not None, "quality_score not filled"
assert isinstance(rep['faults'], list), "faults not a list"
assert 0 <= rep['quality_score'] <= 100, "quality_score out of range"
print(f"Test 6 PASS — scorer fills quality_score={rep['quality_score']} "
      f"faults={rep['faults']}")

# ── Test 7: bad rep gets lower score than good rep ───────────────
good_rep = make_rep()
bad_rep  = make_rep(min_angle=80, duration_s=0.3)
scorer.score_rep(good_rep, calibrated_flex_thresh=55)
scorer.score_rep(bad_rep,  calibrated_flex_thresh=55)
assert good_rep['quality_score'] >= bad_rep['quality_score'], \
    (f"Good rep should score >= bad rep: "
     f"{good_rep['quality_score']} vs {bad_rep['quality_score']}")
print(f"Test 7 PASS — good rep ({good_rep['quality_score']}) "
      f">= bad rep ({bad_rep['quality_score']})")

print("\nAll 7 tests passed ✅")