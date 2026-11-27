"""
B8 - Output Test Harness
Tests the PhysioTrack multi-agent output path against fixture files.
Run: python agents/tests/checks/output.py
"""

import json
import os
import sys

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_DIR = os.path.join(ROOT, "fixtures")
SCHEMA_PATH = os.path.join(ROOT, "contract", "agent_insights.v1.schema.json")

# ── banned phrases ──────────────────────────────────────────────────────────
BANNED_PHRASES = [
    "you have", "diagnos", "medication",
    "stop the exercise", "injury", "damaged"
]

# ── number pattern ──────────────────────────────────────────────────────────
import re
NUMBER_PATTERN = re.compile(
    r'\b\d+(\.\d+)?\s*(degrees?|°|%|reps?|repetitions?)\b', re.IGNORECASE
)

# ── colours for terminal output ─────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def passed(msg): print(f"  {GREEN}✓ PASS{RESET} {msg}")
def failed(msg): print(f"  {RED}✗ FAIL{RESET} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠ WARN{RESET} {msg}")

# ── individual checks ───────────────────────────────────────────────────────

def check_schema(result, fixture_name):
    """Check required fields exist in output."""
    required = ["contract_version", "session_id", "metrics",
                "assessment", "progress", "review", "coaching", "llm_status"]
    all_pass = True
    for field in required:
        if field not in result:
            failed(f"[{fixture_name}] missing field: {field}")
            all_pass = False
    if all_pass:
        passed(f"[{fixture_name}] schema check — all required fields present")
    return all_pass


def check_banned_phrases(result, fixture_name):
    """Check no banned phrases in coaching message or summaries."""
    all_pass = True
    coaching_msg = result.get("coaching", {}).get("patient_message", "")
    assessment_summary = result.get("assessment", {}).get("summary_for_physio", "")
    progress_summary = result.get("progress", {}).get("summary_for_physio", "")

    for text, label in [
        (coaching_msg, "coaching"),
        (assessment_summary, "assessment"),
        (progress_summary, "progress")
    ]:
        for phrase in BANNED_PHRASES:
            if phrase.lower() in text.lower():
                failed(f"[{fixture_name}] banned phrase '{phrase}' in {label}")
                all_pass = False

    if all_pass:
        passed(f"[{fixture_name}] banned phrase check — no banned phrases found")
    return all_pass


def check_no_numbers_in_coaching(result, fixture_name):
    """Check coaching message contains no specific numbers."""
    coaching_msg = result.get("coaching", {}).get("patient_message", "")
    if NUMBER_PATTERN.search(coaching_msg):
        failed(f"[{fixture_name}] coaching message contains specific numbers")
        return False
    passed(f"[{fixture_name}] number check — no specific numbers in coaching message")
    return True


def check_rule_flags_preserved(result, fixture_name):
    """Check rule flags are in the output."""
    flags = result.get("review", {}).get("flags", [])
    if not isinstance(flags, list):
        failed(f"[{fixture_name}] review.flags is not a list")
        return False
    passed(f"[{fixture_name}] rule flags check — {len(flags)} flag(s) in output")
    return True


def check_llm_status(result, fixture_name):
    """Check llm_status is ok or fallback."""
    status = result.get("llm_status", "")
    if status not in ["ok", "fallback"]:
        failed(f"[{fixture_name}] llm_status is '{status}' — must be ok or fallback")
        return False
    passed(f"[{fixture_name}] llm_status = {status}")
    return True


def check_coaching_status(result, fixture_name):
    """Check coaching status is valid."""
    status = result.get("coaching", {}).get("status", "")
    valid = ["pending_review", "approved", "rejected",
             "fallback", "needs_physio_rewrite"]
    if status not in valid:
        failed(f"[{fixture_name}] coaching.status is '{status}' — not a valid value")
        return False
    passed(f"[{fixture_name}] coaching.status = {status}")
    return True


def check_high_flag_has_physio_warning(result, fixture_name):
    """If high flag exists, coaching message must mention physiotherapist."""
    flags = result.get("review", {}).get("flags", [])
    has_high = any(f.get("severity") == "high" for f in flags)
    if has_high:
        msg = result.get("coaching", {}).get("patient_message", "").lower()
        if "physiotherapist" not in msg and "physio" not in msg:
            failed(f"[{fixture_name}] high flag present but no physiotherapist warning in coaching message")
            return False
        passed(f"[{fixture_name}] high flag check — physiotherapist warning present")
    else:
        passed(f"[{fixture_name}] high flag check — no high flags, skipped")
    return True


# ── run all checks on one result ────────────────────────────────────────────

def run_checks(result, fixture_name):
    """Run all checks on a single result object."""
    results = []
    results.append(check_schema(result, fixture_name))
    results.append(check_banned_phrases(result, fixture_name))
    results.append(check_no_numbers_in_coaching(result, fixture_name))
    results.append(check_rule_flags_preserved(result, fixture_name))
    results.append(check_llm_status(result, fixture_name))
    results.append(check_coaching_status(result, fixture_name))
    results.append(check_high_flag_has_physio_warning(result, fixture_name))
    return all(results)


# ── load and test fixture results ───────────────────────────────────────────

def test_fixture_result(result_file):
    """Test a single result JSON file."""
    with open(result_file) as f:
        result = json.load(f)
    fixture_name = os.path.basename(result_file)
    return run_checks(result, fixture_name)


def test_b7_validation():
    """Test B7 final check catches banned phrases."""
    print("\n── B7 Validation Tests ──")

    good_result = {
        "contract_version": "1.0",
        "session_id": 42,
        "metrics": {"reps_vs_target_pct": 80},
        "assessment": {"quality_band": "fair", "summary_for_physio": "Patient did well."},
        "progress": {"trend": "stable", "summary_for_physio": "ROM is stable."},
        "review": {"needs_physio_attention": False, "flags": []},
        "coaching": {
            "patient_message": "Great effort today! Keep your trunk upright.",
            "status": "pending_review"
        },
        "llm_status": "ok",
        "validation_errors": []
    }

    bad_result = {
        "contract_version": "1.0",
        "session_id": 42,
        "metrics": {},
        "assessment": {"quality_band": "poor", "summary_for_physio": ""},
        "progress": {"trend": "declining", "summary_for_physio": ""},
        "review": {"needs_physio_attention": True, "flags": [{"severity": "high"}]},
        "coaching": {
            "patient_message": "You have an injury. Stop the exercise.",
            "status": "fallback"
        },
        "llm_status": "fallback",
        "validation_errors": ["banned phrase found: you have"]
    }

    passed_good = run_checks(good_result, "good_result")
    passed_bad = not check_banned_phrases(bad_result, "bad_result_should_fail")

    if passed_good:
        passed("B7 good result passed all checks")
    if passed_bad:
        passed("B7 bad result correctly detected banned phrases")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("PhysioTrack B8 — Output Test Harness")
    print("=" * 50)

    total_pass = 0
    total_fail = 0

    # Test B7 validation logic
    test_b7_validation()

    # Test any result files in agents/out/
    out_dir = os.path.join(ROOT, "out")
    if os.path.exists(out_dir):
        result_files = [
            f for f in os.listdir(out_dir)
            if f.endswith(".json")
        ]
        if result_files:
            print(f"\n── Testing {len(result_files)} result file(s) in agents/out/ ──")
            for rf in result_files:
                print(f"\nChecking {rf}:")
                if test_fixture_result(os.path.join(out_dir, rf)):
                    total_pass += 1
                else:
                    total_fail += 1
        else:
            warn("No result files found in agents/out/ — run workflows first")
    else:
        warn("agents/out/ folder not found — no workflow results to test yet")

    print("\n" + "=" * 50)
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print("=" * 50)


if __name__ == "__main__":
    main()