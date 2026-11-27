"""
B8 - Output Test Harness
Tests the PhysioTrack multi-agent output path against fixture files.
Run: python agents/tests/checks/output.py
"""

import json
import os
import re

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    print("WARNING: jsonschema not installed. Run: pip install jsonschema")

# ── paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURES_DIR = os.path.join(ROOT, "fixtures")
SCHEMA_PATH = os.path.join(ROOT, "contract", "agent_insights.v1.schema.json")
OUT_DIR = os.path.join(ROOT, "out")

# ── banned phrases ──────────────────────────────────────────────────────────
BANNED_PHRASES = [
    "you have", "diagnos", "medication",
    "stop the exercise", "damaged"
]

# ── number pattern ──────────────────────────────────────────────────────────
NUMBER_PATTERN = re.compile(
    r'\b\d+(\.\d+)?\s*(degrees?|°|%|reps?|repetitions?)\b', re.IGNORECASE
)

# ── colours ─────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def passed(msg): print(f"  {GREEN}✓ PASS{RESET} {msg}")
def failed(msg): print(f"  {RED}✗ FAIL{RESET} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠ WARN{RESET} {msg}")

# ── load schema ──────────────────────────────────────────────────────────────
def load_schema():
    if not os.path.exists(SCHEMA_PATH):
        warn(f"Schema not found at {SCHEMA_PATH}")
        return None
    with open(SCHEMA_PATH) as f:
        return json.load(f)

# ── checks ───────────────────────────────────────────────────────────────────

def check_schema_validation(result, fixture_name, schema):
    """Validate result against agent_insights.v1.schema.json."""
    if not HAS_JSONSCHEMA:
        warn(f"[{fixture_name}] schema validation skipped — jsonschema not installed")
        return True
    if schema is None:
        warn(f"[{fixture_name}] schema validation skipped — schema not found")
        return True
    try:
        jsonschema.validate(result, schema)
        passed(f"[{fixture_name}] schema validation — matches agent_insights.v1")
        return True
    except jsonschema.ValidationError as e:
        failed(f"[{fixture_name}] schema validation failed: {e.message}")
        return False


def check_required_fields(result, fixture_name):
    """Check required fields exist in output."""
    required = ["contract_version", "request_id", "session_id", "metrics",
                "assessment", "progress", "review", "coaching", "meta"]
    all_pass = True
    for field in required:
        if field not in result:
            failed(f"[{fixture_name}] missing field: {field}")
            all_pass = False
    if all_pass:
        passed(f"[{fixture_name}] required fields — all present")
    return all_pass


def check_banned_phrases(result, fixture_name):
    """Check no banned phrases in any text output."""
    all_pass = True
    texts = {
        "coaching": result.get("coaching", {}).get("patient_message", ""),
        "assessment": result.get("assessment", {}).get("summary_for_physio", ""),
        "progress": result.get("progress", {}).get("summary_for_physio", "")
    }
    for label, text in texts.items():
        for phrase in BANNED_PHRASES:
            if phrase.lower() in text.lower():
                failed(f"[{fixture_name}] banned phrase '{phrase}' in {label}")
                all_pass = False
    if all_pass:
        passed(f"[{fixture_name}] banned phrases — none found")
    return all_pass


def check_no_numbers_in_coaching(result, fixture_name):
    """Check coaching message contains no specific numbers."""
    msg = result.get("coaching", {}).get("patient_message", "")
    if NUMBER_PATTERN.search(msg):
        failed(f"[{fixture_name}] coaching message contains specific numbers")
        return False
    passed(f"[{fixture_name}] number check — no specific numbers in coaching")
    return True


def check_rule_flags_preserved(result, fixture_name):
    """Check rule flags are in the output."""
    flags = result.get("review", {}).get("flags", [])
    if not isinstance(flags, list):
        failed(f"[{fixture_name}] review.flags is not a list")
        return False
    passed(f"[{fixture_name}] rule flags — {len(flags)} flag(s) in output")
    return True


def check_llm_status(result, fixture_name):
    """Check llm_status is ok or fallback."""
    status = result.get("meta", {}).get("llm_status", "")
    if status not in ["ok", "fallback"]:
        failed(f"[{fixture_name}] meta.llm_status is '{status}'")
        return False
    passed(f"[{fixture_name}] llm_status = {status}")
    return True


def check_high_flag_physio_warning(result, fixture_name):
    """If high flag exists, coaching must mention physiotherapist."""
    flags = result.get("review", {}).get("flags", [])
    has_high = any(f.get("severity") == "high" for f in flags)
    if has_high:
        msg = result.get("coaching", {}).get("patient_message", "").lower()
        if "physiotherapist" not in msg and "physio" not in msg:
            failed(f"[{fixture_name}] high flag present but no physiotherapist warning")
            return False
        passed(f"[{fixture_name}] high flag — physiotherapist warning present")
    else:
        passed(f"[{fixture_name}] high flag — no high flags, skipped")
    return True


def check_agent_trace(result, fixture_name):
    """Check agent_trace exists and is a list."""
    trace = result.get("agent_trace", None)
    if trace is None:
        warn(f"[{fixture_name}] agent_trace missing")
        return True
    if not isinstance(trace, list):
        failed(f"[{fixture_name}] agent_trace is not a list")
        return False
    passed(f"[{fixture_name}] agent_trace — {len(trace)} step(s) logged")
    return True


def check_coaching_status(result, fixture_name):
    """Check coaching status is valid."""
    status = result.get("coaching", {}).get("status", "")
    valid = ["pending_review", "approved", "rejected",
             "fallback", "needs_physio_rewrite"]
    if status not in valid:
        failed(f"[{fixture_name}] coaching.status '{status}' not valid")
        return False
    passed(f"[{fixture_name}] coaching.status = {status}")
    return True


# ── run all checks ───────────────────────────────────────────────────────────

def run_all_checks(result, fixture_name, schema):
    results = []
    results.append(check_required_fields(result, fixture_name))
    results.append(check_schema_validation(result, fixture_name, schema))
    results.append(check_banned_phrases(result, fixture_name))
    results.append(check_no_numbers_in_coaching(result, fixture_name))
    results.append(check_rule_flags_preserved(result, fixture_name))
    results.append(check_llm_status(result, fixture_name))
    results.append(check_high_flag_physio_warning(result, fixture_name))
    results.append(check_agent_trace(result, fixture_name))
    results.append(check_coaching_status(result, fixture_name))
    return all(results)


# ── inline tests ─────────────────────────────────────────────────────────────

def test_inline():
    """Test B7 validation logic with hardcoded examples."""
    print("\n── Inline Validation Tests ──")
    schema = load_schema()

    good = {
        "contract_version": "1.0",
        "request_id": "test-001",
        "session_id": 42,
        "metrics": {"reps_vs_target_pct": 80},
        "assessment": {"quality_band": "fair",
                       "summary_for_physio": "Patient did well."},
        "progress": {"trend": "stable",
                     "summary_for_physio": "ROM is stable."},
        "review": {"needs_physio_attention": False, "flags": []},
        "coaching": {
            "patient_message": "Great effort! Keep your trunk upright.",
            "status": "pending_review"
        },
        "meta": {"llm_status": "ok", "latency_ms": 500,
                 "validation_errors": []},
        "agent_trace": []
    }

    bad = {
        "contract_version": "1.0",
        "request_id": "test-002",
        "session_id": 42,
        "metrics": {},
        "assessment": {"quality_band": "poor", "summary_for_physio": ""},
        "progress": {"trend": "declining", "summary_for_physio": ""},
        "review": {
            "needs_physio_attention": True,
            "flags": [{"code": "ROM_DROP", "severity": "high",
                       "source": "rule", "evidence": "dropped"}]
        },
        "coaching": {
            "patient_message": "You have an injury. Stop the exercise.",
            "status": "fallback"
        },
        "meta": {"llm_status": "fallback", "validation_errors":
                 ["banned phrase found"]},
        "agent_trace": []
    }

    print("\nGood result (should all pass):")
    good_pass = run_all_checks(good, "good_result", schema)

    print("\nBad result banned phrase check (should detect issues):")
    bad_detected = not check_banned_phrases(bad, "bad_result")
    bad_flag = not check_high_flag_physio_warning(bad, "bad_result")

    if good_pass:
        passed("Good result passed all checks")
    if bad_detected:
        passed("Bad result: banned phrases correctly detected")
    if bad_flag:
        passed("Bad result: missing physio warning correctly detected")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("PhysioTrack B8 — Output Test Harness")
    print("=" * 55)

    schema = load_schema()
    if schema:
        passed(f"Loaded schema from {os.path.basename(SCHEMA_PATH)}")
    
    # Inline tests
    test_inline()

    # Test result files in agents/out/
    total_pass = 0
    total_fail = 0

    if os.path.exists(OUT_DIR):
        result_files = [f for f in os.listdir(OUT_DIR) if f.endswith(".json")]
        if result_files:
            print(f"\n── Testing {len(result_files)} result file(s) in agents/out/ ──")
            for rf in result_files:
                print(f"\nChecking {rf}:")
                with open(os.path.join(OUT_DIR, rf)) as f:
                    result = json.load(f)
                if run_all_checks(result, rf, schema):
                    total_pass += 1
                else:
                    total_fail += 1
        else:
            warn("No result files in agents/out/ — run workflows first")
    else:
        warn("agents/out/ folder not found — no workflow results to test yet")

    print("\n" + "=" * 55)
    print(f"Results: {total_pass} passed, {total_fail} failed")
    print("=" * 55)


if __name__ == "__main__":
    main()