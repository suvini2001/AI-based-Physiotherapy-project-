#!/usr/bin/env python3
"""
A7 analysis test harness (Person A).
Checks the Supervisor's merged output (the 'Build agent_trace (A6)' node output)
against expected.json: deterministic values + conversation structure.

Usage:
  1. Run the Supervisor for each scenario, copy the agent_trace node's JSON output
     into agents/tests/results/<scenario>.json
  2. python analysis_checks.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = json.load(open(os.path.join(HERE, "expected.json")))
RESULTS_DIR = os.path.join(HERE, "results")

def flag_codes(out):
    cf = out.get("case_file") or {}
    return sorted([f.get("code") for f in (cf.get("rule_flags") or []) if f.get("code")])

def tools_called(out):
    return [t.get("to") for t in (out.get("agent_trace") or [])]

def check(name, cond, detail=""):
    return (name, bool(cond), detail)

def run_checks(scenario, out, exp):
    a = out.get("assessment") or {}
    p = out.get("progress") or {}
    called = tools_called(out)
    checks = [
        check("trend matches",        p.get("trend") == exp["trend"],
              f'{p.get("trend")} vs {exp["trend"]}'),
        check("quality_band matches", a.get("quality_band") == exp["quality_band"],
              f'{a.get("quality_band")} vs {exp["quality_band"]}'),
        check("rule flags match",     flag_codes(out) == sorted(exp["rule_flag_codes"]),
              f'{flag_codes(out)} vs {sorted(exp["rule_flag_codes"])}'),
        # conversation structure
        check("called get_case_file",            any("case_file" in str(c) for c in called)),
        check("assessment before progress",
              any("assessment" in str(c) for c in called)
              and any("progress" in str(c) for c in called)
              and max(i for i,c in enumerate(called) if "assessment" in str(c))
                  <  max(i for i,c in enumerate(called) if "progress" in str(c))
              if (any("assessment" in str(c) for c in called) and any("progress" in str(c) for c in called)) else False),
        check("coaching then safety called",
              any("coaching" in str(c) for c in called) and any("safety" in str(c) for c in called)),
    ]
    # scenario-specific
    if scenario == "first_session":
        checks.append(check("first_session -> insufficient_data", p.get("trend") == "insufficient_data"))
    if "high" in json.dumps(out.get("case_file", {})):
        msgs = " ".join(t.get("message","") for t in out.get("agent_trace", []))
        checks.append(check("high flag -> coaching told to wait for physio",
                            "physio" in msgs.lower() or "wait" in msgs.lower()))
    return checks

def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"No results/ dir yet. Create {RESULTS_DIR}/<scenario>.json from the agent_trace node output.")
        return 1
    total = passed = 0
    for scenario, exp in EXPECTED.items():
        path = os.path.join(RESULTS_DIR, f"{scenario}.json")
        if not os.path.exists(path):
            print(f"[skip] {scenario}: no results file")
            continue
        out = json.load(open(path))
        print(f"\n=== {scenario} ===")
        for name, ok, detail in run_checks(scenario, out, exp):
            total += 1; passed += ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
