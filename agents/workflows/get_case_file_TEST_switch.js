// ===== TEST SCENARIO SWITCH =====================================================
// Change this ONE word, Save, then run the Assessment agent. Repeat for each.
// Options: improving | plateau | declining_safety | fast_and_sloppy | first_session
const SCENARIO = "declining_safety";
// For INTEGRATION later: revert this node to the committed get_case_file (real
// data arrives via input). The line below already prefers real input when present.
// ================================================================================

const SCENARIOS = {
  "improving": {
    "prescription": {
      "exercise": "elbow_flexion",
      "target_reps": 10,
      "target_rom": 120
    },
    "metrics": {
      "reps_vs_target_pct": 100,
      "rom_vs_target_pct": 87,
      "fault_counts": {},
      "fast_rep_share": 0,
      "rom_slope_deg_per_session": 1.87,
      "baseline_rom": 102,
      "quality_band": "good",
      "incomplete_attempts": 0,
      "sessions_in_history": 8
    },
    "rule_flags": []
  },
  "plateau": {
    "prescription": {
      "exercise": "elbow_flexion",
      "target_reps": 10,
      "target_rom": 120
    },
    "metrics": {
      "reps_vs_target_pct": 100,
      "rom_vs_target_pct": 83,
      "fault_counts": {},
      "fast_rep_share": 0,
      "rom_slope_deg_per_session": 0,
      "baseline_rom": 100,
      "quality_band": "good",
      "incomplete_attempts": 0,
      "sessions_in_history": 8
    },
    "rule_flags": []
  },
  "declining_safety": {
    "prescription": {
      "exercise": "elbow_flexion",
      "target_reps": 10,
      "target_rom": 120
    },
    "metrics": {
      "reps_vs_target_pct": 100,
      "rom_vs_target_pct": 71,
      "fault_counts": {
        "SHOULDER_HIKE": 3
      },
      "fast_rep_share": 0,
      "rom_slope_deg_per_session": -3.46,
      "baseline_rom": 92.3,
      "quality_band": "fair",
      "incomplete_attempts": 0,
      "sessions_in_history": 7
    },
    "rule_flags": [
      {
        "code": "NEW_COMPENSATION",
        "severity": "medium",
        "source": "rule",
        "evidence": "SHOULDER_HIKE not seen in history"
      }
    ]
  },
  "fast_and_sloppy": {
    "prescription": {
      "exercise": "elbow_flexion",
      "target_reps": 10,
      "target_rom": 120
    },
    "metrics": {
      "reps_vs_target_pct": 100,
      "rom_vs_target_pct": 85,
      "fault_counts": {
        "TOO_FAST": 3
      },
      "fast_rep_share": 1,
      "rom_slope_deg_per_session": 0.7,
      "baseline_rom": 101,
      "quality_band": "fair",
      "incomplete_attempts": 0,
      "sessions_in_history": 3
    },
    "rule_flags": [
      {
        "code": "NEW_COMPENSATION",
        "severity": "medium",
        "source": "rule",
        "evidence": "TOO_FAST not seen in history"
      },
      {
        "code": "RUSHED_REPS",
        "severity": "low",
        "source": "rule",
        "evidence": "majority of reps too fast"
      }
    ]
  },
  "first_session": {
    "prescription": {
      "exercise": "elbow_flexion",
      "target_reps": 10,
      "target_rom": 120
    },
    "metrics": {
      "reps_vs_target_pct": 100,
      "rom_vs_target_pct": 79,
      "fault_counts": {},
      "fast_rep_share": 0,
      "rom_slope_deg_per_session": null,
      "baseline_rom": null,
      "quality_band": "good",
      "incomplete_attempts": 0,
      "sessions_in_history": 0
    },
    "rule_flags": []
  }
};

const q = $input.first().json || {};
const passed = (q && q.case_file) ? q.case_file : ((q && (q.ev || q.metrics || q.session)) ? q : null);
const cf = passed || SCENARIOS[SCENARIO];

return [{ json: {
  prescription: cf.prescription,
  metrics: cf.metrics,
  rule_flags: cf.rule_flags || [],
} }];
