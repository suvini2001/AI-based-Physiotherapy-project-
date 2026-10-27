"""patch_fsm.py — run this on your local machine to fix exercise_state.py"""

src = open('exercise_state.py', encoding='utf-8').read()

# The old block appears TWICE (ElbowFSM and ShoulderFSM)
# We replace both occurrences

OLD = (
    '            "quality_score": None,\n'
    '            "faults":        [],\n'
    '        }'
)

NEW = (
    '            "quality_score": None,\n'
    '            "faults":        [],\n'
    '            "_angle_series": [round(float(a), 1) for a in self._rep_angles],\n'
    '        }'
)

count = src.count(OLD)
print(f"Found {count} occurrence(s) to replace")

if count == 0:
    print("ERROR: pattern not found — check indentation in exercise_state.py")
else:
    new_src = src.replace(OLD, NEW)
    open('exercise_state.py', 'w', encoding='utf-8').write(new_src)
    print("Done — exercise_state.py updated")