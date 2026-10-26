from exercise_state import ElbowFSM
import numpy as np

two_dips = (
    [162]*20 +
    list(np.linspace(162, 40, 30)) +
    list(np.linspace(40, 130, 20)) +
    list(np.linspace(130, 40, 20)) +
    list(np.linspace(40, 162, 30)) +
    [162]*20
)

fsm = ElbowFSM()
prev_state = fsm.state
for i, a in enumerate(two_dips):
    state, done = fsm.update(float(a), i/30.0)
    if state != prev_state or done:
        print(f"frame {i:3d}  angle={a:6.1f}  state={state}  done={done}")
        prev_state = state

fsm.flush(len(two_dips)/30.0)
print(f"Total reps: {fsm.rep_count}")
for r in fsm.reps:
    print(f"  Rep {r['rep_number']}: ROM={r['rom']}  dur={r['duration_s']}")