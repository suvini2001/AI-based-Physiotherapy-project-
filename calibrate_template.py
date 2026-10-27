# run this once — save as calibrate_template.py
import json
from assessment.dtw import calibrate_template

# load your real reps
session = json.load(open('session_log.json'))

# get angle series from clean reps only
clean_series = [
    r['_angle_series']
    for r in session['reps']
    if not r.get('faults') and r.get('_angle_series')
]

print(f"Calibrating from {len(clean_series)} clean reps")

if len(clean_series) < 3:
    print("Not enough clean reps — do more reps first")
else:
    # update mean trajectory and mu/sigma
    import numpy as np
    from assessment.dtw import normalise_series

    # compute mean trajectory from clean reps
    normalised = [normalise_series(s) for s in clean_series]
    mean_traj  = np.mean(normalised, axis=0).tolist()

    # update template with real mean trajectory
    template = json.load(open('template.json'))
    template['mean_trajectory'] = mean_traj
    template['recorded_by']     = 'real_session'
    template['n_reps']          = len(clean_series)
    json.dump(template, open('template.json', 'w'), indent=2)

    # now calibrate mu and sigma
    calibrate_template(clean_series, 'template.json')
    print("Done — run demo_cv.py again to see varied quality scores")