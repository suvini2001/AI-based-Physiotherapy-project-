# calibrate_template.py
import json
import numpy as np
from assessment.dtw import calibrate_template, normalise_series

MIN_PLAUSIBLE_ELBOW = 25.0   # degrees: human elbow cannot physically flex below ~30 deg; reject 2D collapse

# load real session reps
session = json.load(open('session_log.json'))

# get angle series from clean reps only, filtering out 2D collapse (<25 deg)
clean_series = [
    r['_angle_series']
    for r in session['reps']
    if not r.get('faults') and r.get('_angle_series')
    and r.get('min_angle', 0) >= MIN_PLAUSIBLE_ELBOW
]

print(f"Found {len(clean_series)} valid physiologically plausible clean reps")

if len(clean_series) < 3:
    print("Not enough clean reps (>= 25° min_angle) — do more reps first with side view")
else:
    # compute mean trajectory from clean reps
    normalised = [normalise_series(s) for s in clean_series]
    mean_traj  = np.mean(normalised, axis=0).tolist()

    # update template with real mean trajectory
    template = json.load(open('template.json'))
    template['mean_trajectory'] = mean_traj
    template['recorded_by']     = 'physiotherapist_reference'
    template['n_reps']          = len(clean_series)
    json.dump(template, open('template.json', 'w'), indent=2)

    # now calibrate mu and sigma
    calibrate_template(clean_series, 'template.json')
    print("Done — calibrated template saved to template.json")