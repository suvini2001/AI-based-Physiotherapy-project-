# scripts/make_template.py
"""
Creates a synthetic template.json for testing.
Replace with a real recorded template later.
Run: python scripts/make_template.py
"""
import numpy as np, json, os

n = 100
# clean elbow flexion: 162 -> 40 -> 162
half = n // 2
trajectory = np.concatenate([
    np.linspace(162, 40, half),
    np.linspace(40, 162, n - half)
]).tolist()

template = {
    "exercise":    "elbow_flexion",
    "recorded_by": "synthetic",
    "n_reps":       1,
    "mean_trajectory": trajectory,
    "std_trajectory":  [5.0] * n,
    "mu_dtw":   0.0,   # will be updated by calibrate_template()
    "sigma_dtw": 1.0,  # will be updated by calibrate_template()
}

os.makedirs("scripts", exist_ok=True)
path = "template.json"
with open(path, 'w') as f:
    json.dump(template, f, indent=2)
print(f"template.json written ({n} points)")
print("Run calibrate_template() after recording real reps to set mu/sigma.")