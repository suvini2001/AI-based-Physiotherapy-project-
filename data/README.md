# Data files and licences

- `calib_pairs_rehab24.npz`: 2D raw vs. motion capture angle pairs extracted from the REHAB24-6 dataset (open research dataset).
- `population_curve.json`: Fitted monotonic calibration curve (REHAB24-6 Profile view, raw 2D → mocap angle).
- `user_calibration.json`: Personal calibration output from `capture/calibrate_ui.py`.
- `elbow_A_laptop.mp4`, `elbow_A_landmarks.npy`: Sample test clip (301 frames, 29.5 FPS) and extracted landmarks.
- `elbow_A_angles*.csv`: Raw and filtered elbow angle series.
- `sample_world_landmarks.npz`: Example 3D world landmarks for offline testing.

## Licences
- **REHAB24-6**: Open research dataset for rehabilitation exercise analysis.
- **Fit3D**: Research-only dataset (Fieraru et al., CVPR 2023); biceps curls used for curve extension.
