# Data files and licences

- `calib_pairs_rehab24.npz`: 2D raw vs. motion capture angle pairs extracted from the REHAB24-6 dataset (open research dataset).
- `population_curve.json`: Fitted monotonic calibration curve (REHAB24-6, camera-relative side-on views: cam17 "Profile" + cam18 "Front"; raw 2D → mocap angle). Fallback angle.
- `feat_rehab24.npz`: Per-frame I_ML features and mocap truth for REHAB24-6 (benchmark STEP 6 export).
- `iml_nolook_rehab24_exp_torso.txt` (+ `.json`): Deployed I_ML model (LightGBM, 14 features, expanding session statistics, torso scale), trained on REHAB24-6 streaming features only.
- `user_calibration.json`: Personal calibration output from `capture/calibrate_ui.py`.
- `elbow_A_laptop.mp4`, `elbow_A_landmarks.npy`: Sample test clip (301 frames, 29.5 FPS) and extracted landmarks.
- `elbow_A_angles*.csv`: Raw and filtered elbow angle series.
- `sample_world_landmarks.npz`: Example 3D world landmarks for offline testing.

## Licences
- **REHAB24-6** (Černek et al., 2024): CC BY-NC 4.0. Files above derived from it are shared under the same terms (non-commercial, with attribution).
- **Fit3D** (Fieraru et al., AIFit, CVPR 2021): research licence, single user, no redistribution. Used for evaluation and model selection only (notebooks 01–04b); its data and every derived file live in the git-ignored `data/fit3d/` and are not shared. No deployed file is trained on it.
