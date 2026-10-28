# AI-based Physiotherapy — webcam exercise monitoring

Real-time, single-webcam monitoring of upper-limb home exercises (elbow flexion,
shoulder abduction). MediaPipe Pose → cleaned joint angle → repetition counting
(finite-state machine) → per-rep assessment (posture rules + DTW similarity) →
on-screen and spoken feedback.

## Setup

- **Python 3.10–3.12** (`mediapipe==0.10.14` is needed for the solutions API and is not available on 3.13+)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q             # all tests should pass
```

## Run

**1. Camera setup (elbow flexion).** Turn fully side-on to the camera, camera at
shoulder height, 2–3 m away, whole upper body in frame. The arm nearest the
camera is measured: right side to the camera → `--side right`.

**2. Personal calibration (straight-arm check).** Loads `data/population_curve.json`
and saves `data/user_calibration.json`. The status must be `ok`; `warn` means the
camera position is wrong (for example a straight arm reading ~90°).

```bash
python capture/calibrate_ui.py --side right
```

**3. Exercise session.**

```bash
python demo_cv.py --exercise elbow --side right --variant A \
    --calibration data/user_calibration.json
```

No `WARNING:` lines should be printed at start-up. Phases: READY (stand still) →
CALIB (move through your full range; FSM thresholds adapt to you) → COUNT.
Press `q` to finish; the session is saved to `--session-out` (default `session_log.json`).

Angle variants: `A` 2D pixel angle (used with the calibration curve), `B` MediaPipe
3D world landmarks, `C` yaw-corrected 3D.

## How each part works

| Part | Files | Method |
|---|---|---|
| Pose + cleaning | `pose/`, `visibility.py`, `vision/gap_fill.py`, `filters.py` | Visibility mask 0.70, gap fill ≤ 5 frames, median-5 + One Euro filter |
| Angle calibration | `calibration.py`, `scripts/fit_calibration.py`, `capture/calibrate_ui.py` | Monotonic population curve (raw 2D → mocap angle), profile view, fitted on REHAB24-6, LOPO MAE 20.1° → 16.3° |
| Rep counting | `exercise_state.py` | 4-state FSM per exercise, thresholds tuned from the CALIB phase |
| Posture rules | `assessment/rules.py` | Shoulder elevation, trunk lean, asymmetry (ideas adapted from Pavlikov et al. 2026; thresholds preliminary), ROM, speed |
| DTW score | `assessment/dtw.py`, `templates/` | Labchurat et al. 2026: Z = (d − μ)/σ from healthy reps, score = 100 / max(Z, 1) |
| Quality score | `assessment/scorer.py` | 0.5·DTW + 0.5·rules when the template is calibrated, rules only otherwise |

## Building a real DTW template

The templates in `templates/` are **synthetic** (`"calibrated": false`), so the
DTW part is switched off. To calibrate, record healthy volunteers (one file per person):

```bash
python demo_cv.py --exercise elbow --side right --variant A \
    --calibration data/user_calibration.json --session-out data/healthy/p01.json
python scripts/build_template.py data/healthy/*.json --exercise elbow_flexion \
    --recorded-by healthy_volunteers --out templates/elbow_flexion.json
python scripts/evaluate_dtw.py --lopo data/healthy/*.json --abnormal data/abnormal/*.json
```

Participant recordings are git-ignored. See `data/README.md` for dataset licences.

## Status and known limits

- Calibration curve: REHAB24-6 profile data covers truth angles 64–175°; below 64° the curve is extrapolated. To be extended with Fit3D biceps curls and checked with a goniometer on our own setup.
- DTW templates and rule thresholds still need real recordings.
- Benchmark notebooks: `benchmark_3d_world.ipynb` (angle variants, learned correction), `rehab24_validation_v4.ipynb`.

Earlier milestone notes: `docs/week1_notes.md`.
