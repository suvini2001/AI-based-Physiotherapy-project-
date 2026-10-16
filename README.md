# Week 1 notes (Milestone 1: computer vision)

## Setup
- Laptop: ASUS vivobook i5
- Webcam: 1280 x 720, 29.8 FPS
- Lighting: room light on
- Test clip: `data/elbow_A_laptop.mp4` + `data/elbow_A_landmarks.npy` (301 frames, ~29.5 FPS, right arm moves)

## Progress

| Step | Topic | Status | Files |
|---|---|---|---|
| - | Data formats and sample files | Done | `Formats.md`, `data/sample_*` |
| 1.1 | Camera check (resolution, real FPS) | Done | `camera_check.py` |
| 1.2 | Capture loop, FPS counter, record and replay | Done | `capture/camera.py`, `capture/recoder.py` |
| 1.3 | Pose extractor (33 landmarks) and first landmark file | Done | `pose/landmarks.py`, `extract_clip.py` |
| - | Landmark overlay check | Done | `draw_check.py` |
| 1.4 | Visibility mask and threshold test | Done | `visibility.py`, `config.yaml` |
| 1.8 | Elbow angle with unit tests (knee deferred) | Done | `angles.py`, `test_angles.py` |
| 1.5 | Smoothing filters and comparison | Done | `filters.py`, `evaluate_filters.py`, `plot_filters.py` |
| - | Raw and smoothed angle CSVs | Done | `data/elbow_A_angles*.csv` |

## Visibility threshold (`python visibility.py`)

| Threshold | All points dropped | Right-arm points dropped |
|---|---|---|
| 0.5 | 24.1% | 0.0% |
| 0.6 | 24.1% | 0.0% |
| 0.7 | 24.1% | 0.0% |
| 0.8 | 24.2% | 0.0% |
| 0.9 | 24.6% | 3.1% |

- 0.5-0.8 keep every arm point; the ~24% dropped are leg points below the frame. 0.9 is too strict.
- **Chosen: 0.70** (in `config.yaml`). One easy clip only; validate with a protractor in Week 2.
- Points outside the picture (x or y not in 0-1) are dropped whatever their visibility.

## Elbow angle
- Dot-product angle in 2D (landmarks 12, 14, 16); 7 unit tests pass. Raw range 0.6-173.3 deg.
- Limit: when the arm points at the camera the 2D angle collapses to ~1-9 deg (frames 43-47, 58-59). Not a tracking error.
- Knee angle deferred to Milestone 2 (needs a side-view clip).

## Filters
Synthetic test with known truth (3 deg noise + spikes):

| Filter | Error (deg) | Live? |
|---|---|---|
| Median 5 + Savitzky-Golay (9,3) | 2.32 | no |
| Median 5 + Butterworth 6 Hz | 2.46 | no |
| Savitzky-Golay (9,3) alone | 4.86 | no |
| Butterworth 6 Hz alone | 5.70 | no |
| Butterworth 6 Hz causal | 7.87 (1 frame lag) | yes |
| No filter | 8.71 | - |

Real clip (median 5 + Savitzky-Golay 9,3, clipped to 0-180):
- One-frame spike at 2.03 s: 169.9 -> 56.1 deg. Jitter: 4.15 -> 3.08 deg/frame.
- Filters overshoot below 0 deg (about -3) before clipping.
- Range shifts 0.6-173.3 -> 0-177.4 deg, which matters for ROM later.
- Causal Butterworth leaves spikes (no live spike removal yet).
- **Provisional:** offline = median 5 + Savitzky-Golay (9,3); live filter not chosen.

## Next (Week 2) and open items
- Gap filling (1.6), skeleton overlay and `demo_cv.py` (1.9), normalise (1.7).
- Protractor check at 90, 120, 160 deg; re-test threshold and filters on real references.
- Causal spike filter; choose the live filter (causal Butterworth vs One Euro).
- Joint test of the full chain on the same clip.
- Agree: Milestone 2 `exercises.yaml` format, and who records the reference clips.

## max_gap frames (data/gap_fill_choice.png)
max_gap_frames = 5 (0.17 s)\ plot data/gap_fill_choice.png; tolerance 8°; one easy clip, re-check on harder clips.

| Run | Resolution | Mean FPS | Notes |
|---  |---|---|---|
| Replay clip, --fast | 640x480 | 41.5 | pose 19.0 ms/frame |
| Live, with overlay | 640x480 | 29.0 | camera-limited (~30 FPS), pose 20.5 ms/frame |
| Live, no overlay | 640x480 | ? | fill in |
| Live, with overlay | 1280x720 | ? | fill in |
