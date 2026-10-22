# Person A: Setup and Recording Notes

## Machine
- Laptop: ASUS vivobook i5
- OS: Windows 11
- Python: 3.12.10 (.venv), MediaPipe 0.10.14, OpenCV 5.0, NumPy 2.5.3
- Libraries pinned in requirements.txt

## Camera
- Webcam: laptop built-in
- Setting used: 640x480, MJPG, requested 30 FPS
- Real speed (measured): about 30 FPS in good light
- In a dim room it drops to 7.5-15 FPS
- 720p only gave 10 FPS, so I use 640x480
- Camera backend: DirectShow (CAP_DSHOW)

## Camera test results (camera_check2.py, dim room)
- 640x480 default: 7.5 FPS
- 640x480 MJPG: 15.0 FPS
- 1280x720 default: 10.0 FPS
- 1280x720 MJPG: 10.0 FPS
- Good light (outside / bright window): about 30 FPS

## Recording spot (elbow flexion)
- Laptop position: [on a table / on books], camera at about [hip / chest] height
- Distance from laptop to my feet: [__] m (marked with tape)
- Facing: the camera, frontal view
- Frame: head to hips visible, whole arm in frame with the arm out
- Light: [near a bright window / outside in shade], in front of me
- Background: [plain wall]
- Clothing: [fitted top, blue white strips]
- Arm used: [right ]

## Movement
- Upper arm out to the side at about shoulder height, kept still
- Elbow bends and straightens, about 2-3 s each way, short pause at the top and bottom
- 3-4 reps per 10-second clip

## Clips and files
| File | Camera | Setting | FPS | Notes |
|---   |---     |---      |---  |---    |
| data/elbow_A_laptop.mp4 | laptop webcam | 640x480 MJPG | 30.0 | 300 frames, 10 s, good light |
| data/elbow_A_landmarks.npy | (made from the clip) | | | shape (301, 33, 4) |

## Landmark check (draw_check.py)
- Person found in 301 of 301 frames
- Left arm: shoulder, elbow, wrist visibility >= 0.7 in 100% of frames
- Right arm: shoulder, elbow, wrist visibility >= 0.7 in 100% of frames

## Lessons learned
- Run all commands from the main project folder, not from inside capture/ or pose/
- The recorder script needs the data/ folder to exist
- Replaying a clip needs a wait of 1000/FPS ms per frame, or it plays too fast
- MediaPipe 0.10.14 needs Python 3.12 or older
- The on-screen FPS during a replay is only the window speed, not the clip speed

## Week 2 results (Person A/B combined — Chamodi)

### Variants implemented
| Variant | File | Description |
|---|---|---|
| A | angles.py | 2D pixel angle (baseline from Week 1) |
| B | angles_world.py | 3D world landmark angle |
| C | angles_3d.py | 3D angle ÷ cos(body yaw), capped at 60° |
| E (gate) | angles_3d.py | bone-length gate, rejects frames with >40% jump or <5% min |

### Pipeline additions this week
- `pose_normalise.py` — distance/size invariant landmark normalisation (A)
- `calibration.py` — population correction curve with LOPO cross-validation (A)
- `capture/calibrate_ui.py` — countdown calibration screen, saves user_calibration.json (A)
- `gap_fill.py` — linear interpolation for gaps ≤ 5 frames (merged Week 2 PR)
- `demo_cv.py` — full live pipeline: variants A, B, C; calibration; yaw warning; gate
- `angles_3d.py` — body_yaw, elbow_angle_yaw_corrected, bone_length_gate (B)
- `capture/view_check.py` — real-time yaw warning overlay (B)

### Open items
- B2: cam18 extraction (needs Person B's videos on Drive)
- B5/A6: benchmark runs on REHAB24-6 dataset (needs dataset access in Colab)
- B6: threshold tuning on PM_013, PM_014, PM_025, PM_040 recordings
- Protractor validation at 90/120/160 deg (carried from Week 1)
- Live filter final decision (One Euro currently used as placeholder)
- exercises.yaml format agreement for Milestone 2