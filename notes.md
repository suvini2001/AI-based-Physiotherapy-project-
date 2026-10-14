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

## Open questions for B
- Who records the reference clips in Milestone 2, and on which camera?
- Knee exercise: standing knee flexion (heel to buttock) or mini-squat?
- Keep 640x480 or try for 720p with a better light or an external camera?

## Next (week 2)
- 1.6 Gap-filling function
- 1.9 Skeleton overlay and demo_cv.py
- Joint test with B