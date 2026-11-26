Data formats (agreed between Person A and Person B)

Rule: neither of us changes these without telling the other.

1. Landmark file (A makes it, B reads it)
- File type: .npy (a saved NumPy array)
- Shape: (frames, 33, 4)
- frames = one entry per video frame
- 33 = one entry per body point (MediaPipe Pose order)
- 4 = x, y, z, visibility
- x and y are 0 to 1 across the picture; visibility is 0 to 1
- If no person is found in a frame, all its numbers are NaN
- Sample file: data/sample_landmarks.npy

2. Angle file (B makes it, later steps read it)
- File type: .csv
- One row per frame
- Columns: frame, time_s, elbow_angle, valid
- frame = frame number starting at 0
- time_s = time in seconds
- elbow_angle = angle in degrees
- valid = 1 if the angle is usable, 0 if not
- Sample file: data/sample_angles.csv

Naming clips and files: person_exercise_camera, for example elbow_A_laptop.mp4

# Live Frame Binary Layout — v1

Fixed 20-byte frame. Little-endian.

| Offset | Bytes | Type   | Field        | Notes                          |
|--------|-------|--------|--------------|--------------------------------|
| 0      | 1     | uint8  | version      | Always 1                       |
| 1      | 4     | float32| timestamp    | Seconds since session start    |
| 5      | 4     | float32| joint_angle  | Degrees                        |
| 9      | 1     | uint8  | state        | 0=READY 1=CALIB 2=COUNT        |
| 10     | 2     | uint16 | rep_count    | 0–65535                        |
| 12     | 1     | uint8  | fault_code   | 0=none, see fault table below  |
| 13     | 7     | bytes  | reserved     | Zero-padded, for future use    |

## Fault codes
0 = no fault
1 = elbow_too_low
2 = body_sway
3 = incomplete_rom

## State codes
0 = READY
1 = CALIB
2 = COUNT