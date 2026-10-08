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