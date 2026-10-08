import numpy as np

# Fake landmark file: 100 frames, 33 body points, 4 numbers each (x, y, z, visibility)
frames = 100
landmarks = np.random.rand(frames, 33, 4).astype(np.float32)
np.save("data/sample_landmarks.npy", landmarks)

# Fake angle file: one row per frame
with open("data/sample_angles.csv", "w") as f:
    f.write("frame,time_s,elbow_angle,valid\n")
    for i in range(frames):
        f.write(f"{i},{i / 30:.3f},{90 + 50 * np.sin(i / 10):.1f},1\n")

print("Made data/sample_landmarks.npy", landmarks.shape)
print("Made data/sample_angles.csv")