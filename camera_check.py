import sys
import time
import cv2

# On Windows use the DirectShow backend; on Mac/Linux the default is fine.
backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
cap = cv2.VideoCapture(0, backend)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("Camera did not open. Close Zoom/Teams/browser tabs and try again.")
    sys.exit(1)

# Read back what the camera REALLY gives us (not what we asked for).
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Measure the real FPS over 90 frames.
frames = 0
start = time.time()
while frames < 90:
    ok, frame = cap.read()
    if not ok:
        break
    frames += 1
seconds = time.time() - start
cap.release()

print(f"Resolution: {width} x {height}")
print(f"Measured FPS: {frames / seconds:.1f}")
print("Write these two numbers and your laptop model in your notes.")