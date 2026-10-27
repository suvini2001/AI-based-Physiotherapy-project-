import sys
import time
import cv2

backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

# (width, height, use MJPG?)  MJPG is a compressed format many webcams need for 30 FPS.
settings = [
    (640, 480, False),
    (640, 480, True),
    (1280, 720, False),
    (1280, 720, True),
]

for width, height, mjpg in settings:
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        print("Camera did not open")
        sys.exit(1)
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Warm-up: the first frames are slow while the camera adjusts exposure.
    for _ in range(30):
        cap.read()

    frames = 0
    start = time.time()
    while frames < 90:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
    seconds = time.time() - start

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    name = "MJPG" if mjpg else "default"
    print(f"asked {width}x{height} {name:7s} -> got {real_w}x{real_h}, "
          f"{frames / seconds:.1f} FPS")