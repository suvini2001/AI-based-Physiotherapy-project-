# check_fps.py
import cv2, time, sys

backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY

# on Windows, open with MJPEG flag directly
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# must set FOURCC before setting resolution
mjpg = cv2.VideoWriter_fourcc(*"MJPG")
cap.set(cv2.CAP_PROP_FOURCC, mjpg)

# verify it took
actual_codec = int(cap.get(cv2.CAP_PROP_FOURCC))
codec_str = actual_codec.to_bytes(4, 'little').decode('ascii', errors='?')
print(f"Codec after set: {codec_str}")

if codec_str.strip() != "MJPG":
    print("Driver rejected MJPG — trying with resolution change first")
    cap.release()
    cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)
    cap.set(cv2.CAP_PROP_FPS, 30)
    codec_str = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(
        4, 'little').decode('ascii', errors='?')
    print(f"Codec after retry: {codec_str}")

print(f"FPS setting: {cap.get(cv2.CAP_PROP_FPS):.0f}")
print(f"Resolution : {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x "
      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
print()
print("Measuring FPS over 60 frames...")

n, t0, times = 0, time.perf_counter(), []
for _ in range(60):
    t = time.perf_counter()
    ok, _ = cap.read()
    times.append((time.perf_counter()-t)*1000)
    if ok: n += 1

elapsed = time.perf_counter() - t0
print(f"Actual FPS     : {n/elapsed:.1f}")
print(f"Mean read time : {sum(times)/len(times):.1f} ms")
cap.release()