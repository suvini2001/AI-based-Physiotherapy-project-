import sys
import time
import cv2


class FrameSource:
    """One interface for the live webcam AND a saved clip.

    source = 0            -> live laptop webcam
    source = "clip.mp4"   -> replay a saved video file
    """

    def __init__(self, source=0, width=640, height=480, fps=30):
        self.is_live = isinstance(source, int)
        if self.is_live:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(source, backend)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {source}")

    def read_frame(self):
        """Return (ok, frame). ok is False when the clip ends or the camera fails."""
        return self.cap.read()

    def fps(self):
        value = self.cap.get(cv2.CAP_PROP_FPS)
        return value if value and value > 0 else 30.0

    def size(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def release(self):
        self.cap.release()


class FpsCounter:
    """Smoothed frames-per-second, so the number on screen does not flicker."""

    def __init__(self):
        self.last = time.time()
        self.value = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt > 0:
            self.value = 0.9 * self.value + 0.1 * (1.0 / dt) if self.value else 1.0 / dt
        return self.value


if __name__ == "__main__":
    source = 0 if len(sys.argv) < 2 else sys.argv[1]
    src = FrameSource(source)
    counter = FpsCounter()
    while True:
        ok, frame = src.read_frame()
        if not ok:
            break
        fps = counter.tick()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("camera (press q to quit)", frame)
        # Live camera: no extra wait. Saved clip: wait one frame-time (e.g. 33 ms).
        delay = 1 if src.is_live else int(1000 / src.fps())
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break
    src.release()
    cv2.destroyAllWindows()