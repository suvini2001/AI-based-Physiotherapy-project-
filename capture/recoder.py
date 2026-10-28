import sys
import time
import cv2
try:
    from .camera import FrameSource
except ImportError:
    from capture.camera import FrameSource


def record(path, seconds=10, countdown=5):
    """Record a clip from the live webcam, saved at the REAL frame rate."""
    src = FrameSource(0)
    width, height = src.size()

    # 1) Countdown: show big numbers while you walk back to your spot.
    #    We also count frames here to measure the camera's real FPS.
    count_frames = 0
    count_start = time.time()
    end_count = count_start + countdown
    while time.time() < end_count:
        ok, frame = src.read_frame()
        if not ok:
            break
        count_frames += 1
        left = int(end_count - time.time()) + 1
        cv2.putText(frame, str(left), (width // 2 - 60, height // 2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 12)
        cv2.imshow("recording", frame)
        cv2.waitKey(1)
    measured_fps = count_frames / (time.time() - count_start)
    print(f"Measured camera speed: {measured_fps:.1f} FPS")

    # 2) Create the video file. If this fails, stop and say why.
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                             measured_fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: cannot save to {path}. Does the folder exist? "
              "Run this from the main project folder.")
        src.release()
        cv2.destroyAllWindows()
        return

    # 3) Record, saving the video at the measured speed.
    rec_frames = 0
    rec_start = time.time()
    end_rec = rec_start + seconds
    while time.time() < end_rec:
        ok, frame = src.read_frame()
        if not ok:
            break
        writer.write(frame)
        rec_frames += 1
        shown = frame.copy()
        cv2.circle(shown, (50, 50), 20, (0, 0, 255), -1)  # red dot = recording
        cv2.imshow("recording", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    rec_seconds = time.time() - rec_start

    writer.release()
    src.release()
    cv2.destroyAllWindows()

    actual_fps = rec_frames / rec_seconds
    print(f"Saved {path}: {rec_frames} frames in {rec_seconds:.1f} s "
          f"= {actual_fps:.1f} FPS")
    if abs(actual_fps - measured_fps) > 0.15 * measured_fps:
        print("Warning: the speed changed during recording (the light may have "
              "changed). Check the clip plays at normal speed, or record again.")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/elbow_A_laptop.mp4"
    record(out)