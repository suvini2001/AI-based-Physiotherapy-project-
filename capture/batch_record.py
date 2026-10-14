import os
import sys
import time
import cv2
from camera import FrameSource

# (file name, what to do, seconds to record)
CLIPS = {
    "elbow": [
        ("elbow_A_slow",     "CORRECT, SLOW: 4-5 s each way",                 12),
        ("elbow_A_fast",     "TOO FAST: about 1 s each way",                  10),
        ("elbow_A_half",     "WRONG: bend only halfway",                      10),
        ("elbow_A_shoulder", "WRONG: shoulder lifts/moves while bending",     10),
        ("elbow_A_pause",    "Pause 2 s in the middle of a rep",              12),
        ("elbow_A_gap",      "Move wrist out of frame briefly, then back",    10),
        ("elbow_A_long",     "CORRECT, normal speed, 8 reps",                 25),
    ],
}


def put_lines(frame, lines, y=30):
    for text in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        y += 28


def record_one(src, path, label, seconds, countdown=8):
    width, height = src.size()
    name = os.path.basename(path)

    # Countdown (also measures the real camera speed). s = skip, q = quit all.
    frames, start = 0, time.time()
    end = start + countdown
    while time.time() < end:
        ok, frame = src.read_frame()
        if not ok:
            return "quit"
        frames += 1
        left = int(end - time.time()) + 1
        put_lines(frame, [f"NEXT: {name}", label, "s = skip, q = quit"])
        cv2.putText(frame, str(left), (width // 2 - 40, height // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 8)
        cv2.imshow("batch recording", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            return "skip"
        if key == ord("q"):
            return "quit"
    fps = frames / (time.time() - start)

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    if not writer.isOpened():
        print(f"ERROR: cannot save to {path}. Run from the main project folder.")
        return "quit"

    rec_frames, rec_start = 0, time.time()
    while time.time() - rec_start < seconds:
        ok, frame = src.read_frame()
        if not ok:
            break
        writer.write(frame)
        rec_frames += 1
        shown = frame.copy()
        cv2.circle(shown, (width - 30, 30), 14, (0, 0, 255), -1)
        put_lines(shown, [name], y=30)
        cv2.imshow("batch recording", shown)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    writer.release()
    real = rec_frames / (time.time() - rec_start)
    print(f"Saved {path}: {rec_frames} frames, {real:.1f} FPS (countdown said {fps:.1f})")
    return "ok"


def main(exercise):
    src = FrameSource(0)
    for name, label, seconds in CLIPS[exercise]:
        path = f"data/{name}.mp4"
        if os.path.exists(path):
            print(f"Already exists, skipping {path}")
            continue
        result = record_one(src, path, label, seconds)
        if result == "quit":
            break
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CLIPS:
        print("Usage: python capture/batch_record.py elbow   (or knee)")
        sys.exit(1)
    main(sys.argv[1])