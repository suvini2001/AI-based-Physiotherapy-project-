import cv2
import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    """Frame in -> array of 33 landmarks (x, y, z, visibility) out."""

    def __init__(self, model_complexity=1):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr):
        # MediaPipe wants RGB; OpenCV gives BGR. Do NOT mirror-flip before this.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        out = np.full((33, 4), np.nan, dtype=np.float32)  # NaN = no person found
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                out[i] = [lm.x, lm.y, lm.z, lm.visibility]
        return out

    def close(self):
        self.pose.close()