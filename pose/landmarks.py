import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Pose module එක කෙලින්ම Import කරගැනීම
try:
    import mediapipe.python.solutions.pose as mp_pose
except ImportError:
    mp_pose = mp.solutions.pose

class LandmarkExtractor:
    """Frame in -> array of 33 landmarks (x, y, z, visibility) out."""

    def __init__(self, model_complexity=1):
        # mp.solutions.pose.Pose වෙනුවට mp_pose.Pose පාවිච්චි කරන්න
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            smooth_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5    
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


    def process_with_world(self, frame_bgr):
        """Return (norm, world). Both float32, shape (33, 4), all NaN if no person.
        norm  = x, y (0-1 of the image), z (relative depth), visibility   <- same as process()
        world = x, y, z in METRES (origin = middle of the hips), visibility
        Both come from the SAME MediaPipe call, so this adds almost no time."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)   # do NOT mirror-flip before this
        result = self.pose.process(rgb)
        norm = np.full((33, 4), np.nan, dtype=np.float32)
        world = np.full((33, 4), np.nan, dtype=np.float32)
        if result.pose_landmarks:
            for i, lm in enumerate(result.pose_landmarks.landmark):
                norm[i] = [lm.x, lm.y, lm.z, lm.visibility]
        if result.pose_world_landmarks:
            for i, lm in enumerate(result.pose_world_landmarks.landmark):
                world[i] = [lm.x, lm.y, lm.z, lm.visibility]
        return norm, world
    def close(self):
        self.pose.close()