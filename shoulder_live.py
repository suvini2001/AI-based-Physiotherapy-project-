"""shoulder_live.py - shoulder I_ML (learned elevation angle) for the live webcam pipeline.

Same idea as iml_live.IMLStream for the elbow, but with the shoulder features:
ShoulderFeatureStream builds one row per frame from past frames only, exactly as
it did when the training data was made (S01b / S02), and the LightGBM model turns
that row into the shoulder elevation angle (0 = arm down, 90 = horizontal,
180 = overhead).

The model works for front AND side views (S02: E3 median error front 5.6 deg,
side 9.1 deg), so a shoulder session no longer has to face the camera.
"""
import json

import numpy as np

from shoulder_features import SH_FEATURE_NAMES, ShoulderFeatureStream


def load_shoulder_model(path):
    import lightgbm as lgb
    with open(path + ".json", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("joint", "shoulder") != "shoulder":
        raise ValueError(f"{path} is a {meta.get('joint')} model, not a shoulder model")
    unknown = [f for f in meta.get("features", []) if f not in SH_FEATURE_NAMES]
    if unknown or not meta.get("features"):
        raise ValueError(f"model expects features this code does not build: {unknown or 'none listed'}")
    return lgb.Booster(model_file=path), meta


class ShoulderIMLStream:
    """norm, world (33, 4) per frame  ->  shoulder elevation in degrees, or None.

    None for the first `min_frames` frames (session statistics not ready yet) and
    whenever the shoulder / elbow / hips are not seen; demo_cv then falls back to the
    curve or the raw 2D angle for that frame."""

    def __init__(self, model_path, side="right", size=(640, 480), fps=30.0,
                 threshold=0.70, min_frames=30):
        self.booster, self.meta = load_shoulder_model(model_path)
        self.cols = [SH_FEATURE_NAMES.index(f) for f in self.meta["features"]]
        self.min_frames = min_frames
        self.fs = ShoulderFeatureStream(side, size, fps, threshold, min_frames=min_frames)
        self.last_features = None

    @property
    def ready(self):
        return len(self.fs) >= self.min_frames

    def step(self, norm, world):
        row = self.fs.push(norm, world)
        self.last_features = row
        # training rows always had the 2D angle (S02 dropped rows without it)
        if not self.ready or not np.isfinite(row[SH_FEATURE_NAMES.index("angle_2d")]):
            return None
        pred = self.booster.predict(row[self.cols][None], num_threads=1)[0]
        return float(min(max(pred, 0.0), 180.0))
