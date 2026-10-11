'''The overall flow is:
1. Load the landmark data.
2. Read the visibility threshold from the configuration.
3. Mark low-visibility and out-of-frame landmarks.
4. Replace rejected points with NaN.
5. Preserve the array's shape.
6. Compare how much data is lost at each threshold.
'''

import numpy as np
import yaml

#  load the config file
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def mask_low_visibility(landmarks, threshold=None):
    """(frames, 33, 4) in -> (frames, 33, 4) out.
    
    A point becomes NaN if its visibility is below the threshold,
    or if its x or y is outside the picture (below 0 or above 1)."""
    if threshold is None:
        threshold = load_config()["visibility_threshold"]     

    # original data will not be changed - otherwise can't compare original and masked data
    out = landmarks.copy()   
    #  take index 3 - visibility
    low = out[..., 3] < threshold
    #  Find coordinates outside the image
    outside = (out[..., 0] < 0) | (out[..., 0] > 1) | (out[..., 1] < 0) | (out[..., 1] > 1)
    # replace rejected points with NaN
    out[low | outside] = np.nan        # whole point (x, y, z, vis) -> NaN
    return out


if __name__ == "__main__":
    data = np.load("data/elbow_A_landmarks.npy")
    arm = [12, 14, 16]                 # right shoulder, elbow, wrist (the moving arm)
    print("threshold | all points dropped | right-arm points dropped")
    for t in [0.5, 0.6, 0.7, 0.8]:
        masked = mask_low_visibility(data, t)
        assert masked.shape == data.shape
        all_drop = np.isnan(masked[..., 0]).mean() * 100
        arm_drop = np.isnan(masked[:, arm, 0]).mean() * 100
        print(f"  {t}     |      {all_drop:5.1f}%        |      {arm_drop:5.1f}%")