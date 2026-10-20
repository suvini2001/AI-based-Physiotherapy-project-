import numpy as np
import pytest
from angles_world import elbow_angle_3d


def test_elbow_angle_3d_planar():
    # Synthetic right arm flexed at 90 degrees in 3D meters
    # Shoulder (12), Elbow (14), Wrist (16)
    world = np.zeros((1, 33, 4), dtype=np.float32)
    world[0, :, 3] = 0.9  # visibility
    
    world[0, 12, :3] = [0.0, 0.3, 0.0]  # Shoulder
    world[0, 14, :3] = [0.0, 0.0, 0.0]  # Elbow
    world[0, 16, :3] = [0.3, 0.0, 0.0]  # Wrist
    
    ang = elbow_angle_3d(world, side="right")
    assert pytest.approx(ang[0], 0.1) == 90.0


def test_elbow_angle_3d_yaw_invariance():
    # Verify that rotating a 60 degree elbow around Y axis (body turn)
    # leaves the 3D angle unchanged at 60 degrees.
    base_shoulder = np.array([0.0, 0.3, 0.0])
    base_elbow = np.array([0.0, 0.0, 0.0])
    base_wrist = np.array([0.3 * np.cos(np.radians(30)), 0.3 * np.sin(np.radians(30)), 0.0])
    
    n_frames = 4
    yaws = [0, 20, 40, 60]
    world = np.zeros((n_frames, 33, 4), dtype=np.float32)
    world[:, :, 3] = 0.9
    
    for i, yaw in enumerate(yaws):
        rad = np.radians(yaw)
        R_y = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
        world[i, 12, :3] = R_y @ base_shoulder
        world[i, 14, :3] = R_y @ base_elbow
        world[i, 16, :3] = R_y @ base_wrist

    angles = elbow_angle_3d(world, side="right")
    for a in angles:
        assert pytest.approx(a, 0.1) == 60.0


def test_elbow_angle_3d_nan_handling():
    world = np.full((1, 33, 4), np.nan, dtype=np.float32)
    ang = elbow_angle_3d(world, side="right")
    assert np.isnan(ang[0])