# This is a unit test file for your physiotherapy project. It checks whether the functions in angles.py 
# correctly calculate angles between body points, especially the elbow angle extracted from MediaPipe landmarks.


import numpy as np
from angles import angle, elbow_angle


def test_right_angle():
    assert np.isclose(angle([1, 0], [0, 0], [0, 1]), 90)


def test_straight_line():
    assert np.isclose(angle([-1, 0], [0, 0], [1, 0]), 180)


def test_forty_five_degrees():
    assert np.isclose(angle([1, 0], [0, 0], [1, 1]), 45)


def test_zero_length_gives_nan():
    assert np.isnan(angle([0, 0], [0, 0], [1, 1]))


def test_nan_point_gives_nan():
    assert np.isnan(angle([np.nan, np.nan], [0, 0], [1, 1]))


def test_many_frames_at_once():
    a = np.array([[1, 0], [-1, 0]])
    b = np.zeros((2, 2))
    c = np.array([[0, 1], [1, 0]])
    assert np.allclose(angle(a, b, c), [90, 180])


def test_elbow_angle_shape_and_value():
    lm = np.full((3, 33, 4), np.nan)
    lm[:, 12, :2] = [0.5, 0.5]    # shoulder
    lm[:, 14, :2] = [0.5, 0.6]    # elbow
    lm[:, 16, :2] = [0.6, 0.6]    # wrist -> right angle at the elbow
    result = elbow_angle(lm, "right", size=(100, 100))
    assert result.shape == (3,)
    assert np.allclose(result, 90)