import numpy as np
from gap_fill import fill_gaps


def make_line(frames=30):
    """Landmark 0 moves in a perfect straight line, so the right answer is known."""
    arr = np.zeros((frames, 33, 4), dtype=np.float32)
    arr[:, :, 3] = 0.9
    arr[:, 0, 0] = np.linspace(0.2, 0.8, frames)
    arr[:, 0, 1] = np.linspace(0.3, 0.5, frames)
    return arr


def test_shape_and_dtype_kept():
    a = make_line()
    out = fill_gaps(a)
    assert out.shape == a.shape and out.dtype == a.dtype


def test_no_nan_means_no_change():
    a = make_line()
    assert np.array_equal(fill_gaps(a), a)


def test_short_gap_is_filled_on_the_line():
    a = make_line()
    a[:, 0, 3] = np.linspace(0.8, 0.9, len(a))     # visibility changes smoothly too
    truth = a.copy()
    a[10:14, 0, :] = np.nan                        # all 4 numbers, like B's mask
    out = fill_gaps(a, max_gap=5)
    assert not np.isnan(out).any()
    assert np.allclose(out[10:14, 0, :], truth[10:14, 0, :], atol=1e-5)


def test_long_gap_stays_missing():
    a = make_line()
    a[5:20, 0, :] = np.nan
    out = fill_gaps(a, max_gap=9)
    assert np.isnan(out[5:20, 0, :]).all()


def test_gap_at_start_or_end_stays_missing():
    a = make_line()
    a[0:3, 0, :] = np.nan
    a[-2:, 0, :] = np.nan
    out = fill_gaps(a)
    assert np.isnan(out[0:3, 0, :]).all() and np.isnan(out[-2:, 0, :]).all()


def test_input_not_modified():
    a = make_line()
    a[10:12, 0, :] = np.nan
    before = a.copy()
    fill_gaps(a)
    assert np.array_equal(a, before, equal_nan=True)


def test_gap_of_exactly_max_gap_is_filled():
    a = make_line()
    a[10:19, 0, :] = np.nan                        # 9 frames
    assert not np.isnan(fill_gaps(a, max_gap=9)[10:19, 0, :]).any()


def test_other_landmarks_not_touched():
    a = make_line()
    a[10:12, 0, :] = np.nan
    out = fill_gaps(a)
    assert np.array_equal(out[:, 1:], a[:, 1:])