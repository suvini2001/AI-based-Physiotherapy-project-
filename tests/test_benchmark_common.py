import numpy as np
import pandas as pd
import pytest
from benchmark_common import (
    load_truth_array,
    truth_elbow_3d,
    overlap,
    best_lag,
    metrics,
    make_row,
    person_of,
    summary,
    results_table,
    to_markdown,
    delta,
    ROW_COLUMNS
)


def test_person_of():
    assert person_of("PM_013_arm_ex1") == "PM_013"
    assert person_of("PM_005_cam17") == "PM_005"
    assert person_of("subject_x") == "subject_x"


def test_overlap():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50, 60])
    
    # Positive lag
    x, y = overlap(a, b, lag=2)
    np.testing.assert_array_equal(x, [1, 2, 3, 4])
    np.testing.assert_array_equal(y, [30, 40, 50, 60])

    # Negative lag
    x, y = overlap(a, b, lag=-1)
    np.testing.assert_array_equal(x, [2, 3, 4, 5])
    np.testing.assert_array_equal(y, [10, 20, 30, 40])


def test_best_lag():
    t = np.linspace(0, 4 * np.pi, 100)
    truth = 90 + 30 * np.sin(t)
    
    # Shift pred by +3 frames (lag = -3 in overlap signature)
    pred = np.roll(truth, 3)
    pred[:3] = np.nan
    
    lag, r = best_lag(pred, truth, max_lag=10)
    assert lag == -3  # Corrected lag expected value
    assert r > 0.95


def test_metrics():
    # Provided at least 10 valid frames so ok.sum() >= 10
    pred = np.array([100.0, 110.0, 120.0, 130.0, np.nan, 150.0, 160.0, 170.0, 160.0, 150.0, 140.0, 130.0])
    truth = np.array([100.0, 105.0, 120.0, 155.0, 140.0, 150.0, 160.0, 170.0, 160.0, 150.0, 140.0, 130.0])
    
    m = metrics(pred, truth, big_error=20.0)
    assert m is not None
    assert m["n"] == 11
    assert pytest.approx(m["coverage"], 0.1) == (11 / 12) * 100.0
    assert "mae" in m
    assert "share_gt20" in m


def test_make_row():
    pred = np.linspace(90, 150, 50)
    truth = pred + 2.0
    meta = {
        "video": "test_vid",
        "side": "right",
        "cam": "cam17",
        "ex": 1,
        "view": "front",
        "person": "PM_001"
    }
    
    row = make_row("variant_A", pred, truth, meta)
    assert row is not None
    assert row["variant"] == "variant_A"
    assert row["person"] == "PM_001"
    assert pytest.approx(row["mae"], 0.01) == 2.0


def test_results_table_and_markdown():
    data = [
        {"variant": "A", "view": "front", "mae": 8.0, "share_gt20": 10.0, "coverage": 90.0},
        {"variant": "A", "view": "side", "mae": 25.0, "share_gt20": 45.0, "coverage": 85.0},
        {"variant": "B", "view": "front", "mae": 8.2, "share_gt20": 11.0, "coverage": 90.0},
        {"variant": "B", "view": "side", "mae": 12.0, "share_gt20": 18.0, "coverage": 85.0},
    ]
    df = pd.DataFrame(data)
    tbl = results_table(df, min_coverage=50.0)
    assert "front" in tbl.columns
    assert "side" in tbl.columns
    
    md = to_markdown(tbl)
    assert "| Variant |" in md
    assert "8.0 (10%)" in md


def test_delta():
    # Included share_gt20 column to satisfy summary() aggregation requirements
    data = [
        {"variant": "A", "view": "side", "mae": 25.0, "share_gt20": 40.0, "coverage": 90.0},
        {"variant": "B", "view": "side", "mae": 15.0, "share_gt20": 20.0, "coverage": 90.0},
    ]
    df = pd.DataFrame(data)
    d = delta(df, new="B", old="A", metric="mae")
    assert pytest.approx(d["side"], 0.01) == -10.0