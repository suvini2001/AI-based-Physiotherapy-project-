import os
import tempfile
import pandas as pd
from benchmark_common import save_rows, load_variants, ROW_COLUMNS


def test_save_and_load_variants():
    rows = [
        {
            "variant": "variant_A",
            "video": "vid1",
            "side": "right",
            "cam": "cam17",
            "ex": 1,
            "view": "front",
            "person": "PM_001",
            "n": 100,
            "coverage": 100.0,
            "mae": 8.5,
            "rmse": 10.0,
            "bias": 1.2,
            "r": 0.95,
            "share_gt20": 5.0
        }
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "variant_A.csv")
        save_rows(rows, csv_path)
        
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert list(df.columns) == ROW_COLUMNS
        assert df.iloc[0]["variant"] == "variant_A"