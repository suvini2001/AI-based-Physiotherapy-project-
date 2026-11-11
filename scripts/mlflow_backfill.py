"""Record the experiments ALREADY DONE (10 Nov 2026) in MLflow, from the files they wrote.

Nothing is re-run and no number is typed in by hand except the two console results
that were never saved to a file (marked source=console_output).

    pip install mlflow
    python scripts/mlflow_backfill.py            # writes mlflow.db + mlruns/ in the repo root
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
    -> open http://127.0.0.1:5000

What it creates in experiment "Physio-ROM-Estimation":
  curve-vs-iml-nested-2026-11-10          parent run (decision, settings, artefacts)
    E1/E2a/E2b/E3 x raw/curve/iml/iml_nolook   16 child runs, metrics from the per-recording CSV
  curve-refit-sideon-2026-11-10           profile curve refitted on camera-relative views
  benchmark-causal-rerun-2026-11-10       REHAB24-6 benchmark after the causal-filter fix
  iml-nolook-rehab24-deployable           the model file that goes into the live system

Run it once. Running again creates duplicates (delete them in the UI, or use --fresh).
mlflow.db and mlruns/ are git-ignored: they may hold Fit3D-derived results (licence).
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIT3D = os.path.join(ROOT, "data", "fit3d")
EXPERIMENT = "Physio-ROM-Estimation"
DATE = "2026-11-10"
COMMON_TAGS = {"result_type": "historical_import", "original_experiment_date": DATE}


def need(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} - run 02_curve_vs_iml.ipynb (TUNE = True) first, or fix the path")
    return path


def per_method_metrics(df, m):
    e = df[m]
    q25, q75 = np.percentile(e, [25, 75])
    return {"median_mae_deg": float(np.median(e)), "mean_mae_deg": float(e.mean()),
            "iqr_mae_deg": float(q75 - q25), "n_recordings": int(len(e)),
            "n_people": int(df["person"].nunique())}


def log_nested(mlflow, args):
    rec = pd.read_csv(need(os.path.join(FIT3D, "curve_vs_iml_per_recording.csv")))
    res = json.load(open(need(os.path.join(FIT3D, "curve_vs_iml_results.json"))))
    methods = [m for m in ("raw", "curve", "iml", "iml_nolook") if m in rec.columns]
    with mlflow.start_run(run_name=f"curve-vs-iml-nested-{DATE}") as parent:
        mlflow.set_tags({**COMMON_TAGS, "stage": "offline_evaluation",
                         "source": "notebooks/02_curve_vs_iml.ipynb", "choice": res.get("choice", ""),
                         "decision_reason": res.get("why", "")[:500]})
        mlflow.log_params({"protocol": "LOSO + cross-dataset, nested CV (inner GroupKFold 4)",
                           "datasets": "REHAB24-6 (22 people) + Fit3D train (8 people)",
                           "views": "camera-relative (REHAB cam18 labels converted)",
                           "filter": "causal Butterworth 2nd order 6 Hz",
                           "decision_rule": ">=4 deg better on profile, both datasets, p<0.05",
                           "tune": True})
        for k, v in (res.get("final_hyperparameters") or {}).items():
            mlflow.log_param(f"final_{k}", v)
        for row in res.get("paired_profile", []):
            exp = str(row.get("experiment", "")).split()[0]
            for key in ("gain_median", "wilcoxon_p"):
                v = row.get(key)
                if v is not None and np.isfinite(float(v)):
                    mlflow.log_metric(f"profile_{exp}_{key}", float(v))
        for f in ("curve_vs_iml_results.json", "curve_vs_iml_per_recording.csv",
                  "tuning_log.csv", "curve_vs_iml.png"):
            p = os.path.join(FIT3D, f)
            if os.path.exists(p):
                mlflow.log_artifact(p, artifact_path="results")

        for exp, d in rec.groupby("experiment"):
            short = exp.split()[0]
            prof = d[d["vclass"] == "profile"]
            for m in methods:
                with mlflow.start_run(run_name=f"{short}-{m}", nested=True):
                    mlflow.set_tags({**COMMON_TAGS, "stage": "offline_evaluation", "method": m,
                                     "experiment_split": exp, "parent": parent.info.run_id})
                    mlflow.log_params({"method": m, "experiment": exp,
                                       "features": {"iml": 16, "iml_nolook": 14}.get(m, 1)})
                    mlflow.log_metrics(per_method_metrics(d, m))
                    if len(prof):
                        mlflow.log_metric("profile_median_mae_deg", float(prof[m].median()))
                        mlflow.log_metric("profile_n_recordings", int(len(prof)))
                    for vc, g in d.groupby("vclass"):
                        mlflow.log_metric(f"{vc}_median_mae_deg", float(g[m].median()))
    print(f"logged nested comparison: {rec['experiment'].nunique()} experiments x {len(methods)} methods")


def log_curve_refit(mlflow):
    path = need(os.path.join(ROOT, "data", "population_curve.json"))
    curve = json.load(open(path))
    with mlflow.start_run(run_name=f"curve-refit-sideon-{DATE}"):
        mlflow.set_tags({**COMMON_TAGS, "stage": "calibration", "method": "curve",
                         "source": "scripts/fit_calibration.py console_output"})
        mlflow.log_params({"views": "Profile + Front/cam18 (camera-relative side-on)",
                           "pairs": "data/calib_pairs_rehab24.npz", "people": 16,
                           "knots": len(curve.get("knots_raw", []))})
        # printed by fit_calibration.py on 10 Nov (not saved to a file)
        mlflow.log_metrics({"lopo_raw_mae_deg": 23.34, "lopo_calibrated_mae_deg": 15.77,
                            "truth_p5_deg": 49, "truth_p95_deg": 173,
                            "previous_curve_lopo_raw_mae_deg": 20.1,
                            "previous_curve_lopo_calibrated_mae_deg": 16.3})
        mlflow.log_artifact(path, artifact_path="curve")
        old = os.path.join(ROOT, "data", "population_curve_old_mixed.json")
        if os.path.exists(old):
            mlflow.log_artifact(old, artifact_path="curve_previous")
    print("logged curve refit")


def log_benchmark(mlflow):
    # REHAB24-6 benchmark (benchmark_3d_world.ipynb) re-run with the causal filter.
    # Median MAE over 56 recordings, printed in Colab on 10 Nov (variant CSVs stay on Drive).
    causal = {"I_ML_Features": 10.58, "D_2D_PerView": 13.38, "H_Hybrid_PerView": 13.78,
              "D_2D_Calibrated": 14.21, "D_3D_Calibrated": 15.07, "A_2D_Base": 16.16,
              "G_Hybrid": 16.36, "F_Lift_PerView": 18.91, "E_2D_Lifted": 19.28, "B_3D_World": 21.90}
    before = {"I_ML_Features": 10.79, "D_2D_PerView": 13.33, "A_2D_Base": 16.12, "B_3D_World": 21.93}
    with mlflow.start_run(run_name=f"benchmark-causal-rerun-{DATE}"):
        mlflow.set_tags({**COMMON_TAGS, "stage": "offline_evaluation",
                         "source": "benchmark_3d_world.ipynb console_output (Colab)"})
        mlflow.log_params({"dataset": "REHAB24-6", "recordings": 56, "cv": "GroupKFold 5 by person",
                           "filter": "causal Butterworth (lfilter) after fix d051530"})
        mlflow.log_metrics({f"{k}_median_mae_deg": v for k, v in causal.items()})
        mlflow.log_metrics({f"{k}_median_mae_deg_filtfilt": v for k, v in before.items()})
    print("logged benchmark re-run")


def log_model(mlflow):
    path = os.path.join(ROOT, "data", "iml_nolook_rehab24.txt")
    if not os.path.exists(path):
        print("skip model run: data/iml_nolook_rehab24.txt not found (run the deployable-model cell)")
        return
    meta = json.load(open(path + ".json"))
    with mlflow.start_run(run_name="iml-nolook-rehab24-deployable"):
        mlflow.set_tags({**COMMON_TAGS, "stage": "model", "method": "iml_nolook",
                         "trained_on": meta.get("trained_on", "REHAB24-6 only"), "licence": "CC BY-NC"})
        params = meta.get("params", {})
        mlflow.log_params({"features": len(meta["features"]), "feature_names": ",".join(meta["features"]),
                           **{k: params[k] for k in ("n_estimators", "num_leaves", "learning_rate",
                                                     "min_child_samples", "objective") if k in params}})
        for k, v in (meta.get("cv") or {}).items():
            mlflow.log_metric(f"cv_{k}_deg", float(v))
        mlflow.log_metric("model_size_kb", os.path.getsize(path) / 1024)
        mlflow.log_artifact(path, artifact_path="model")
        mlflow.log_artifact(path + ".json", artifact_path="model")
    print("logged deployable model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default="sqlite:///" + os.path.join(ROOT, "mlflow.db").replace("\\", "/"))
    ap.add_argument("--fresh", action="store_true", help="delete earlier backfilled runs first")
    ap.add_argument("--only", nargs="+", choices=["nested", "curve", "benchmark", "model"],
                    help="log only these parts, e.g. --only model")
    a = ap.parse_args()
    import mlflow
    mlflow.set_tracking_uri(a.tracking_uri)
    exp = mlflow.set_experiment(EXPERIMENT)
    if a.fresh:
        old = mlflow.search_runs([exp.experiment_id], "tags.result_type = 'historical_import'")
        for rid in old.get("run_id", []):
            mlflow.delete_run(rid)
        print(f"deleted {len(old)} earlier backfilled runs")
    parts = {"nested": lambda: log_nested(mlflow, a), "curve": lambda: log_curve_refit(mlflow),
             "benchmark": lambda: log_benchmark(mlflow), "model": lambda: log_model(mlflow)}
    for name in (a.only or parts):
        parts[name]()
    print(f"\ndone. view with:  mlflow ui --backend-store-uri {a.tracking_uri} --port 5000")


if __name__ == "__main__":
    main()