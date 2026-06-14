"""Run controlled 25/50/75 km label-buffer experiments on one fixed grid."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments"
STAGES = [
    "04_preprocess.py",
    "05_eda.py",
    "06_model_rf.py",
    "06_model_cnn.py",
    "06_ensemble.py",
    "07_evaluate.py",
    "07_shap_gradcam.py",
    "08_risk_map.py",
]


def run_stage(script: str, env: dict[str, str]) -> None:
    print(f"\nRunning {script}")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )


def replace_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def make_cnn_payload_self_contained(destination: Path) -> None:
    payload_path = destination / "models" / "model_cnn.pkl"
    if not payload_path.exists():
        return
    payload = joblib.load(payload_path)
    if payload.get("backend") == "tensorflow_resnet50":
        keras_path = destination / "models" / "model_cnn_resnet50.keras"
        payload["model_path"] = str(keras_path.resolve())
        joblib.dump(payload, payload_path)


def collect_summary(buffer_km: int, destination: Path) -> dict:
    metadata = json.loads(
        (PROJECT_ROOT / "data" / "processed" / "dataset_metadata.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (PROJECT_ROOT / "outputs" / "evaluation_metrics.json").read_text(encoding="utf-8")
    )
    weights = json.loads(
        (PROJECT_ROOT / "models" / "ensemble_weights.json").read_text(encoding="utf-8")
    )
    rf_payload = joblib.load(PROJECT_ROOT / "models" / "model_rf.pkl")
    splits = json.loads(
        (PROJECT_ROOT / "models" / "split_indices.json").read_text(encoding="utf-8")
    )
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "dataset_ml.parquet")

    split_positive = {
        name: int(df.iloc[index]["risk_label"].sum())
        for name, index in splits.items()
    }
    summary = {
        "risk_buffer_km": buffer_km,
        "grid_resolution_degrees": metadata["grid_resolution_degrees"],
        "n_grid_cells": metadata["n_grid_cells"],
        "n_positive": metadata["n_positive"],
        "positive_rate": metadata["positive_rate"],
        "split_sizes": {name: len(index) for name, index in splits.items()},
        "split_positive": split_positive,
        "rf_training_metrics": rf_payload["metrics"],
        "ensemble_weights": weights,
        "test_metrics": evaluation,
    }
    (destination / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffers", nargs="+", type=int, default=[25, 50, 75])
    args = parser.parse_args()
    buffers = list(dict.fromkeys(args.buffers))
    if 50 in buffers:
        buffers = [50] + [value for value in buffers if value != 50]

    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for buffer_km in buffers:
        print(f"\n{'=' * 72}\nCONTROLLED BUFFER EXPERIMENT: {buffer_km} km\n{'=' * 72}")
        env = os.environ.copy()
        env["RISK_BUFFER_KM"] = str(buffer_km)
        for stage in STAGES:
            run_stage(stage, env)

        destination = EXPERIMENT_ROOT / f"buffer_{buffer_km}km"
        destination.mkdir(parents=True, exist_ok=True)
        replace_tree(PROJECT_ROOT / "outputs", destination / "outputs")
        replace_tree(PROJECT_ROOT / "models", destination / "models")
        replace_tree(PROJECT_ROOT / "data" / "processed", destination / "processed")
        make_cnn_payload_self_contained(destination)
        summaries.append(collect_summary(buffer_km, destination))

    ranked = sorted(
        summaries,
        key=lambda item: (
            item["ensemble_weights"]["validation_metrics"]["f1_macro"],
            item["ensemble_weights"]["validation_metrics"]["average_precision"],
        ),
        reverse=True,
    )
    comparison = {
        "selection_rule": (
            "Highest validation F1-macro, with validation Average Precision as the tie-breaker. "
            "Test metrics are reported after selection and are not used to choose the buffer."
        ),
        "preferred_buffer_km": ranked[0]["risk_buffer_km"],
        "experiments": summaries,
    }
    (EXPERIMENT_ROOT / "buffer_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame([
        {
            "buffer_km": item["risk_buffer_km"],
            "n_grid_cells": item["n_grid_cells"],
            "n_positive": item["n_positive"],
            "positive_rate": item["positive_rate"],
            "validation_f1_macro": item["ensemble_weights"]["validation_metrics"]["f1_macro"],
            "validation_average_precision": item["ensemble_weights"]["validation_metrics"]["average_precision"],
            "test_auc_roc": item["test_metrics"]["model_c_late_fusion"]["auc_roc"],
            "test_f1_macro": item["test_metrics"]["model_c_late_fusion"]["f1_macro"],
            "test_average_precision": item["test_metrics"]["model_c_late_fusion"]["average_precision"],
        }
        for item in summaries
    ]).sort_values("buffer_km").to_csv(
        EXPERIMENT_ROOT / "buffer_comparison.csv",
        index=False,
    )

    baseline = EXPERIMENT_ROOT / "buffer_50km"
    if baseline.exists():
        replace_tree(baseline / "outputs", PROJECT_ROOT / "outputs")
        replace_tree(baseline / "models", PROJECT_ROOT / "models")
        replace_tree(baseline / "processed", PROJECT_ROOT / "data" / "processed")
        make_cnn_payload_self_contained(PROJECT_ROOT)
        print("Restored the main project artefacts to the default 50 km baseline.")

    print(f"\nPreferred buffer from validation metrics: {ranked[0]['risk_buffer_km']} km")
    print(f"Wrote {EXPERIMENT_ROOT / 'buffer_comparison.json'}")


if __name__ == "__main__":
    main()
