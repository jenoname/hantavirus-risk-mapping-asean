from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, MODEL_DIR, OUTPUT_DIR, RANDOM_SEED, ensure_directories
from scripts.common import ensure_region_column, load_dataset, split_frame
from scripts.patch_model import predict_patch_model


def metric_block(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    rho = spearmanr(y, p).statistic
    return {
        "auc_roc": float(roc_auc_score(y, p)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "average_precision": float(average_precision_score(y, p)),
        "spearman_spatial_rho": None if np.isnan(rho) else float(rho),
    }


def loro_cross_validation(df: pd.DataFrame, feature_cols: list[str], random_seed: int) -> dict:
    df = ensure_region_column(df).copy()
    regions = sorted(df["asean_region"].dropna().unique())
    results = {}
    for held_out in regions:
        train_mask = df["asean_region"] != held_out
        test_mask = df["asean_region"] == held_out
        y_test = df.loc[test_mask, "risk_label"].to_numpy()
        print(
            f"LORO: held_out={held_out}, train={train_mask.sum()}, "
            f"test={test_mask.sum()}, pos={y_test.sum()}"
        )
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                random_state=random_seed,
                n_jobs=1,
            )),
        ])
        pipe.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, "risk_label"])
        if test_mask.sum() < 5:
            results[str(held_out)] = {"skipped": "too_few_samples", "n": int(test_mask.sum())}
            continue
        if y_test.sum() == 0:
            p = pipe.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
            results[str(held_out)] = {
                "note": "no_positive_labels_in_region",
                "mean_predicted_risk": float(p.mean()),
                "n_cells": int(test_mask.sum()),
            }
            continue
        p = pipe.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        results[str(held_out)] = metric_block(y_test, p)
    return results


def main() -> None:
    ensure_directories()
    df = load_dataset()
    train_df, val_df, test_df, splits = split_frame(df)
    rf = joblib.load(MODEL_DIR / "model_rf.pkl")
    cnn = joblib.load(MODEL_DIR / "model_cnn.pkl")
    if cnn.get("backend") != "tensorflow_resnet50":
        print("WARNING: Model B is running as MLP fallback — AUC may be near random (0.5).")
        print("         Install tensorflow>=2.15 and re-run 06_model_cnn.py to use ResNet-50.")
    weights = json.loads((MODEL_DIR / "ensemble_weights.json").read_text(encoding="utf-8"))
    p_a = rf["model"].predict_proba(test_df[rf["features"]])[:, 1]
    p_b = predict_patch_model(cnn, test_df)
    p_c = weights["w_model_a_rf"] * p_a + weights["w_model_b_cnn"] * p_b
    y = test_df["risk_label"].to_numpy()
    train_val_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    loro_results = loro_cross_validation(train_val_df, FEATURE_COLUMNS, RANDOM_SEED)
    metrics = {
        "model_a_random_forest": metric_block(y, p_a),
        "model_b_cnn": {
            "backend": cnn.get("backend"),
            "is_resnet50": cnn.get("backend") == "tensorflow_resnet50",
            "metrics": metric_block(y, p_b),
            "note": cnn.get("note", "TensorFlow ResNet-50 patch model."),
            "warning": (
                None if cnn.get("backend") == "tensorflow_resnet50"
                else "Model B is running as MLP fallback because TensorFlow is not available. "
                     "Install tensorflow>=2.15 and re-run 06_model_cnn.py for ResNet-50 results."
            ),
        },
        "model_c_late_fusion": metric_block(y, p_c),
        "loro_spatial_cv_by_asean_region": loro_results,
        "ablation": "Model A (RF) vs Model B (CNN) vs Model C (Ensemble late fusion)",
        "evaluated_on": "held-out test split only",
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
    }
    out = OUTPUT_DIR / "evaluation_metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
