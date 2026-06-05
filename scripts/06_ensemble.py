from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from config import MODEL_DIR, OUTPUT_DIR, ensure_directories
from scripts.common import load_dataset, split_frame
from scripts.patch_model import predict_patch_model


def main() -> None:
    ensure_directories()
    df = load_dataset()
    train_df, val_df, test_df, splits = split_frame(df)
    rf = joblib.load(MODEL_DIR / "model_rf.pkl")
    cnn = joblib.load(MODEL_DIR / "model_cnn.pkl")
    p_a_val = rf["model"].predict_proba(val_df[rf["features"]])[:, 1]
    p_b_val = predict_patch_model(cnn, val_df)
    y_val = val_df["risk_label"].to_numpy()

    def objective(w: float) -> float:
        p = w * p_a_val + (1 - w) * p_b_val
        return -f1_score(y_val, (p >= 0.5).astype(int), average="macro")

    result = minimize_scalar(objective, bounds=(0, 1), method="bounded")
    w = float(result.x)
    p_a = rf["model"].predict_proba(df[rf["features"]])[:, 1]
    p_b = predict_patch_model(cnn, df)
    proba = w * p_a + (1 - w) * p_b
    out = {
        "formula": "P(C) = w * P(A) + (1 - w) * P(B)",
        "w_model_a_rf": w,
        "w_model_b_cnn": 1 - w,
        "optimized_on": "validation split only",
        "threshold": 0.5,
        "risk_tiers": {"low": "<0.3", "medium": "0.3-0.6", "high": ">0.6"},
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
    }
    (MODEL_DIR / "ensemble_weights.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pred = df[["cell_id", "lon", "lat", "risk_label"]].copy()
    pred["split"] = "unused"
    for split_name, idx in splits.items():
        pred.loc[idx, "split"] = split_name
    pred["risk_probability"] = proba
    pred["risk_tier"] = pred["risk_probability"].map(lambda p: "low" if p < 0.3 else "medium" if p <= 0.6 else "high")
    pred.to_csv(OUTPUT_DIR / "ensemble_predictions.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
