"""Train Model A: Random Forest tabular classifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, MODEL_DIR, RANDOM_SEED, ensure_directories
from scripts.common import load_dataset, split_frame


def main() -> None:
    ensure_directories()
    df = load_dataset()
    train_df, val_df, test_df, splits = split_frame(df)
    X_train_raw = train_df[FEATURE_COLUMNS]
    y_train_raw = train_df["risk_label"]

    pos = int(y_train_raw.sum())
    neg = int((y_train_raw == 0).sum())
    if neg / max(pos, 1) > 10:
        print(f"Applying SMOTE: {pos} positive vs {neg} negative samples")
        k = min(5, pos - 1)
        smote = SMOTE(
            random_state=RANDOM_SEED,
            k_neighbors=k,
            sampling_strategy=0.2,
        )
        X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
        print(f"After SMOTE: {int(y_train.sum())} positive vs {int((y_train==0).sum())} negative")
    else:
        X_train, y_train = X_train_raw, y_train_raw
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)),
    ])
    search = RandomizedSearchCV(
        pipe,
        {
            "rf__n_estimators": [100, 200, 300, 500],
            "rf__max_depth": [6, 10, 14, 20, None],
            "rf__min_samples_leaf": [1, 2, 3, 5, 10],
            "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
        },
        n_iter=20,
        cv=5,
        scoring="f1_macro",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    val_proba = search.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    test_proba = search.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    pred = (test_proba >= 0.5).astype(int)
    metrics = {
        "validation_auc_roc": float(roc_auc_score(val_df["risk_label"], val_proba)),
        "test_auc_roc": float(roc_auc_score(test_df["risk_label"], test_proba)),
        "test_f1_macro": float(f1_score(test_df["risk_label"], pred, average="macro")),
        "test_average_precision": float(average_precision_score(test_df["risk_label"], test_proba)),
        "best_params": search.best_params_,
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
    }
    joblib.dump({"model": search.best_estimator_, "metrics": metrics, "features": FEATURE_COLUMNS}, MODEL_DIR / "model_rf.pkl")
    print(metrics)


if __name__ == "__main__":
    main()
