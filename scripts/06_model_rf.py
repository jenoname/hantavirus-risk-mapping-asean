"""Train Model A: Random Forest tabular classifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, MODEL_DIR, OUTPUT_DIR, RANDOM_SEED, ensure_directories
from scripts.common import load_dataset, split_frame


N_ESTIMATORS_CANDIDATES = [100, 200, 300, 500]


def main() -> None:
    ensure_directories()
    df = load_dataset()
    train_df, val_df, test_df, splits = split_frame(df)
    X_train_raw = train_df[FEATURE_COLUMNS]
    y_train_raw = train_df["risk_label"]

    pos = int(y_train_raw.sum())
    neg = int((y_train_raw == 0).sum())
    use_smote = neg / max(pos, 1) > 10 and pos >= 2
    if use_smote:
        print(f"Applying SMOTE: {pos} positive vs {neg} negative samples")
        k = min(5, pos - 1)
        smote_step = SMOTE(
            random_state=RANDOM_SEED,
            k_neighbors=k,
            sampling_strategy=0.2,
        )
        expected_positive = int(neg * 0.2)
        print(
            f"SMOTE runs inside each CV training fold; the full training-set target "
            f"would be approximately {expected_positive} positive vs {neg} negative samples."
        )
    else:
        smote_step = "passthrough"
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("smote", smote_step),
        ("rf", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED, n_jobs=1)),
    ])
    search = RandomizedSearchCV(
        pipe,
        {
            "rf__n_estimators": N_ESTIMATORS_CANDIDATES,
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
    search.fit(X_train_raw, y_train_raw)

    randomized_best = search.best_params_
    controlled_pipe = Pipeline([
        ("scale", StandardScaler()),
        ("smote", smote_step),
        ("rf", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
            max_depth=randomized_best["rf__max_depth"],
            min_samples_leaf=randomized_best["rf__min_samples_leaf"],
            max_features=randomized_best["rf__max_features"],
        )),
    ])
    estimator_search = GridSearchCV(
        controlled_pipe,
        {"rf__n_estimators": N_ESTIMATORS_CANDIDATES},
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
        return_train_score=True,
    )
    estimator_search.fit(X_train_raw, y_train_raw)
    final_model = estimator_search.best_estimator_

    comparison = pd.DataFrame(estimator_search.cv_results_)[[
        "param_rf__n_estimators",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
        "mean_train_score",
    ]].rename(columns={
        "param_rf__n_estimators": "n_estimators",
        "mean_test_score": "mean_cv_f1_macro",
        "std_test_score": "std_cv_f1_macro",
        "rank_test_score": "rank",
        "mean_train_score": "mean_train_f1_macro",
    }).sort_values("n_estimators")
    comparison.to_csv(OUTPUT_DIR / "rf_n_estimators_comparison.csv", index=False)

    val_proba = final_model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    test_proba = final_model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    pred = (test_proba >= 0.5).astype(int)
    metrics = {
        "validation_auc_roc": float(roc_auc_score(val_df["risk_label"], val_proba)),
        "test_auc_roc": float(roc_auc_score(test_df["risk_label"], test_proba)),
        "test_f1_macro": float(f1_score(test_df["risk_label"], pred, average="macro")),
        "test_average_precision": float(average_precision_score(test_df["risk_label"], test_proba)),
        "randomized_search_best_params": randomized_best,
        "controlled_n_estimators_candidates": N_ESTIMATORS_CANDIDATES,
        "controlled_best_n_estimators": int(estimator_search.best_params_["rf__n_estimators"]),
        "best_params": {
            "rf__n_estimators": int(estimator_search.best_params_["rf__n_estimators"]),
            "rf__max_depth": randomized_best["rf__max_depth"],
            "rf__min_samples_leaf": randomized_best["rf__min_samples_leaf"],
            "rf__max_features": randomized_best["rf__max_features"],
        },
        "split_sizes": {name: len(idx) for name, idx in splits.items()},
    }
    joblib.dump(
        {"model": final_model, "metrics": metrics, "features": FEATURE_COLUMNS},
        MODEL_DIR / "model_rf.pkl",
    )
    print(metrics)


if __name__ == "__main__":
    main()
