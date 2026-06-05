"""Train Model B on 32x32x3 raster patches.

TensorFlow environments use a ResNet-50 transfer-learning model. Lightweight
classroom environments without TensorFlow use an explicit MLP patch baseline
and report it as non-ResNet, so the output is never disguised as Grad-CAM/CNN.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from config import FEATURE_COLUMNS, MODEL_DIR, RANDOM_SEED, ensure_directories
from scripts.common import load_dataset, split_frame
from scripts.patch_model import build_patch_baseline, flatten_patch_features, make_raster_patches


def _metric_block(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y, proba)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "average_precision": float(average_precision_score(y, proba)),
    }


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
        X_train_smote, y_train_smote = smote.fit_resample(X_train_raw, y_train_raw)
        print(f"After SMOTE: {int(y_train_smote.sum())} positive vs {int((y_train_smote==0).sum())} negative")
        train_df_smote = pd.DataFrame(X_train_smote, columns=FEATURE_COLUMNS)
        train_df_smote["risk_label"] = y_train_smote.values
    else:
        train_df_smote = train_df

    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models

        x_train = make_raster_patches(train_df_smote)
        x_val = make_raster_patches(val_df)
        x_test = make_raster_patches(test_df)
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=(32, 32, 3),
            pooling="avg",
        )
        base.trainable = True
        inputs = layers.Input(shape=(32, 32, 3))
        x = layers.RandomFlip("horizontal_and_vertical")(inputs)
        x = layers.RandomRotation(0.15)(x)
        x = layers.GaussianNoise(0.03)(x)
        x = base(x, training=True)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0),
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )
        model.fit(
            x_train,
            train_df_smote["risk_label"].to_numpy(),
            validation_data=(x_val, val_df["risk_label"].to_numpy()),
            epochs=8,
            batch_size=64,
            verbose=2,
        )
        model.save(MODEL_DIR / "model_cnn_resnet50.keras")
        val_proba = model.predict(x_val, verbose=0).ravel()
        test_proba = model.predict(x_test, verbose=0).ravel()
        payload = {
            "backend": "tensorflow_resnet50",
            "model_path": str(MODEL_DIR / "model_cnn_resnet50.keras"),
            "features": "32x32x3 raster patches: MODIS LST, land-cover index, CHIRPS precipitation",
            "split_sizes": {name: len(idx) for name, idx in splits.items()},
        }
    except Exception as exc:
        x_train = flatten_patch_features(train_df_smote)
        x_val = flatten_patch_features(val_df)
        x_test = flatten_patch_features(test_df)
        model = build_patch_baseline(RANDOM_SEED)
        model.fit(x_train, train_df_smote["risk_label"])
        val_proba = model.predict_proba(x_val)[:, 1]
        test_proba = model.predict_proba(x_test)[:, 1]
        payload = {
            "backend": "sklearn_patch_mlp_fallback",
            "model": model,
            "features": x_train.columns.tolist(),
            "split_sizes": {name: len(idx) for name, idx in splits.items()},
            "note": (
                "TensorFlow ResNet-50 could not run in this environment, so this is an explicit "
                f"non-ResNet patch baseline. Original TensorFlow error: {exc}"
            ),
        }

    metrics = {
        "validation": _metric_block(val_df["risk_label"], val_proba),
        "test": _metric_block(test_df["risk_label"], test_proba),
    }
    payload["metrics"] = metrics
    joblib.dump(payload, MODEL_DIR / "model_cnn.pkl")
    print(metrics)


if __name__ == "__main__":
    main()
