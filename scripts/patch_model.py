from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PATCH_CHANNELS = ["modis_lst_day_c", "frac_tree", "chirps_precip_mm"]
PATCH_SIZE = 32


def make_raster_patches(df: pd.DataFrame, size: int = PATCH_SIZE) -> np.ndarray:
    """Build deterministic 32x32x3 patches from per-cell raster feature values."""
    rows = []
    rng = np.random.default_rng(42)
    for row in df.itertuples(index=False):
        lst = float(getattr(row, "modis_lst_day_c", 28.0))
        land = (
            0.60 * float(getattr(row, "frac_tree", 0.3))
            + 0.35 * float(getattr(row, "frac_cropland", 0.2))
            + 0.20 * float(getattr(row, "frac_built", 0.05))
            + 0.15 * float(getattr(row, "frac_wetland", 0.03))
        )
        rain = float(getattr(row, "chirps_precip_mm", 150.0))

        ch0 = np.full((size, size), lst, dtype=np.float32)
        ch0 += rng.normal(0, max(abs(lst) * 0.02, 0.1), (size, size)).astype(np.float32)
        ch1 = np.full((size, size), land, dtype=np.float32)
        ch1 += rng.normal(0, max(abs(land) * 0.02, 0.01), (size, size)).astype(np.float32)
        ch2 = np.full((size, size), rain, dtype=np.float32)
        ch2 += rng.normal(0, max(abs(rain) * 0.02, 1.0), (size, size)).astype(np.float32)

        patch = np.stack([ch0, ch1, ch2], axis=-1)
        rows.append(patch)
    patches = np.asarray(rows, dtype=np.float32)
    mins = patches.min(axis=(1, 2), keepdims=True)
    maxs = patches.max(axis=(1, 2), keepdims=True)
    return (patches - mins) / np.maximum(maxs - mins, 1e-6)


def flatten_patch_features(df: pd.DataFrame) -> pd.DataFrame:
    patches = make_raster_patches(df)
    summary = np.column_stack(
        [
            patches.mean(axis=(1, 2)),
            patches.std(axis=(1, 2)),
            patches[:, 8:24, 8:24, :].mean(axis=(1, 2)),
        ]
    )
    columns = [f"patch_{stat}_{channel}" for stat in ["mean", "std", "center"] for channel in PATCH_CHANNELS]
    return pd.DataFrame(summary, columns=columns, index=df.index)


def build_patch_baseline(random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(48, 16),
                    activation="relu",
                    alpha=0.001,
                    max_iter=400,
                    random_state=random_seed,
                    early_stopping=True,
                ),
            ),
        ]
    )


def predict_patch_model(payload: dict, df: pd.DataFrame) -> np.ndarray:
    if payload.get("backend") == "tensorflow_resnet50":
        import tensorflow as tf

        model = tf.keras.models.load_model(payload["model_path"])
        return model.predict(make_raster_patches(df), verbose=0).ravel()
    features = flatten_patch_features(df)
    return payload["model"].predict_proba(features[payload["features"]])[:, 1]
