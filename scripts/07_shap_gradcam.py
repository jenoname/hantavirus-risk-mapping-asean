from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill

from config import FIGURE_DIR, MODEL_DIR, ensure_directories
from scripts.common import load_dataset, split_frame
from scripts.patch_model import make_raster_patches


def main() -> None:
    ensure_directories()
    df = load_dataset()
    _, _, test_df, _ = split_frame(df)
    rf = joblib.load(MODEL_DIR / "model_rf.pkl")
    model = rf["model"].named_steps["rf"]
    try:
        import shap

        sample = test_df[rf["features"]].sample(min(500, len(test_df)), random_state=42)
        sample_for_shap = rf["model"].named_steps["scale"].transform(sample)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample_for_shap)
        if isinstance(values, list):
            values = values[1]
        elif isinstance(values, np.ndarray) and values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        importances = dict(sorted(zip(rf["features"], mean_abs), key=lambda item: item[1], reverse=True))
        shap_label = "SHAP mean absolute value"

        plt.figure(figsize=(9, 8))
        shap.summary_plot(
            values,
            sample,
            feature_names=rf["features"],
            max_display=min(15, len(rf["features"])),
            show=False,
        )
        plt.gcf().subplots_adjust(left=0.30, right=0.95, top=0.96, bottom=0.10)
        plt.savefig(FIGURE_DIR / "shap_beeswarm_plot.png", dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        importances = dict(sorted(zip(rf["features"], model.feature_importances_), key=lambda item: item[1], reverse=True))
        shap_label = f"Fallback RF feature importance; SHAP unavailable: {exc}"
    (FIGURE_DIR / "shap_feature_importance.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")

    top = list(importances.items())[:10]
    plt.figure(figsize=(8.5, 5.5))
    plt.barh([x[0] for x in top[::-1]], [x[1] for x in top[::-1]], color="#2f7f6f")
    plt.title(fill(shap_label, width=58), pad=12)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close()

    cnn = joblib.load(MODEL_DIR / "model_cnn.pkl")
    patches = make_raster_patches(test_df.iloc[:1])
    if cnn.get("backend") == "tensorflow_resnet50":
        import tensorflow as tf

        tf_model = tf.keras.models.load_model(cnn["model_path"])
        def _is_conv_layer(layer):
            try:
                shape = layer.output_shape
            except AttributeError:
                shape = getattr(getattr(layer, "output", None), "shape", None)
            try:
                return len(shape) == 4
            except AttributeError:
                return False

        base_model = next(
            layer for layer in tf_model.layers
            if getattr(layer, "name", "") == "resnet50" or hasattr(layer, "layers")
        )
        last_conv_layer = next(
            layer for layer in reversed(base_model.layers)
            if _is_conv_layer(layer)
        )
        conv_model = tf.keras.models.Model(base_model.inputs, [last_conv_layer.output, base_model.output])
        with tf.GradientTape() as tape:
            x = patches
            seen_base = False
            for layer in tf_model.layers[1:]:
                if layer is base_model:
                    conv_outputs, x = conv_model(x)
                    seen_base = True
                    continue
                if seen_base:
                    x = layer(x, training=False)
                elif layer is not base_model:
                    x = layer(x, training=False)
            predictions = x
            loss = predictions[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heat = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1).numpy()
        heat = np.maximum(heat, 0)
        heat = heat / max(float(heat.max()), 1e-6)
        title = "Grad-CAM: Model B ResNet-50"
        out_name = "gradcam_resnet50_patch.png"
    else:
        patch = patches[0]
        heat = np.abs(patch[..., 0] - patch[..., 0].mean())
        heat = heat / max(float(heat.max()), 1e-6)
        title = "Patch sensitivity map\n(not Grad-CAM; non-ResNet fallback)"
        out_name = "patch_sensitivity_not_gradcam.png"
    plt.figure(figsize=(5.2, 5.2))
    plt.imshow(heat, cmap="inferno")
    plt.axis("off")
    plt.title(title, pad=12)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / out_name, dpi=180, bbox_inches="tight")
    plt.close()

    patch = patches[0]
    sensitivity = np.abs(patch[..., 0] - patch[..., 0].mean())
    sensitivity = sensitivity / max(float(sensitivity.max()), 1e-6)
    plt.figure(figsize=(5.2, 5.2))
    plt.imshow(sensitivity, cmap="inferno")
    plt.axis("off")
    plt.title("Patch sensitivity map\n(non-Grad-CAM diagnostic)", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "patch_sensitivity_not_gradcam.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote explainability artifacts to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
