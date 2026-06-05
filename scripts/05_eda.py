"""Exploratory data analysis for grid labels and environmental features."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURE_COLUMNS, FIGURE_DIR, ensure_directories
from scripts.common import load_dataset
from scripts.patch_model import make_raster_patches


def main() -> None:
    ensure_directories()
    df = load_dataset()
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5.5))
    sns.scatterplot(data=df, x="lon", y="lat", hue="risk_label", size="presence_count", palette=["#2c7bb6", "#d7191c"], alpha=0.75)
    plt.title("GBIF Presence and Risk Label Grid")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_spatial_distribution.png", dpi=180, bbox_inches="tight")
    plt.close()

    corr = df[FEATURE_COLUMNS + ["risk_label"]].corr(method="spearman")
    plt.figure(figsize=(13, 10))
    sns.heatmap(corr, cmap="vlag", center=0, square=False)
    plt.title("Spearman Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_spearman_correlation.png", dpi=180, bbox_inches="tight")
    plt.close()

    selected = ["chirps_precip_mm", "modis_lst_day_c", "frac_tree", "frac_cropland"]
    melted = df.melt(id_vars="risk_label", value_vars=selected, var_name="feature", value_name="value")
    plt.figure(figsize=(11, 6.5))
    sns.boxplot(data=melted, x="feature", y="value", hue="risk_label")
    plt.xticks(rotation=20, ha="right")
    plt.title("Feature Distribution by Risk Class")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_boxplots_by_class.png", dpi=180, bbox_inches="tight")
    plt.close()

    distribution_features = [
        "chirps_precip_mm",
        "modis_lst_day_c",
        "srtm_elevation_m",
        "frac_tree",
        "frac_cropland",
        "frac_built",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, feature in zip(axes.ravel(), distribution_features):
        sns.histplot(
            data=df,
            x=feature,
            hue="risk_label",
            stat="density",
            common_norm=False,
            bins=30,
            element="step",
            fill=False,
            palette=["#2c7bb6", "#d7191c"],
            ax=ax,
        )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
    fig.suptitle("Normalized Feature Histograms by Risk Class", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURE_DIR / "eda_normalized_histograms.png", dpi=180, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, feature in zip(axes.ravel(), distribution_features):
        for label, color in [(0, "#2c7bb6"), (1, "#d7191c")]:
            subset = df.loc[df["risk_label"] == label, feature].dropna()
            if len(subset) >= 5 and subset.nunique() > 1:
                sns.kdeplot(
                    x=subset,
                    ax=ax,
                    color=color,
                    label=f"class {label}",
                    fill=False,
                    linewidth=1.6,
                    warn_singular=False,
                )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle("Kernel Density Estimates by Risk Class", y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURE_DIR / "eda_kde_by_class.png", dpi=180, bbox_inches="tight")
    plt.close()

    df["risk_label"].value_counts(normalize=True).sort_index().to_json(FIGURE_DIR / "eda_class_balance.json", indent=2)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    counts = df["risk_label"].value_counts().sort_index()
    bars = ax.bar(
        ["Low risk (0)", "High risk (1)"],
        counts.values,
        color=["#2c7bb6", "#d7191c"],
        edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{val:,}\n({val/len(df)*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Class Imbalance: Risk Label Distribution")
    ax.set_ylabel("Number of grid cells")
    ax.set_ylim(0, max(counts) * 1.22)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_class_imbalance.png", dpi=180, bbox_inches="tight")
    plt.close()

    patches = make_raster_patches(df)
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.8))
    for row_i, label in enumerate([0, 1]):
        candidates = df.index[df["risk_label"] == label].tolist()[:3]
        for col_i, idx in enumerate(candidates):
            axes[row_i, col_i].imshow(patches[idx])
            axes[row_i, col_i].set_title(f"class {label}", pad=8)
            axes[row_i, col_i].axis("off")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "eda_sample_32x32_raster_patches.png", dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Wrote EDA outputs to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
