"""
Generate initial Phase 2 EDA outputs from the GEE-enriched dataset.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "processed" / "Kerala_Soil_Environmental_Features.csv"
RESULTS_DIR = REPO_ROOT / "results" / "EDA"

NUTRIENT_COLUMNS = [
    "pH",
    "EC",
    "Organic_Carbon",
    "P",
    "K",
    "Ca",
    "Mg",
    "S",
    "Fe",
    "Mn",
    "Zn",
    "Cu",
    "B",
]

ENV_COLUMNS = [
    "Elevation",
    "Slope",
    "Aspect",
    "NDVI",
    "NDWI",
    "Rainfall",
    "Temperature_C",
    "Soil_Moisture",
    "LULC",
]


def save_correlation_matrix(df: pd.DataFrame, method: str) -> None:
    columns = [column for column in NUTRIENT_COLUMNS + ENV_COLUMNS if column in df.columns]
    corr = df[columns].corr(method=method)
    corr.to_csv(RESULTS_DIR / f"{method}_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(14, 11))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.set_title(f"{method.title()} Correlation Matrix")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"{method}_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def save_distribution_plots(df: pd.DataFrame) -> None:
    columns = [column for column in NUTRIENT_COLUMNS if column in df.columns]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for axis, column in zip(axes, columns):
        axis.hist(df[column].dropna(), bins=30, color="#2f6f73", edgecolor="white")
        axis.set_title(column)
        axis.set_xlabel("")
        axis.set_ylabel("Count")

    for axis in axes[len(columns):]:
        axis.axis("off")

    fig.suptitle("Nutrient Distributions", fontsize=16)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "nutrient_distributions.png", dpi=200)
    plt.close(fig)


def save_boxplots(df: pd.DataFrame) -> None:
    columns = [column for column in NUTRIENT_COLUMNS if column in df.columns]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for axis, column in zip(axes, columns):
        axis.boxplot(df[column].dropna(), vert=True, patch_artist=True)
        axis.set_title(column)
        axis.set_xticks([])

    for axis in axes[len(columns):]:
        axis.axis("off")

    fig.suptitle("Nutrient Boxplots", fontsize=16)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "nutrient_boxplots.png", dpi=200)
    plt.close(fig)


def save_sample_location_map(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 10))
    for block, block_df in df.groupby("Block"):
        ax.scatter(
            block_df["Longitude"],
            block_df["Latitude"],
            s=8,
            alpha=0.65,
            label=block,
        )

    ax.set_title("Soil Sample Locations")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_location_map.png", dpi=200)
    plt.close(fig)


def save_feature_target_scatter(df: pd.DataFrame) -> None:
    pairs = [
        ("NDVI", "Organic_Carbon"),
        ("Rainfall", "P"),
        ("Elevation", "K"),
        ("Soil_Moisture", "pH"),
        ("Temperature_C", "Ca"),
        ("Slope", "Mg"),
    ]
    pairs = [(x, y) for x, y in pairs if x in df.columns and y in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for axis, (x_col, y_col) in zip(axes, pairs):
        axis.scatter(df[x_col], df[y_col], s=8, alpha=0.45, color="#5b5f97")
        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        axis.set_title(f"{y_col} vs {x_col}")
        axis.grid(True, alpha=0.2)

    for axis in axes[len(pairs):]:
        axis.axis("off")

    fig.suptitle("Initial Environmental Feature vs Nutrient Relationships", fontsize=16)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_target_relationships.png", dpi=200)
    plt.close(fig)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    summary_columns = [
        column for column in NUTRIENT_COLUMNS + ENV_COLUMNS if column in df.columns
    ]
    df[summary_columns].describe().T.to_csv(RESULTS_DIR / "nutrient_environment_summary.csv")

    save_correlation_matrix(df, "pearson")
    save_correlation_matrix(df, "spearman")
    save_distribution_plots(df)
    save_boxplots(df)
    save_sample_location_map(df)
    save_feature_target_scatter(df)

    print(f"EDA outputs written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
