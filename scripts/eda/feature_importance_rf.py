"""
Generate Phase 2 feature-importance outputs.

The current environment has scikit-learn available, but not XGBoost or SHAP.
This script therefore runs a reproducible Random Forest baseline for all
available soil targets and records dependency availability for later steps.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.modeling_config import ENVIRONMENTAL_FEATURE_COLUMNS, TARGET_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "processed" / "Kerala_Soil_Environmental_Features.csv"
RESULTS_DIR = REPO_ROOT / "results" / "EDA" / "feature_importance"

FEATURE_COLUMNS = ENVIRONMENTAL_FEATURE_COLUMNS


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": mape(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
    }


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def cross_validate_target(
    df: pd.DataFrame, target: str, feature_columns: list[str]
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    X = df[feature_columns]
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    groups = df.loc[valid, "Block"].astype(str)

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)

    metric_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[test_idx])
        metrics = regression_metrics(y.iloc[test_idx].to_numpy(), predictions)
        test_blocks = ",".join(sorted(groups.iloc[test_idx].unique()))
        metric_rows.append(
            {
                "model": "RandomForest",
                "target": target,
                "fold": fold_id,
                "test_blocks": test_blocks,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **metrics,
            }
        )

    final_model = make_model()
    final_model.fit(X, y)
    fitted_rf = final_model.named_steps["model"]
    importance = pd.DataFrame(
        {
            "model": "RandomForest",
            "target": target,
            "feature": feature_columns,
            "importance": fitted_rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return metric_rows, importance


def write_dependency_status() -> None:
    status = pd.DataFrame(
        [
            {
                "package": "scikit-learn",
                "available": find_spec("sklearn") is not None,
                "used_in_this_run": True,
                "note": "Used for Random Forest feature importance and spatial group CV.",
            },
            {
                "package": "xgboost",
                "available": find_spec("xgboost") is not None,
                "used_in_this_run": False,
                "note": "Required for the planned XGBoost feature-importance step.",
            },
            {
                "package": "shap",
                "available": find_spec("shap") is not None,
                "used_in_this_run": False,
                "note": "Required for the planned SHAP explanation step.",
            },
        ]
    )
    status.to_csv(RESULTS_DIR / "dependency_status.csv", index=False)


def save_importance_plots(importances: pd.DataFrame) -> None:
    aggregate = (
        importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(aggregate["feature"], aggregate["importance"], color="#356859")
    ax.set_title("Mean Random Forest Feature Importance Across Targets")
    ax.set_xlabel("Mean impurity-based importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "rf_mean_feature_importance.png", dpi=200)
    plt.close(fig)

    top = (
        importances.sort_values(["target", "importance"], ascending=[True, False])
        .groupby("target")
        .head(5)
    )
    top.to_csv(RESULTS_DIR / "rf_top5_features_by_target.csv", index=False)


def write_interpretation_report(
    metrics: pd.DataFrame, importances: pd.DataFrame, correlations: pd.DataFrame
) -> None:
    metric_summary = (
        metrics.groupby("target", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
            mape=("mape", "mean"),
            bias=("bias", "mean"),
            explained_variance=("explained_variance", "mean"),
        )
        .sort_values("r2", ascending=False)
    )
    metric_summary.to_csv(RESULTS_DIR / "rf_spatial_cv_metric_summary.csv", index=False)

    aggregate_importance = (
        importances.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    aggregate_importance.to_csv(RESULTS_DIR / "rf_mean_feature_importance.csv", index=False)

    strongest_correlations = []
    for target in TARGET_COLUMNS:
        for feature in FEATURE_COLUMNS:
            strongest_correlations.append(
                {
                    "target": target,
                    "feature": feature,
                    "spearman_correlation": correlations.loc[target, feature],
                    "abs_correlation": abs(correlations.loc[target, feature]),
                }
            )
    corr_summary = (
        pd.DataFrame(strongest_correlations)
        .sort_values("abs_correlation", ascending=False)
        .head(20)
    )
    corr_summary.to_csv(RESULTS_DIR / "top_environment_correlations.csv", index=False)

    lines = [
        "# Phase 2 Feature Importance Report",
        "",
        "## Dataset Decision",
        "",
        "- Proceeding with `Kerala_Soil_Environmental_Features.csv` using 4,908 samples.",
        "- The 4-row GEE extraction difference is accepted for current EDA and baseline modeling.",
        "",
        "## Current Run",
        "",
        "- Random Forest feature importance completed using environmental covariates only.",
        "- Coordinates are not used as ML input features; they remain metadata for mapping and spatial validation.",
        "- Spatial validation uses grouped folds by `Block`; with four blocks available, this run uses 4 folds.",
        "- XGBoost and SHAP were not run because those packages are not installed in the active Python environment.",
        "",
        "## Strongest Environmental Signals From Spearman Correlation",
        "",
    ]
    for row in corr_summary.head(10).itertuples(index=False):
        lines.append(
            f"- {row.target}: {row.feature} "
            f"({row.spearman_correlation:.3f})"
        )

    lines.extend(
        [
            "",
            "## Mean Random Forest Feature Importance",
            "",
        ]
    )
    for row in aggregate_importance.itertuples(index=False):
        lines.append(f"- {row.feature}: {row.importance:.4f}")

    lines.extend(
        [
            "",
            "## Spatial CV Metric Snapshot",
            "",
        ]
    )
    for row in metric_summary.itertuples(index=False):
        lines.append(
            f"- {row.target}: R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f}"
        )

    (RESULTS_DIR / "phase2_feature_importance_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    feature_columns = [column for column in FEATURE_COLUMNS if column in df.columns]
    target_columns = [column for column in TARGET_COLUMNS if column in df.columns]
    missing_features = sorted(set(FEATURE_COLUMNS) - set(feature_columns))
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    metrics = []
    importances = []
    for target in target_columns:
        target_metrics, target_importances = cross_validate_target(
            df, target, feature_columns
        )
        metrics.extend(target_metrics)
        importances.append(target_importances)

    metrics_df = pd.DataFrame(metrics)
    importances_df = pd.concat(importances, ignore_index=True)

    metrics_df.to_csv(RESULTS_DIR / "rf_spatial_cv_metrics.csv", index=False)
    importances_df.to_csv(RESULTS_DIR / "rf_feature_importance_long.csv", index=False)
    write_dependency_status()
    save_importance_plots(importances_df)

    spearman_path = REPO_ROOT / "results" / "EDA" / "spearman_correlation_matrix.csv"
    correlations = pd.read_csv(spearman_path, index_col=0)
    write_interpretation_report(metrics_df, importances_df, correlations)

    print(f"Feature-importance outputs written to: {RESULTS_DIR}")
    print(metrics_df.groupby("target")["r2"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
