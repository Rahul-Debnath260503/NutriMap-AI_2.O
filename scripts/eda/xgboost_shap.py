"""
Run Phase 2 XGBoost feature importance and SHAP summaries.

Coordinates are intentionally excluded from model input features. `Block` is
used only for grouped spatial cross-validation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.modeling_config import ENVIRONMENTAL_FEATURE_COLUMNS, TARGET_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "processed" / "Kerala_Soil_Environmental_Features.csv"
RESULTS_DIR = REPO_ROOT / "results" / "EDA" / "xgboost_shap"


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


def make_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="rmse",
    )


def run_target(
    df: pd.DataFrame, target: str, feature_columns: list[str]
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    X = df[feature_columns].copy()
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    groups = df.loc[valid, "Block"].astype(str)

    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    metric_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[test_idx])
        metric_rows.append(
            {
                "model": "XGBoost",
                "target": target,
                "fold": fold_id,
                "test_blocks": ",".join(sorted(groups.iloc[test_idx].unique())),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                **regression_metrics(y.iloc[test_idx].to_numpy(), predictions),
            }
        )

    final_model = make_model()
    final_model.fit(X, y)

    importance = pd.DataFrame(
        {
            "model": "XGBoost",
            "target": target,
            "feature": feature_columns,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    shap_sample = X.sample(n=min(500, len(X)), random_state=42)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(shap_sample)
    shap_importance = pd.DataFrame(
        {
            "model": "XGBoost",
            "target": target,
            "feature": feature_columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    return metric_rows, importance, shap_importance


def save_bar_plot(df: pd.DataFrame, value_column: str, output_name: str, title: str) -> None:
    aggregate = (
        df.groupby("feature", as_index=False)[value_column]
        .mean()
        .sort_values(value_column, ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(aggregate["feature"], aggregate[value_column], color="#395b8f")
    ax.set_title(title)
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / output_name, dpi=200)
    plt.close(fig)


def write_report(metrics: pd.DataFrame, importance: pd.DataFrame, shap_importance: pd.DataFrame) -> None:
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
    metric_summary.to_csv(RESULTS_DIR / "xgb_spatial_cv_metric_summary.csv", index=False)

    mean_importance = (
        importance.groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    mean_importance.to_csv(RESULTS_DIR / "xgb_mean_feature_importance.csv", index=False)

    mean_shap = (
        shap_importance.groupby("feature", as_index=False)["mean_abs_shap"]
        .mean()
        .sort_values("mean_abs_shap", ascending=False)
    )
    mean_shap.to_csv(RESULTS_DIR / "xgb_mean_abs_shap_importance.csv", index=False)

    lines = [
        "# Phase 2 XGBoost And SHAP Report",
        "",
        "## Modeling Policy",
        "",
        "- Input features are environmental covariates only.",
        "- Coordinates are not used as ML features.",
        "- `Block` is used only for grouped spatial cross-validation.",
        "",
        "## Mean XGBoost Feature Importance",
        "",
    ]
    for row in mean_importance.itertuples(index=False):
        lines.append(f"- {row.feature}: {row.importance:.4f}")

    lines.extend(["", "## Mean Absolute SHAP Importance", ""])
    for row in mean_shap.itertuples(index=False):
        lines.append(f"- {row.feature}: {row.mean_abs_shap:.4f}")

    lines.extend(["", "## Spatial CV Metric Snapshot", ""])
    for row in metric_summary.itertuples(index=False):
        lines.append(
            f"- {row.target}: R2={row.r2:.3f}, RMSE={row.rmse:.3f}, MAE={row.mae:.3f}"
        )

    (RESULTS_DIR / "phase2_xgb_shap_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    feature_columns = [column for column in ENVIRONMENTAL_FEATURE_COLUMNS if column in df.columns]
    target_columns = [column for column in TARGET_COLUMNS if column in df.columns]

    metrics = []
    importance_tables = []
    shap_tables = []
    for target in target_columns:
        target_metrics, target_importance, target_shap = run_target(
            df, target, feature_columns
        )
        metrics.extend(target_metrics)
        importance_tables.append(target_importance)
        shap_tables.append(target_shap)

    metrics_df = pd.DataFrame(metrics)
    importance_df = pd.concat(importance_tables, ignore_index=True)
    shap_df = pd.concat(shap_tables, ignore_index=True)

    metrics_df.to_csv(RESULTS_DIR / "xgb_spatial_cv_metrics.csv", index=False)
    importance_df.to_csv(RESULTS_DIR / "xgb_feature_importance_long.csv", index=False)
    shap_df.to_csv(RESULTS_DIR / "xgb_shap_importance_long.csv", index=False)

    save_bar_plot(
        importance_df,
        "importance",
        "xgb_mean_feature_importance.png",
        "Mean XGBoost Feature Importance Across Targets",
    )
    save_bar_plot(
        shap_df,
        "mean_abs_shap",
        "xgb_mean_abs_shap_importance.png",
        "Mean Absolute SHAP Importance Across Targets",
    )
    write_report(metrics_df, importance_df, shap_df)

    print(f"XGBoost/SHAP outputs written to: {RESULTS_DIR}")
    print(metrics_df.groupby("target")["r2"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
