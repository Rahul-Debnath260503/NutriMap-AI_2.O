"""
Run Phase 3 Ordinary Kriging spatial-CV baselines.

Ordinary Kriging is a geostatistical baseline and therefore uses coordinates
to model spatial autocorrelation. This is separate from ML/DL modeling, where
coordinates remain excluded from predictive feature columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.modeling_config import TARGET_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "results" / "OK" / "readiness" / "kerala_soil_environmental_projected.csv"
RESULTS_DIR = REPO_ROOT / "results" / "OK" / "baseline"
VARIOGRAM_MODELS = ["spherical", "exponential", "gaussian", "linear"]


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


def sample_training_rows(train_df: pd.DataFrame, max_train: int, random_state: int) -> pd.DataFrame:
    if len(train_df) <= max_train:
        return train_df
    return train_df.sample(n=max_train, random_state=random_state)


def krige_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    variogram_model: str,
    n_closest_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    ok = OrdinaryKriging(
        train_df["x_utm43n_m"].to_numpy(),
        train_df["y_utm43n_m"].to_numpy(),
        train_df[target].to_numpy(),
        variogram_model=variogram_model,
        coordinates_type="euclidean",
        verbose=False,
        enable_plotting=False,
    )
    pred, variance = ok.execute(
        "points",
        test_df["x_utm43n_m"].to_numpy(),
        test_df["y_utm43n_m"].to_numpy(),
        backend="loop",
        n_closest_points=n_closest_points,
    )
    return np.asarray(pred, dtype=float), np.asarray(variance, dtype=float)


def run_target(
    df: pd.DataFrame,
    target: str,
    variogram_models: list[str],
    max_train: int,
    n_closest_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df = df.dropna(subset=[target, "x_utm43n_m", "y_utm43n_m", "Block"]).copy()
    groups = target_df["Block"].astype(str)
    splitter = GroupKFold(n_splits=min(5, groups.nunique()))
    metric_rows = []
    prediction_rows = []

    for variogram_model in variogram_models:
        for fold_id, (train_idx, test_idx) in enumerate(
            splitter.split(target_df, target_df[target], groups), start=1
        ):
            train_df = target_df.iloc[train_idx]
            test_df = target_df.iloc[test_idx]
            sampled_train = sample_training_rows(
                train_df, max_train=max_train, random_state=42 + fold_id
            )

            try:
                predictions, variance = krige_predictions(
                    sampled_train,
                    test_df,
                    target,
                    variogram_model,
                    n_closest_points=n_closest_points,
                )
                metrics = regression_metrics(test_df[target].to_numpy(), predictions)
                status = "success"
                error = ""
            except Exception as exc:  # PyKrige can fail for unstable variograms.
                predictions = np.full(len(test_df), np.nan)
                variance = np.full(len(test_df), np.nan)
                metrics = {
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "mape": np.nan,
                    "bias": np.nan,
                    "explained_variance": np.nan,
                }
                status = "failed"
                error = str(exc)

            metric_rows.append(
                {
                    "model": "OrdinaryKriging",
                    "target": target,
                    "variogram_model": variogram_model,
                    "fold": fold_id,
                    "test_blocks": ",".join(sorted(test_df["Block"].astype(str).unique())),
                    "n_train_available": len(train_df),
                    "n_train_used": len(sampled_train),
                    "n_test": len(test_df),
                    "n_closest_points": n_closest_points,
                    "status": status,
                    "error": error,
                    **metrics,
                }
            )

            fold_predictions = test_df[
                ["District", "Block", "Latitude", "Longitude", "x_utm43n_m", "y_utm43n_m", target]
            ].copy()
            fold_predictions["model"] = "OrdinaryKriging"
            fold_predictions["target"] = target
            fold_predictions["variogram_model"] = variogram_model
            fold_predictions["fold"] = fold_id
            fold_predictions["prediction"] = predictions
            fold_predictions["kriging_variance"] = variance
            prediction_rows.append(fold_predictions)

    return pd.DataFrame(metric_rows), pd.concat(prediction_rows, ignore_index=True)


def write_report(metrics: pd.DataFrame, targets: list[str]) -> None:
    successful = metrics[metrics["status"] == "success"].copy()
    if successful.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            successful.groupby(["target", "variogram_model"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                mape=("mape", "mean"),
                bias=("bias", "mean"),
                explained_variance=("explained_variance", "mean"),
            )
            .sort_values(["target", "rmse"])
        )
    summary.to_csv(RESULTS_DIR / "ordinary_kriging_metric_summary.csv", index=False)

    best_rows = []
    if not summary.empty:
        for target, target_summary in summary.groupby("target"):
            best_rows.append(target_summary.sort_values("rmse").iloc[0])
    best = pd.DataFrame(best_rows)
    best.to_csv(RESULTS_DIR / "ordinary_kriging_best_variogram_by_target.csv", index=False)

    lines = [
        "# Phase 3 Ordinary Kriging Baseline Report",
        "",
        "## Modeling Policy",
        "",
        "- Coordinates are used here only for the geostatistical OK baseline.",
        "- Coordinates remain excluded from ML/DL predictive input features.",
        "",
        "## Run Configuration",
        "",
        f"- Targets: {', '.join(targets)}",
        "- Validation: grouped spatial CV by `Block`.",
        "",
        "## Best Variogram By Target",
        "",
    ]
    if best.empty:
        lines.append("- No successful OK fits.")
    else:
        for row in best.itertuples(index=False):
            lines.append(
                f"- {row.target}: {row.variogram_model}, "
                f"RMSE={row.rmse:.3f}, MAE={row.mae:.3f}, R2={row.r2:.3f}"
            )

    failures = metrics[metrics["status"] != "success"]
    if not failures.empty:
        lines.extend(["", "## Failures", ""])
        for row in failures.itertuples(index=False):
            lines.append(
                f"- {row.target} {row.variogram_model} fold {row.fold}: {row.error}"
            )

    (RESULTS_DIR / "phase3_ordinary_kriging_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        nargs="+",
        default=TARGET_COLUMNS,
        help="Target columns to krige. Default covers all soil response variables.",
    )
    parser.add_argument(
        "--variogram-models",
        nargs="+",
        default=VARIOGRAM_MODELS,
        choices=VARIOGRAM_MODELS,
    )
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--n-closest-points", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Projected input not found: {INPUT_FILE}. Run phase3_kriging_readiness.py first."
        )

    invalid_targets = sorted(set(args.targets) - set(TARGET_COLUMNS))
    if invalid_targets:
        raise ValueError(f"Unknown target columns: {invalid_targets}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)

    metric_tables = []
    prediction_tables = []
    for target in args.targets:
        print(f"Running OK for {target}...")
        target_metrics, target_predictions = run_target(
            df=df,
            target=target,
            variogram_models=args.variogram_models,
            max_train=args.max_train,
            n_closest_points=args.n_closest_points,
        )
        metric_tables.append(target_metrics)
        prediction_tables.append(target_predictions)

    metrics = pd.concat(metric_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    metrics.to_csv(RESULTS_DIR / "ordinary_kriging_spatial_cv_metrics.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "ordinary_kriging_spatial_cv_predictions.csv", index=False)
    write_report(metrics, args.targets)

    print(f"Ordinary Kriging outputs written to: {RESULTS_DIR}")
    print(
        metrics[metrics["status"] == "success"]
        .groupby(["target", "variogram_model"])["rmse"]
        .mean()
        .sort_values()
        .to_string()
    )


if __name__ == "__main__":
    main()
