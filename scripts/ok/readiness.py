"""
Prepare Phase 3 Ordinary Kriging inputs and readiness reports.

True Ordinary Kriging will require a geostatistical library such as PyKrige or
GSTools. This script creates the projected coordinate dataset and spatial
diagnostics needed before fitting variograms and kriging models.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    raise ImportError("geopandas is required for coordinate projection.") from exc

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.modeling_config import TARGET_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "processed" / "Kerala_Soil_Environmental_Features.csv"
RESULTS_DIR = REPO_ROOT / "results" / "OK" / "readiness"
PROJECTED_CRS = "EPSG:32643"


def project_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["Longitude"], df["Latitude"]),
        crs="EPSG:4326",
    )
    projected = gdf.to_crs(PROJECTED_CRS)
    output = pd.DataFrame(df.copy())
    output["x_utm43n_m"] = projected.geometry.x
    output["y_utm43n_m"] = projected.geometry.y
    return output


def nearest_neighbor_summary(df: pd.DataFrame) -> pd.DataFrame:
    coords = df[["x_utm43n_m", "y_utm43n_m"]].to_numpy()
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    nn = distances[:, 1]
    return pd.DataFrame(
        [
            {
                "samples": len(df),
                "min_nearest_neighbor_m": float(np.min(nn)),
                "median_nearest_neighbor_m": float(np.median(nn)),
                "mean_nearest_neighbor_m": float(np.mean(nn)),
                "max_nearest_neighbor_m": float(np.max(nn)),
            }
        ]
    )


def extent_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "crs": PROJECTED_CRS,
                "x_min_m": df["x_utm43n_m"].min(),
                "x_max_m": df["x_utm43n_m"].max(),
                "y_min_m": df["y_utm43n_m"].min(),
                "y_max_m": df["y_utm43n_m"].max(),
                "width_m": df["x_utm43n_m"].max() - df["x_utm43n_m"].min(),
                "height_m": df["y_utm43n_m"].max() - df["y_utm43n_m"].min(),
            }
        ]
    )


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        series = pd.to_numeric(df[target], errors="coerce")
        rows.append(
            {
                "target": target,
                "samples": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
                "zero_count": int((series == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def dependency_status() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "package": "geopandas",
                "available": find_spec("geopandas") is not None,
                "purpose": "Coordinate projection to UTM.",
            },
            {
                "package": "scipy",
                "available": find_spec("scipy") is not None,
                "purpose": "Nearest-neighbor diagnostics.",
            },
            {
                "package": "pykrige",
                "available": find_spec("pykrige") is not None,
                "purpose": "Ordinary Kriging and variogram modeling.",
            },
            {
                "package": "gstools",
                "available": find_spec("gstools") is not None,
                "purpose": "Variogram modeling and kriging alternative.",
            },
        ]
    )


def write_target_variogram_inputs(df: pd.DataFrame) -> None:
    target_dir = RESULTS_DIR / "target_inputs"
    target_dir.mkdir(parents=True, exist_ok=True)
    for target in TARGET_COLUMNS:
        columns = ["x_utm43n_m", "y_utm43n_m", "Longitude", "Latitude", target]
        df[columns].dropna().to_csv(target_dir / f"{target}_kriging_input.csv", index=False)


def write_report(
    extent: pd.DataFrame,
    nn_summary: pd.DataFrame,
    targets: pd.DataFrame,
    deps: pd.DataFrame,
) -> None:
    kriging_ready = bool(
        deps.loc[deps["package"].isin(["pykrige", "gstools"]), "available"].any()
    )
    lines = [
        "# Phase 3 Ordinary Kriging Readiness Report",
        "",
        "## Dataset",
        "",
        "- Source: `data/processed/Kerala_Soil_Environmental_Features.csv`",
        "- Working samples: 4,908",
        f"- Projected CRS for distance-based geostatistics: `{PROJECTED_CRS}`",
        "",
        "## Readiness Decision",
        "",
    ]
    if kriging_ready:
        lines.append("- Ordinary Kriging dependencies are available; variogram fitting can proceed.")
    else:
        lines.append(
            "- Ordinary Kriging dependencies are not installed. Install `pykrige` or `gstools` before fitting OK models."
        )

    lines.extend(
        [
            "",
            "## Spatial Extent",
            "",
            extent.to_csv(index=False),
            "",
            "## Nearest Neighbor Summary",
            "",
            nn_summary.to_csv(index=False),
            "",
            "## Target Summary",
            "",
            targets.to_csv(index=False),
            "",
            "## Dependency Status",
            "",
            deps.to_csv(index=False),
        ]
    )
    (RESULTS_DIR / "phase3_kriging_readiness_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_FILE)
    projected = project_coordinates(df)
    projected.to_csv(RESULTS_DIR / "kerala_soil_environmental_projected.csv", index=False)
    write_target_variogram_inputs(projected)

    extent = extent_summary(projected)
    nn = nearest_neighbor_summary(projected)
    targets = target_summary(projected)
    deps = dependency_status()

    extent.to_csv(RESULTS_DIR / "spatial_extent.csv", index=False)
    nn.to_csv(RESULTS_DIR / "nearest_neighbor_summary.csv", index=False)
    targets.to_csv(RESULTS_DIR / "target_summary.csv", index=False)
    deps.to_csv(RESULTS_DIR / "dependency_status.csv", index=False)
    write_report(extent, nn, targets, deps)

    print(f"Kriging readiness outputs written to: {RESULTS_DIR}")
    print(deps.to_string(index=False))


if __name__ == "__main__":
    main()
