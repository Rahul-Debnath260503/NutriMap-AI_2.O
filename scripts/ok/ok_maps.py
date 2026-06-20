"""
Generate area-specific Ordinary Kriging maps for the best variogram per target.

The user can export maps for an entire district/block or for every available
block separately. This keeps each map focused on one location instead of mixing
multiple study areas in the same surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.common.nutrient_groups import ALL_TARGETS


REPO_ROOT = Path(__file__).resolve().parents[2]
READY_FILE = REPO_ROOT / "results" / "OK" / "readiness" / "kerala_soil_environmental_projected.csv"
BEST_VARIGRAM_FILE = REPO_ROOT / "results" / "OK" / "baseline" / "ordinary_kriging_best_variogram_by_target.csv"
OUTPUT_DIR = REPO_ROOT / "results" / "OK" / "maps"
DEFAULT_GRID_RESOLUTION_M = 100


def safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def build_area_groups(df: pd.DataFrame, district: str | None, block: str | None) -> list[tuple[str, str, pd.DataFrame]]:
    if district and block:
        area_df = df[(df["District"] == district) & (df["Block"] == block)].copy()
        return [(district, block, area_df)]
    if district:
        area_groups = []
        for block_name, area_df in df[df["District"] == district].groupby("Block"):
            area_groups.append((district, str(block_name), area_df.copy()))
        return area_groups
    groups = []
    for (district_name, block_name), area_df in df.groupby(["District", "Block"]):
        groups.append((str(district_name), str(block_name), area_df.copy()))
    return groups


def krige_area(
    area_df: pd.DataFrame,
    target: str,
    variogram_model: str,
    grid_resolution_m: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok = OrdinaryKriging(
        area_df["x_utm43n_m"].to_numpy(),
        area_df["y_utm43n_m"].to_numpy(),
        area_df[target].to_numpy(),
        variogram_model=variogram_model,
        coordinates_type="euclidean",
        verbose=False,
        enable_plotting=False,
    )

    x_min, x_max = area_df["x_utm43n_m"].min(), area_df["x_utm43n_m"].max()
    y_min, y_max = area_df["y_utm43n_m"].min(), area_df["y_utm43n_m"].max()

    x_pad = max(grid_resolution_m, (x_max - x_min) * 0.05)
    y_pad = max(grid_resolution_m, (y_max - y_min) * 0.05)
    x_values = np.arange(x_min - x_pad, x_max + x_pad + grid_resolution_m, grid_resolution_m)
    y_values = np.arange(y_min - y_pad, y_max + y_pad + grid_resolution_m, grid_resolution_m)

    z, variance = ok.execute("grid", x_values, y_values)
    return x_values, y_values, np.asarray(z, dtype=float), np.asarray(variance, dtype=float)


def save_png(
    path: Path,
    area_df: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
    surface: np.ndarray,
    target: str,
    district: str,
    block: str,
    variogram_model: str,
    show_samples: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    extent = [x_values.min(), x_values.max(), y_values.min(), y_values.max()]
    image = ax.imshow(
        np.flipud(surface),
        extent=extent,
        cmap="viridis",
        aspect="auto",
    )
    if show_samples:
        ax.scatter(
            area_df["x_utm43n_m"],
            area_df["y_utm43n_m"],
            s=6,
            c="white",
            edgecolors="black",
            linewidths=0.2,
            alpha=0.8,
        )
    ax.set_title(f"{target} OK Map - {district} / {block} ({variogram_model})")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.colorbar(image, ax=ax, label=target)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", help="Generate maps for one district only.")
    parser.add_argument("--block", help="Generate maps for one block only.")
    parser.add_argument("--grid-resolution-m", type=int, default=100)
    parser.add_argument(
        "--save-geotiff",
        action="store_true",
        help="Also export GeoTIFFs. Disabled by default because local PROJ installs can conflict.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=ALL_TARGETS,
        help="Targets to map. Defaults to all nutrients in the OK workflow.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Overlay the sample locations on the PNG maps.",
    )
    args = parser.parse_args()

    if not READY_FILE.exists():
        raise FileNotFoundError(f"Projected dataset not found: {READY_FILE}")
    if not BEST_VARIGRAM_FILE.exists():
        raise FileNotFoundError(f"Best variogram table not found: {BEST_VARIGRAM_FILE}")

    df = pd.read_csv(READY_FILE)
    best = pd.read_csv(BEST_VARIGRAM_FILE)
    best_lookup = best.set_index("target")["variogram_model"].to_dict()

    missing_targets = sorted(set(args.targets) - set(best_lookup))
    if missing_targets:
        raise ValueError(f"No best variogram available for: {missing_targets}")

    areas = build_area_groups(df, args.district, args.block)
    if not areas:
        raise ValueError("No matching area rows found.")

    manifest_rows = []
    for district, block, area_df in areas:
        if len(area_df) < 10:
            continue
        for target in args.targets:
            target_df = area_df.dropna(subset=[target, "x_utm43n_m", "y_utm43n_m"]).copy()
            if len(target_df) < 10:
                continue
            variogram_model = best_lookup[target]
            try:
                x_values, y_values, surface, variance = krige_area(
                    target_df, target, variogram_model, args.grid_resolution_m
                )
            except Exception as exc:
                manifest_rows.append(
                    {
                        "district": district,
                        "block": block,
                        "target": target,
                        "variogram_model": variogram_model,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                continue

            safe_district = safe_name(district)
            safe_block = safe_name(block)
            output_root = OUTPUT_DIR / safe_district / safe_block / target
            png_path = output_root / f"{target}_ok_best.png"
            stats_path = output_root / f"{target}_ok_best_stats.csv"

            save_png(
                png_path,
                target_df,
                x_values,
                y_values,
                surface,
                target,
                district,
                block,
                variogram_model,
                args.show_samples,
            )
            tif_path = ""
            if args.save_geotiff:
                tif_path = str(output_root / f"{target}_ok_best.tif")
            pd.DataFrame(
                [
                    {
                        "district": district,
                        "block": block,
                        "target": target,
                        "variogram_model": variogram_model,
                        "grid_resolution_m": args.grid_resolution_m,
                        "pred_min": float(np.nanmin(surface)),
                        "pred_max": float(np.nanmax(surface)),
                        "variance_min": float(np.nanmin(variance)),
                        "variance_max": float(np.nanmax(variance)),
                        "sample_count": len(target_df),
                    }
                ]
            ).to_csv(stats_path, index=False)

            manifest_rows.append(
                {
                    "district": district,
                    "block": block,
                    "target": target,
                    "variogram_model": variogram_model,
                    "status": "success",
                    "error": "",
                    "tif_path": tif_path,
                    "png_path": str(png_path),
                    "stats_path": str(stats_path),
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_DIR / "ok_map_manifest.csv", index=False)
    print(f"OK map outputs written to: {OUTPUT_DIR}")
    print(manifest.groupby(["district", "block", "status"]).size().to_string())


if __name__ == "__main__":
    main()
