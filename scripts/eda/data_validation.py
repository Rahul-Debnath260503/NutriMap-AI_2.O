"""
Generate reproducible Phase 1 data validation reports.
"""

from pathlib import Path
import sys

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = REPO_ROOT / "results" / "data_validation"

DATASETS = {
    "clean_final": PROCESSED_DIR / "Kerala_Soil_Clean_Final.csv",
    "ml_ready": PROCESSED_DIR / "Kerala_Soil_ML_Ready.csv",
    "environmental_features": PROCESSED_DIR / "Kerala_Soil_Environmental_Features.csv",
}

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


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return pd.read_csv(path)


def dataset_summary(name: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "dataset": name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "latitude_min": df["Latitude"].min() if "Latitude" in df else None,
        "latitude_max": df["Latitude"].max() if "Latitude" in df else None,
        "longitude_min": df["Longitude"].min() if "Longitude" in df else None,
        "longitude_max": df["Longitude"].max() if "Longitude" in df else None,
    }


def numeric_ranges(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in df.columns]
    rows = []
    for column in available:
        series = pd.to_numeric(df[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "missing": int(series.isna().sum()),
                "zero_count": int((series == 0).sum()),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
            }
        )
    return pd.DataFrame(rows)


def block_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["District", "Block"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["District", "Block"])
    )


def coordinate_key(df: pd.DataFrame, decimals: int) -> pd.Series:
    lat = pd.to_numeric(df["Latitude"], errors="coerce").round(decimals).astype(str)
    lon = pd.to_numeric(df["Longitude"], errors="coerce").round(decimals).astype(str)
    return lat + "," + lon


def compare_ml_to_environment(
    ml_ready: pd.DataFrame, env: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ml_counts = block_counts(ml_ready).rename(columns={"rows": "ml_ready_rows"})
    env_counts = block_counts(env).rename(columns={"rows": "environmental_rows"})
    count_delta = ml_counts.merge(env_counts, on=["District", "Block"], how="outer").fillna(0)
    count_delta["row_delta"] = (
        count_delta["ml_ready_rows"] - count_delta["environmental_rows"]
    )

    rows = []
    for decimals in [6, 5, 4, 3]:
        ml_keys = set(coordinate_key(ml_ready, decimals))
        env_keys = set(coordinate_key(env, decimals))
        rows.append(
            {
                "coordinate_rounding_decimals": decimals,
                "ml_ready_unique_coordinates": len(ml_keys),
                "environmental_unique_coordinates": len(env_keys),
                "ml_ready_coordinates_not_in_environmental": len(ml_keys - env_keys),
                "environmental_coordinates_not_in_ml_ready": len(env_keys - ml_keys),
            }
        )
    coordinate_delta = pd.DataFrame(rows)
    return count_delta, coordinate_delta


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    text_df = df.copy()
    text_df = text_df.where(pd.notnull(text_df), "")
    text_df = text_df.astype(str)

    headers = list(text_df.columns)
    rows = text_df.values.tolist()
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(values: list[str]) -> str:
        cells = [
            str(value).ljust(widths[index])
            for index, value in enumerate(values)
        ]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def write_markdown_report(
    summaries: pd.DataFrame,
    range_tables: dict[str, pd.DataFrame],
    count_delta: pd.DataFrame,
    coordinate_delta: pd.DataFrame,
) -> None:
    report_path = RESULTS_DIR / "phase1_data_validation_report.md"
    lines = [
        "# Phase 1 Data Validation Report",
        "",
        "## Dataset Summary",
        "",
        markdown_table(summaries),
        "",
        "## ML-Ready vs Environmental Feature Row Counts",
        "",
        markdown_table(count_delta),
        "",
        "## Coordinate Match Sensitivity",
        "",
        markdown_table(coordinate_delta),
        "",
        "## Key Notes",
        "",
        "- `environmental_features` has 4 fewer rows than `ml_ready`; the difference is isolated to Chelannur.",
        "- Exact coordinate joins are unreliable because GEE export can alter coordinate precision.",
        "- Zero values are reported, not automatically removed; soil-science review is required before treating zeros as missing.",
        "",
    ]

    for name, table in range_tables.items():
        lines.extend([f"## Numeric Ranges: {name}", "", markdown_table(table), ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = {name: read_dataset(path) for name, path in DATASETS.items()}

    summaries = pd.DataFrame(dataset_summary(name, df) for name, df in data.items())
    summaries.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    range_tables = {}
    for name, df in data.items():
        columns = NUTRIENT_COLUMNS + ENV_COLUMNS
        ranges = numeric_ranges(df, columns)
        ranges.to_csv(RESULTS_DIR / f"{name}_numeric_ranges.csv", index=False)
        range_tables[name] = ranges

        counts = block_counts(df)
        counts.to_csv(RESULTS_DIR / f"{name}_block_counts.csv", index=False)

    count_delta, coordinate_delta = compare_ml_to_environment(
        data["ml_ready"], data["environmental_features"]
    )
    count_delta.to_csv(RESULTS_DIR / "ml_ready_vs_environmental_block_delta.csv", index=False)
    coordinate_delta.to_csv(RESULTS_DIR / "coordinate_match_sensitivity.csv", index=False)

    write_markdown_report(summaries, range_tables, count_delta, coordinate_delta)

    print(f"Data validation reports written to: {RESULTS_DIR}")
    print(summaries.to_string(index=False))


if __name__ == "__main__":
    main()
