"""
Final data cleaning for ML/DL training.

This script reads the merged clean soil dataset, applies domain-informed
quality filters, and writes the ML-ready dataset. Paths are resolved from the
repository root so the workflow is reproducible across machines.
"""

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = REPO_ROOT / "data" / "processed" / "Kerala_Soil_Clean_Final.csv"
OUTPUT_FILE = REPO_ROOT / "data" / "processed" / "Kerala_Soil_ML_Ready.csv"

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


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    original_rows = len(df)

    print("=" * 70)
    print("STARTING DATA CLEANING FOR ML/DL")
    print("=" * 70)
    print(f"\nInput file: {INPUT_FILE}")
    print(f"Initial shape: {df.shape}")

    print("\n[STEP 1] Removing NULL values...")
    before = len(df)
    df = df.dropna(subset=["Ca"])
    print(f"  Ca NULL removed: {before - len(df)} rows")

    print("\n[STEP 2] Removing invalid pH values...")
    before = len(df)
    df = df[(df["pH"] >= 3.5) & (df["pH"] <= 8.5)]
    print(f"  pH outliers removed: {before - len(df)} rows")
    print(f"  pH range after: {df['pH'].min():.2f} - {df['pH'].max():.2f}")

    print("\n[STEP 3] Removing micro-nutrient extremes...")
    thresholds = {
        "B": 5,
        "Cu": 15,
        "Zn": 30,
        "Mn": 350,
    }
    for column, threshold in thresholds.items():
        before = len(df)
        df = df[df[column] <= threshold]
        print(f"  {column} > {threshold} removed: {before - len(df)} rows")

    print("\n[STEP 4] Validating macro-nutrients...")
    macro_thresholds = {
        "Ca": 2000,
        "K": 1000,
    }
    for column, threshold in macro_thresholds.items():
        before = len(df)
        df = df[df[column] <= threshold]
        print(f"  {column} > {threshold} removed: {before - len(df)} rows")

    print(f"  Mg range: {df['Mg'].min():.2f} - {df['Mg'].max():.2f} ppm")

    print("\n[STEP 5] Removing zero/negative values where inappropriate...")
    before = len(df)
    df = df[df["Organic_Carbon"] > 0]
    print(f"  Organic_Carbon <= 0 removed: {before - len(df)} rows")

    print("\n[STEP 6] Final verification...")
    df = df.reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        if existing.equals(df):
            print(f"  Existing output is unchanged: {OUTPUT_FILE}")
        else:
            df.to_csv(OUTPUT_FILE, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n  Final shape: {df.shape}")
    print(f"  Rows kept: {len(df)} ({len(df) / original_rows * 100:.1f}% of original)")
    print(f"  Missing values: {int(df.isnull().sum().sum())}")
    print(f"  Duplicates: {int(df.duplicated().sum())}")

    print("\nSUCCESS")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Final samples: {len(df)}")

    print("\n" + "=" * 70)
    print("FINAL DATA SUMMARY (ML/DL READY)")
    print("=" * 70)
    print("\nNutrient ranges:")
    for column in NUTRIENT_COLUMNS:
        if column in df.columns:
            valid = df[column].dropna()
            print(
                f"  {column:20s} -> {valid.min():8.3f} "
                f"to {valid.max():8.3f} (mean: {valid.mean():8.3f})"
            )
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
