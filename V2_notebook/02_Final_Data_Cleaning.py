"""
FINAL DATA CLEANING FOR ML/DL TRAINING
Removes outliers and prepares data for model training
Based on senior data analyst review
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv(r'D:\NutriMap-AI\data\raw\Kerala_Soil_Clean_Final.csv')

print("="*70)
print("STARTING DATA CLEANING FOR ML/DL")
print("="*70)
print(f"\nInitial Shape: {df.shape}")

# ============================================================
# STEP 1: REMOVE NULL VALUES
# ============================================================
print("\n[STEP 1] Removing NULL values...")
before = len(df)
df = df.dropna(subset=['Ca'])  # Remove 1 Ca missing value
print(f"  Ca NULL removed: {before - len(df)} rows")

# ============================================================
# STEP 2: REMOVE INVALID pH OUTLIERS
# ============================================================
print("\n[STEP 2] Removing invalid pH values...")
before = len(df)
df = df[(df['pH'] >= 3.5) & (df['pH'] <= 8.5)]
print(f"  pH outliers removed: {before - len(df)} rows")
print(f"  pH range after: {df['pH'].min():.2f} - {df['pH'].max():.2f}")

# ============================================================
# STEP 3: HANDLE MICRO-NUTRIENT OUTLIERS (B, Cu, Zn)
# ============================================================
# Using domain-appropriate caps based on soil science standards

print("\n[STEP 3] Capping micro-nutrient extremes...")

# Boron: cap at 5 ppm (normal max ~2, but allowing for extreme variations)
before = len(df)
df = df[df['B'] <= 5]
print(f"  B > 5 ppm removed: {before - len(df)} rows")

# Copper: cap at 15 ppm (normal max ~5, allowing extremes)
before = len(df)
df = df[df['Cu'] <= 15]
print(f"  Cu > 15 ppm removed: {before - len(df)} rows")

# Zinc: cap at 30 ppm (normal max ~10, allowing extremes)
before = len(df)
df = df[df['Zn'] <= 30]
print(f"  Zn > 30 ppm removed: {before - len(df)} rows")

# Manganese: cap at 350 ppm (extreme but possible)
before = len(df)
df = df[df['Mn'] <= 350]
print(f"  Mn > 350 ppm removed: {before - len(df)} rows")

# ============================================================
# STEP 4: VALIDATE MACRO-NUTRIENTS
# ============================================================
print("\n[STEP 4] Validating macro-nutrients...")

# Ca cap at 2000 ppm (reasonable agricultural soil max)
before = len(df)
df = df[df['Ca'] <= 2000]
print(f"  Ca > 2000 ppm removed: {before - len(df)} rows (likely data errors)")

# K cap at 1000 ppm (reasonable max)
before = len(df)
df = df[df['K'] <= 1000]
print(f"  K > 1000 ppm removed: {before - len(df)} rows")

# Mg max usually 500 (already present as cap)
print(f"  Mg range: {df['Mg'].min():.2f} - {df['Mg'].max():.2f} ppm")

# ============================================================
# STEP 5: REMOVE ZERO/NEGATIVE PROBLEMATIC VALUES
# ============================================================
print("\n[STEP 5] Removing zero/negative values (where inappropriate)...")

# Organic Carbon should be > 0
before = len(df)
df = df[df['Organic_Carbon'] > 0]
print(f"  Organic_Carbon <= 0 removed: {before - len(df)} rows")

# ============================================================
# STEP 6: FINAL RESET AND VERIFICATION
# ============================================================
print("\n[STEP 6] Final verification...")
df = df.reset_index(drop=True)

print(f"\n  Final Shape: {df.shape}")
print(f"  Rows kept: {len(df)} ({len(df)/5493*100:.1f}% of original)")
print(f"  Missing values: {df.isnull().sum().sum()}")
print(f"  Duplicates: {df.duplicated().sum()}")

# ============================================================
# STEP 7: EXPORT CLEANED DATA
# ============================================================
output_file = r'D:\NutriMap-AI\data\processed\Kerala_Soil_ML_Ready.csv'

df.to_csv(output_file, index=False)

print(f"\n✅ SUCCESS")
print(f"Saved: {output_file}")
print(f"Final samples: {len(df)}")

# ============================================================
# STEP 8: SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("FINAL DATA SUMMARY (ML/DL READY)")
print("="*70)

nutrient_cols = ["pH", "EC", "Organic_Carbon", "P", "K", "Ca", "Mg", "S", "Fe", "Mn", "Zn", "Cu", "B"]

print("\nNutrient Ranges:")
for col in nutrient_cols:
    if col in df.columns:
        valid = df[col].dropna()
        print(f"  {col:20s} → {valid.min():8.3f} to {valid.max():8.3f} (mean: {valid.mean():8.3f})")

print("\n" + "="*70)
