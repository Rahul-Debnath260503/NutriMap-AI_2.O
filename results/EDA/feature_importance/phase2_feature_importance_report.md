# Phase 2 Feature Importance Report

## Dataset Decision

- Proceeding with `Kerala_Soil_Environmental_Features.csv` using 4,908 samples.
- The 4-row GEE extraction difference is accepted for current EDA and baseline modeling.

## Current Run

- Random Forest feature importance completed using environmental covariates only.
- Spatial validation uses grouped folds by `Block`; with four blocks available, this run uses 4 folds.
- XGBoost and SHAP were not run because those packages are not installed in the active Python environment.

## Strongest Environmental Signals From Spearman Correlation

- Zn: Rainfall (-0.462)
- Fe: Rainfall (-0.409)
- Cu: Rainfall (-0.403)
- Zn: Soil_Moisture (-0.360)
- Cu: Soil_Moisture (-0.346)
- Fe: Soil_Moisture (-0.332)
- pH: Rainfall (-0.326)
- pH: Soil_Moisture (-0.310)
- Mg: Elevation (0.288)
- B: Rainfall (-0.283)

## Mean Random Forest Feature Importance

- Elevation: 0.1603
- NDWI: 0.1477
- NDVI: 0.1463
- Rainfall: 0.1452
- Aspect: 0.1353
- Slope: 0.0928
- Soil_Moisture: 0.0851
- Temperature_C: 0.0844
- LULC: 0.0029

## Spatial CV Metric Snapshot

- Organic_Carbon: R2=-0.065, RMSE=1.039, MAE=0.792
- EC: R2=-0.099, RMSE=0.241, MAE=0.144
- pH: R2=-0.264, RMSE=0.810, MAE=0.666
- S: R2=-0.266, RMSE=26.507, MAE=18.381
- K: R2=-0.383, RMSE=191.117, MAE=154.479
- P: R2=-0.705, RMSE=51.626, MAE=42.072
- Mg: R2=-1.143, RMSE=136.564, MAE=115.716
- Ca: R2=-1.172, RMSE=375.312, MAE=323.672
- B: R2=-2.681, RMSE=0.793, MAE=0.615
- Cu: R2=-2.782, RMSE=2.714, MAE=2.159
- Fe: R2=-7.683, RMSE=42.626, MAE=33.816
- Mn: R2=-8.691, RMSE=29.574, MAE=24.128
- Zn: R2=-69.160, RMSE=5.966, MAE=5.072