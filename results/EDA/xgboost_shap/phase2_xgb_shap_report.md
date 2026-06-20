# Phase 2 XGBoost And SHAP Report

## Modeling Policy

- Input features are environmental covariates only.
- Coordinates are not used as ML features.
- `Block` is used only for grouped spatial cross-validation.

## Mean XGBoost Feature Importance

- Temperature_C: 0.2327
- Rainfall: 0.1986
- Soil_Moisture: 0.1892
- Elevation: 0.0785
- NDWI: 0.0679
- NDVI: 0.0668
- Aspect: 0.0665
- Slope: 0.0609
- LULC: 0.0389

## Mean Absolute SHAP Importance

- Rainfall: 10.1944
- Temperature_C: 10.0525
- Elevation: 4.4526
- Soil_Moisture: 3.9451
- NDWI: 2.7918
- NDVI: 2.5237
- Aspect: 2.2915
- Slope: 1.9204
- LULC: 0.1244

## Spatial CV Metric Snapshot

- Organic_Carbon: R2=-0.085, RMSE=1.048, MAE=0.803
- S: R2=-0.262, RMSE=26.520, MAE=17.964
- pH: R2=-0.331, RMSE=0.831, MAE=0.685
- K: R2=-0.440, RMSE=194.939, MAE=157.505
- EC: R2=-0.638, RMSE=0.276, MAE=0.148
- P: R2=-0.746, RMSE=52.333, MAE=42.285
- Mg: R2=-1.586, RMSE=149.120, MAE=127.202
- Ca: R2=-1.774, RMSE=411.202, MAE=357.020
- Cu: R2=-2.920, RMSE=2.812, MAE=2.220
- B: R2=-3.529, RMSE=0.905, MAE=0.714
- Fe: R2=-9.538, RMSE=46.301, MAE=37.839
- Mn: R2=-9.759, RMSE=30.212, MAE=24.171
- Zn: R2=-66.389, RMSE=6.383, MAE=5.439