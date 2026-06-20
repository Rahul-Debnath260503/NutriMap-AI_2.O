# Phase 3 Ordinary Kriging Baseline Report

## Modeling Policy

- Coordinates are used here only for the geostatistical OK baseline.
- Coordinates remain excluded from ML/DL predictive input features.

## Run Configuration

- Targets: pH, EC, P, K, Organic_Carbon, Ca, Mg, S, Fe, Mn, Zn, Cu, B
- Validation: grouped spatial CV by `Block`.

## Best Variogram By Target

- B: gaussian, RMSE=0.676, MAE=0.484, R2=-5.290
- Ca: gaussian, RMSE=307.974, MAE=254.733, R2=-0.238
- Cu: spherical, RMSE=2.881, MAE=2.364, R2=-3.500
- EC: linear, RMSE=0.236, MAE=0.137, R2=-0.047
- Fe: spherical, RMSE=41.740, MAE=31.486, R2=-4.901
- K: exponential, RMSE=213.994, MAE=176.889, R2=-0.874
- Mg: gaussian, RMSE=113.995, MAE=90.344, R2=-0.296
- Mn: exponential, RMSE=29.835, MAE=24.394, R2=-8.740
- Organic_Carbon: linear, RMSE=1.069, MAE=0.851, R2=-0.141
- P: exponential, RMSE=45.595, MAE=34.977, R2=-0.058
- S: spherical, RMSE=24.949, MAE=15.926, R2=-0.086
- Zn: exponential, RMSE=5.863, MAE=4.910, R2=-66.287
- pH: linear, RMSE=0.889, MAE=0.741, R2=-0.583