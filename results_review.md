# NutriMap-AI Results Review

## Short Version

- The dataset is now stable enough to work with `4,908` GEE-enriched samples.
- Environmental covariates alone are informative, but not strong enough to predict every nutrient well under spatial validation.
- Rainfall, temperature, soil moisture, and elevation are the most influential environmental drivers in the current EDA.
- Ordinary Kriging is now run for all current soil-response targets, with the best variogram selected per target and area-specific map outputs produced.
- Coordinates are metadata for maps and geostatistics, not input features for ML/DL models.
- `K` is potassium and is present in the data. If you want a nitrogen-proxy target for your thesis framing, use `Organic_Carbon` as a separate labeled target; it is not potassium itself.
- Use `scripts/common/nutrient_groups.py` for target grouping and `scripts/common/spatial_scope.py` for area-specific map generation.

## What The Main Results Mean

### Data validation

- `clean_final` has `5,494` rows, `1` missing value, and `0` duplicates.
- `ml_ready` has `4,912` rows and is clean after the filtering rules.
- `environmental_features` has `4,908` rows, so 4 samples were lost during GEE enrichment.
- Those 4 missing rows are all in `Chelannur`.

### Correlation analysis

- Pearson measures straight-line relationships.
- Spearman measures monotonic relationships and is often better for soil data because the relationships are not perfectly linear.
- Rainfall is strongly related to several nutrients, especially `Fe`, `Zn`, `Cu`, `pH`, and `K`.
- Temperature and soil moisture also matter a lot, especially in the XGBoost and SHAP outputs.

### Feature importance

- Random Forest and XGBoost agree on the general story: `Elevation`, `Rainfall`, `NDVI`, `NDWI`, `Soil_Moisture`, and `Temperature_C` are the key environmental predictors.
- `LULC` is consistently weak in the current setup.
- The very poor spatial-CV R2 values for many targets mean the model is learning something, but not enough to generalize well across blocks.

### Ordinary Kriging

- `P` is the stronger kriging target in the current baseline.
- `K` is harder to interpolate with the current sampling pattern.
- The best variogram for both `P` and `K` was `exponential`.
- The OK baseline is not a winner everywhere, but it is scientifically valuable because it captures spatial autocorrelation directly.

## Folder Guide

### `results/data_validation`

- `dataset_summary.csv`  
  Row count, column count, missing values, duplicate rows, and coordinate bounds for the clean, ML-ready, and GEE-enriched datasets.

- `clean_final_block_counts.csv`  
  Number of samples per `District` and `Block` in the clean final dataset.

- `clean_final_numeric_ranges.csv`  
  Min, max, mean, standard deviation, missing count, and zero count for nutrient and environmental numeric columns in the clean final dataset.

- `ml_ready_block_counts.csv`  
  Same as above, but for the ML-ready dataset.

- `ml_ready_numeric_ranges.csv`  
  Numeric summary for the ML-ready dataset after filtering.

- `environmental_features_block_counts.csv`  
  Sample counts by district and block for the GEE-enriched dataset.

- `environmental_features_numeric_ranges.csv`  
  Numeric summary for the enriched dataset, including GEE-derived variables.

- `ml_ready_vs_environmental_block_delta.csv`  
  Shows the row-count difference between ML-ready and enriched data by block. This is where the 4-row Chelannur gap is visible.

- `coordinate_match_sensitivity.csv`  
  Diagnostic table showing how sensitive exact row matching is to coordinate precision. This explains why exact merges can miss rows even when the data are effectively the same.

### `results/EDA`

- `nutrient_environment_summary.csv`  
  Descriptive statistics for the nutrient and environmental variables in the enriched dataset.

- `pearson_correlation_matrix.csv`  
  Full Pearson correlation matrix for nutrients and environmental variables.

- `spearman_correlation_matrix.csv`  
  Full Spearman correlation matrix for nutrients and environmental variables.

### `results/EDA/feature_importance`

- `dependency_status.csv`  
  Shows which libraries were available when the Random Forest feature-importance run was executed.

- `rf_feature_importance_long.csv`  
  Long-format Random Forest importance table, with one row per target-feature pair.

- `rf_mean_feature_importance.csv`  
  Average Random Forest feature importance across all modeled targets.

- `rf_spatial_cv_metrics.csv`  
  Fold-level spatial cross-validation metrics for each Random Forest target.

- `rf_spatial_cv_metric_summary.csv`  
  Target-level summary of the Random Forest spatial CV results.

- `rf_top5_features_by_target.csv`  
  Top five Random Forest features for each target.

- `top_environment_correlations.csv`  
  Top environmental correlation pairs from the Spearman matrix.

### `results/EDA/xgboost_shap`

- `xgb_feature_importance_long.csv`  
  Long-format XGBoost feature importance table.

- `xgb_mean_feature_importance.csv`  
  Mean XGBoost feature importance across targets.

- `xgb_shap_importance_long.csv`  
  Long-format mean absolute SHAP importance table.

- `xgb_mean_abs_shap_importance.csv`  
  Mean absolute SHAP importance averaged across targets.

- `xgb_spatial_cv_metrics.csv`  
  Fold-level spatial CV metrics for XGBoost.

- `xgb_spatial_cv_metric_summary.csv`  
  Target-level summary for XGBoost spatial CV.

### `results/OK/readiness`

- `dependency_status.csv`  
  Confirms whether geostatistical packages are available in the project environment.

- `kerala_soil_environmental_projected.csv`  
  The enriched dataset projected into `EPSG:32643` with `x_utm43n_m` and `y_utm43n_m` added for geostatistics.

- `spatial_extent.csv`  
  Bounding box and width/height of the projected study area.

- `nearest_neighbor_summary.csv`  
  Nearest-neighbor spacing summary, useful for understanding point density and kriging behavior.

- `target_summary.csv`  
  Per-target summary of sample count, missing count, zero count, min, max, mean, and standard deviation.

- `target_inputs/*.csv`  
  One kriging-ready point table per nutrient. These are the inputs used for Ordinary Kriging baselines.

### `results/OK/baseline`

- `ordinary_kriging_spatial_cv_metrics.csv`  
  Fold-level metrics for each target and variogram model.

- `ordinary_kriging_metric_summary.csv`  
  Aggregated metrics by target and variogram model.

- `ordinary_kriging_best_variogram_by_target.csv`  
  Best variogram choice for each target based on RMSE.

- `ordinary_kriging_spatial_cv_predictions.csv`  
  Row-level prediction file with actual values, predicted values, kriging variance, block fold, and variogram label.

- `phase3_ordinary_kriging_report.md`  
  Human-readable summary of the OK run, including the best variogram and any failures.

### `results/OK/maps`

- `ok_map_manifest.csv`  
  Manifest of all exported OK maps, with district, block, target, status, and output paths.

- `Kottayam/...`, `Kozhikode/...`, `Thiruvananthapuram/...`  
  Area-specific map folders. Each district/block folder contains one subfolder per target.

- `*_ok_best.png`  
  The rendered OK prediction map for the best variogram chosen for that target.

- `*_ok_best_stats.csv`  
  Summary statistics for the rendered map, such as grid size and numeric range.

## Target Grouping Policy

- `primary` = `P`, `K`, `Organic_Carbon`
- `secondary` = `Ca`, `Mg`, `S`
- `micro` = `Fe`, `Mn`, `Zn`, `Cu`, `B`
- `Environmental features` = terrain, vegetation, moisture, climate, and land-cover variables only

This is the cleanest way to think about model input/output:

- For primary-target models, inputs are `secondary + micro + environmental features` if you choose a meta-model setup.
- For secondary-target models, inputs are `primary + micro + environmental features`.
- For micronutrient models, inputs are `primary + secondary + environmental features`.
- Coordinates stay out of ML/DL features and remain metadata/geostatistical inputs only.

## Phase Status

- Phase 3 OK is complete for the current dataset and current targets.
- Phase 4 ML can start next, using environmental variables only as features.
- Coordinates stay available for maps and spatial validation, not for model input matrices.

## Map Scope Policy

- Generate maps per district/block scope, not as one state-wide blend.
- If the deliverable is a map for `Kottayam/Vazhoor`, the prediction grid, kriging surface, and exported figures should be limited to that scope.
- The current data currently supports `Thiruvananthapuram/Nedumangad`, `Kottayam/Pampady`, `Kottayam/Vazhoor`, and `Kozhikode/Chelannur`.
- If you later add `Kollam` data, add a new scope entry and keep the same per-area map rule.
- The same rule applies for OK, ML, and DL output maps so the visual story stays geographically consistent.

## Scientific Interpretation

### 1. The covariates are useful, but not enough on their own

Random Forest and XGBoost both show that the environmental variables are informative. The problem is spatial generalization: under block-based CV, the models still struggle to predict many nutrients well.

### 2. Spatial structure matters

The OK baseline is important because it uses coordinates directly to model spatial autocorrelation. That is allowed for geostatistics and is exactly what makes OK a different family from ML/DL.

### 3. `P` is easier than `K` in the current baseline

Ordinary Kriging improved `P` enough to be competitive with or better than the ML baselines, but `K` remains difficult. That usually means the potassium surface is more complex or the current sampling layout is not capturing its structure cleanly.

### 4. `Organic_Carbon` is a valid extra target, not a potassium replacement

If you want a separate alternative target line, model `Organic_Carbon`. Do not label it as `K`. Potassium and organic carbon mean different things scientifically, and the dataset already contains both.

### 5. Coordinate policy

For ML and DL, coordinates stay out of the feature matrix. They are still kept in the CSVs for mapping, reporting, spatial validation, and kriging workflows.

## Script Layout

- `scripts/common/`  
  Shared configuration and constants.

- `scripts/eda/`  
  Validation, correlation, feature importance, and SHAP workflows.

- `scripts/ok/`  
  Ordinary Kriging readiness and baseline workflows.

Run the tools from the repository root with the project interpreter, for example:

```powershell
.\Nutrimap_AI\Scripts\python.exe -m scripts.eda.data_validation
.\Nutrimap_AI\Scripts\python.exe -m scripts.eda.feature_importance_rf
.\Nutrimap_AI\Scripts\python.exe -m scripts.eda.xgboost_shap
.\Nutrimap_AI\Scripts\python.exe -m scripts.ok.readiness
.\Nutrimap_AI\Scripts\python.exe -m scripts.ok.ordinary_kriging
```

## Next Step

- Start Phase 4 ML with a clean feature pipeline that excludes coordinate columns.
- Use the current OK outputs as the geostatistical baseline for later comparison against ML and hybrid models.
