# NutriMap-AI Memory

## Current Status

- Project A is the immediate priority: close Phase 1 data preparation, then begin Phase 2 EDA.
- Available processed datasets:
  - `data/processed/Kerala_Soil_Clean_Final.csv`: 5,494 rows, 17 columns.
  - `data/processed/Kerala_Soil_ML_Ready.csv`: 4,912 rows, 17 columns.
  - `data/processed/Kerala_Soil_Environmental_Features.csv`: 4,908 rows, 28 columns.
- Project B backend/frontend work has not started.

## Completed Tasks

- 2026-06-20: Read `plan.md` and inspected current repository status.
- 2026-06-20: Confirmed source is minimal: merge notebook, final cleaning script, GEE extraction script, and untracked SQL schema.
- 2026-06-20: Confirmed `Nutrimap_AI/` is a local virtual environment and should remain ignored.
- 2026-06-20: Cleaned `.gitignore` merge conflict markers and stopped ignoring essential Markdown/text project files.
- 2026-06-20: Created this tracked project memory/progress file.
- 2026-06-20: Replaced hard-coded paths in `notebook/02_Final_Data_Cleaning.py` with repository-relative paths.
- 2026-06-20: Added reproducible Phase 1 validation script at `scripts/data_validation.py`.
- 2026-06-20: Added initial Phase 2 EDA script at `scripts/eda/phase2_eda.py`.
- 2026-06-20: Verified syntax for cleaning, validation, and EDA scripts with AST parsing.
- 2026-06-20: Reran the cleaning workflow successfully; generated output matched existing `Kerala_Soil_ML_Ready.csv`, so no dataset rewrite was needed.
- 2026-06-20: Generated Phase 1 validation outputs under `results/data_validation/`.
- 2026-06-20: Generated initial Phase 2 EDA outputs under `results/EDA/`.
- 2026-06-20: User approved proceeding with 4,908 GEE-enriched samples instead of re-exporting Chelannur immediately.
- 2026-06-20: Reviewed Pearson/Spearman outputs; strongest environmental correlations involve rainfall, soil moisture, temperature, and elevation.
- 2026-06-20: Added Phase 2 Random Forest feature-importance script at `scripts/eda/feature_importance_rf.py`.
- 2026-06-20: Ran Phase 2 Random Forest feature importance; outputs saved under `results/EDA/feature_importance/`.
- 2026-06-20: Added Phase 3 kriging-readiness script at `scripts/ok/readiness.py`.
- 2026-06-20: Ran Phase 3 readiness; projected coordinates and per-target kriging input files saved under `results/OK/readiness/`.
- 2026-06-20: Cleaned `requirements.txt` and added planned missing packages: `optuna`, `shap`, `pykrige`, and `gstools`.
- 2026-06-20: Confirmed installed dependencies are available in project virtual environment `Nutrimap_AI\\Scripts\\python.exe`, not global Python.
- 2026-06-20: Added shared modeling config at `scripts/modeling_config.py` to keep coordinates out of ML/DL feature sets.
- 2026-06-20: Reran Phase 3 readiness with project venv; PyKrige and GSTools are now available there.
- 2026-06-20: Added Phase 2 XGBoost + SHAP script at `scripts/eda/xgboost_shap.py`.
- 2026-06-20: Ran XGBoost + SHAP; outputs saved under `results/EDA/xgboost_shap/`.
- 2026-06-20: Added Phase 3 Ordinary Kriging baseline script at `scripts/ok/ordinary_kriging.py`.
- 2026-06-20: Restructured the script tree into `scripts/common/`, `scripts/eda/`, and `scripts/ok/`.
- 2026-06-20: Added `results_review.md` summarizing result folders, CSV meanings, and interpretation.
- 2026-06-20: Ran first true Ordinary Kriging baseline for available primary nutrients `P` and `K`.
- 2026-06-20: Added `scripts/common/nutrient_groups.py` and `scripts/common/spatial_scope.py` to formalize target groups and area-specific map scoping.
- 2026-06-20: Pinned `requirements.txt` to exact installed versions from the project virtual environment, including notebook stack versions.
- 2026-06-20: Expanded the Ordinary Kriging baseline to all current soil-response targets: `pH`, `EC`, `P`, `K`, `Organic_Carbon`, `Ca`, `Mg`, `S`, `Fe`, `Mn`, `Zn`, `Cu`, and `B`.
- 2026-06-20: Generated area-specific OK prediction maps and stats for `Kottayam/Pampady`, `Kottayam/Vazhoor`, `Kozhikode/Chelannur`, and `Thiruvananthapuram/Nedumangad`.
- 2026-06-20: Confirmed OK outputs are written under `results/OK/baseline/` and `results/OK/maps/`, with map exports split by district/block and best variogram selected per target.
- 2026-06-20: Confirmed the DB schema file `create_tables.sql` is present for later model registry and metric persistence work.
- 2026-06-20: Updated OK map export so sample markers are optional instead of drawn by default on the PNG maps.

## Errors And Blockers

- `.gitignore` had unresolved merge-conflict markers and ignored `*.md`/`*.txt`, hiding important project files from Git.
- `notebook/02_Final_Data_Cleaning.py` used hard-coded Windows paths and pointed to `data/raw/Kerala_Soil_Clean_Final.csv`, while the available clean file is under `data/processed/`.
- `python -m py_compile notebook/02_Final_Data_Cleaning.py` failed because the sandbox could not create `notebook/__pycache__`; use AST-based syntax checks instead.
- 2026-06-20: First cleaning rerun failed with `PermissionError` while overwriting `data/processed/Kerala_Soil_ML_Ready.csv`. Script was updated to skip rewriting when the generated output is unchanged.
- 2026-06-20: First validation run failed because `DataFrame.to_markdown()` requires optional dependency `tabulate`. Script was updated with an internal Markdown table renderer.
- 2026-06-20: `environmental_features` has 4 fewer rows than `ml_ready`; validation shows the difference is isolated to Chelannur.
- 2026-06-20: Active Python environment has scikit-learn and matplotlib, but not XGBoost or SHAP. Phase 2 can run Random Forest now; XGBoost/SHAP require dependency installation later.
- 2026-06-20: PyKrige and GSTools are not installed in the active Python environment, so true Ordinary Kriging cannot be run yet.
- 2026-06-20: Global Python still does not see `pykrige`, `gstools`, `xgboost`, `shap`, or `optuna`; run advanced scripts through `Nutrimap_AI\\Scripts\\python.exe`.
## Data Decisions

- Keep `.env`, virtual environments, local datasets, model binaries, and large geospatial exports ignored.
- Treat `memory.md`, `plan.md`, `requirements.txt`, and project docs as trackable repository artifacts.
- Proceed with the 4,908-row GEE-enriched dataset for current EDA and baseline modeling.
- Do not substitute IDW for Ordinary Kriging in Phase 3; prepare kriging inputs now and run OK once `pykrige` or `gstools` is available.
- Random Forest with environmental covariates only gives negative block-spatial-CV R2 for all current targets; this supports the need for geostatistical and hybrid spatial models.
- ML/DL models must not use `Latitude`, `Longitude`, `x_utm43n_m`, or `y_utm43n_m` as input features. Coordinates are metadata for maps, spatial CV, Ordinary Kriging baselines, and residual kriging only.
- `K` means potassium. If we want an alternate target in the thesis or platform experiments, `Organic_Carbon` should be modeled separately rather than treated as K.
- First OK result: exponential variogram performed best for both `P` and `K`; `P` RMSE=45.595 and `K` RMSE=213.994 under grouped block CV.
- Maps should be generated per district/block scope. For example, `Kottayam/Vazhoor` should produce its own map instead of blending all districts into one visualization.
- Target grouping policy: `primary = [P, K, Organic_Carbon]`, `secondary = [Ca, Mg, S]`, `micro = [Fe, Mn, Zn, Cu, B]`, environmental features stay separate.
- Phase 3 is complete for the current dataset: readiness, full OK baselines for all current targets, and area-specific OK maps are all in place.
- ML will use environmental features only. Coordinates stay out of ML/DL feature matrices and remain reserved for mapping, spatial validation, and kriging workflows.
- OK PNG maps should stay clean by default; sample locations only appear when the export flag explicitly asks for them.
- Current zero-value review items in ML-ready/enriched data: one `Ca` zero, one `EC` zero, and one `Mn` zero. Do not automatically remove them until domain validity is confirmed.
- Exact coordinate joins between ML-ready and GEE-enriched outputs are unreliable because GEE export changes coordinate precision; use block counts and tolerant spatial matching.

## Best Next Step

- Move to Phase 4 ML with environmental-only features, starting with a clean train/validation strategy that respects the no-coordinate rule.
- Use `create_tables.sql` and `.env` later for model registry and metric persistence once the modeling pipeline starts writing outputs.
