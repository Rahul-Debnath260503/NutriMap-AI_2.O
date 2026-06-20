# NutriMap-AI Master Project Plan

## Overview

NutriMap-AI consists of two interconnected projects:

### Project A – MSc Research Thesis

**Title**

Comparative Evaluation of Geostatistical, Machine Learning, and Deep Learning Approaches for Spatial Prediction of Soil Nutrients Using Environmental Covariates in Kerala

**Purpose**

To scientifically evaluate and compare geostatistical, machine learning, and deep learning methods for soil nutrient prediction and digital soil mapping.

### Project B – NutriMap-AI Intelligent Prediction Platform

**Purpose**

To develop a production-ready soil nutrient prediction platform capable of predicting missing nutrient groups using environmental variables retrieved from Google Earth Engine (GEE) and user-provided soil samples.

The platform will expose prediction services through FastAPI and provide visualization, map generation, reporting, and model management capabilities.

---

# Project A: MSc Research Thesis

## Research Objectives

Compare the performance of:

### Geostatistical Models

* Ordinary Kriging (OK)

### Machine Learning Models

* Random Forest (RF)
* XGBoost (XGB)
* CatBoost
* Gradient Boosting (GB)

### Deep Learning Models

* Dense Neural Network (Dense-NN)
* Deep Neural Network (Deep-NN)
* ResNet-like Neural Network
* Wide & Deep Neural Network

for:

* Soil nutrient prediction
* Digital soil mapping
* Spatial interpolation
* Spatial accuracy assessment

---

## Research Questions

### RQ1

Can environmental covariates improve soil nutrient prediction compared to Ordinary Kriging?

### RQ2

Which model family provides the highest predictive performance?

* Geostatistical
* Machine Learning
* Deep Learning

### RQ3

Which approach produces the most spatially realistic and smooth nutrient maps?

### RQ4

Which nutrient group is easiest and hardest to predict?

* Primary Nutrients
* Secondary Nutrients
* Micronutrients

---

# Dataset

## Primary Dataset

```text
Kerala_Soil_Environmental_Features.csv
```

## Nutrient Groups

### Primary Nutrients

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)

### Secondary Nutrients

* Calcium (Ca)
* Magnesium (Mg)
* Sulfur (S)

### Micronutrients

* Iron (Fe)
* Manganese (Mn)
* Zinc (Zn)
* Copper (Cu)
* Boron (B)
* Molybdenum (Mo)

---

## Environmental Variables

Examples:

* Elevation
* Slope
* Aspect
* Rainfall
* Temperature
* NDVI
* LST
* Soil Moisture
* Land Cover
* TWI
* SPI
* Clay
* Sand
* SOC

Additional environmental variables may be incorporated as available.

---


# Phase 1: Data Preparation

## Objectives

Create a clean and analysis-ready dataset.

## Tasks

### Data Quality Assessment

* Missing value analysis
* Duplicate detection
* Data consistency checks

### Outlier Detection

Methods:

* IQR
* Z-score
* Isolation Forest (optional)

### Feature Engineering

Examples:

* Nutrient ratios
* Topographic indices
* Climate derivatives

### Feature Scaling

Methods:

* StandardScaler
* MinMaxScaler

### Outputs

* Clean dataset
* Data quality report
* Preprocessing report

---

# Phase 2: Exploratory Data Analysis

## Statistical Analysis

### Correlation Analysis

Generate:

* Pearson correlation matrix
* Spearman correlation matrix

### Distribution Analysis

Generate:

* Histograms
* KDE plots
* Boxplots
* Violin plots

---

## Spatial Analysis

Generate:

* Sample location maps
* Nutrient distribution maps

---

## Feature Importance Analysis

Methods:

* Random Forest
* XGBoost
* SHAP

Outputs:

* Feature ranking
* SHAP summary plots
* SHAP dependence plots

---

# Phase 3: Ordinary Kriging

## Objective

Establish geostatistical baseline performance.

### Variogram Models

Evaluate:

* Spherical
* Exponential
* Gaussian
* Linear

### Validation

Method:

* Leave-One-Out Cross Validation (LOOCV)

### Metrics

* MAE
* RMSE
* R²
* MAPE
* NSE

### Outputs

* Variogram plots
* Prediction maps
* Error maps
* Cross-validation reports

---

# Phase 4: Machine Learning Modeling

## Models

### Random Forest

### XGBoost

### CatBoost

### Gradient Boosting

---

## Hyperparameter Optimization

Tool:

* Optuna

Store:

* Best parameters
* Optimization history
* Trial results

---

## Validation Strategy

### Spatial Cross Validation

Use:

* 5-Fold Spatial CV

Avoid:

* Random train-test split

Reason:

Spatial datasets require spatially independent validation.

---

## Outputs

* Tuned models
* Feature importance plots
* SHAP explanations
* Model reports

---

# Phase 5: Deep Learning Modeling

## Model 1: Dense Neural Network

Architecture:

* Dense Layers
* Batch Normalization
* Dropout

---

## Model 2: Deep Neural Network

Architecture:

* 6–10 hidden layers
* Residual skip connections

---

## Model 3: ResNet-like Architecture

Architecture:

* Residual blocks
* Deep tabular learning

---

## Model 4: Wide & Deep Network

Architecture:

* Wide linear component
* Deep neural component

---

## Training Enhancements

Implement:

* EarlyStopping
* ReduceLROnPlateau
* ModelCheckpoint
* TensorBoard
* Mixed Precision Training

---

## Outputs

Save:

* Training loss curves
* Validation loss curves
* Learning rate curves
* Training logs
* Best model checkpoints

---

# Phase 6: Model Comparison

## Evaluation Metrics

### Regression Metrics

* MAE
* RMSE
* R²
* MAPE
* NSE
* Bias
* Explained Variance

---

## Comparison Table

| Model | MAE | RMSE | R² | MAPE | NSE | Bias | Explained Variance |
| ----- | --- | ---- | -- | ---- | --- | ---- | ------------------ |

---

# Spatial Accuracy Assessment

## Raster Smoothness Analysis

Compute:

* Roughness Index

---

## Spatial Autocorrelation

Compute:

* Moran's I

---

## Prediction Surface Analysis

Compute:

* Semivariogram of predicted raster

Purpose:

Evaluate spatial realism and map quality.

---

# Phase 7: Best Model Selection

Select best-performing models for:

## Primary Nutrients

Example:

* XGBoost

## Secondary Nutrients

Example:

* CatBoost

## Micronutrients

Example:

* ResNet

Store all selected models for deployment.

---

# Phase 8: Hybrid Modeling (Regression Kriging & Residual Kriging)

## Objective

Develop and evaluate hybrid spatial prediction models that combine:

* Machine Learning
* Deep Learning
* Geostatistics

to determine whether hybrid approaches can outperform standalone Ordinary Kriging, Machine Learning, and Deep Learning models for soil nutrient prediction and digital soil mapping.

---

# Scientific Motivation

Traditional Machine Learning and Deep Learning models can effectively learn nonlinear relationships between soil nutrients and environmental variables.

However, they often fail to fully capture:

* Spatial autocorrelation
* Local spatial variability
* Geographic dependency

Ordinary Kriging captures spatial structure but ignores environmental covariates.

Hybrid approaches attempt to leverage both:

### Environmental Learning

Captured by:

* RF
* XGBoost
* CatBoost
* Deep Learning

### Spatial Dependency

Captured by:

* Ordinary Kriging
* Residual Kriging

This often results in superior prediction accuracy and more realistic spatial maps.

---

# Hybrid Approaches to Evaluate

## Model H1: Regression Kriging (RK)

### Workflow

Step 1

Train regression model:

```text id="n5n0z7"
Environmental Variables
            ↓
     Regression Model
            ↓
   Initial Prediction
```

Step 2

Calculate residuals:

```text id="2b6uqw"
Residual = Actual − Predicted
```

Step 3

Apply Ordinary Kriging to residuals.

Step 4

Combine:

```text id="m2q2ae"
Final Prediction
=
Regression Prediction
+
Kriged Residual
```

---

## Model H2: Random Forest + Residual Kriging

### Workflow

```text id="pr2r0n"
Environmental Variables
            ↓
      Random Forest
            ↓
     RF Prediction
            ↓
     Residual Computation
            ↓
 Ordinary Kriging on Residuals
            ↓
      Final Prediction
```

---

## Model H3: XGBoost + Residual Kriging

### Workflow

```text id="z60r4j"
Environmental Variables
            ↓
         XGBoost
            ↓
    XGB Prediction
            ↓
         Residuals
            ↓
 Ordinary Kriging Residuals
            ↓
     Final Prediction
```

Expected to be one of the strongest performers.

---

## Model H4: CatBoost + Residual Kriging

### Workflow

```text id="w6b2s8"
Environmental Variables
            ↓
         CatBoost
            ↓
     Residual Kriging
            ↓
      Final Prediction
```

---

## Model H5: Best Deep Learning Model + Residual Kriging

### Workflow

```text id="ez2n8q"
Best DL Model
(DenseNN / DeepNN / ResNet / WideDeep)
            ↓
      DL Prediction
            ↓
         Residuals
            ↓
 Ordinary Kriging Residuals
            ↓
      Final Prediction
```

---

# Hybrid Model Development Pipeline

## Step 1

Train base model.

Examples:

* RF
* XGB
* CatBoost
* ResNet

---

## Step 2

Generate predictions on training set.

---

## Step 3

Calculate residuals.

Formula:

```text id="s5b4wu"
Residual = Actual − Predicted
```

---

## Step 4

Perform variogram analysis on residuals.

Evaluate:

* Spherical
* Exponential
* Gaussian
* Linear

---

## Step 5

Perform Ordinary Kriging on residuals.

---

## Step 6

Generate residual prediction surface.

---

## Step 7

Create final hybrid prediction.

Formula:

```text id="9x3d2m"
Hybrid Prediction
=
ML Prediction
+
Kriged Residual
```

---

# Validation Strategy

Use same validation framework as previous phases.

## Spatial Cross Validation

Recommended:

* 5-Fold Spatial Blocking

---

## Evaluation Metrics

### Prediction Metrics

* MAE
* RMSE
* R²
* MAPE
* NSE
* Bias
* Explained Variance

---

## Spatial Metrics

### Moran's I

Assess spatial realism.

### Raster Roughness Index

Assess smoothness.

### Prediction Surface Variogram

Assess spatial continuity.

---

# Model Comparison Framework

Compare:

| Category       | Model             |
| -------------- | ----------------- |
| Geostatistical | Ordinary Kriging  |
| ML             | RF                |
| ML             | XGBoost           |
| ML             | CatBoost          |
| ML             | Gradient Boosting |
| DL             | DenseNN           |
| DL             | DeepNN            |
| DL             | ResNet            |
| DL             | Wide & Deep       |
| Hybrid         | RF + RK           |
| Hybrid         | XGB + RK          |
| Hybrid         | CatBoost + RK     |
| Hybrid         | DL + RK           |

---

# Uncertainty Mapping

Generate uncertainty products for hybrid models.

## Maps

### Prediction Mean

```text id="mqr34m"
predicted_nutrient.tif
```

### Prediction Standard Deviation

```text id="h7zv7v"
prediction_uncertainty.tif
```

### Confidence Interval Maps

```text id="kz4q7f"
lower_bound.tif
upper_bound.tif
```

---

# Explainable Hybrid Models

Apply SHAP to:

* RF
* XGB
* CatBoost
* Best DL Model

Objectives:

* Explain environmental influence
* Identify dominant predictors
* Improve scientific interpretability

---

# Expected Research Contributions

## Contribution 1

Comprehensive comparison among:

* Geostatistical
* Machine Learning
* Deep Learning
* Hybrid Models

within a single framework.

---

## Contribution 2

Evaluation of spatial realism and map quality.

---

## Contribution 3

Explainable AI framework for soil nutrient prediction.

---

## Contribution 4

Assessment of residual spatial structure after ML and DL prediction.

---

## Contribution 5

Identification of optimal modeling strategy for Digital Soil Mapping in Kerala.

---

# Publication Potential

This phase has strong publication potential because many soil mapping studies stop at:

* Kriging
* Random Forest
* XGBoost

Few studies comprehensively compare:

```text id="umh5cr"
OK
vs
ML
vs
DL
vs
Hybrid RK Models
```

within a single spatial validation framework.

If hybrid models significantly outperform standalone models, this phase can become the core contribution of a journal article.

---

# Expected Outcome

Hypothesis:

```text id="g9r0rx"
XGBoost + Residual Kriging
or
CatBoost + Residual Kriging
```

will achieve:

* Lowest RMSE
* Highest R²
* Best spatial realism
* Most publication-worthy results

while retaining interpretability through SHAP analysis.

---

# Advanced Research Extensions

## SHAP Explainability

Explain model behavior.

---

## Uncertainty Mapping

Generate:

* Mean prediction maps
* Standard deviation maps
* Confidence interval maps

---

## Ensemble Modeling

Develop stacking ensemble using:

* RF
* XGB
* CatBoost

---

## Hybrid Geostatistical Models

Investigate:

### Residual Kriging

ML Prediction + Kriging Residuals

### OK + XGBoost Hybrid

Potential publication-quality contribution.

---

# Results Directory Structure

```text
results/

├── OK
│   ├── Primary
│   ├── Secondary
│   └── Micro
│
├── ML
│   ├── RF
│   ├── XGBoost
│   ├── CatBoost
│   └── GB
│
├── DL
│   ├── DenseNN
│   ├── DeepNN
│   ├── ResNet
│   └── WideDeep
│
├── Comparison
├── Best_Models
│   ├── Primary
│   ├── Secondary
│   └── Micro
│
├── Maps
├── Reports
└── SHAP
```

---

# Project B: NutriMap-AI Deployment Platform

## Objective

Create an intelligent nutrient prediction platform.

---

# Prediction Scenarios

## Scenario 1

Input:

* Secondary Nutrients
* Micronutrients

Predict:

* Primary Nutrients

---

## Scenario 2

Input:

* Primary Nutrients
* Micronutrients

Predict:

* Secondary Nutrients

---

## Scenario 3

Input:

* Primary Nutrients
* Secondary Nutrients

Predict:

* Micronutrients

---

# Meta-Model Development

## Model Set A

Target:

Primary Nutrients

Features:

* Secondary Nutrients
* Micronutrients
* Environmental Variables

Output:

```text
best_primary_model.pkl
```

---

## Model Set B

Target:

Secondary Nutrients

Features:

* Primary Nutrients
* Micronutrients
* Environmental Variables

Output:

```text
best_secondary_model.pkl
```

---

## Model Set C

Target:

Micronutrients

Features:

* Primary Nutrients
* Secondary Nutrients
* Environmental Variables

Output:

```text
best_micro_model.pkl
```

---

# FastAPI System Architecture

```text
Frontend
    ↓
FastAPI
    ↓
Prediction Engine
    ↓
GEE Environmental Fetcher
    ↓
Model Registry
    ↓
PostgreSQL + PostGIS
```

---

# Technology Stack

## Backend

* FastAPI

## Database

* PostgreSQL
* PostGIS

## Geospatial Processing

* GDAL
* Rasterio
* GeoPandas

## Machine Learning

* Scikit-Learn
* XGBoost
* CatBoost

## Deep Learning

* TensorFlow/Keras
* PyTorch

## MLOps

* MLflow

## Containerization

* Docker

## Deployment

* AWS EC2
* Azure VM

---

# Database Components

## Model Registry

Store:

* Model metadata
* Performance metrics
* Hyperparameters
* File paths

---

## Prediction History

Store:

* User predictions
* Input datasets
* Generated maps
* Download history

---

# Future Enhancements

## Recommendation Engine

Generate:

* Fertility classification
* Nutrient deficiency status
* Fertilizer recommendations

---

## Interactive Mapping

Support:

* WebGIS
* Raster visualization
* Layer comparison

---

## Batch Processing

Allow:

* CSV uploads
* Bulk prediction

---

## Real-Time GEE Integration

Automatically fetch:

* NDVI
* Rainfall
* LST
* Elevation
* Soil Moisture

for uploaded coordinates.

---

# Final Deliverables

## MSc Thesis

* Complete preprocessing workflow
* OK vs ML vs DL comparison
* Spatial validation framework
* Nutrient maps
* Publication-quality figures

---

## NutriMap-AI Product

* FastAPI backend
* PostgreSQL/PostGIS database
* Model registry
* GEE integration
* Nutrient prediction API
* Interactive web interface
* Dockerized deployment

---

# Success Criteria

1. Publishable MSc thesis.
2. Reproducible research pipeline.
3. Production-ready prediction platform.
4. Explainable AI integration using SHAP.
5. Scalable deployment architecture.
6. Support for real-time nutrient prediction and map generation.


----
# Phase 0: Literature Review & Research Gap Analysis

## Objective

Establish the scientific foundation of the research, identify current state-of-the-art methodologies in Digital Soil Mapping (DSM), determine existing research gaps, and justify the selection of geostatistical, machine learning, deep learning, and hybrid modeling approaches for soil nutrient prediction.

---

# 0.1 Literature Review Strategy

## Databases

Conduct systematic literature review using:

* Scopus
* Web of Science
* ScienceDirect
* SpringerLink
* IEEE Xplore
* Google Scholar
* Wiley Online Library
* MDPI
* Nature
* Elsevier

---

## Search Keywords

### Digital Soil Mapping

```text
Digital Soil Mapping
DSM Soil Nutrient Prediction
Environmental Covariates Soil Mapping
Predictive Soil Mapping
```

### Geostatistics

```text
Ordinary Kriging
Regression Kriging
Universal Kriging
Geostatistical Soil Mapping
Spatial Interpolation of Soil Nutrients
```

### Machine Learning

```text
Random Forest Soil Prediction
XGBoost Soil Nutrient Mapping
CatBoost Soil Science
Gradient Boosting Environmental Modeling
```

### Deep Learning

```text
Deep Learning for Soil Mapping
Neural Networks Soil Nutrient Prediction
Tabular Deep Learning Geospatial Data
Wide and Deep Networks Environmental Modeling
```

### Explainable AI

```text
SHAP Soil Science
Explainable AI Environmental Modeling
Interpretable Machine Learning Agriculture
```

### Validation

```text
Spatial Cross Validation
Spatial Blocking
Environmental Modeling Validation
Spatial Machine Learning Evaluation
```

---

# 0.2 Digital Soil Mapping (DSM)

## Objective

Review the evolution of soil mapping methodologies and identify the role of environmental covariates in nutrient prediction.

### Topics to Cover

* Traditional soil survey methods
* Digital Soil Mapping framework
* SCORPAN model
* Environmental covariate integration
* GIS and Remote Sensing in DSM
* Soil nutrient prediction techniques

### Expected Outcome

Demonstrate why DSM is preferred over traditional interpolation approaches for large-scale soil nutrient mapping.

---

# 0.3 Geostatistical Approaches

## Objective

Evaluate strengths and limitations of kriging-based methods.

### Review Topics

#### Ordinary Kriging (OK)

Study:

* Assumptions
* Variogram modeling
* Spatial autocorrelation
* Advantages
* Limitations

#### Universal Kriging

Study:

* Trend incorporation
* Environmental gradients

#### Regression Kriging (RK)

Study:

* Regression component
* Spatial residual component
* Integration with environmental variables

---

## Comparison Table

Create literature summary table:

| Method | Strengths | Weaknesses | Environmental Variables |
| ------ | --------- | ---------- | ----------------------- |
| OK     |           |            |                         |
| UK     |           |            |                         |
| RK     |           |            |                         |

---

## Research Gap

Typical gaps:

* OK ignores environmental variables.
* Many studies compare only OK vs RF.
* Limited comparisons involving modern boosting algorithms.
* Few studies evaluate deep learning alongside kriging.

---

# 0.4 Machine Learning for Soil Nutrient Prediction

## Objective

Review machine learning approaches used in Digital Soil Mapping.

### Models to Review

#### Random Forest

Topics:

* Ensemble learning
* Feature importance
* Robustness to noise

#### XGBoost

Topics:

* Gradient boosting
* Regularization
* High predictive accuracy

#### CatBoost

Topics:

* Ordered boosting
* Handling categorical variables
* Reduced overfitting

#### Gradient Boosting

Topics:

* Sequential learning
* Prediction refinement

---

## Literature Analysis

For each paper extract:

* Study area
* Dataset size
* Environmental variables
* Validation method
* Metrics used
* Best model

---

## Research Gap

Typical observations:

* Many studies use random train-test splits.
* Few studies apply spatial cross-validation.
* Limited explainability analysis.
* Lack of reproducible workflows.

---

# 0.5 Deep Learning for Tabular Geospatial Data

## Objective

Investigate deep learning suitability for structured environmental datasets.

### Topics

#### Dense Neural Networks

* Feedforward architecture
* Regression tasks

#### Deep Neural Networks

* Multi-layer representations
* Feature interactions

#### ResNet-like Architectures

* Residual learning
* Gradient stability

#### Wide & Deep Networks

* Memorization and generalization
* Structured tabular data

---

## Literature Analysis

Evaluate:

* Dataset characteristics
* Model architecture
* Performance improvements
* Computational requirements

---

## Research Gap

Current challenges:

* Limited use in soil nutrient prediction.
* Small sample datasets often lead to overfitting.
* Few comparisons with boosting algorithms.
* Limited explainability studies.

---

# 0.6 Explainable AI (XAI) in Soil Science

## Objective

Review explainability methods used in environmental prediction models.

### Methods

#### SHAP

Topics:

* Global explainability
* Local explainability
* Feature contribution analysis

#### Permutation Importance

#### Partial Dependence Plots

#### Feature Interaction Analysis

---

## Research Gap

Common issues:

* Most studies focus solely on accuracy.
* Limited interpretation of environmental drivers.
* Lack of transparent prediction systems.

---

## Research Contribution

This study will integrate SHAP analysis to explain nutrient prediction mechanisms.

---

# 0.7 Spatial Cross-Validation in Environmental Modeling

## Objective

Evaluate validation strategies used in spatial prediction studies.

### Methods

#### Random Split

Advantages:

* Simple

Limitations:

* Spatial leakage
* Over-optimistic results

---

#### K-Fold Cross Validation

Advantages:

* Stable estimates

Limitations:

* Ignores spatial dependency

---

#### Spatial Cross Validation

Methods:

* Spatial Blocking
* Buffering
* Leave-Location-Out

Advantages:

* More realistic performance estimation

---

## Research Gap

Many published studies:

* Use random splits
* Ignore spatial autocorrelation
* Report inflated accuracy

---

## Research Contribution

This study will implement spatial cross-validation for all ML and DL models to ensure realistic and publication-quality evaluation.

---

# 0.8 Identification of Research Gaps

Based on the literature review, identify key gaps:

### Gap 1

Limited studies comparing:

* Geostatistics
* Machine Learning
* Deep Learning

within a single framework.

---

### Gap 2

Limited evaluation of modern boosting algorithms:

* XGBoost
* CatBoost

for soil nutrient prediction.

---

### Gap 3

Insufficient use of spatial cross-validation.

---

### Gap 4

Limited explainability analysis using SHAP.

---

### Gap 5

Few studies assess spatial realism of generated maps using:

* Moran's I
* Raster roughness
* Prediction semivariograms

---

### Gap 6

Limited research on hybrid models:

* Regression Kriging
* Residual Kriging
* ML + Kriging combinations

---

# 0.9 Research Hypotheses

## H1

Machine Learning models will outperform Ordinary Kriging due to integration of environmental covariates.

---

## H2

Boosting algorithms (XGBoost and CatBoost) will outperform traditional Random Forest.

---

## H3

Deep Learning models may outperform ML models when complex nonlinear relationships exist.

---

## H4

Residual Kriging will outperform standalone geostatistical and machine learning models.

---

## H5

Spatial cross-validation will produce lower but more realistic accuracy estimates than random validation.

---

# 0.10 Expected Scientific Contributions

This research aims to contribute:

1. Comprehensive comparison of OK, ML, DL, and hybrid approaches.
2. Evaluation of environmental covariate importance.
3. Explainable AI framework using SHAP.
4. Spatially robust validation methodology.
5. Assessment of spatial realism and map quality.
6. Deployment-ready nutrient prediction framework.
7. Foundation for NutriMap-AI production platform.

---

# Deliverables

## Outputs

* Literature review chapter
* Gap analysis matrix
* Research framework diagram
* Conceptual methodology diagram
* Research hypotheses
* Publication strategy

## Recommended Target Publications

### Paper 1

Comparative Evaluation of Geostatistical, Machine Learning, and Deep Learning Models for Soil Nutrient Prediction.

### Paper 2

Explainable AI and Spatial Validation Framework for Digital Soil Mapping.

### Paper 3 (Future)

NutriMap-AI: A Real-Time Soil Nutrient Prediction Platform Using Environmental Covariates and Hybrid AI Models.
