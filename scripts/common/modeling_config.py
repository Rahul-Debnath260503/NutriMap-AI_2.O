"""
Shared modeling configuration for NutriMap-AI research scripts.

Coordinates are metadata for mapping, spatial validation, and geostatistical
baselines. They must not be used as predictive input features for ML/DL models.
"""

from __future__ import annotations

from .nutrient_groups import (
    ALL_TARGETS,
    ENVIRONMENTAL_VARIABLES,
    SOIL_PROPERTIES,
)


TARGET_COLUMNS = ALL_TARGETS
ENVIRONMENTAL_FEATURE_COLUMNS = ENVIRONMENTAL_VARIABLES
SOIL_PROPERTY_COLUMNS = SOIL_PROPERTIES

METADATA_COLUMNS = [
    "system:index",
    "District",
    "Block",
    "Latitude",
    "Longitude",
    ".geo",
]

COORDINATE_COLUMNS = [
    "Latitude",
    "Longitude",
    "x_utm43n_m",
    "y_utm43n_m",
]
