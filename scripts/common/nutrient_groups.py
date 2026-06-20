"""
Explicit nutrient group definitions for NutriMap-AI.

Organic_Carbon is kept as a primary-level target/proxy in this project because
the dataset does not contain direct nitrogen measurements.
"""

from __future__ import annotations


SOIL_PROPERTIES = ["pH", "EC"]
PRIMARY_NUTRIENTS = ["P", "K", "Organic_Carbon"]
SECONDARY_NUTRIENTS = ["Ca", "Mg", "S"]
MICRONUTRIENTS = ["Fe", "Mn", "Zn", "Cu", "B"]

ENVIRONMENTAL_VARIABLES = [
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

ALL_TARGETS = SOIL_PROPERTIES + PRIMARY_NUTRIENTS + SECONDARY_NUTRIENTS + MICRONUTRIENTS

MODEL_INPUT_SCHEMES = {
    "primary_plus_secondary_plus_env_to_micro": {
        "inputs": PRIMARY_NUTRIENTS + SECONDARY_NUTRIENTS + ENVIRONMENTAL_VARIABLES,
        "targets": MICRONUTRIENTS,
    },
    "primary_plus_micro_plus_env_to_secondary": {
        "inputs": PRIMARY_NUTRIENTS + MICRONUTRIENTS + ENVIRONMENTAL_VARIABLES,
        "targets": SECONDARY_NUTRIENTS,
    },
    "secondary_plus_micro_plus_env_to_primary": {
        "inputs": SECONDARY_NUTRIENTS + MICRONUTRIENTS + ENVIRONMENTAL_VARIABLES,
        "targets": PRIMARY_NUTRIENTS,
    },
}
