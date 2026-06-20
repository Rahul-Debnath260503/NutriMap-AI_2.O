"""
Spatial scoping helpers for area-specific map generation.

Prediction maps should be generated per district/block scope rather than for
the entire state at once when the goal is area-specific visualization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class MapScope:
    name: str
    district: str
    block: str | None = None


DEFAULT_MAP_SCOPES = [
    MapScope(name="Thiruvananthapuram_Nedumangad", district="Thiruvananthapuram", block="Nedumangad"),
    MapScope(name="Kottayam_Pampady", district="Kottayam", block="Pampady"),
    MapScope(name="Kottayam_Vazhoor", district="Kottayam", block="Vazhoor"),
    MapScope(name="Kozhikode_Chelannur", district="Kozhikode", block="Chelannur"),
]


def filter_scope(df: pd.DataFrame, scope: MapScope) -> pd.DataFrame:
    subset = df[df["District"].astype(str).str.casefold() == scope.district.casefold()]
    if scope.block:
        subset = subset[subset["Block"].astype(str).str.casefold() == scope.block.casefold()]
    return subset.copy()


def scope_names(scopes: Iterable[MapScope]) -> list[str]:
    return [scope.name for scope in scopes]

