"""Spectral indices and the per-pixel feature stack."""
from __future__ import annotations

import numpy as np


def normalized_difference(a, b):
    """(a - b) / (a + b), NaN where the denominator is zero or an input is missing."""
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / den
    return np.where(den == 0, np.nan, out).astype("float32")


def spectral_indices(bands: dict) -> dict:
    """Indices computed from a dict of Sentinel-2 reflectance arrays.

    NDVI  = (B08 - B04) / (B08 + B04)      Rouse et al. (1974)
    NDMI  = (B8A - B11) / (B8A + B11)      Gao (1996) NDWI, often called NDMI
    NDWI  = (B03 - B08) / (B03 + B08)      McFeeters (1996)
    MNDWI = (B03 - B11) / (B03 + B11)      Xu (2006)
    """
    return {
        "NDVI": normalized_difference(bands["B08"], bands["B04"]),
        "NDMI": normalized_difference(bands["B8A"], bands["B11"]),
        "NDWI": normalized_difference(bands["B03"], bands["B08"]),
        "MNDWI": normalized_difference(bands["B03"], bands["B11"]),
    }


def build_feature_stack(seasons: dict[str, dict]) -> tuple[np.ndarray, list[str]]:
    """Stack bands + indices for each season, plus the NDVI seasonal amplitude.

    ``seasons`` maps a season label (e.g. "summer", "winter") to a dict of band arrays.
    Returns (array of shape [n_features, rows, cols], feature names).
    """
    layers, names = [], []
    ndvi = {}
    for season, bands in seasons.items():
        idx = spectral_indices(bands)
        ndvi[season] = idx["NDVI"]
        for name, arr in list(bands.items()) + list(idx.items()):
            layers.append(arr.astype("float32"))
            names.append(f"{season}_{name}")
    if len(ndvi) >= 2:
        first, second = list(ndvi)[:2]
        layers.append((ndvi[first] - ndvi[second]).astype("float32"))
        names.append(f"NDVI_{first}_minus_{second}")
    return np.stack(layers), names
