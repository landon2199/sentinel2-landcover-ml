"""Data access: Sentinel-2 L2A composites and ESA WorldCover labels from Planetary Computer.

Everything is read straight from cloud-optimized GeoTIFFs through the Planetary Computer
STAC API (no account needed). Results are cached as GeoTIFFs in ``data/`` so reruns are fast.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# Sentinel-2 bands used as features (10 m bands are averaged to the working resolution).
S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Scene Classification Layer values kept as "clear": 2 dark area, 4 vegetation,
# 5 bare soil, 6 water, 7 unclassified. Dropped: 0 no data, 1 saturated/defective,
# 3 cloud shadow, 8/9 cloud medium/high probability, 10 thin cirrus, 11 snow/ice.
CLEAR_SCL = (2, 4, 5, 6, 7)

# Threads used to read COG windows (override with LANDCOVER_LOAD_THREADS).
LOAD_THREADS = int(os.environ.get("LANDCOVER_LOAD_THREADS", "8")) or None

# ESA WorldCover v200 classes with the official legend colors (Zanaga et al. 2022).
WORLDCOVER = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}


@dataclass
class Grid:
    """Pixel grid shared by every raster in a run."""

    crs: str
    transform: tuple  # affine coefficients (a, b, c, d, e, f)
    shape: tuple  # (rows, cols)
    resolution: float


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS 84 / UTM zone containing a point."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _catalog():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_sentinel2(bbox, datetime: str, max_cloud: float = 10.0, max_items: int = 12):
    """Least-cloudy Sentinel-2 L2A items intersecting ``bbox`` within ``datetime``."""
    items = list(
        _catalog()
        .search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=datetime,
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        .items()
    )
    items.sort(key=lambda it: it.properties["eo:cloud_cover"])
    return items[:max_items]


def search_worldcover(bbox, year: int):
    """WorldCover item(s) for ``year`` (2020 = v100, 2021 = v200)."""
    items = list(
        _catalog()
        .search(collections=["esa-worldcover"], bbox=bbox, datetime=f"{year}-01-01/{year}-12-31")
        .items()
    )
    if not items:
        raise RuntimeError(f"No ESA WorldCover product for {year}; available years are 2020 and 2021.")
    return items


def _boa_offset(item) -> float:
    """Processing baseline 04.00+ (from Jan 2022) adds a -1000 DN offset to L2A reflectance."""
    baseline = item.properties.get("s2:processing_baseline", "00.00")
    return -1000.0 if float(baseline) >= 4.0 else 0.0


def load_composite(items, bbox, epsg: int, resolution: float):
    """Cloud-masked median composite of ``items``.

    Returns (bands dict of float32 surface reflectance arrays, clear-observation count, Grid).
    """
    import odc.stac

    ds = odc.stac.load(
        items,
        bands=S2_BANDS + ["SCL"],
        bbox=bbox,
        crs=f"EPSG:{epsg}",
        resolution=resolution,
        groupby="solar_day",
        resampling={"SCL": "nearest", "*": "average"},
        dtype="uint16",
        nodata=0,
        pool=LOAD_THREADS,
        progress=None,
    )
    # One offset per acquisition day (tiles from the same day share a baseline).
    offsets = {}
    for it in items:
        offsets[np.datetime64(it.datetime.date())] = _boa_offset(it)
    day_offset = np.array([offsets.get(np.datetime64(t, "D"), 0.0) for t in ds.time.values.astype("datetime64[D]")])

    scl = ds["SCL"].values
    clear = np.isin(scl, CLEAR_SCL)
    bands = {}
    for b in S2_BANDS:
        dn = ds[b].values.astype("float32")
        valid = clear & (dn > 0)
        refl = (dn + day_offset[:, None, None].astype("float32")) / 10000.0
        refl[~valid] = np.nan
        with warnings.catch_warnings():  # all-NaN pixels (never clear) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            bands[b] = np.nanmedian(refl, axis=0).astype("float32")
    n_clear = clear.sum(axis=0).astype("int16")
    gb = ds.odc.geobox
    t = gb.affine
    grid = Grid(crs=f"EPSG:{epsg}", transform=(t.a, t.b, t.c, t.d, t.e, t.f), shape=tuple(gb.shape), resolution=resolution)
    return bands, n_clear, grid


def load_worldcover(items, grid: Grid) -> np.ndarray:
    """WorldCover map resampled (majority) onto ``grid``."""
    import odc.stac
    from affine import Affine
    from odc.geo.geobox import GeoBox

    gb = GeoBox(grid.shape, Affine(*grid.transform), grid.crs)
    ds = odc.stac.load(items, bands=["map"], geobox=gb, resampling="mode", dtype="uint8", nodata=0, pool=LOAD_THREADS)
    return ds["map"].values[0]


# --------------------------------------------------------------------------- caching
def save_stack(path: Path, arrays: np.ndarray, names: list[str], grid: Grid, nodata=np.nan):
    """Write a multi-band GeoTIFF (one band per feature/label) with band descriptions."""
    import rasterio
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = np.asarray(arrays)
    if arrays.ndim == 2:
        arrays = arrays[None]
    profile = dict(
        driver="GTiff",
        height=grid.shape[0],
        width=grid.shape[1],
        count=arrays.shape[0],
        dtype=arrays.dtype.name,
        crs=grid.crs,
        transform=Affine(*grid.transform),
        compress="deflate",
        tiled=True,
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arrays)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)


def read_stack(path: Path):
    """Inverse of :func:`save_stack`; returns (array, names, Grid)."""
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read()
        names = list(src.descriptions)
        t = src.transform
        grid = Grid(
            crs=src.crs.to_string(),
            transform=(t.a, t.b, t.c, t.d, t.e, t.f),
            shape=(src.height, src.width),
            resolution=abs(t.a),
        )
    return arr, names, grid


def configure_gdal():
    """Point GDAL at its data files when the conda env was not activated, and make HTTP reads
    time out and retry instead of stalling on a dropped connection."""
    for key, value in {"GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_MAX_RETRY": "5", "GDAL_HTTP_RETRY_DELAY": "2"}.items():
        os.environ.setdefault(key, value)
    if "GDAL_DATA" not in os.environ:
        import sys

        for cand in (Path(sys.prefix) / "Library" / "share" / "gdal", Path(sys.prefix) / "share" / "gdal"):
            if cand.exists():
                os.environ["GDAL_DATA"] = str(cand)
                break
