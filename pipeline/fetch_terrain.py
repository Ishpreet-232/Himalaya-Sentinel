# =============================================================
# pipeline/fetch_terrain.py
# Downloads DEM from OpenTopography and computes:
#   slope, aspect, ruggedness (TRI)
# =============================================================

import os
import requests
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import generic_filter
import warnings
warnings.filterwarnings("ignore")


def fetch_dem(south, north, west, east, api_key, dem_type="SRTMGL1", output_dir="."):
    """
    Download DEM GeoTIFF from OpenTopography API.
    Returns path to saved .tif file.
    """
    print(f"  [DEM] Fetching {dem_type} for bbox: S={south} N={north} W={west} E={east}")

    url = (
        f"https://portal.opentopography.org/API/globaldem"
        f"?demtype={dem_type}"
        f"&south={south}&north={north}&west={west}&east={east}"
        f"&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )

    response = requests.get(url, timeout=120)

    if response.status_code == 401:
        raise ValueError(
            "OpenTopography API key invalid or missing.\n"
            "Get a free key at: https://portal.opentopography.org/requestApiKey\n"
            "Then set OPENTOPO_API_KEY in config.py"
        )
    if response.status_code != 200:
        raise ConnectionError(f"OpenTopography API error: {response.status_code} — {response.text[:200]}")

    out_path = os.path.join(output_dir, f"dem_{south}_{north}_{west}_{east}.tif")
    with open(out_path, "wb") as f:
        f.write(response.content)

    # Verify it's a valid raster
    with rasterio.open(out_path) as ds:
        print(f"  [DEM] Downloaded: {ds.width}×{ds.height} pixels, CRS={ds.crs}")

    return out_path


def compute_slope(dem_array, cell_size_m=30):
    """
    Compute slope in degrees from DEM array.
    Uses Horn's method (same as QGIS/ArcGIS default).
    """
    # Gradient in x and y
    dy, dx = np.gradient(dem_array.astype(float), cell_size_m)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg


def compute_aspect(dem_array):
    """
    Compute aspect in degrees (0=North, 90=East, 180=South, 270=West).
    """
    dy, dx = np.gradient(dem_array.astype(float))
    aspect = np.degrees(np.arctan2(-dy, dx))
    # Convert to 0-360 compass bearing
    aspect = 90.0 - aspect
    aspect[aspect < 0] += 360
    aspect[aspect > 360] -= 360
    return aspect


def compute_tri(dem_array):
    """
    Terrain Ruggedness Index (TRI) — Riley et al. 1999
    Mean absolute difference from focal cell to its 8 neighbours.
    This is what QGIS 'Ruggedness Index' computes.
    """
    def tri_focal(window):
        center = window[4]  # middle of 3x3
        return np.sqrt(np.sum((window - center) ** 2))

    tri = generic_filter(dem_array.astype(float), tri_focal, size=3, mode='nearest')
    return tri


def process_dem(dem_path, output_dir="."):
    """
    Load DEM, compute slope/aspect/TRI, save each as a GeoTIFF.
    Returns dict of {layer_name: array} and the rasterio profile.
    """
    print(f"  [Terrain] Computing slope, aspect, ruggedness...")

    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(float)
        profile = ds.profile.copy()
        transform = ds.transform
        nodata = ds.nodata

        # Estimate cell size in metres from transform
        # transform.a = pixel width in degrees; ~111,000m per degree
        cell_size_m = abs(transform.a) * 111_000

    # Replace nodata with median
    if nodata is not None:
        dem[dem == nodata] = np.nan
    dem_filled = np.where(np.isnan(dem), np.nanmedian(dem), dem)

    slope   = compute_slope(dem_filled, cell_size_m)
    aspect  = compute_aspect(dem_filled)
    tri     = compute_tri(dem_filled)

    layers = {
        "dem":          dem_filled,
        "slope":        slope,
        "aspect":       aspect,
        "ruggedness":   tri
    }

    # Save each layer
    profile.update(dtype=rasterio.float32, count=1, nodata=None)
    base = os.path.splitext(dem_path)[0]

    for name, arr in layers.items():
        out_path = os.path.join(output_dir, f"{name}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr.astype(np.float32), 1)

    print(f"  [Terrain] Done. Slope max={slope.max():.1f}°, TRI max={tri.max():.1f}m")
    return layers, profile
