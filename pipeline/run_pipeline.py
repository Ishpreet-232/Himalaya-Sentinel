# =============================================================
# pipeline/run_pipeline.py
# Main entry point — given a bounding box, runs full pipeline:
#   DEM → terrain features → NDVI → road/river distances → grid CSV
#
# Usage:
#   python run_pipeline.py --area chamoli
#   python run_pipeline.py --bbox 30.2 30.6 79.1 79.7 --name my_area
# =============================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
import warnings
warnings.filterwarnings("ignore")

# Add parent dir to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    OPENTOPO_API_KEY, COPERNICUS_CLIENT_ID, COPERNICUS_CLIENT_SECRET,
    DEM_TYPE, OUTPUT_DIR, STUDY_AREAS
)
from pipeline.fetch_terrain import fetch_dem, process_dem
from pipeline.fetch_ndvi    import get_ndvi
from pipeline.fetch_osm     import get_road_river_distances


def resample_to_reference(src_array, src_profile, ref_profile):
    """
    Resample src_array to match ref_profile's shape and transform.
    Used to align NDVI (different resolution) to DEM grid.
    """
    import rasterio.transform as rtransform

    ref_height = ref_profile["height"]
    ref_width  = ref_profile["width"]

    if src_array.shape == (ref_height, ref_width):
        return src_array  # already aligned

    # Simple bilinear interpolation via zoom
    from scipy.ndimage import zoom
    zoom_y = ref_height / src_array.shape[0]
    zoom_x = ref_width  / src_array.shape[1]
    resampled = zoom(src_array, (zoom_y, zoom_x), order=1)  # order=1 = bilinear
    return resampled


def normalize(arr):
    """Min-max normalize to [0, 1]."""
    rng = arr.max() - arr.min()
    if rng < 1e-9:
        return np.zeros_like(arr)
    return (arr - arr.min()) / rng


def run_pipeline(south, north, west, east, area_name="region"):
    """
    Full automated pipeline for one bounding box.
    Returns a DataFrame with one row per grid cell, ready for model.
    """
    print("\n" + "=" * 60)
    print(f"HIMALAYAN RISK PIPELINE — {area_name.upper()}")
    print(f"Bbox: S={south} N={north} W={west} E={east}")
    print("=" * 60)

    # Create output folder for this area
    area_dir = os.path.join(OUTPUT_DIR, area_name.replace(" ", "_"))
    os.makedirs(area_dir, exist_ok=True)

    # ----------------------------------------------------------
    # STEP 1: Fetch DEM and compute terrain features
    # ----------------------------------------------------------
    print("\n[STEP 1/4] Terrain")

    if OPENTOPO_API_KEY == "YOUR_OPENTOPO_KEY_HERE":
        print("  ⚠  No OpenTopography API key set in config.py")
        print("     Get free key at: https://portal.opentopography.org/requestApiKey")
        print("     For now, loading demo DEM if available...")

        # Check if DEM already downloaded
        demo_path = os.path.join(area_dir, "dem.tif")
        if not os.path.exists(demo_path):
            raise ValueError(
                "Set OPENTOPO_API_KEY in config.py before running pipeline.\n"
                "Free registration: https://portal.opentopography.org"
            )
        dem_path = demo_path
    else:
        dem_path = fetch_dem(
            south, north, west, east,
            api_key=OPENTOPO_API_KEY,
            dem_type=DEM_TYPE,
            output_dir=area_dir
        )

    terrain_layers, ref_profile = process_dem(dem_path, output_dir=area_dir)

    dem_arr      = terrain_layers["dem"]
    slope_arr    = terrain_layers["slope"]
    aspect_arr   = terrain_layers["aspect"]
    rugged_arr   = terrain_layers["ruggedness"]
    h, w         = dem_arr.shape
    ref_profile.update(height=h, width=w)

    # ----------------------------------------------------------
    # STEP 2: Fetch NDVI
    # ----------------------------------------------------------
    print("\n[STEP 2/4] Vegetation (NDVI)")

    ndvi_arr, ndvi_path = get_ndvi(
        south, north, west, east,
        client_id=COPERNICUS_CLIENT_ID,
        client_secret=COPERNICUS_CLIENT_SECRET,
        output_dir=area_dir,
        dem_array=dem_arr
    )

    # Resample NDVI to match DEM grid if needed
    if ndvi_arr.shape != (h, w):
        print(f"  [NDVI] Resampling {ndvi_arr.shape} → {(h, w)}")
        ndvi_arr = resample_to_reference(ndvi_arr, {}, ref_profile)

    # ----------------------------------------------------------
    # STEP 3: Fetch road and river distances from OSM
    # ----------------------------------------------------------
    print("\n[STEP 3/4] Road & River distances (OpenStreetMap)")

    road_dist, river_dist = get_road_river_distances(
        south, north, west, east,
        profile=ref_profile,
        output_dir=area_dir
    )

    # Resample if needed
    if road_dist.shape  != (h, w): road_dist  = resample_to_reference(road_dist,  {}, ref_profile)
    if river_dist.shape != (h, w): river_dist = resample_to_reference(river_dist, {}, ref_profile)

    # ----------------------------------------------------------
    # STEP 4: Build grid DataFrame (same format as chamoli_dataset)
    # ----------------------------------------------------------
    print("\n[STEP 4/4] Building feature grid...")

    transform = ref_profile["transform"]
    west_edge  = transform.c
    north_edge = transform.f
    px_deg     = abs(transform.a)

    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    lat_centres = north_edge - (rows + 0.5) * px_deg
    lon_centres = west_edge  + (cols + 0.5) * px_deg

    df = pd.DataFrame({
        "row_index":       rows.ravel(),
        "col_index":       cols.ravel(),
        "latitude":        lat_centres.ravel(),
        "longitude":       lon_centres.ravel(),
        "elevation":       dem_arr.ravel(),
        "slope_mean":      slope_arr.ravel(),
        "aspect_mean":     aspect_arr.ravel(),
        "ruggedness_mean": rugged_arr.ravel(),
        "nvdi_mean":       ndvi_arr.ravel(),
        "road_dist":       road_dist.ravel(),
        "river_dist":      river_dist.ravel(),
        # landslide_label left as NaN — to be added from NGDR/geodataindia data
        "landslide_label": np.nan,
        "area_name":       area_name,
        "bbox_south":      south,
        "bbox_north":      north,
        "bbox_west":       west,
        "bbox_east":       east,
    })

    # Basic sanity filter
    df = df[df["slope_mean"] >= 0]
    df = df[df["ruggedness_mean"] >= 0]
    df = df[df["nvdi_mean"].between(-1, 1)]

    # Save
    csv_path = os.path.join(area_dir, f"{area_name}_features.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n✅ Pipeline complete!")
    print(f"   Area:       {area_name}")
    print(f"   Grid cells: {len(df):,}")
    print(f"   Shape:      {h} rows × {w} cols")
    print(f"   Saved to:   {csv_path}")
    print(f"\n   Feature summary:")
    print(f"     Slope:        {df['slope_mean'].mean():.1f}° mean, {df['slope_mean'].max():.1f}° max")
    print(f"     Ruggedness:   {df['ruggedness_mean'].mean():.1f} mean")
    print(f"     NDVI:         {df['nvdi_mean'].mean():.3f} mean")
    print(f"     Road dist:    {df['road_dist'].mean():.0f}m mean")
    print(f"     River dist:   {df['river_dist'].mean():.0f}m mean")

    print(f"\n   Next step: Add landslide_label column from geodataindia.gov.in")
    print(f"   Then merge with Chamoli data and retrain the model.")

    return df, csv_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Himalayan Risk Pipeline")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--area", choices=list(STUDY_AREAS.keys()),
                       help="Named study area from config.py")
    group.add_argument("--bbox", nargs=4, type=float,
                       metavar=("SOUTH", "NORTH", "WEST", "EAST"),
                       help="Custom bounding box in decimal degrees")

    parser.add_argument("--name", type=str, default=None,
                        help="Name for custom bbox area")

    args = parser.parse_args()

    if args.area:
        area_cfg = STUDY_AREAS[args.area]
        south, north, west, east = area_cfg["bbox"]
        name = args.area
    else:
        south, north, west, east = args.bbox
        name = args.name or f"custom_{south}_{west}"

    df, path = run_pipeline(south, north, west, east, area_name=name)
