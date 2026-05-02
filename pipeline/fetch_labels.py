# =============================================================
# pipeline/fetch_labels.py
#
# Automatically fetches landslide inventory points for any
# bounding box using TWO sources:
#
#   Source 1 — NASA COOLR REST API (no key, works anywhere)
#              Global landslide catalog 2007–present
#              URL: gis.earthdata.nasa.gov (ArcGIS FeatureServer)
#
#   Source 2 — Local GeoJSON (your geodataindia/NGDR download)
#              Higher quality GSI field data for India
#              Used when file path is provided
#
# Both sources are merged and deduplicated.
# Points are matched to grid cells → landslide_label column added.
# =============================================================

import os
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# NASA COOLR ArcGIS REST API endpoints
# No authentication required — completely public
# -------------------------------------------------------------------
COOLR_REPORTS_URL = (
    "https://gis.earthdata.nasa.gov/gis05/rest/services/"
    "Landslides/COOLR_Reports_Points/MapServer/0/query"
)
COOLR_EVENTS_URL = (
    "https://gis.earthdata.nasa.gov/gis05/rest/services/"
    "Landslides/COOLR_Events_Points/MapServer/0/query"
)


def fetch_coolr_points(south, north, west, east, max_records=2000):
    """
    Query NASA COOLR REST API for landslide points within bbox.
    Returns GeoDataFrame with point geometries, or None if failed.

    Uses ArcGIS REST spatial query — no API key needed.
    MaxRecordCount per request is 2000, so we paginate if needed.
    """
    print("  [COOLR] Querying NASA COOLR REST API...")

    # ArcGIS spatial filter — envelope in WGS84
    geometry_filter = f"{west},{south},{east},{north}"

    params = {
        "where":           "1=1",
        "geometry":        geometry_filter,
        "geometryType":    "esriGeometryEnvelope",
        "inSR":            "4326",
        "spatialRel":      "esriSpatialRelIntersects",
        "outFields":       "event_id,event_date,landslide_category,country_name,loc_accuracy",
        "returnGeometry":  "true",
        "outSR":           "4326",
        "f":               "geojson",
        "resultRecordCount": max_records,
    }

    all_points = []

    for url, source_name in [
        (COOLR_REPORTS_URL, "COOLR Reports"),
        (COOLR_EVENTS_URL,  "COOLR Events"),
    ]:
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                features = data.get("features", [])
                if features:
                    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                    gdf["source"] = source_name
                    all_points.append(gdf)
                    print(f"  [COOLR] {source_name}: {len(gdf)} points found")
                else:
                    print(f"  [COOLR] {source_name}: 0 points in this bbox")
            else:
                print(f"  [COOLR] {source_name}: HTTP {r.status_code} — skipping")
        except Exception as e:
            print(f"  [COOLR] {source_name}: failed ({e}) — skipping")
        time.sleep(1)  # be polite to NASA servers

    if not all_points:
        print("  [COOLR] No points retrieved from NASA COOLR")
        return None

    combined = pd.concat(all_points, ignore_index=True)
    print(f"  [COOLR] Total COOLR points: {len(combined)}")
    return combined


def load_local_geojson(geojson_path, south, north, west, east):
    """
    Load local GeoJSON (from geodataindia / NGDR) and clip to bbox.
    Returns GeoDataFrame or None.
    """
    if not geojson_path or not os.path.exists(geojson_path):
        return None

    print(f"  [Local] Loading local GeoJSON: {os.path.basename(geojson_path)}")
    gdf = gpd.read_file(geojson_path)

    # Keep only points
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()

    # Clip to bbox
    gdf = gdf.cx[west:east, south:north]
    gdf["source"] = "GSI_NGDR"

    print(f"  [Local] Points in bbox: {len(gdf)}")
    return gdf if len(gdf) > 0 else None


def merge_inventory_sources(*gdfs):
    """
    Merge multiple GeoDataFrames of landslide points.
    Deduplicates by proximity (points within 0.001° considered duplicate).
    Prioritises GSI_NGDR over COOLR when duplicates exist.
    """
    valid = [g for g in gdfs if g is not None and len(g) > 0]
    if not valid:
        return None

    merged = pd.concat(valid, ignore_index=True)
    merged = merged[merged.geometry.notna()]

    print(f"  [Merge] Total before dedup: {len(merged)}")

    # Simple grid dedup: round coords to 0.001° (~110m) and keep first
    merged["lat_r"] = merged.geometry.y.round(3)
    merged["lon_r"] = merged.geometry.x.round(3)

    # Sort so GSI data is prioritised (kept first in dedup)
    source_order = {"GSI_NGDR": 0, "COOLR Reports": 1, "COOLR Events": 2}
    merged["src_order"] = merged["source"].map(source_order).fillna(3)
    merged = merged.sort_values("src_order")

    merged = merged.drop_duplicates(subset=["lat_r", "lon_r"], keep="first")
    merged = merged.drop(columns=["lat_r", "lon_r", "src_order"])

    print(f"  [Merge] After dedup: {len(merged)} unique landslide points")
    return merged


def match_points_to_grid(df_grid, gdf_points, px_deg=None):
    """
    For each landslide point, find which grid cell it falls in.
    Sets landslide_label = 1 for that cell.

    df_grid   : pipeline feature DataFrame with latitude/longitude columns
    gdf_points: GeoDataFrame of landslide points
    px_deg    : grid cell size in degrees (auto-detected if None)
    """
    df_grid = df_grid.copy()
    df_grid["landslide_label"] = 0

    if gdf_points is None or len(gdf_points) == 0:
        print("  [Labels] No landslide points to match — all labels set to 0")
        return df_grid

    # Auto-detect pixel size from grid spacing
    if px_deg is None:
        lats = df_grid["latitude"].sort_values().unique()
        if len(lats) > 1:
            px_deg = abs(lats[1] - lats[0])
        else:
            px_deg = 0.00027  # ~30m default

    half = px_deg / 2
    matched = 0

    for _, pt in gdf_points.iterrows():
        lon = pt.geometry.x
        lat = pt.geometry.y

        mask = (
            (df_grid["latitude"]  >= lat - half) &
            (df_grid["latitude"]  <= lat + half) &
            (df_grid["longitude"] >= lon - half) &
            (df_grid["longitude"] <= lon + half)
        )

        if mask.sum() > 0:
            df_grid.loc[mask, "landslide_label"] = 1
            matched += 1

    total_pts = len(gdf_points)
    label_cells = df_grid["landslide_label"].sum()

    print(f"\n  [Labels] Points matched to grid: {matched}/{total_pts}")
    print(f"  [Labels] Grid cells labelled = 1: {label_cells}")
    print(f"  [Labels] Total grid cells:         {len(df_grid)}")
    print(f"  [Labels] Landslide rate:           {label_cells/len(df_grid)*100:.2f}%")

    if matched < total_pts * 0.5:
        print(f"\n  ⚠  WARNING: Only {matched}/{total_pts} points matched.")
        print("     Possible reasons:")
        print("     1. Points are outside the pipeline bbox")
        print("     2. Grid resolution doesn't align with point density")
        print("     3. CRS mismatch in local GeoJSON")

    return df_grid


# -------------------------------------------------------------------
# MAIN ENTRY POINT
# -------------------------------------------------------------------

def add_landslide_labels(
    features_csv,
    south, north, west, east,
    local_geojson=None,
    output_csv=None
):
    """
    Full automated labelling pipeline.

    features_csv  : path to *_features.csv from run_pipeline.py
    south/north/west/east : bounding box (same as pipeline)
    local_geojson : path to local GeoJSON if available (optional)
    output_csv    : output path (defaults to *_labelled.csv)

    Returns labelled DataFrame.
    """
    print("\n" + "=" * 60)
    print("AUTOMATED LABEL FETCHER")
    print("=" * 60)

    # Load feature grid
    print(f"\nLoading feature grid: {os.path.basename(features_csv)}")
    df = pd.read_csv(features_csv)
    print(f"Grid cells: {len(df):,}")

    # Source 1: NASA COOLR (always attempted — no key needed)
    coolr_gdf = fetch_coolr_points(south, north, west, east)

    # Source 2: Local GeoJSON (GSI/NGDR — better quality for India)
    local_gdf = load_local_geojson(local_geojson, south, north, west, east)

    # Merge and dedup
    print("\n  Merging sources...")
    all_points = merge_inventory_sources(local_gdf, coolr_gdf)

    if all_points is not None:
        # Show source breakdown
        print("\n  Source breakdown:")
        print(all_points["source"].value_counts().to_string())

    # Match to grid
    print("\n  Matching points to grid cells...")
    df_labelled = match_points_to_grid(df, all_points)

    # Save
    if output_csv is None:
        base = os.path.splitext(features_csv)[0]
        output_csv = base + "_labelled.csv"

    df_labelled.to_csv(output_csv, index=False)
    print(f"\n✅ Labelled dataset saved: {output_csv}")

    return df_labelled


# -------------------------------------------------------------------
# CLI — run directly for any area
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import STUDY_AREAS, OUTPUT_DIR

    import argparse
    parser = argparse.ArgumentParser(description="Add landslide labels to pipeline output")
    parser.add_argument("--area", required=True,
                        choices=list(STUDY_AREAS.keys()),
                        help="Study area name")
    parser.add_argument("--geojson", default=None,
                        help="Path to local GeoJSON from geodataindia/NGDR (optional)")
    args = parser.parse_args()

    area_cfg = STUDY_AREAS[args.area]
    south, north, west, east = area_cfg["bbox"]

    features_csv = os.path.join(
        OUTPUT_DIR, args.area,
        f"{args.area}_features.csv"
    )

    if not os.path.exists(features_csv):
        print(f"ERROR: Features CSV not found: {features_csv}")
        print(f"Run pipeline first: python pipeline/run_pipeline.py --area {args.area}")
        sys.exit(1)

    df = add_landslide_labels(
        features_csv=features_csv,
        south=south, north=north, west=west, east=east,
        local_geojson=args.geojson,
    )

    print("\nFinal label distribution:")
    print(df["landslide_label"].value_counts())