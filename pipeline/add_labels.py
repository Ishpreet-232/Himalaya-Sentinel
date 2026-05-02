# pipeline/add_labels.py
# Matches landslide inventory points to pipeline grid cells
# Run this for each district after downloading inventory from geodataindia.gov.in

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import sys, os

def add_labels(features_csv, geojson_path, output_csv=None):
    """
    features_csv : path to chamoli_features.csv from pipeline
    geojson_path : path to landslide points GeoJSON from geodataindia
    output_csv   : where to save labelled CSV (defaults to same folder)
    """
    print(f"Loading feature grid: {features_csv}")
    df = pd.read_csv(features_csv)
    df['landslide_label'] = 0  # default = no landslide

    print(f"Loading landslide inventory: {geojson_path}")
    gdf = gpd.read_file(geojson_path)

    # Filter to only points (ignore any polygon/line features)
    gdf = gdf[gdf.geometry.geom_type == 'Point'].copy()
    print(f"  Landslide points found: {len(gdf)}")

    # Get grid cell size from first two rows
    if len(df) > 1:
        lat_diff = abs(df['latitude'].iloc[0] - df['latitude'].iloc[1])
        px_deg = lat_diff if lat_diff > 0 else 0.00027  # ~30m default
    else:
        px_deg = 0.00027

    half = px_deg / 2
    matched = 0

    for _, point in gdf.iterrows():
        lon = point.geometry.x
        lat = point.geometry.y

        # Find all grid cells whose centre is within half a pixel of this point
        mask = (
            (df['latitude']  >= lat - half) & (df['latitude']  <= lat + half) &
            (df['longitude'] >= lon - half) & (df['longitude'] <= lon + half)
        )

        if mask.sum() > 0:
            df.loc[mask, 'landslide_label'] = 1
            matched += 1

    print(f"  Matched {matched}/{len(gdf)} points to grid cells")
    print(f"  Landslide cells (label=1): {df['landslide_label'].sum()}")
    print(f"  Total cells:               {len(df)}")
    print(f"  Landslide rate:            {df['landslide_label'].mean()*100:.2f}%")

    if output_csv is None:
        base = os.path.splitext(features_csv)[0]
        output_csv = base + "_labelled.csv"

    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved labelled dataset: {output_csv}")
    return df


if __name__ == "__main__":
    # -------------------------------------------------------
    # EDIT THESE TWO PATHS
    # -------------------------------------------------------
    FEATURES_CSV  = r"C:\Users\Ishpreet\Downloads\himalaya_risk\chamoli\chamoli_features.csv"
    GEOJSON_PATH  = r"C:\Users\Ishpreet\Downloads\your_landslide_points.geojson"
    # -------------------------------------------------------

    df = add_labels(FEATURES_CSV, GEOJSON_PATH)
    print("\nLabel distribution:")
    print(df['landslide_label'].value_counts())