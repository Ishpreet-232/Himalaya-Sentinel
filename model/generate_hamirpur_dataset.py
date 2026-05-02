# =============================================================
# generate_hamirpur_dataset.py
#
# Fetches real terrain data for Hamirpur via the pipeline,
# then generates landslide pseudo-labels from:
#   1. Confirmed events from NASA COOLR / ISRO Bhuvan (if available)
#   2. Rule-based susceptibility thresholds as a fallback
#      (slope > 25°, bare soil, near river → high susceptibility)
#
# Output: ~/Downloads/hamirpur_dataset.xlsx
#         (same schema as chamoli_dataset.xlsx — ready for train_model.py)
#
# Run ONCE to generate the dataset, then add it to TRAINING_FILES
# in train_model.py and retrain.
#
# Usage:
#   cd E:\himalaya_assessment
#   python generate_hamirpur_dataset.py
# =============================================================

import os, sys, json, tempfile
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Hamirpur bounding box ──────────────────────────────────────
# Central Hamirpur district, HP — foothill terrain, clay-rich, monsoon-prone
SOUTH, NORTH = 31.60, 31.90
WEST,  EAST  = 76.30, 76.70
AREA_NAME    = "Hamirpur, Himachal Pradesh"

OUTPUT_PATH  = os.path.join(os.path.expanduser("~"), "Downloads",
                             "hamirpur_dataset.xlsx")

# ── Susceptibility thresholds for rule-based pseudo-labelling ──
# Foothills have lower slopes but high risk due to clay + water
# These are deliberately conservative (flag ~5-10% cells as positive)
LABEL_RULES = {
    "slope_min_deg":      18.0,   # lower than Chamoli gorges (was ~30°)
    "river_dist_max_m": 400.0,   # close to water courses
    "veg_bare_min":       0.35,   # moderate bare / degraded vegetation
    "ruggedness_min":     15.0,   # some local relief
}

# ── Confirmed event search via NASA COOLR (no auth needed) ────
COOLR_URL = ("https://maps.nccs.nasa.gov/arcgis/rest/services/"
             "NCCS_COOLR/MapServer/0/query"
             "?where=1%3D1&geometry={w}%2C{s}%2C{e}%2C{n}"
             "&geometryType=esriGeometryEnvelope&spatialRel=esriSpatialRelIntersects"
             "&outFields=*&returnGeometry=true&f=json")


def fetch_coolr_events(south, north, west, east):
    """Try to get confirmed landslide points from NASA COOLR."""
    try:
        url = COOLR_URL.format(s=south, n=north, w=west, e=east)
        resp = requests.get(url, timeout=20)
        data = resp.json()
        features = data.get("features", [])
        events = []
        for feat in features:
            geom = feat.get("geometry", {})
            if geom.get("x") and geom.get("y"):
                events.append({"lon": float(geom["x"]), "lat": float(geom["y"])})
        print(f"  [COOLR] Found {len(events)} confirmed events")
        return events
    except Exception as e:
        print(f"  [COOLR] Failed ({e}) — will use rule-based labels only")
        return []


def snap_events_to_grid(events, df):
    """
    For each confirmed event point, find the nearest grid cell
    and mark it as landslide_label=1.
    """
    if not events:
        return df
    lats = df['latitude'].values
    lons = df['longitude'].values
    for ev in events:
        dist = np.sqrt((lats - ev['lat'])**2 + (lons - ev['lon'])**2)
        idx  = int(np.argmin(dist))
        df.loc[idx, 'landslide_label'] = 1
    print(f"  [Labels] Snapped {len(events)} COOLR events to grid")
    return df


def rule_based_labels(df):
    """
    Rule-based pseudo-labelling for foothill terrain.
    Conservative: only flag cells with multiple converging risk factors.
    Returns Series of 0/1 labels.
    """
    rules = LABEL_RULES
    cond = (
        (df['slope_mean']       >= rules['slope_min_deg']) &
        (df['river_dist']       <= rules['river_dist_max_m']) &
        (1 - df['nvdi_mean']    >= rules['veg_bare_min']) &
        (df['ruggedness_mean']  >= rules['ruggedness_min'])
    )
    n_flagged = cond.sum()
    print(f"  [Labels] Rule-based: {n_flagged} cells flagged "
          f"({n_flagged/len(df)*100:.1f}%)")
    return cond.astype(int)


def main():
    print("=" * 60)
    print(f"Generating Hamirpur training dataset")
    print(f"  BBox: {SOUTH}°N–{NORTH}°N, {WEST}°E–{EAST}°E")
    print("=" * 60)

    from config import (OPENTOPO_API_KEY, COPERNICUS_CLIENT_ID,
                        COPERNICUS_CLIENT_SECRET, DEM_TYPE)
    from pipeline.fetch_terrain import fetch_dem, process_dem
    from pipeline.fetch_ndvi    import get_ndvi
    from pipeline.fetch_osm     import get_road_river_distances
    from scipy.ndimage          import zoom as scipy_zoom

    tmp = tempfile.mkdtemp()

    # 1 ── DEM
    print("\n[1/4] Fetching DEM...")
    dem_path = fetch_dem(SOUTH, NORTH, WEST, EAST,
                         api_key=OPENTOPO_API_KEY,
                         dem_type=DEM_TYPE,
                         output_dir=tmp)
    terrain, ref_profile = process_dem(dem_path, output_dir=tmp)
    h, w = int(terrain["dem"].shape[0]), int(terrain["dem"].shape[1])
    print(f"  DEM shape: {h}×{w}")

    # 2 ── NDVI
    print("\n[2/4] Fetching NDVI...")
    ndvi_arr, _ = get_ndvi(SOUTH, NORTH, WEST, EAST,
                            client_id=COPERNICUS_CLIENT_ID,
                            client_secret=COPERNICUS_CLIENT_SECRET,
                            output_dir=tmp,
                            dem_array=terrain["dem"])

    # 3 ── OSM
    print("\n[3/4] Fetching roads/rivers...")
    road_dist, river_dist = get_road_river_distances(
        SOUTH, NORTH, WEST, EAST,
        profile=ref_profile, output_dir=tmp)

    # 4 ── Resize to DEM shape
    def resize(arr):
        a = np.array(arr, dtype=np.float64)
        ah, aw = a.shape
        if ah == h and aw == w:
            return a
        return scipy_zoom(a, (h/ah, w/aw), order=1)

    ndvi_arr   = resize(ndvi_arr)
    road_dist  = resize(road_dist)
    river_dist = resize(river_dist)

    # 5 ── Downsample to ~100m grid (step to keep ≤200k cells)
    total = h * w
    step  = int(np.ceil(np.sqrt(total / 200_000))) if total > 200_000 else 1
    sl    = np.s_[::step, ::step]

    slope_s  = terrain["slope"][sl].ravel()
    aspect_s = terrain["aspect"][sl].ravel()
    rugged_s = terrain["ruggedness"][sl].ravel()
    ndvi_s   = ndvi_arr[sl].ravel()
    road_s   = road_dist[sl].ravel()
    river_s  = river_dist[sl].ravel()

    hs, ws   = terrain["slope"][sl].shape
    t        = ref_profile["transform"]
    px       = float(abs(t.a)) * step

    rows_idx = np.arange(hs).reshape(-1,1) * np.ones((1,ws))
    cols_idx = np.ones((hs,1)) * np.arange(ws).reshape(1,-1)
    lats = float(t.f) - (rows_idx + 0.5) * px
    lons = float(t.c) + (cols_idx + 0.5) * px

    # Also make row/col index columns (same convention as chamoli_dataset)
    row_idx_flat = rows_idx.ravel().astype(int)
    col_idx_flat = cols_idx.ravel().astype(int)

    df = pd.DataFrame({
        'row_index':        row_idx_flat,
        'col_index':        col_idx_flat,
        'latitude':         lats.ravel(),
        'longitude':        lons.ravel(),
        'slope_mean':       slope_s,
        'aspect_mean':      aspect_s,
        'ruggedness_mean':  rugged_s,
        'nvdi_mean':        ndvi_s,
        'road_dist':        road_s,
        'river_dist':       river_s,
        'landslide_label':  0,          # will be filled below
    })

    # ── Clean ──────────────────────────────────────────────────
    for col in ['slope_mean','aspect_mean','ruggedness_mean',
                'nvdi_mean','road_dist','river_dist']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    df = df[df['slope_mean'] >= 0].reset_index(drop=True)
    print(f"\n  Grid: {len(df):,} cells (step={step})")

    # 6 ── Labels: COOLR first, then rule-based ─────────────────
    print("\n[4/4] Generating labels...")
    events = fetch_coolr_events(SOUTH, NORTH, WEST, EAST)
    if events:
        df = snap_events_to_grid(events, df)

    # Supplement with rule-based (don't overwrite confirmed events)
    rule_labels = rule_based_labels(df)
    df['landslide_label'] = np.maximum(
        df['landslide_label'].values,
        rule_labels.values
    )

    n_pos = int(df['landslide_label'].sum())
    print(f"\n  Final labels: {n_pos} positives / {len(df)} total "
          f"({n_pos/len(df)*100:.1f}%)")

    if n_pos < 10:
        print("\n  [WARN] Very few positive labels — "
              "consider relaxing LABEL_RULES thresholds or collecting "
              "manual inventory for Hamirpur.")

    # 7 ── Save ─────────────────────────────────────────────────
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved: {OUTPUT_PATH}")
    print(f"  Rows: {len(df):,} | Positives: {n_pos:,}")
    print(f"\nNext steps:")
    print(f"  1. Open {OUTPUT_PATH} in QGIS or Excel and verify labels")
    print(f"  2. Uncomment the Hamirpur entry in TRAINING_FILES in train_model.py")
    print(f"  3. Run: python model/train_model.py")


if __name__ == "__main__":
    main()