# =============================================================
# pipeline/fetch_osm.py — Fixed v2
# Fixes: 406 errors from Overpass API (wrong Content-Type / query format)
# =============================================================

import numpy as np
import requests
import time
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt

MAX_RETRIES = 4
RETRY_DELAY = 6   # seconds between retries

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

def _build_query(south, north, west, east):
    """Build Overpass QL query for roads and rivers."""
    bbox = f"{south},{west},{north},{east}"
    return f"""
[out:json][timeout:60];
(
  way["highway"~"^(primary|secondary|tertiary|residential|trunk|motorway|unclassified|road)$"]({bbox});
  way["waterway"~"^(river|stream|canal|drain)$"]({bbox});
  relation["waterway"="river"]({bbox});
);
out geom;
"""

def _fetch_overpass(query):
    """Try each Overpass endpoint with retries. Returns parsed JSON or raises."""
    for attempt in range(1, MAX_RETRIES + 1):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                print(f"  [OSM] Attempt {attempt} → {endpoint.split('/')[2]}")
                # KEY FIX: Use POST with data= (not json=), correct Content-Type
                resp = requests.post(
                    endpoint,
                    data={"data": query},          # form-encoded, not JSON
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "HimalayanRiskSystem/2.0 (research project)",
                    },
                    timeout=90,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"  [OSM] Got {len(data.get('elements', []))} elements")
                    return data
                elif resp.status_code == 429:
                    print(f"  [OSM] Rate limited (429), waiting {RETRY_DELAY*2}s...")
                    time.sleep(RETRY_DELAY * 2)
                elif resp.status_code == 504:
                    print(f"  [OSM] Gateway timeout (504), retrying...")
                    time.sleep(RETRY_DELAY)
                elif resp.status_code == 406:
                    print(f"  [OSM] 406 Not Acceptable — trying next endpoint...")
                    # 406 = query format rejected by this mirror, try another
                    continue
                else:
                    print(f"  [OSM] HTTP {resp.status_code}, retrying...")
                    time.sleep(RETRY_DELAY)
            except requests.Timeout:
                print(f"  [OSM] Timeout on {endpoint.split('/')[2]}, trying next...")
            except Exception as e:
                print(f"  [OSM] Error: {e}, retrying...")
                time.sleep(RETRY_DELAY)

        time.sleep(RETRY_DELAY)  # wait before next full round

    raise RuntimeError("Overpass API unavailable after all attempts and endpoints.")


def _coords_to_mask(elements, south, north, west, east, h, w, tag_filter):
    """
    Rasterise LineString elements onto a boolean mask of shape (h, w).
    tag_filter: function(element) → True if this element should be included.
    """
    mask = np.zeros((h, w), dtype=bool)
    lat_range = north - south
    lon_range = east  - west
    if lat_range <= 0 or lon_range <= 0:
        return mask

    for el in elements:
        if el.get("type") != "way":
            continue
        if not tag_filter(el.get("tags", {})):
            continue
        geom = el.get("geometry", [])
        if len(geom) < 2:
            continue
        # Walk the line segments and mark pixels
        for i in range(len(geom) - 1):
            lat1, lon1 = geom[i]["lat"], geom[i]["lon"]
            lat2, lon2 = geom[i+1]["lat"], geom[i+1]["lon"]
            # Sample along segment (simple Bresenham-like approach)
            dist = max(
                abs(lat2 - lat1) / lat_range * h,
                abs(lon2 - lon1) / lon_range * w,
            )
            n_samples = max(int(dist * 3), 2)
            for t in np.linspace(0, 1, n_samples):
                lat = lat1 + t * (lat2 - lat1)
                lon = lon1 + t * (lon2 - lon1)
                row = int((north - lat) / lat_range * h)
                col = int((lon  - west) / lon_range * w)
                if 0 <= row < h and 0 <= col < w:
                    mask[row, col] = True
    return mask


def get_road_river_distances(south, north, west, east,
                              profile=None, output_dir=None,
                              grid_h=None, grid_w=None):
    """
    Returns (road_dist_m, river_dist_m) as 2D float32 arrays.

    grid_h / grid_w: output raster size.
    If profile is provided (rasterio profile dict), use its height/width.
    Falls back to a 300×300 default grid.
    """
    # ── Determine output grid size ──────────────────────────────
    if profile is not None:
        h = int(profile.get("height", 300))
        w = int(profile.get("width",  300))
    elif grid_h is not None and grid_w is not None:
        h, w = int(grid_h), int(grid_w)
    else:
        # Sensible default: ~100m resolution
        h = max(10, int((north - south) * 111_000 / 100))
        w = max(10, int((east  - west)  * 111_000 * np.cos(np.radians((north+south)/2)) / 100))
        h, w = min(h, 1000), min(w, 1000)

    print(f"  [OSM] Fetching roads from OpenStreetMap... (grid {h}×{w})")

    # ── Fetch from Overpass ─────────────────────────────────────
    query   = _build_query(south, north, west, east)
    data    = _fetch_overpass(query)
    elements = data.get("elements", [])

    # ── Separate roads vs rivers ────────────────────────────────
    def is_road(tags):
        return "highway" in tags

    def is_river(tags):
        return "waterway" in tags

    road_mask  = _coords_to_mask(elements, south, north, west, east, h, w, is_road)
    river_mask = _coords_to_mask(elements, south, north, west, east, h, w, is_river)

    n_roads  = road_mask.sum()
    n_rivers = river_mask.sum()
    print(f"  [OSM] Road pixels: {n_roads}, River pixels: {n_rivers}")

    # ── Fallback if no features found ──────────────────────────
    # (remote / data-sparse areas: use centre-of-area as a single point)
    if n_roads == 0:
        print("  [OSM] No roads found — using centre fallback.")
        road_mask[h // 2, w // 2] = True
    if n_rivers == 0:
        print("  [OSM] No rivers found — using centre fallback.")
        river_mask[h // 2, w // 2] = True

    # ── Euclidean distance transform → pixel distance ───────────
    road_dist_px  = distance_transform_edt(~road_mask).astype(np.float32)
    river_dist_px = distance_transform_edt(~river_mask).astype(np.float32)

    # ── Convert pixels → metres ─────────────────────────────────
    lat_m = (north - south) * 111_000 / h                       # m per pixel (lat)
    lon_m = (east  - west)  * 111_000 * np.cos(np.radians((north+south)/2)) / w
    px_m  = (lat_m + lon_m) / 2.0

    road_dist_m  = (road_dist_px  * px_m).astype(np.float32)
    river_dist_m = (river_dist_px * px_m).astype(np.float32)

    print(f"  [OSM] Done. Road dist max={road_dist_m.max():.0f}m, "
          f"River dist max={river_dist_m.max():.0f}m")
    return road_dist_m, river_dist_m