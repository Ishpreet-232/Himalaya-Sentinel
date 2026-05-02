# =============================================================
# api/main.py — FastAPI Backend v3
# Run: python -m uvicorn api.main:app --reload --port 8000
# =============================================================

import os, sys, json, pickle, time, tempfile
import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =============================================================
# LOAD MODEL ONCE AT STARTUP
# =============================================================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")
FRONT_DIR = os.path.join(BASE_DIR, "frontend")

print("Loading model...")
with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(MODEL_DIR, "shap_explainer.pkl"), "rb") as f:
    EXPLAINER = pickle.load(f)
with open(os.path.join(MODEL_DIR, "model_info.json"), "r") as f:
    MODEL_INFO = json.load(f)

FEATURES     = MODEL_INFO["features"]
RAW_FEATURES = MODEL_INFO["raw_features"]
THRESHOLD    = MODEL_INFO["best_threshold"]
RISK_WEIGHTS = MODEL_INFO["risk_weights"]
print(f"Model ready. AUC={MODEL_INFO['auc_roc']}")
# =============================================================
# APP
# =============================================================
app = FastAPI(title="Himalayan Risk Intelligence API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

if os.path.exists(FRONT_DIR):
    app.mount("/static", StaticFiles(directory=FRONT_DIR), name="static")

# =============================================================
# REQUEST MODEL
# =============================================================
class AnalyseRequest(BaseModel):
    south: float
    north: float
    west:  float
    east:  float
    area_name: Optional[str] = "Selected Region"

# =============================================================
# FEATURE ENGINEERING
# =============================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same transformations as training. Must match train_model.py exactly."""
    df = df.copy()

    def norm(arr):
        a = np.array(arr, dtype=np.float64)
        rng = a.max() - a.min()
        if rng < 1e-9:
            return np.zeros_like(a)
        return (a - a.min()) / rng

    df['slope_sin']      = np.sin(np.radians(df['slope_mean'].values))
    df['aspect_ns']      = np.cos(np.radians(df['aspect_mean'].values))
    df['rugged_n']       = norm(df['ruggedness_mean'].values)
    df['veg_bare']       = (1 - df['nvdi_mean'].values.clip(0, 1)).astype(np.float64)
    df['slope_x_rugged'] = df['slope_sin'].values * df['rugged_n'].values
    df['slope_x_bare']   = df['slope_sin'].values * df['veg_bare'].values
    df['road_dist_n']    = norm(df['road_dist'].values)
    df['river_dist_n']   = norm(df['river_dist'].values)

    return df.reset_index(drop=True)

# =============================================================
# RISK COMPUTATION
# =============================================================
def compute_risks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5 risk components from engineered features."""
    df = df.copy().reset_index(drop=True)

    def rn(arr):
        a = np.array(arr, dtype=np.float64)
        lo, hi = a.min(), a.max()
        if (hi - lo) < 1e-9:
            return np.zeros(len(a))
        return (a - lo) / (hi - lo)

    # ── Landslide susceptibility from ML model ──
    X = np.array(df[FEATURES].values, dtype=np.float64)
    probs = MODEL.predict_proba(X)[:, 1]
    df['landslide_risk'] = rn(probs)

    # ── Flood risk: low slope + close to river ──
    slope_safe = np.where(df['slope_mean'].values > 0.5,
                          df['slope_mean'].values, 0.5)
    df['flood_risk'] = rn(
        (1.0 / (df['river_dist'].values + 1.0)) * 0.7 +
        (1.0 / slope_safe) * 0.3
    )

    # ── Soil erosion: steep + bare ──
    df['erosion_risk'] = rn(
        df['veg_bare'].values * 0.6 +
        df['slope_sin'].values * 0.4
    )

    # ── Accessibility: far from road = hard to reach ──
    df['accessibility_risk'] = rn(df['road_dist'].values)

    # ── Terrain stability: steep + rough + bare ──
    df['terrain_stability'] = rn(
        df['slope_sin'].values * 0.40 +
        df['rugged_n'].values  * 0.35 +
        df['veg_bare'].values  * 0.25
    )

    # ── Combined weighted risk ──
    df['combined_risk'] = rn(
        RISK_WEIGHTS['landslide']     * df['landslide_risk'].values +
        RISK_WEIGHTS['flood']         * df['flood_risk'].values +
        RISK_WEIGHTS['erosion']       * df['erosion_risk'].values +
        RISK_WEIGHTS['accessibility'] * df['accessibility_risk'].values +
        RISK_WEIGHTS['terrain']       * df['terrain_stability'].values
    )

    # ── Risk level ──
    levels = np.where(df['combined_risk'].values < 0.30, 'Low',
              np.where(df['combined_risk'].values < 0.60, 'Medium', 'High'))
    df['risk_level'] = levels

    return df

# =============================================================
# SHAP EXPLANATION
# =============================================================
def get_shap(df: pd.DataFrame):
    try:
        X = df[FEATURES].values.astype(np.float64)
        sample = X[:min(200, len(X))]

        shap_values = EXPLAINER(sample)

        # Extract values safely
        vals = shap_values.values  # this is always numpy

        # Ensure 2D
        if vals.ndim == 3:
            # (samples, features, classes)
            vals = vals[:, :, 1]

        if vals.ndim == 1:
            vals = vals.reshape(1, -1)

        # ----------------------------
        # COMPUTE IMPORTANCE
        # ----------------------------
        mean_abs = np.abs(vals).mean(axis=0)
        mean_sign = vals.mean(axis=0)

        sorted_idx = np.argsort(mean_abs)[::-1]

        LABELS = {
            'slope_sin': 'Slope Steepness',
            'aspect_ns': 'North-Facing Aspect',
            'rugged_n': 'Terrain Ruggedness',
            'veg_bare': 'Bare Vegetation',
            'slope_x_rugged': 'Slope × Ruggedness',
            'slope_x_bare': 'Slope × Bare Land',
            'road_dist_n': 'Road Distance',
            'river_dist_n': 'River Distance',
        }

        result = []
        for i in sorted_idx[:5]:
            result.append({
                "feature": FEATURES[int(i)],
                "description": LABELS.get(FEATURES[int(i)], FEATURES[int(i)]),
                "importance": float(round(mean_abs[int(i)], 4)),
                "direction": "increases" if float(mean_sign[int(i)]) > 0 else "decreases"
            })

        return result

    except Exception as e:
        print("[SHAP ERROR]:", str(e))
        return []
# def get_shap(df: pd.DataFrame):
#     try:
#         # Pull raw numpy array — no pandas, no index issues
#         feat_array = df[FEATURES].values.astype(np.float64)
#         sample = feat_array[:min(300, len(feat_array))]
#
#         vals = EXPLAINER.shap_values(sample)
#         if isinstance(vals, list): vals = vals[1]
#
#         # vals is now a pure numpy 2D array
#         mean_abs = np.abs(vals).mean(axis=0)  # shape (n_features,)
#         mean_sign = vals.mean(axis=0)  # shape (n_features,)
#         sorted_idx = np.argsort(mean_abs)[::-1]
#
#         LABELS = {
#             'slope_sin': 'Slope Steepness', 'aspect_ns': 'N-Facing Aspect',
#             'rugged_n': 'Terrain Ruggedness', 'veg_bare': 'Bare Vegetation',
#             'slope_x_rugged': 'Slope × Ruggedness', 'slope_x_bare': 'Slope × Bare Land',
#             'road_dist_n': 'Road Distance', 'river_dist_n': 'River Distance',
#         }
#         return [{
#             "feature": FEATURES[int(i)],
#             "description": LABELS.get(FEATURES[int(i)], FEATURES[int(i)]),
#             "importance": round(float(mean_abs[int(i)]), 4),
#             "direction": "increases" if float(mean_sign[int(i)]) > 0 else "decreases",
#         } for i in sorted_idx[:4]]
#     except Exception as e:
#         print(f"  [SHAP] Failed: {e}")
#         return []

# =============================================================
# GEOJSON BUILDER
# =============================================================
def df_to_geojson(df: pd.DataFrame) -> dict:
    feats = []
    for _, row in df.iterrows():
        feats.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['longitude']), float(row['latitude'])]
            },
            "properties": {
                "risk_level":         str(row.get('risk_level', 'Low')),
                "combined_risk":      round(float(row.get('combined_risk', 0)), 3),
                "landslide_risk":     round(float(row.get('landslide_risk', 0)), 3),
                "flood_risk":         round(float(row.get('flood_risk', 0)), 3),
                "erosion_risk":       round(float(row.get('erosion_risk', 0)), 3),
                "accessibility_risk": round(float(row.get('accessibility_risk', 0)), 3),
                "terrain_stability":  round(float(row.get('terrain_stability', 0)), 3),
                "slope_deg":          round(float(row.get('slope_mean', 0)), 1),
                "ndvi":               round(float(row.get('nvdi_mean', 0)), 3),
                "road_dist_m":        int(round(float(row.get('road_dist', 0)))),
                "river_dist_m":       int(round(float(row.get('river_dist', 0)))),
            }
        })
    return {"type": "FeatureCollection", "features": feats}

# =============================================================
# RAINFALL (Open-Meteo — free, no key needed)
# =============================================================
def fetch_rainfall(lat: float, lon: float) -> float:
    try:
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat:.4f}&longitude={lon:.4f}"
               f"&daily=precipitation_sum&forecast_days=1&timezone=auto")
        r = requests.get(url, timeout=8)
        val = r.json()["daily"]["precipitation_sum"][0]
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0

# =============================================================
# TERRAIN PIPELINE  (runs for any bbox)
# =============================================================
def run_pipeline_for_bbox(south, north, west, east):
    """
    Fetch DEM + NDVI + roads/rivers for any bounding box.
    Returns clean DataFrame ready for engineer_features().
    """
    from config import (OPENTOPO_API_KEY, COPERNICUS_CLIENT_ID,
                        COPERNICUS_CLIENT_SECRET, DEM_TYPE)
    from pipeline.fetch_terrain import fetch_dem, process_dem
    from pipeline.fetch_ndvi    import get_ndvi
    from pipeline.fetch_osm     import get_road_river_distances
    from scipy.ndimage          import zoom as scipy_zoom

    tmp = tempfile.mkdtemp()

    # 1 ── DEM
    dem_path = fetch_dem(south, north, west, east,
                         api_key=OPENTOPO_API_KEY,
                         dem_type=DEM_TYPE,
                         output_dir=tmp)
    terrain, ref_profile = process_dem(dem_path, output_dir=tmp)
    h, w = int(terrain["dem"].shape[0]), int(terrain["dem"].shape[1])

    # 2 ── NDVI
    ndvi_arr, _ = get_ndvi(south, north, west, east,
                            client_id=COPERNICUS_CLIENT_ID,
                            client_secret=COPERNICUS_CLIENT_SECRET,
                            output_dir=tmp,
                            dem_array=terrain["dem"])

    # 3 ── Roads / Rivers
    road_dist, river_dist = get_road_river_distances(
        south, north, west, east,
        profile=ref_profile, output_dir=tmp)

    # 4 ── Resize everything to DEM shape
    def resize(arr):
        a = np.array(arr, dtype=np.float64)
        ah, aw = int(a.shape[0]), int(a.shape[1])
        if ah == h and aw == w:
            return a
        zy = float(h) / float(ah)
        zx = float(w) / float(aw)
        return scipy_zoom(a, (zy, zx), order=1)

    ndvi_arr  = resize(ndvi_arr)
    road_dist = resize(road_dist)
    river_dist= resize(river_dist)

    # 5 ── Downsample: aim for ~200k cells max for speed
    total = h * w
    MAX   = 200_000
    step  = int(np.ceil(np.sqrt(total / MAX))) if total > MAX else 1

    sl = np.s_[::step, ::step]   # slice object

    slope_s  = np.array(terrain["slope"][sl],      dtype=np.float64)
    aspect_s = np.array(terrain["aspect"][sl],     dtype=np.float64)
    rugged_s = np.array(terrain["ruggedness"][sl], dtype=np.float64)
    ndvi_s   = np.array(ndvi_arr[sl],              dtype=np.float64)
    road_s   = np.array(road_dist[sl],             dtype=np.float64)
    river_s  = np.array(river_dist[sl],            dtype=np.float64)

    hs, ws = int(slope_s.shape[0]), int(slope_s.shape[1])

    # 6 ── Build lat/lon grid
    t   = ref_profile["transform"]
    px  = float(abs(t.a)) * float(step)

    rows_idx = np.arange(hs, dtype=np.float64).reshape(-1, 1) * np.ones((1, ws))
    cols_idx = np.ones((hs, 1)) * np.arange(ws, dtype=np.float64).reshape(1, -1)

    lats = float(t.f) - (rows_idx + 0.5) * px
    lons = float(t.c) + (cols_idx + 0.5) * px

    # 7 ── Build DataFrame
    df = pd.DataFrame({
        "latitude":        lats.ravel(),
        "longitude":       lons.ravel(),
        "slope_mean":      slope_s.ravel(),
        "aspect_mean":     aspect_s.ravel(),
        "ruggedness_mean": rugged_s.ravel(),
        "nvdi_mean":       ndvi_s.ravel(),
        "road_dist":       road_s.ravel(),
        "river_dist":      river_s.ravel(),
    })

    # 8 ── Clean
    for col in RAW_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)

    df = df[df['slope_mean'] >= 0].reset_index(drop=True)
    print(f"  Grid built: {len(df):,} cells (step={step})")
    return df

# =============================================================
# SUMMARY BUILDER
# =============================================================
def build_summary(df, area_name, rain_mm, proc_time, note=""):
    dist  = df['risk_level'].value_counts().to_dict()
    total = max(len(df), 1)
    ms = {
        "landslide":     round(float(df['landslide_risk'].mean()),     3),
        "flood":         round(float(df['flood_risk'].mean()),         3),
        "erosion":       round(float(df['erosion_risk'].mean()),       3),
        "accessibility": round(float(df['accessibility_risk'].mean()), 3),
        "terrain":       round(float(df['terrain_stability'].mean()),  3),
        "combined":      round(float(df['combined_risk'].mean()),      3),
    }
    return {
        "area_name":         area_name,
        "total_cells":       total,
        "risk_distribution": dist,
        "high_pct":          round(dist.get("High",   0) / total * 100, 1),
        "medium_pct":        round(dist.get("Medium", 0) / total * 100, 1),
        "low_pct":           round(dist.get("Low",    0) / total * 100, 1),
        "mean_scores":       ms,
        "rainfall_mm":       round(float(rain_mm), 1),
        "model_info": {
            "auc_roc":         MODEL_INFO["auc_roc"],
            "training_region": MODEL_INFO["training_region"],
            "note": note or (
                "Model trained on Chamoli GSI inventory. "
                "Terrain fetched live for this region. "
                "Results are indicative — not certifiable hazard maps."
            )
        },
        "processing_time_s": round(float(proc_time), 1),
    }

# =============================================================
# ROUTES
# =============================================================

@app.get("/")
def root():
    idx = os.path.join(FRONT_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "Himalayan Risk Intelligence API v3", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok",
            "model_auc": MODEL_INFO["auc_roc"],
            "training_region": MODEL_INFO["training_region"]}

@app.get("/areas")
def areas():
    return {"areas": [
        {"id":"chamoli",     "name":"Chamoli, Uttarakhand",
         "bbox":{"south":30.39,"north":30.47,"west":79.37,"east":79.45}},
        {"id":"rudraprayag", "name":"Rudraprayag, Uttarakhand",
         "bbox":{"south":30.3,"north":30.7,"west":78.7,"east":79.3}},
        {"id":"kullu",       "name":"Kullu, Himachal Pradesh",
         "bbox":{"south":31.7,"north":32.3,"west":76.9,"east":77.6}},
        {"id":"shimla",      "name":"Shimla, Himachal Pradesh",
         "bbox":{"south":30.9,"north":31.3,"west":77.0,"east":77.6}},
        {"id":"kinnaur",     "name":"Kinnaur, Himachal Pradesh",
         "bbox":{"south":31.4,"north":31.9,"west":77.6,"east":78.5}},
    ]}

# ── shared pipeline logic ────────────────────────────────────
def _run_full(req: AnalyseRequest):
    t0   = time.time()
    cLat = (req.south + req.north) / 2
    cLon = (req.west  + req.east)  / 2
    rain = fetch_rainfall(cLat, cLon)

    df = run_pipeline_for_bbox(req.south, req.north, req.west, req.east)
    df = engineer_features(df)
    df = compute_risks(df)

    # Downsample dots for response (max 6000 points)
    if len(df) > 6000:
        step = len(df) // 6000 + 1
        df   = df.iloc[::step].reset_index(drop=True)

    shap_exp = get_shap(df)
    summary  = build_summary(df, req.area_name, rain, time.time() - t0)
    summary["shap_explanation"] = shap_exp
    summary["rainfall_mm"]      = round(rain, 1)

    return {"summary": summary, "geojson": df_to_geojson(df)}


@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    """Full live pipeline for any bbox."""
    if req.north <= req.south or req.east <= req.west:
        raise HTTPException(400, "Invalid bounding box")
    if (req.north - req.south) > 2.0 or (req.east - req.west) > 2.0:
        raise HTTPException(400, "Area too large. Max 2°×2°. Zoom in.")
    try:
        return _run_full(req)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {str(e)}")


@app.post("/analyse/demo")
async def analyse_demo(req: AnalyseRequest):
    """
    Demo endpoint — tries real pipeline first, falls back to
    re-projected Chamoli data if pipeline fails.
    """
    # ── Try real pipeline ──
    try:
        from config import OPENTOPO_API_KEY
        if OPENTOPO_API_KEY and OPENTOPO_API_KEY != "YOUR_OPENTOPO_KEY_HERE":
            return _run_full(req)
    except Exception as e:
        print(f"  Pipeline failed, using cached fallback: {e}")

    # ── Fallback: re-project Chamoli data to requested bbox ──
    t0   = time.time()
    cLat = (req.south + req.north) / 2
    cLon = (req.west  + req.east)  / 2
    rain = fetch_rainfall(cLat, cLon)

    try:
        demo_path = os.path.join(
            os.path.expanduser("~"), "Downloads", "chamoli_risk_output.xlsx"
        )
        df = pd.read_excel(demo_path)

        # Re-project row/col to requested bbox
        rows = int(df['row_index'].max()) + 1
        cols = int(df['col_index'].max()) + 1
        df['latitude']  = req.south + (df['row_index'].values / rows) * (req.north - req.south)
        df['longitude'] = req.west  + (df['col_index'].values / cols) * (req.east  - req.west)

        if 'nvdi_mean' not in df.columns:
            df['nvdi_mean'] = 0.5

        df = engineer_features(df)
        df = compute_risks(df)

        if len(df) > 6000:
            step = len(df) // 6000 + 1
            df   = df.iloc[::step].reset_index(drop=True)

        summary = build_summary(
            df, req.area_name, rain, time.time() - t0,
            note="Demo mode — Chamoli data projected to selected region. "
                 "Set OpenTopography API key in config.py for live analysis."
        )
        summary["shap_explanation"] = []
        summary["rainfall_mm"]      = round(rain, 1)
        return {"summary": summary, "geojson": df_to_geojson(df)}

    except Exception as e:
        raise HTTPException(500, f"Fallback failed: {str(e)}")