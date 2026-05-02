# =============================================================
# config.py — All settings in one place
# Change API keys and paths here, nowhere else
# =============================================================

# ----------------------------
# API KEYS
# Get free OpenTopography key at: https://portal.opentopography.org/requestApiKey
# Get free Copernicus key at: https://dataspace.copernicus.eu (register → dashboard)

# ----------------------------
# DEM settings
# SRTMGL1 = 30m resolution (recommended)
# SRTMGL3 = 90m resolution (faster, lower quality)
# COP30    = Copernicus 30m (best quality, same key)
# ----------------------------
DEM_TYPE = "SRTMGL1"
# ----------------------------
# Grid resolution in metres
# 30 = matches DEM resolution (recommended)
# 100 = faster processing, coarser grid
# ----------------------------
GRID_RESOLUTION_M = 100
# ----------------------------
# Output directory
# ----------------------------
from dotenv import load_dotenv
import os

load_dotenv()

COPERNICUS_CLIENT_ID = os.getenv("COPERNICUS_ID")
COPERNICUS_CLIENT_SECRET = os.getenv("COPERNICUS_CLIENT")
OPENTOPO_API_KEY = os.getenv("OPENTOPO_API")
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "himalaya_risk")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Known study areas (for quick testing)
# Format: (south, north, west, east) in decimal degrees WGS84
# ----------------------------
STUDY_AREAS = {
    "chamoli": {
        "bbox": (30.39, 30.46, 79.37, 79.44),
        "name": "Chamoli, Uttarakhand"
    },
    "rudraprayag": {
        "bbox": (30.3, 30.7, 78.7, 79.3),
        "name": "Rudraprayag, Uttarakhand"
    },
    "kinnaur": {
        "bbox": (31.4, 31.9, 77.6, 78.5),
        "name": "Kinnaur, Himachal Pradesh"
    },
    "kullu": {
        "bbox": (31.7, 32.3, 76.9, 77.6),
        "name": "Kullu, Himachal Pradesh"
    },
    "shimla": {
        "bbox": (30.9, 31.3, 77.0, 77.6),
        "name": "Shimla, Himachal Pradesh"
    }
}
