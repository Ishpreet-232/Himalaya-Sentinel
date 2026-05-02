# 🏔️ Himalaya Sentinel

A Multi-Hazard Urban Risk Assessment System for Himalayan Regions using Geospatial Analysis and Machine Learning.

---

## 📌 Overview

Himalaya Sentinel is a geospatial intelligence system that evaluates disaster risk in mountainous regions by combining:

- Terrain analysis
- Vegetation data
- Hydrological proximity
- Infrastructure impact

The system predicts landslide susceptibility and generates a **combined risk score**.

---

## 🎯 Objectives

- Predict landslide-prone regions
- Build multi-hazard risk model
- Use open-source geospatial data
- Create a scalable and explainable system

---

## 🌍 Data Sources

- OpenTopography (DEM - terrain data)
- Copernicus (NDVI - vegetation)
- NGDR (landslide data)
- OpenStreetMap (roads & rivers)

---

## 🧰 Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn

### ML & Data
- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Visualization
- Matplotlib
- QGIS

### Frontend (planned)
- Leaflet.js

---

## 🧠 Key Features

- Grid-based spatial modeling
- Feature engineering using terrain physics
- Class imbalance handling
- Threshold tuning
- Multi-hazard risk scoring
- Explainable ML (SHAP)

---

## 📊 Model Performance

- AUC-ROC: ~0.93–0.95
- Recall (landslides): ~60–70%
- Spatial validation using GIS

---

## ⚙️ How It Works

1. User selects region (lat/lon bounding box)
2. Fetch geospatial data via APIs
3. Generate features (slope, NDVI, distances)
4. Run ML model
5. Output risk score and map

---

## 🚀 Running the Backend

```bash
python -m uvicorn api.main:app --reload --port 8000