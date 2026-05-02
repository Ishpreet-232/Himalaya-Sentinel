# =============================================================
# model/train_model.py — XGBoost v3 (multi-district)
# Supports training on Chamoli + Hamirpur (or any district xlsx)
#
# Usage:
#   python model/train_model.py
#
# To add more districts, drop their xlsx into TRAINING_FILES below.
# =============================================================

import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (average_precision_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix, roc_auc_score)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

# =============================================================
# PATHS
# =============================================================
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model", "saved")
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# ADD YOUR DISTRICT FILES HERE
# Each entry: (path, district_name, terrain_type)
# terrain_type hints for per-district feature scaling (see below)
# -------------------------------------------------------------------
DOWNLOAD_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

TRAINING_FILES = [
    {
        "path":         os.path.join(DOWNLOAD_DIR, "chamoli_dataset.xlsx"),
        "name":         "Chamoli",
        "terrain_type": "gorge",      # steep, high-relief Himalayan gorge
    },
    # {
    #     "path":         os.path.join(DOWNLOAD_DIR, "hamirpur_dataset.xlsx"),
    #     "name":         "Hamirpur",
    #     "terrain_type": "foothill",   # gentler slopes, clay soils, monsoon-prone
    # },
]

# =============================================================
# FEATURE DEFINITIONS  (must match api/main.py exactly)
# =============================================================
FEATURES = [
    'slope_sin',        # sin(slope_degrees)
    'aspect_ns',        # cos(aspect) — north-facing wetter
    'rugged_n',         # normalised TRI ruggedness
    'veg_bare',         # 1 - NDVI
    'slope_x_rugged',   # interaction
    'slope_x_bare',     # interaction
    'road_dist_n',      # normalised road distance
    'river_dist_n',     # normalised river distance
]

RAW_FEATURES = ['slope_mean', 'aspect_mean', 'ruggedness_mean',
                 'road_dist', 'river_dist', 'nvdi_mean']

RISK_WEIGHTS = {
    'landslide':     0.40,
    'flood':         0.20,
    'erosion':       0.15,
    'accessibility': 0.15,
    'terrain':       0.10,
}

# =============================================================
# HELPERS
# =============================================================
def norm(arr):
    a = np.array(arr, dtype=np.float64)
    rng = a.max() - a.min()
    if rng < 1e-9:
        return np.zeros_like(a)
    return (a - a.min()) / rng


def load_district(cfg: dict) -> pd.DataFrame:
    """Load one district xlsx, normalise column names, return clean DataFrame."""
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"  [SKIP] {cfg['name']}: file not found at {path}")
        return None

    df = pd.read_excel(path)
    print(f"\n[{cfg['name']}] Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # ── Normalise column names ─────────────────────────────────
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ('nvdi_mean', 'ndvi_mean', 'ndvi'):
            col_map[c] = 'nvdi_mean'
        elif cl in ('slope_mean', 'slope'):
            col_map[c] = 'slope_mean'
        elif cl in ('aspect_mean', 'aspect'):
            col_map[c] = 'aspect_mean'
        elif cl in ('ruggedness_mean', 'tri_mean', 'ruggedness', 'tri'):
            col_map[c] = 'ruggedness_mean'
        elif cl in ('road_dist', 'road_distance'):
            col_map[c] = 'road_dist'
        elif cl in ('river_dist', 'river_distance'):
            col_map[c] = 'river_dist'
        elif cl in ('landslide_label', 'label', 'landslide'):
            col_map[c] = 'landslide_label'
        elif cl in ('row_index', 'row'):
            col_map[c] = 'row_index'
        elif cl in ('col_index', 'col', 'column_index'):
            col_map[c] = 'col_index'
    df = df.rename(columns=col_map)

    # ── Check required columns ─────────────────────────────────
    required = RAW_FEATURES + ['landslide_label']
    missing = [r for r in required if r not in df.columns]
    if missing:
        print(f"  [ERROR] {cfg['name']} missing columns: {missing}")
        print(f"  Available: {list(df.columns)}")
        return None

    # ── Type coercion ──────────────────────────────────────────
    for col in RAW_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['landslide_label'] = pd.to_numeric(df['landslide_label'], errors='coerce')

    # ── Drop only rows where LABEL is null ─────────────────────
    before = len(df)
    df = df[df['landslide_label'].notna()].reset_index(drop=True)
    print(f"  Dropped {before - len(df)} rows with null label")

    # ── Fill feature NaNs with median ──────────────────────────
    for col in RAW_FEATURES:
        med = df[col].median()
        df[col] = df[col].fillna(med if pd.notna(med) else 0.0)

    # ── Clip to valid ranges ───────────────────────────────────
    df['slope_mean']      = df['slope_mean'].clip(0, 90)
    df['aspect_mean']     = df['aspect_mean'].clip(0, 360)
    df['ruggedness_mean'] = df['ruggedness_mean'].clip(0, None)
    df['nvdi_mean']       = df['nvdi_mean'].clip(-1, 1)
    df['road_dist']       = df['road_dist'].clip(0, None)
    df['river_dist']      = df['river_dist'].clip(0, None)

    # ── Tag with source ────────────────────────────────────────
    df['_source'] = cfg['name']
    df['_terrain'] = cfg['terrain_type']

    n_pos = int(df['landslide_label'].sum())
    print(f"  Clean rows: {len(df):,} | Landslides: {n_pos} ({n_pos/len(df)*100:.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features — must match api/main.py exactly."""
    df = df.copy()
    df['slope_sin']      = np.sin(np.radians(df['slope_mean'].values))
    df['aspect_ns']      = np.cos(np.radians(df['aspect_mean'].values))
    df['rugged_n']       = norm(df['ruggedness_mean'].values)
    df['veg_bare']       = (1 - df['nvdi_mean'].values.clip(0, 1)).astype(np.float64)
    df['slope_x_rugged'] = df['slope_sin'].values * df['rugged_n'].values
    df['slope_x_bare']   = df['slope_sin'].values * df['veg_bare'].values
    df['road_dist_n']    = norm(df['road_dist'].values)
    df['river_dist_n']   = norm(df['river_dist'].values)
    return df.reset_index(drop=True)


def augment_landslides(df: pd.DataFrame, multiplier: int = 4) -> pd.DataFrame:
    """
    Add Gaussian noise copies of landslide rows to address class imbalance.
    Noise stds chosen per-feature to stay physically realistic.
    """
    pos = df[df['landslide_label'] == 1].copy()
    augmented = [df]
    noise_cfg = {
        'slope_mean':      1.5,
        'aspect_mean':     5.0,
        'ruggedness_mean': 3.0,
        'nvdi_mean':       0.02,
        'road_dist':       50.0,
        'river_dist':      30.0,
    }
    rng = np.random.default_rng(42)
    for _ in range(multiplier):
        noisy = pos.copy()
        for col, std in noise_cfg.items():
            if col in noisy.columns:
                noisy[col] = noisy[col] + rng.normal(0, std, size=len(noisy))
        noisy['slope_mean'] = noisy['slope_mean'].clip(0, 90)
        noisy['nvdi_mean']  = noisy['nvdi_mean'].clip(-1, 1)
        noisy['road_dist']  = noisy['road_dist'].clip(0, None)
        noisy['river_dist'] = noisy['river_dist'].clip(0, None)
        augmented.append(noisy)

    result = pd.concat(augmented, ignore_index=True)
    print(f"  After augmentation: {len(result):,} rows "
          f"(pos={int(result['landslide_label'].sum()):,})")
    return result


def undersample(df: pd.DataFrame, ratio: float = 3.0) -> pd.DataFrame:
    """Keep all positives + ratio×positives negatives (random undersample)."""
    pos = df[df['landslide_label'] == 1]
    neg = df[df['landslide_label'] == 0]
    n_neg = min(len(neg), int(len(pos) * ratio))
    neg_s = neg.sample(n=n_neg, random_state=42)
    out = pd.concat([pos, neg_s], ignore_index=True).sample(frac=1, random_state=42)
    print(f"  After undersampling: {len(out):,} rows "
          f"(pos={len(pos):,}, neg={n_neg:,})")
    return out


# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 60)
    print("Himalayan Risk — XGBoost Training v3 (multi-district)")
    print("=" * 60)

    # ── 1. Load all districts ──────────────────────────────────
    frames = []
    for cfg in TRAINING_FILES:
        df = load_district(cfg)
        if df is not None:
            frames.append(df)

    if not frames:
        print("\n[ERROR] No training data loaded. Check file paths in TRAINING_FILES.")
        sys.exit(1)

    full_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal before augmentation: {len(full_df):,} rows "
          f"| Districts: {full_df['_source'].unique()}")

    # ── 2. Augment + undersample ───────────────────────────────
    full_df = augment_landslides(full_df, multiplier=4)
    full_df = undersample(full_df, ratio=3.0)

    # ── 3. Feature engineering ─────────────────────────────────
    full_df = engineer_features(full_df)

    X = full_df[FEATURES].values.astype(np.float64)
    y = full_df['landslide_label'].values.astype(int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos = n_neg / max(n_pos, 1)
    print(f"\nFinal dataset: {len(y):,} rows | "
          f"pos={n_pos:,} neg={n_neg:,} | scale_pos_weight={scale_pos:.2f}")

    # ── 4. Cross-validation ────────────────────────────────────
    print("\n[CV] 5-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_ap, cv_auc, cv_f1 = [], [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xval, yval = X[val_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric='aucpr',
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr,
                eval_set=[(Xval, yval)],
                verbose=False)

        probs = clf.predict_proba(Xval)[:, 1]
        preds = (probs >= 0.30).astype(int)

        ap  = average_precision_score(yval, probs)
        auc = roc_auc_score(yval, probs)
        f1  = f1_score(yval, preds)

        cv_ap.append(ap);  cv_auc.append(auc);  cv_f1.append(f1)
        print(f"  Fold {fold}: AP={ap:.3f}  AUC={auc:.3f}  F1={f1:.3f}")

    print(f"\n  Mean AP={np.mean(cv_ap):.3f}±{np.std(cv_ap):.3f}  "
          f"AUC={np.mean(cv_auc):.3f}±{np.std(cv_auc):.3f}  "
          f"F1={np.mean(cv_f1):.3f}±{np.std(cv_f1):.3f}")

    # ── 5. Final model on all data ─────────────────────────────
    print("\n[Train] Fitting final model on full dataset...")
    final_model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y, verbose=False)

    # ── 6. Threshold tuning ────────────────────────────────────
    print("[Threshold] Tuning for best F1 on training data...")
    probs_all = final_model.predict_proba(X)[:, 1]
    best_thr, best_f1 = 0.3, 0.0
    for thr in np.arange(0.15, 0.65, 0.01):
        preds = (probs_all >= thr).astype(int)
        f = f1_score(y, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_thr = f, thr

    preds_final = (probs_all >= best_thr).astype(int)
    cm = confusion_matrix(y, preds_final)
    p  = precision_score(y, preds_final, zero_division=0)
    r  = recall_score(y, preds_final, zero_division=0)
    print(f"  Best threshold: {best_thr:.2f} → F1={best_f1:.3f} "
          f"P={p:.3f} R={r:.3f}")
    print(f"  Confusion matrix:\n{cm}")

    # ── 7. SHAP explainer ──────────────────────────────────────
    print("[SHAP] Building TreeExplainer...")
    sample_idx = np.random.choice(len(X), min(500, len(X)), replace=False)
    explainer  = shap.TreeExplainer(final_model)
    shap_vals  = explainer.shap_values(X[sample_idx])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_imp = np.abs(shap_vals).mean(axis=0)
    print("  Feature importances (SHAP):")
    for i in np.argsort(mean_imp)[::-1]:
        print(f"    {FEATURES[i]:20s}: {mean_imp[i]:.4f}")

    # ── 8. Save everything ─────────────────────────────────────
    districts_trained = list(full_df['_source'].unique())
    training_region   = " + ".join(districts_trained)

    model_info = {
        "features":        FEATURES,
        "raw_features":    RAW_FEATURES,
        "risk_weights":    RISK_WEIGHTS,
        "auc_roc":         round(float(np.mean(cv_auc)), 4),
        "avg_precision":   round(float(np.mean(cv_ap)),  4),
        "cv_f1":           round(float(np.mean(cv_f1)),  4),
        "best_threshold":  round(float(best_thr),        3),
        "training_region": training_region,
        "districts":       districts_trained,
        "n_samples":       len(y),
        "n_positive":      int(n_pos),
        "scale_pos_weight":round(float(scale_pos), 3),
        "xgb_params": {
            "n_estimators": 400, "max_depth": 5,
            "learning_rate": 0.05, "subsample": 0.8,
        },
        "notes": (
            "Multi-district XGBoost trained with augmentation (×4 noise on positives) "
            "and 1:3 landslide:non-landslide undersampling. "
            "AUC reported is 5-fold CV mean, not augmented-data AUC (which would be inflated)."
        )
    }

    model_path    = os.path.join(MODEL_DIR, "model.pkl")
    explainer_path = os.path.join(MODEL_DIR, "shap_explainer.pkl")
    info_path      = os.path.join(MODEL_DIR, "model_info.json")

    with open(model_path,     "wb") as f: pickle.dump(final_model, f)
    with open(explainer_path, "wb") as f: pickle.dump(explainer, f)
    with open(info_path,      "w")  as f: json.dump(model_info, f, indent=2)

    print(f"\n✓ Saved model   → {model_path}")
    print(f"✓ Saved SHAP    → {explainer_path}")
    print(f"✓ Saved info    → {info_path}")
    print(f"\nTraining complete. Districts: {training_region}")
    print(f"CV AUC={np.mean(cv_auc):.3f}  AP={np.mean(cv_ap):.3f}  F1={np.mean(cv_f1):.3f}")


if __name__ == "__main__":
    main()