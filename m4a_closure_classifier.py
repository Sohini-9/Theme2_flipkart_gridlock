"""
M4a - ROAD CLOSURE CLASSIFIER
=============================

Trains the improved blended road-closure model directly from raw data.csv,
while using M2's corridor reconstruction rule as the single source of truth:

  raw event rows -> M2 RF+KNN corridor_final -> closure feature engineering
                 -> blended ExtraTrees + RandomForest model

Important consistency decisions:
  - No corridor_final_v3 is created here.
  - corridor_final is produced by the same locked M2 rule:
      RF and KNN must agree, and their average confidence must be >= 0.80.
  - M5 loads the single saved closure bundle from this file. The old
    RF-primary/LR-cross-check architecture is intentionally removed.

Input  : data.csv
Output : closure_model_bundle.pkl
         closure_model.pkl                  (same bundle, compatibility alias)
         closure_eval_metrics.json
         closure_feature_importance.csv
"""

import json
import os
import pickle
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
MIN_PRECISION = 0.45
MIN_RECALL = 0.0
CORRIDOR_CONFIDENCE_THRESHOLD = 0.80

DEFAULT_PIPELINE_DIR = "/content/drive/MyDrive/flipkart/"
PIPELINE_DIR = os.environ.get(
    "THEME2_PIPELINE_DIR",
    DEFAULT_PIPELINE_DIR if os.path.isdir(DEFAULT_PIPELINE_DIR) else os.getcwd(),
)

DATA_PATH = os.environ.get("M4A_DATA_PATH", os.path.join(PIPELINE_DIR, "data.csv"))
CORRIDOR_MODEL_PATH = os.environ.get(
    "M2_CORRIDOR_MODEL_PATH", os.path.join(PIPELINE_DIR, "corridor_model.pkl")
)

CLOSURE_BUNDLE_PATH = os.path.join(PIPELINE_DIR, "closure_model_bundle.pkl")
CLOSURE_MODEL_ALIAS_PATH = os.path.join(PIPELINE_DIR, "closure_model.pkl")
METRICS_PATH = os.path.join(PIPELINE_DIR, "closure_eval_metrics.json")
IMPORTANCE_PATH = os.path.join(PIPELINE_DIR, "closure_feature_importance.csv")

ROAD_KEYWORDS = [
    "Outer Ring Road", "Inner Ring Road", "Bellary Road", "Old Airport Road",
    "Old Madras Road", "Mysore Road", "Hosur Road", "Bannerghatta Main Road",
    "Varthur Main Road", "Whitefield Main Road", "Hennur Main Road",
    "Magadi Main Road", "HAL Airport Road", "Mumbai Bengaluru Highway",
    "Tumkur Road", "Sarjapur Road", "Bannerghatta Road",
]

RATE_GROUP_COLS = ["event_cause", "corridor_final", "hour", "police_station"]

SEVERE_KEYWORD_PATTERN = r"block|bandh|bandu|closed|close|divert"
MINOR_KEYWORD_PATTERN = r"normal|no problem|not a traffic|no issue|cleared|clear "
SLOW_KEYWORD_PATTERN = r"slow|one side|single lane|single line"
KEYWORD_FEATURES = ["kw_severe", "kw_minor", "kw_slow"]

STRUCTURED_FEATURES = [
    "latitude", "longitude",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_office_hours", "month",
    "is_planned", "authenticated_flag",
    "event_type_enc", "event_cause_enc", "corridor_enc",
    "police_station_enc", "zone_enc", "junction_enc",
    "planned_x_office_hours", "cause_x_planned",
    "hours_since_last_corridor_incident", "corridor_24h_count",
    "road_name_enc", "corridor_final_enc", "corridor_confidence",
    "corridor_agreement",
] + [f"{c}_closure_rate" for c in RATE_GROUP_COLS] + KEYWORD_FEATURES

LOGGING_TIME_EXTRA_FEATURES = [
    "priority_enc", "direction_enc", "veh_type_enc", "reason_breakdown_enc",
    "has_vehicle_details",
]


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def resolve_existing_path(path: str) -> str:
    """Prefer the configured path; fall back to ./basename for local runs."""
    if os.path.exists(path):
        return path
    local = os.path.join(os.getcwd(), os.path.basename(path))
    return local if os.path.exists(local) else path


def load_pickle_or_joblib(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
        .astype(int)
    )


def extract_road_name(addr):
    if pd.isna(addr):
        return "Unknown"
    addr = str(addr)
    for road in ROAD_KEYWORDS:
        if road.lower() in addr.lower():
            return road
    match = re.search(r"([A-Za-z\s]+(?:Road|Highway|Underpass))", addr, flags=re.IGNORECASE)
    if match:
        road = match.group(1).strip()
        if any(x in road.lower() for x in ["cross road", "main road"]) and len(road.split()) <= 3:
            return "Local Road"
        return road
    return "Unknown"


def fit_label_encoder(df: pd.DataFrame, col: str, encoders: dict) -> None:
    values = df[col].fillna("UNKNOWN").astype(str) if col in df.columns else pd.Series("UNKNOWN", index=df.index)
    encoder = LabelEncoder()
    df[f"{col}_enc"] = encoder.fit_transform(values)
    encoders[col] = encoder


# ---------------------------------------------------------------------------
# M2-CONSISTENT CORRIDOR RECONSTRUCTION
# ---------------------------------------------------------------------------

def train_corridor_model(df: pd.DataFrame) -> dict:
    labeled = df[df["corridor"].notna() & (df["corridor"] != "Non-corridor")].copy()
    if labeled.empty:
        raise ValueError("[M4a] Cannot train corridor model: no labeled corridor rows found.")

    X = labeled[["latitude", "longitude"]].values
    y = labeled["corridor"].astype(str).values

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)

    knn_model = KNeighborsClassifier(n_neighbors=15, weights="distance")
    knn_model.fit(X, y)

    models = {"rf": rf_model, "knn": knn_model}
    os.makedirs(os.path.dirname(CORRIDOR_MODEL_PATH) or ".", exist_ok=True)
    with open(CORRIDOR_MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print(f"[M4a] Saved M2-compatible corridor models -> {CORRIDOR_MODEL_PATH}")
    return models


def load_or_train_corridor_model(df: pd.DataFrame) -> dict:
    path = resolve_existing_path(CORRIDOR_MODEL_PATH)
    if os.path.exists(path):
        models = load_pickle_or_joblib(path)
        if isinstance(models, dict) and {"rf", "knn"}.issubset(models):
            print(f"[M4a] Loaded M2 corridor model bundle -> {path}")
            return models
        print("[M4a] Existing corridor_model.pkl is not the M2 RF+KNN bundle; retraining.")
    return train_corridor_model(df)


def apply_corridor_reconstruction(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    rf_model, knn_model = models["rf"], models["knn"]
    needs_prediction = (df["corridor"].isna() | (df["corridor"] == "Non-corridor")).values

    corridor_final = df["corridor"].fillna("Non-corridor").astype(str).copy()
    corridor_confidence = pd.Series(1.0, index=df.index)
    corridor_agreement = pd.Series(1, index=df.index)
    corridor_source = pd.Series("original", index=df.index)

    if needs_prediction.sum() > 0:
        X_pred = df.loc[needs_prediction, ["latitude", "longitude"]].values

        rf_probs = rf_model.predict_proba(X_pred)
        rf_pred = rf_model.classes_[rf_probs.argmax(axis=1)]
        rf_conf = rf_probs.max(axis=1)

        knn_probs = knn_model.predict_proba(X_pred)
        knn_pred = knn_model.classes_[knn_probs.argmax(axis=1)]
        knn_conf = knn_probs.max(axis=1)

        agreement = (rf_pred == knn_pred).astype(int)
        combined_conf = (rf_conf + knn_conf) / 2
        accept_mask = (agreement == 1) & (combined_conf >= CORRIDOR_CONFIDENCE_THRESHOLD)

        idx_subset = df.index[needs_prediction]
        corridor_confidence.loc[idx_subset] = combined_conf
        corridor_agreement.loc[idx_subset] = agreement
        corridor_source.loc[idx_subset] = np.where(accept_mask, "predicted", "unresolved")
        corridor_final.loc[idx_subset] = np.where(accept_mask, rf_pred, "Non-corridor")

    df["corridor_final"] = corridor_final
    df["corridor_confidence"] = corridor_confidence
    df["corridor_agreement"] = corridor_agreement
    df["corridor_source"] = corridor_source
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def load_and_engineer(path: str):
    path = resolve_existing_path(path)
    df = pd.read_csv(path)
    print(f"[M4a] Loaded raw data: {df.shape[0]} rows from {path}")

    df["target"] = parse_bool_series(df["requires_road_closure"])

    df["start_dt"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df["hour"] = df["start_dt"].dt.hour.fillna(0).astype(int)
    df["day_of_week"] = df["start_dt"].dt.dayofweek.fillna(0).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["start_dt"].dt.month.fillna(1).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_office_hours"] = df["hour"].between(11, 16).astype(int)

    for col in ["event_type", "event_cause", "corridor", "police_station", "zone", "junction"]:
        df[col] = df[col].fillna("UNKNOWN").astype(str) if col in df.columns else "UNKNOWN"

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    corridor_models = load_or_train_corridor_model(df)
    df = apply_corridor_reconstruction(df, corridor_models)

    encoders = {}
    for col in ["event_type", "event_cause", "corridor", "corridor_final", "police_station", "zone", "junction"]:
        fit_label_encoder(df, col, encoders)

    df["is_planned"] = (df["event_type"].str.lower() == "planned").astype(int)
    df["authenticated_flag"] = parse_bool_series(
        df["authenticated"] if "authenticated" in df.columns else pd.Series(False, index=df.index)
    )
    df["planned_x_office_hours"] = df["is_planned"] * df["is_office_hours"]
    df["cause_x_planned"] = df["event_cause_enc"] * df["is_planned"]

    df["road_name"] = df["address"].apply(extract_road_name) if "address" in df.columns else "Unknown"
    fit_label_encoder(df, "road_name", encoders)

    description = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series("", index=df.index)
    address = df["address"].fillna("").astype(str) if "address" in df.columns else pd.Series("", index=df.index)
    df["model_text"] = description + " " + address

    text_lower = df["model_text"].str.lower()
    df["kw_severe"] = text_lower.str.contains(SEVERE_KEYWORD_PATTERN, regex=True).astype(int)
    df["kw_minor"] = text_lower.str.contains(MINOR_KEYWORD_PATTERN, regex=True).astype(int)
    df["kw_slow"] = text_lower.str.contains(SLOW_KEYWORD_PATTERN, regex=True).astype(int)

    for col in ["priority", "direction", "veh_type", "reason_breakdown"]:
        fit_label_encoder(df, col, encoders)

    df["has_vehicle_details"] = (
        df["veh_no"].fillna("").astype(str).str.strip().ne("") if "veh_no" in df.columns
        else pd.Series(False, index=df.index)
    ).astype(int)

    df = df.sort_values("start_dt", na_position="last").reset_index(drop=True)
    df["prev_time_same_corridor"] = df.groupby("corridor_final")["start_dt"].shift(1)
    df["hours_since_last_corridor_incident"] = (
        (df["start_dt"] - df["prev_time_same_corridor"]).dt.total_seconds() / 3600
    ).clip(upper=720).fillna(720)

    df["corridor_24h_count"] = 0.0
    for _, grp in df.groupby("corridor_final"):
        valid = grp["start_dt"].notna()
        times = grp.loc[valid, "start_dt"].values
        idx = grp.loc[valid].index.values
        counts = np.zeros(len(times))
        for i in range(len(times)):
            counts[i] = np.sum(times[:i] >= times[i] - np.timedelta64(24, "h"))
        df.loc[idx, "corridor_24h_count"] = counts

    return df, encoders


def build_rate_maps(train_df: pd.DataFrame, group_cols, target_col="target", smoothing=10):
    global_rate = float(train_df[target_col].mean())
    rate_maps = {}
    for col in group_cols:
        stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_rate * smoothing) / (stats["count"] + smoothing)
        rate_maps[col] = smoothed.to_dict()
    return rate_maps, global_rate


def apply_rate_features(df: pd.DataFrame, rate_maps: dict, global_rate: float):
    for col, mapping in rate_maps.items():
        df[f"{col}_closure_rate"] = df[col].map(mapping).fillna(global_rate)
    return df


def make_feature_matrices(train_df, val_df, test_df):
    structured_features = STRUCTURED_FEATURES + LOGGING_TIME_EXTRA_FEATURES
    X_train_struct = train_df[structured_features].fillna(0).values
    X_val_struct = val_df[structured_features].fillna(0).values
    X_test_struct = test_df[structured_features].fillna(0).values

    tfidf = TfidfVectorizer(max_features=600, min_df=2, ngram_range=(1, 3))
    X_train_text = tfidf.fit_transform(train_df["model_text"]).toarray()
    X_val_text = tfidf.transform(val_df["model_text"]).toarray()
    X_test_text = tfidf.transform(test_df["model_text"]).toarray()
    text_features = [f"desc_tfidf_{w}" for w in tfidf.get_feature_names_out()]

    X_train = np.hstack([X_train_struct, X_train_text])
    X_val = np.hstack([X_val_struct, X_val_text])
    X_test = np.hstack([X_test_struct, X_test_text])

    return (
        X_train, X_val, X_test,
        train_df["target"].values, val_df["target"].values, test_df["target"].values,
        tfidf, structured_features, structured_features + text_features,
    )


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def metric_dict(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def choose_threshold(y_true, y_proba, min_precision=MIN_PRECISION, min_recall=MIN_RECALL):
    rows = [metric_dict(y_true, y_proba, t) for t in np.arange(0.05, 0.951, 0.005)]
    feasible = [r for r in rows if r["precision"] >= min_precision and r["recall"] >= min_recall]
    pool = feasible if feasible else rows
    best = min(pool, key=lambda r: (abs(r["precision"] - r["recall"]), -r["f1"]))
    best["met_constraints"] = bool(feasible)
    return best, rows


def build_blend_models():
    model_a = ExtraTreesClassifier(
        n_estimators=600, max_depth=None, min_samples_leaf=1,
        class_weight={0: 1, 1: 6}, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model_b = RandomForestClassifier(
        n_estimators=450, max_depth=16, min_samples_leaf=1,
        class_weight={0: 1, 1: 4}, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return model_a, model_b, 0.5


def predict_blend_proba(model_a, model_b, weight_a, n_structured, X):
    proba_a = model_a.predict_proba(X)[:, 1]
    proba_b = model_b.predict_proba(X[:, :n_structured])[:, 1]
    return weight_a * proba_a + (1 - weight_a) * proba_b


def blend_feature_importances(model_a, model_b, weight_a, n_structured):
    importances = np.array(model_a.feature_importances_, dtype=float)
    importances[:n_structured] = (
        weight_a * importances[:n_structured]
        + (1 - weight_a) * np.array(model_b.feature_importances_, dtype=float)
    )
    return importances


def build_history_state(df: pd.DataFrame) -> dict:
    history = {}
    for corridor, grp in df.dropna(subset=["start_dt"]).groupby("corridor_final"):
        history[str(corridor)] = [ts.isoformat() for ts in grp["start_dt"].sort_values()]
    return {"corridor_event_times": history}


def run():
    print("=" * 70)
    print("[M4a] Road closure model - improved blend using M2 corridor_final")
    print("=" * 70)

    df, encoders = load_and_engineer(DATA_PATH)
    print(f"[M4a] Dataset: {len(df)} rows | closure rate: {df['target'].mean():.2%}")

    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=RANDOM_STATE
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, stratify=train_val_df["target"], random_state=RANDOM_STATE
    )

    rate_maps, global_rate = build_rate_maps(train_df, RATE_GROUP_COLS)
    train_df = apply_rate_features(train_df.copy(), rate_maps, global_rate)
    val_df = apply_rate_features(val_df.copy(), rate_maps, global_rate)
    test_df = apply_rate_features(test_df.copy(), rate_maps, global_rate)

    X_train, X_val, X_test, y_train, y_val, y_test, tfidf, structured_features, all_features = (
        make_feature_matrices(train_df, val_df, test_df)
    )

    model_a, model_b, weight_a = build_blend_models()
    n_structured = len(structured_features)
    model_a.fit(X_train, y_train)
    model_b.fit(X_train[:, :n_structured], y_train)

    val_proba = predict_blend_proba(model_a, model_b, weight_a, n_structured, X_val)
    chosen, threshold_curve = choose_threshold(y_val, val_proba)
    decision_threshold = chosen["threshold"]

    test_proba = predict_blend_proba(model_a, model_b, weight_a, n_structured, X_test)
    metrics = metric_dict(y_test, test_proba, decision_threshold)
    fixed_threshold_metrics = metric_dict(y_test, test_proba, 0.5)

    model_name = f"blend(ExtraTrees_w6 x{weight_a:.1f} + RandomForest_w4 x{1 - weight_a:.1f})"
    print(f"[M4a] Selected threshold: {decision_threshold:.3f}")
    print(f"[M4a] Test precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
          f"f1={metrics['f1']:.4f} roc_auc={metrics['roc_auc']:.4f}")

    importances = blend_feature_importances(model_a, model_b, weight_a, n_structured)
    importance_df = pd.DataFrame({"feature": all_features, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False)

    bundle = {
        "schema_version": "m4a_blended_closure_v1",
        "model_name": model_name,
        "model_a": model_a,
        "model_b": model_b,
        "weight_a": weight_a,
        "n_structured": n_structured,
        "decision_threshold": decision_threshold,
        "tfidf": tfidf,
        "encoders": encoders,
        "structured_features": structured_features,
        "all_features": all_features,
        "rate_group_cols": RATE_GROUP_COLS,
        "rate_maps": rate_maps,
        "global_rate": global_rate,
        "history": build_history_state(df),
        "road_keywords": ROAD_KEYWORDS,
        "keyword_patterns": {
            "kw_severe": SEVERE_KEYWORD_PATTERN,
            "kw_minor": MINOR_KEYWORD_PATTERN,
            "kw_slow": SLOW_KEYWORD_PATTERN,
        },
    }

    os.makedirs(PIPELINE_DIR, exist_ok=True)
    joblib.dump(bundle, CLOSURE_BUNDLE_PATH)
    joblib.dump(bundle, CLOSURE_MODEL_ALIAS_PATH)
    importance_df.to_csv(IMPORTANCE_PATH, index=False)

    summary = {
        "model_name": model_name,
        "decision_threshold": decision_threshold,
        "feature_pipeline": "raw data.csv + M2 corridor_final + text/rate/temporal features",
        "corridor_source": {
            "column": "corridor_final",
            "rule": "M2 RF+KNN agreement with combined confidence >= 0.80",
        },
        "dataset_rows": int(df.shape[0]),
        "positive_rate": float(df["target"].mean()),
        "validation_selection": chosen,
        "threshold_curve": threshold_curve,
        "metrics": metrics,
        "fixed_threshold_0_50_metrics": fixed_threshold_metrics,
        "feature_count": len(all_features),
        "structured_feature_count": len(structured_features),
        "top_10_features": importance_df.head(10).to_dict(orient="records"),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[M4a] Saved closure bundle -> {CLOSURE_BUNDLE_PATH}")
    print(f"[M4a] Saved metrics -> {METRICS_PATH}")
    return bundle, summary


if __name__ == "__main__":
    run()
