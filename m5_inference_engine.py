"""
M5 - INFERENCE ENGINE
=====================

Takes one live event and runs it through the trained pipeline:

    raw live event -> M2 RF+KNN corridor_final -> police_station
                   -> signal_score / CIS
                   -> M4a improved blended closure model
                   -> one JSON response

Consistency fixes in this version:
  - Live corridor reconstruction uses M2's locked RF+KNN agreement gate
    with combined confidence >= 0.80.
  - The old M4a RF-primary/LR-cross-check logic is removed.
  - Closure inference loads the single M4a closure bundle and reproduces
    the same text, keyword, rate, and temporal features used in training.
  - Live input now accepts optional description and address fields so the
    closure model can use its strongest text signal.
"""

import json
import os
import pickle
import re

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_DIR = "/content/drive/MyDrive/flipkart/"
PIPELINE_DIR = os.environ.get(
    "THEME2_PIPELINE_DIR",
    DEFAULT_PIPELINE_DIR if os.path.isdir(DEFAULT_PIPELINE_DIR) else os.getcwd(),
)

PATHS = {
    "encoding_maps": os.path.join(PIPELINE_DIR, "category_encoding_maps.json"),
    "corridor_model": os.path.join(PIPELINE_DIR, "corridor_model.pkl"),
    "corridor_nn": os.path.join(PIPELINE_DIR, "nearest_corridor_nn.pkl"),
    "police_station_nn": os.path.join(PIPELINE_DIR, "police_station_nn.pkl"),
    "cis_signal_tables": os.path.join(PIPELINE_DIR, "cis_signal_tables.json"),
    "eta_baselines": os.path.join(PIPELINE_DIR, "eta_baselines.json"),
    "closure_bundle": os.path.join(PIPELINE_DIR, "closure_model_bundle.pkl"),
    "closure_model_alias": os.path.join(PIPELINE_DIR, "closure_model.pkl"),
    "closure_metrics": os.path.join(PIPELINE_DIR, "closure_eval_metrics.json"),
    "scorer_model": os.path.join(PIPELINE_DIR, "scorer_model.pkl"),
    "scorer_metrics": os.path.join(PIPELINE_DIR, "scorer_eval_metrics.json"),
    "forecast": os.path.join(PIPELINE_DIR, "forecast.json"),
}

CORRIDOR_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_QUERY_HOUR_BUCKET = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

CIS_FEATURE_COLS = [
    "event_cause_code", "corridor_final_code", "police_station_code",
    "event_type_planned", "priority_high", "was_escalated_int",
    "is_weekend_int", "hour", "dow_num", "month",
]

VALID_EVENT_TYPES = {"planned", "unplanned"}
VALID_PRIORITIES = {"HIGH", "LOW"}

LAT_MIN, LAT_MAX = 12.7, 13.2
LON_MIN, LON_MAX = 77.3, 77.9

ROAD_KEYWORDS = [
    "Outer Ring Road", "Inner Ring Road", "Bellary Road", "Old Airport Road",
    "Old Madras Road", "Mysore Road", "Hosur Road", "Bannerghatta Main Road",
    "Varthur Main Road", "Whitefield Main Road", "Hennur Main Road",
    "Magadi Main Road", "HAL Airport Road", "Mumbai Bengaluru Highway",
    "Tumkur Road", "Sarjapur Road", "Bannerghatta Road",
]


# ---------------------------------------------------------------------------
# ARTIFACT LOADING
# ---------------------------------------------------------------------------

def load_pickle_or_joblib(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


class PipelineArtifacts:
    """Loads all M2-M4 artifacts once for repeated predict_event calls."""

    def __init__(self):
        closure_path = (
            PATHS["closure_bundle"]
            if os.path.exists(PATHS["closure_bundle"])
            else PATHS["closure_model_alias"]
        )

        required = [
            "encoding_maps", "corridor_model", "corridor_nn", "police_station_nn",
            "cis_signal_tables", "eta_baselines", "scorer_model", "scorer_metrics",
        ]
        missing = [name for name in required if not os.path.exists(PATHS[name])]
        if not os.path.exists(closure_path):
            missing.append("closure_model_bundle")
        if missing:
            raise FileNotFoundError(
                f"[M5] Missing required artifacts: {missing}. "
                "Run M1 -> M2 -> M3 -> M4a -> M4b before using M5."
            )

        with open(PATHS["encoding_maps"]) as f:
            self.encoding_maps = json.load(f)

        self.corridor_model = load_pickle_or_joblib(PATHS["corridor_model"])
        if not (isinstance(self.corridor_model, dict) and {"rf", "knn"}.issubset(self.corridor_model)):
            raise ValueError(
                "[M5] corridor_model.pkl must be M2's RF+KNN bundle. "
                "Re-run the updated M2 before using M5."
            )

        with open(PATHS["corridor_nn"], "rb") as f:
            d = pickle.load(f)
            self.corridor_nn_model = d["model"]
            self.corridor_nn_labels = d["labels"]

        with open(PATHS["police_station_nn"], "rb") as f:
            d = pickle.load(f)
            self.police_station_nn_model = d["model"]
            self.police_station_nn_labels = d["labels"]

        with open(PATHS["cis_signal_tables"]) as f:
            self.cis_signal_tables = json.load(f)

        with open(PATHS["eta_baselines"]) as f:
            self.eta_baselines = json.load(f)

        self.closure_bundle = joblib.load(closure_path)
        if self.closure_bundle.get("schema_version") != "m4a_blended_closure_v1":
            raise ValueError(
                "[M5] Closure artifact is not the updated M4a blended closure bundle. "
                "Re-run the updated M4a."
            )

        if os.path.exists(PATHS["closure_metrics"]):
            with open(PATHS["closure_metrics"]) as f:
                self.closure_metrics = json.load(f)
        else:
            self.closure_metrics = {}

        self.scorer_model = load_pickle_or_joblib(PATHS["scorer_model"])
        with open(PATHS["scorer_metrics"]) as f:
            self.scorer_metrics = json.load(f)
        self.scorer_model_class = type(self.scorer_model).__name__

        if os.path.exists(PATHS["forecast"]):
            with open(PATHS["forecast"]) as f:
                self.forecast = json.load(f)
            print("[M5] Loaded M4c forecast context.")
        else:
            self.forecast = None
            print("[M5] NOTE: forecast.json not found; forecast context will be omitted.")

        print(f"[M5] Loaded artifacts from {PIPELINE_DIR}")
        print(f"[M5] Closure model: {self.closure_bundle['model_name']}")
        print(f"[M5] CIS scorer model class: {self.scorer_model_class}")


# ---------------------------------------------------------------------------
# INPUT VALIDATION
# ---------------------------------------------------------------------------

def validate_event_input(event: dict):
    required = ["latitude", "longitude", "event_cause", "event_type", "priority", "timestamp"]
    missing = [k for k in required if k not in event or event[k] is None]
    if missing:
        raise ValueError(f"[M5] Missing required event fields: {missing}")

    lat = float(event["latitude"])
    lon = float(event["longitude"])
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        raise ValueError(
            f"[M5] Coordinates ({lat}, {lon}) fall outside Bengaluru bounds "
            f"(lat {LAT_MIN}-{LAT_MAX}, lon {LON_MIN}-{LON_MAX})."
        )

    event_type = str(event["event_type"]).strip().lower()
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"[M5] event_type must be one of {VALID_EVENT_TYPES}, got '{event_type}'")

    priority = str(event["priority"]).strip().upper()
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"[M5] priority must be one of {VALID_PRIORITIES}, got '{priority}'")

    try:
        ts = pd.to_datetime(event["timestamp"], utc=True)
    except Exception as exc:
        raise ValueError(f"[M5] Could not parse timestamp '{event['timestamp']}': {exc}") from exc

    return {
        "latitude": lat,
        "longitude": lon,
        "event_cause": str(event["event_cause"]).strip().lower(),
        "event_type": event_type,
        "priority": priority,
        "timestamp": ts,
        "was_escalated": bool(event.get("was_escalated", False)),
        "description": str(event.get("description", "") or ""),
        "address": str(event.get("address", "") or ""),
        "authenticated": bool(event.get("authenticated", False)),
        "corridor": str(event.get("corridor", "") or ""),
        "zone": str(event.get("zone", "UNKNOWN") or "UNKNOWN"),
        "junction": str(event.get("junction", "UNKNOWN") or "UNKNOWN"),
        "direction": str(event.get("direction", "UNKNOWN") or "UNKNOWN"),
        "veh_type": str(event.get("veh_type", "UNKNOWN") or "UNKNOWN"),
        "veh_no": str(event.get("veh_no", "") or ""),
        "reason_breakdown": str(event.get("reason_breakdown", "UNKNOWN") or "UNKNOWN"),
    }


# ---------------------------------------------------------------------------
# SHARED FEATURE HELPERS
# ---------------------------------------------------------------------------

def derive_time_features(ts: pd.Timestamp) -> dict:
    return {
        "hour": int(ts.hour),
        "dow_num": int(ts.dayofweek),
        "month": int(ts.month),
        "is_weekend": ts.dayofweek in (5, 6),
    }


def extract_road_name(addr):
    if pd.isna(addr) or str(addr).strip() == "":
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


def safe_label_encode(encoders: dict, col: str, value) -> int:
    encoder = encoders.get(col)
    if encoder is None:
        return 0

    raw = str(value)
    candidates = [raw, raw.lower(), raw.upper(), raw.title(), "UNKNOWN", "Unknown"]
    for candidate in candidates:
        if candidate in encoder.classes_:
            return int(encoder.transform([candidate])[0])

    print(f"[M5] WARNING: '{value}' unseen for M4a encoder '{col}', encoded as 0.")
    return 0


def lookup(table: dict, key, fallback: float):
    return table.get(str(key), fallback)


def lookup_rate(rate_maps: dict, col: str, value, fallback: float) -> float:
    mapping = rate_maps.get(col, {})
    candidates = [value, str(value)]
    if isinstance(value, (int, np.integer, float, np.floating)):
        candidates.append(int(value))
    for candidate in candidates:
        if candidate in mapping:
            return float(mapping[candidate])
    return float(fallback)


def closure_history_features(corridor_final: str, ts: pd.Timestamp, closure_bundle: dict) -> dict:
    history = closure_bundle.get("history", {}).get("corridor_event_times", {})
    raw_times = history.get(str(corridor_final), [])
    if not raw_times:
        return {"hours_since_last_corridor_incident": 720.0, "corridor_24h_count": 0.0}

    times = pd.to_datetime(raw_times, utc=True, errors="coerce")
    times = times[~pd.isna(times)]
    past = times[times < ts]
    if len(past) == 0:
        return {"hours_since_last_corridor_incident": 720.0, "corridor_24h_count": 0.0}

    hours_since = min(float((ts - past.max()).total_seconds() / 3600), 720.0)
    count_24h = float((past >= ts - pd.Timedelta(hours=24)).sum())
    return {
        "hours_since_last_corridor_incident": hours_since,
        "corridor_24h_count": count_24h,
    }


# ---------------------------------------------------------------------------
# LOCATION RESOLUTION
# ---------------------------------------------------------------------------

def resolve_corridor(lat: float, lon: float, artifacts: PipelineArtifacts) -> dict:
    """Matches M2.apply_corridor_reconstruction for a live unlabeled event."""
    rf_model = artifacts.corridor_model["rf"]
    knn_model = artifacts.corridor_model["knn"]
    X = [[lat, lon]]

    rf_probs = rf_model.predict_proba(X)[0]
    rf_idx = rf_probs.argmax()
    rf_pred = rf_model.classes_[rf_idx]
    rf_conf = float(rf_probs[rf_idx])

    knn_probs = knn_model.predict_proba(X)[0]
    knn_idx = knn_probs.argmax()
    knn_pred = knn_model.classes_[knn_idx]
    knn_conf = float(knn_probs[knn_idx])

    agreement = int(rf_pred == knn_pred)
    combined_conf = (rf_conf + knn_conf) / 2
    accepted = agreement == 1 and combined_conf >= CORRIDOR_CONFIDENCE_THRESHOLD

    return {
        "corridor_final": str(rf_pred) if accepted else "Non-corridor",
        "corridor_confidence": float(combined_conf),
        "corridor_agreement": agreement,
        "corridor_source": "predicted" if accepted else "unresolved",
        "rf_corridor_prediction": str(rf_pred),
        "rf_corridor_confidence": round(rf_conf, 4),
        "knn_corridor_prediction": str(knn_pred),
        "knn_corridor_confidence": round(knn_conf, 4),
    }


def resolve_police_station(lat: float, lon: float, artifacts: PipelineArtifacts) -> dict:
    distance, index = artifacts.police_station_nn_model.kneighbors([[lat, lon]])
    station = artifacts.police_station_nn_labels[index.flatten()[0]]
    distance_km = float(distance.flatten()[0]) * 111
    return {"police_station": str(station), "police_station_nn_distance_km": distance_km}


def resolve_baseline_corridor(corridor_final: str, lat: float, lon: float, artifacts: PipelineArtifacts) -> dict:
    if corridor_final != "Non-corridor":
        return {
            "baseline_corridor": corridor_final,
            "baseline_source": "own_corridor",
            "nn_distance_km": 0.0,
        }

    distance, index = artifacts.corridor_nn_model.kneighbors([[lat, lon]])
    nearest_corridor = artifacts.corridor_nn_labels[index.flatten()[0]]
    distance_km = float(distance.flatten()[0]) * 111
    if distance_km > 2.0:
        print(
            f"[M5] WARNING: nearest labeled corridor '{nearest_corridor}' is "
            f"{distance_km:.2f}km away for this Non-corridor event."
        )

    return {
        "baseline_corridor": str(nearest_corridor),
        "baseline_source": "nearest_neighbor_fallback",
        "nn_distance_km": distance_km,
    }


# ---------------------------------------------------------------------------
# CIS FEATURES AND SCORING
# ---------------------------------------------------------------------------

def compute_signal_score(event_cause: str, event_type: str, was_escalated: bool,
                         hour: int, corridor_final: str, artifacts: PipelineArtifacts) -> float:
    t = artifacts.cis_signal_tables
    fb = t["fallback"]
    w1 = lookup(t["w_cause"], event_cause, fb["w_cause"])
    w2 = lookup(t["w_planned"], event_type, fb["w_planned"])
    w3 = lookup(t["w_escalation"], was_escalated, fb["w_escalation"])
    w4 = lookup(t["w_hour"], hour, fb["w_hour"])
    w5 = lookup(t["w_corridor"], corridor_final, fb["w_corridor"])
    return (w1 + w2 + w3 + w4 + w5) / 5


def encode_for_cis(event_cause: str, corridor_final: str, police_station: str,
                   event_type: str, priority: str, was_escalated: bool,
                   is_weekend: bool, artifacts: PipelineArtifacts) -> dict:
    maps = artifacts.encoding_maps
    codes = {}
    for col, val in [
        ("event_cause", event_cause),
        ("corridor_final", corridor_final),
        ("police_station", police_station),
    ]:
        if val in maps[col]:
            codes[f"{col}_code"] = maps[col][val]
        else:
            codes[f"{col}_code"] = -1
            print(f"[M5] WARNING: '{val}' unseen for M2 '{col}', encoded as -1.")

    codes["event_type_planned"] = int(event_type == "planned")
    codes["priority_high"] = int(priority == "HIGH")
    codes["was_escalated_int"] = int(was_escalated)
    codes["is_weekend_int"] = int(is_weekend)
    return codes


def compute_cis(feature_row: dict, signal_score: float, baseline_corridor: str,
                hour: int, artifacts: PipelineArtifacts) -> dict:
    snapped_hour = min(DEFAULT_QUERY_HOUR_BUCKET, key=lambda h: abs(h - hour))
    corridor_data = artifacts.eta_baselines.get(baseline_corridor)
    if corridor_data is None:
        baseline_delay_min = 0.0
        print(f"[M5] WARNING: no eta_baseline for '{baseline_corridor}', using 0.0.")
    else:
        baseline_delay_min = corridor_data.get(str(snapped_hour), {}).get(
            "baseline_delay_2023_min", 0.0
        )

    raw_score = baseline_delay_min * signal_score * 10
    max_possible = artifacts.scorer_metrics.get("max_possible_95th_pct")
    formula_cis = min(raw_score / max_possible * 10, 10.0) if max_possible else None

    X = pd.DataFrame([{c: feature_row[c] for c in CIS_FEATURE_COLS}])
    ml_cis = float(np.clip(artifacts.scorer_model.predict(X)[0], 0.0, 10.0))

    return {
        "cis_formula_based": round(formula_cis, 3) if formula_cis is not None else None,
        "cis_ml_based": round(ml_cis, 3),
        "cis_ml_model_class": artifacts.scorer_model_class,
        "baseline_delay_min": baseline_delay_min,
        "signal_score": round(signal_score, 4),
        "snapped_hour_used_for_baseline": snapped_hour,
    }


# ---------------------------------------------------------------------------
# CLOSURE MODEL FEATURES AND PREDICTION
# ---------------------------------------------------------------------------

def build_closure_feature_row(clean: dict, time_feats: dict, corridor_info: dict,
                              station_info: dict, artifacts: PipelineArtifacts) -> tuple:
    bundle = artifacts.closure_bundle
    encoders = bundle["encoders"]
    corridor_final = corridor_info["corridor_final"]
    hour = time_feats["hour"]
    dow = time_feats["dow_num"]
    is_planned = int(clean["event_type"] == "planned")
    is_office_hours = int(11 <= hour <= 16)

    text = f"{clean.get('description', '')} {clean.get('address', '')}".strip()
    text_lower = text.lower()
    patterns = bundle.get("keyword_patterns", {})
    raw_corridor = clean.get("corridor") or corridor_final

    hist = closure_history_features(corridor_final, clean["timestamp"], bundle)
    road_name = extract_road_name(clean.get("address", ""))

    row = {
        "latitude": clean["latitude"],
        "longitude": clean["longitude"],
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7),
        "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": int(time_feats["is_weekend"]),
        "is_office_hours": is_office_hours,
        "month": time_feats["month"],
        "is_planned": is_planned,
        "authenticated_flag": int(clean.get("authenticated", False)),
        "event_type_enc": safe_label_encode(encoders, "event_type", clean["event_type"]),
        "event_cause_enc": safe_label_encode(encoders, "event_cause", clean["event_cause"]),
        "corridor_enc": safe_label_encode(encoders, "corridor", raw_corridor),
        "police_station_enc": safe_label_encode(encoders, "police_station", station_info["police_station"]),
        "zone_enc": safe_label_encode(encoders, "zone", clean.get("zone", "UNKNOWN")),
        "junction_enc": safe_label_encode(encoders, "junction", clean.get("junction", "UNKNOWN")),
        "planned_x_office_hours": is_planned * is_office_hours,
        "cause_x_planned": safe_label_encode(encoders, "event_cause", clean["event_cause"]) * is_planned,
        "hours_since_last_corridor_incident": hist["hours_since_last_corridor_incident"],
        "corridor_24h_count": hist["corridor_24h_count"],
        "road_name_enc": safe_label_encode(encoders, "road_name", road_name),
        "corridor_final_enc": safe_label_encode(encoders, "corridor_final", corridor_final),
        "corridor_confidence": corridor_info["corridor_confidence"],
        "corridor_agreement": corridor_info["corridor_agreement"],
        "priority_enc": safe_label_encode(encoders, "priority", clean["priority"]),
        "direction_enc": safe_label_encode(encoders, "direction", clean.get("direction", "UNKNOWN")),
        "veh_type_enc": safe_label_encode(encoders, "veh_type", clean.get("veh_type", "UNKNOWN")),
        "reason_breakdown_enc": safe_label_encode(encoders, "reason_breakdown", clean.get("reason_breakdown", "UNKNOWN")),
        "has_vehicle_details": int(bool(clean.get("veh_no", "").strip())),
        "kw_severe": int(bool(re.search(patterns.get("kw_severe", r"$^"), text_lower))),
        "kw_minor": int(bool(re.search(patterns.get("kw_minor", r"$^"), text_lower))),
        "kw_slow": int(bool(re.search(patterns.get("kw_slow", r"$^"), text_lower))),
    }

    rate_values = {
        "event_cause": clean["event_cause"],
        "corridor_final": corridor_final,
        "hour": hour,
        "police_station": station_info["police_station"],
    }
    for col in bundle.get("rate_group_cols", []):
        row[f"{col}_closure_rate"] = lookup_rate(
            bundle.get("rate_maps", {}), col, rate_values[col], bundle.get("global_rate", 0.083)
        )

    return row, text


def predict_closure(clean: dict, time_feats: dict, corridor_info: dict,
                    station_info: dict, artifacts: PipelineArtifacts) -> dict:
    bundle = artifacts.closure_bundle
    row, text = build_closure_feature_row(clean, time_feats, corridor_info, station_info, artifacts)

    structured_features = bundle["structured_features"]
    X_struct = np.array([[row.get(f, 0) for f in structured_features]], dtype=float)
    X_text = bundle["tfidf"].transform([text]).toarray()
    X = np.hstack([X_struct, X_text])

    model_a = bundle["model_a"]
    model_b = bundle["model_b"]
    weight_a = bundle["weight_a"]
    n_structured = bundle["n_structured"]
    proba_a = model_a.predict_proba(X)[0, 1]
    proba_b = model_b.predict_proba(X[:, :n_structured])[0, 1]
    proba = float(weight_a * proba_a + (1 - weight_a) * proba_b)
    threshold = float(bundle["decision_threshold"])
    pred = bool(proba >= threshold)

    metrics = artifacts.closure_metrics.get("metrics", {})
    return {
        "requires_road_closure_predicted": pred,
        "closure_probability": round(proba, 4),
        "decision_threshold": round(threshold, 4),
        "model": bundle["model_name"],
        "model_roc_auc": round(metrics.get("roc_auc"), 4) if metrics.get("roc_auc") is not None else None,
        "text_available": bool(text.strip()),
        "model_a_probability": round(float(proba_a), 4),
        "model_b_structured_probability": round(float(proba_b), 4),
    }


# ---------------------------------------------------------------------------
# FORECAST CONTEXT
# ---------------------------------------------------------------------------

def attach_forecast_context(corridor_final: str, artifacts: PipelineArtifacts) -> dict:
    if artifacts.forecast is None:
        return {"available": False, "note": "M4c has not been run yet"}

    corridor_forecast = artifacts.forecast.get(corridor_final)
    if corridor_forecast is None:
        return {"available": False, "note": f"No forecast entry for '{corridor_final}'"}

    return {
        "available": True,
        "confidence": corridor_forecast.get("confidence"),
        "avg_weekly_incidents_historical": corridor_forecast.get("avg_weekly_incidents_historical"),
        "next_week_forecast": (corridor_forecast.get("next_weeks_forecast") or [None])[0],
    }


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def predict_event(event: dict, artifacts: PipelineArtifacts) -> dict:
    """
    Required event keys:
        latitude, longitude, event_cause, event_type, priority, timestamp

    Optional event keys:
        was_escalated, description, address, authenticated, corridor, zone,
        junction, direction, veh_type, veh_no, reason_breakdown
    """
    clean = validate_event_input(event)

    time_feats = derive_time_features(clean["timestamp"])
    corridor_info = resolve_corridor(clean["latitude"], clean["longitude"], artifacts)
    station_info = resolve_police_station(clean["latitude"], clean["longitude"], artifacts)
    baseline_info = resolve_baseline_corridor(
        corridor_info["corridor_final"], clean["latitude"], clean["longitude"], artifacts
    )

    signal_score = compute_signal_score(
        clean["event_cause"], clean["event_type"], clean["was_escalated"],
        time_feats["hour"], corridor_info["corridor_final"], artifacts
    )

    cis_codes = encode_for_cis(
        clean["event_cause"], corridor_info["corridor_final"], station_info["police_station"],
        clean["event_type"], clean["priority"], clean["was_escalated"],
        time_feats["is_weekend"], artifacts
    )
    cis_feature_row = {
        **cis_codes,
        "hour": time_feats["hour"],
        "dow_num": time_feats["dow_num"],
        "month": time_feats["month"],
    }

    closure_result = predict_closure(clean, time_feats, corridor_info, station_info, artifacts)
    cis_result = compute_cis(
        cis_feature_row, signal_score, baseline_info["baseline_corridor"],
        time_feats["hour"], artifacts
    )
    forecast_context = attach_forecast_context(corridor_info["corridor_final"], artifacts)

    return {
        "input": {
            "latitude": clean["latitude"],
            "longitude": clean["longitude"],
            "event_cause": clean["event_cause"],
            "event_type": clean["event_type"],
            "priority": clean["priority"],
            "timestamp": clean["timestamp"].isoformat(),
            "was_escalated": clean["was_escalated"],
            "description": clean["description"],
            "address": clean["address"],
        },
        "resolved_location": {
            **corridor_info,
            **station_info,
        },
        "baseline_lookup": baseline_info,
        "closure_prediction": closure_result,
        "congestion_impact_score": cis_result,
        "corridor_forecast_context": forecast_context,
    }


def run_demo():
    artifacts = PipelineArtifacts()
    demo_events = [
        {
            "latitude": 12.9716,
            "longitude": 77.5573,
            "event_cause": "vehicle_breakdown",
            "event_type": "unplanned",
            "priority": "LOW",
            "timestamp": "2026-06-20T17:30:00",
            "description": "lorry breakdown on left lane, slow traffic",
            "address": "Mysore Road, Bengaluru",
        },
        {
            "latitude": 13.0050,
            "longitude": 77.5700,
            "event_cause": "vip_movement",
            "event_type": "planned",
            "priority": "HIGH",
            "timestamp": "2026-06-20T09:00:00",
            "description": "traffic diverted and road closed for convoy movement",
            "address": "Bellary Road, Bengaluru",
        },
    ]

    for i, event in enumerate(demo_events, start=1):
        print(f"\n{'=' * 70}\n[M5] DEMO EVENT {i}\n{'=' * 70}")
        result = predict_event(event, artifacts)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_demo()
