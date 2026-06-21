"""
M4b — CONGESTION IMPACT SCORER (CIS)
=======================================
Two stages:

  STAGE 1 — Compute CIS for all 8,024 rows using the locked formula:
      signal_score = (w_cause + w_planned + w_escalation + w_hour + w_corridor) / 5
      baseline     = (eta_traffic - eta_normal) * TGCF   [from M3, real corridor,
                       or NEAREST-NEIGHBOR fallback baseline for Non-corridor rows]
      raw_score    = baseline * signal_score * 10
      CIS          = min(raw_score / max_possible * 10, 10)
      max_possible = 95th percentile of raw_score (LOCKED, outlier-robust)

  STAGE 2 — Train XGBoost regressor: features -> CIS
      Purpose: generalize the formula to corridor/cause combinations never
      explicitly queried against MapMyIndia, without further API calls.

  NON-CORRIDOR FALLBACK (LOCKED):
      Non-corridor incidents have no defined road stretch to query MapMyIndia
      for. Instead of a city-wide average, we use sklearn NearestNeighbors
      trained on ALL labeled incident coordinates (not corridor centroids -
      this preserves the actual shape of each corridor's road network).
      Each Non-corridor incident inherits the M3 baseline of its single
      nearest labeled neighbor's corridor. Distance to that neighbor is
      stored, so weak matches (far neighbor) are visible, not hidden.

Input  : feature_matrix.csv (M2), clean_incidents.csv (M1, for lat/long),
         eta_baselines.json (M3)
Output : cis_scores.csv         (CIS per row + full audit trail)
         scorer_model.pkl       (trained XGBoost)
         scorer_eval_metrics.json
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_PATH = "/content/drive/MyDrive/flipkart/feature_matrix.csv"
ETA_BASELINES_PATH = "/content/drive/MyDrive/flipkart/eta_baselines.json"
CIS_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/cis_scores.csv"
MODEL_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/scorer_model.pkl"
METRICS_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/scorer_eval_metrics.json"
NN_MODEL_PATH = "/content/drive/MyDrive/flipkart/nearest_corridor_nn.pkl"

DEFAULT_QUERY_HOUR_BUCKET = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]  # FIXED: must match M3's TIME_SLOTS (raised 6->12 slots) - was stale at the old 6-slot bucket, which would have silently returned 0.0 baselines for every hour once M3's real eta_baselines.json (keyed by the new 12-slot hours) is loaded


def load_inputs():
    df = pd.read_csv(FEATURE_PATH)
    with open(ETA_BASELINES_PATH) as f:
        eta_baselines = json.load(f)
    print(f"[M4b] Loaded feature matrix: {df.shape[0]} rows")
    print(f"[M4b] Loaded ETA baselines for {len(eta_baselines)-1} corridors "
          f"(excluding _meta)")
    return df, eta_baselines


def nearest_calibrated_hour(hour: int) -> int:
    """M3 only queried 6 time slots; snap any hour to the nearest one queried."""
    return min(DEFAULT_QUERY_HOUR_BUCKET, key=lambda h: abs(h - hour))


# ──────────────────────────────────────────────────────────────────
# NEAREST-NEIGHBOR FALLBACK FOR NON-CORRIDOR ROWS (LOCKED APPROACH)
# ──────────────────────────────────────────────────────────────────

def build_nearest_corridor_lookup(df: pd.DataFrame):
    """
    Trains NearestNeighbors on every row that has a REAL corridor_final
    (not 'Non-corridor'). Used to assign Non-corridor incidents the
    baseline of their closest actual labeled point - not a corridor
    centroid, so the corridor's true road-network shape is respected.
    """
    known = df[df["corridor_final"] != "Non-corridor"]
    X_known = known[["latitude", "longitude"]].values
    y_known = known["corridor_final"].values

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_known)

    with open(NN_MODEL_PATH, "wb") as f:
        pickle.dump({"model": nn, "labels": y_known}, f)
    print(f"[M4b] Trained nearest-neighbor corridor lookup on {len(known)} labeled points")

    return nn, y_known


def assign_non_corridor_baselines(df: pd.DataFrame, nn, y_known) -> pd.DataFrame:
    """
    For every row with corridor_final == 'Non-corridor', find nearest
    labeled point's corridor, and tag it as the inherited corridor for
    baseline lookup purposes. Original corridor_final value is preserved
    separately for ML feature encoding (M4a uses raw corridor_final code).
    """
    is_non = (df["corridor_final"] == "Non-corridor").values
    n_non = is_non.sum()

    inherited = df["corridor_final"].copy()
    nn_distance_km = pd.Series(0.0, index=df.index)
    baseline_source = pd.Series("own_corridor", index=df.index)

    if n_non > 0:
        X_query = df.loc[is_non, ["latitude", "longitude"]].values
        distances, indices = nn.kneighbors(X_query)
        nearest_corridors = y_known[indices.flatten()]
        # distances are in degrees (lat/lon) - convert to approx km
        distances_km = distances.flatten() * 111

        idx_subset = df.index[is_non]
        inherited.loc[idx_subset] = nearest_corridors
        nn_distance_km.loc[idx_subset] = distances_km
        baseline_source.loc[idx_subset] = "nearest_neighbor_fallback"

        print(f"[M4b] Assigned nearest-neighbor baseline corridor to {n_non} "
              f"Non-corridor rows")
        print(f"[M4b] Nearest-neighbor distance stats (km): "
              f"mean={distances_km.mean():.3f}, median={np.median(distances_km):.3f}, "
              f"max={distances_km.max():.3f}")
        far_matches = (distances_km > 2.0).sum()
        print(f"[M4b] WARNING: {far_matches} rows have nearest-neighbor distance > 2km "
              f"(weak match - baseline borrowed from a fairly distant corridor)")

    df["baseline_corridor"] = inherited
    df["nn_distance_km"] = nn_distance_km
    df["baseline_source"] = baseline_source
    return df


# ──────────────────────────────────────────────────────────────────
# CIS FORMULA
# ──────────────────────────────────────────────────────────────────

def get_baseline_delay(corridor: str, hour: int, eta_baselines: dict) -> float:
    snapped_hour = nearest_calibrated_hour(hour)
    corridor_data = eta_baselines.get(corridor)
    if corridor_data is None:
        # should not happen after nearest-neighbor assignment, but guard anyway
        print(f"[M4b] WARNING: no eta_baselines entry for corridor='{corridor}' "
              f"- returning 0.0 baseline (this will silently understate CIS "
              f"for this row - check for a corridor name mismatch)")
        return 0.0
    hour_data = corridor_data.get(str(snapped_hour))
    if hour_data is None:
        print(f"[M4b] WARNING: corridor='{corridor}' has no entry for snapped "
              f"hour={snapped_hour} (requested hour={hour}) - returning 0.0 "
              f"baseline. This usually means DEFAULT_QUERY_HOUR_BUCKET here is "
              f"out of sync with M3's TIME_SLOTS - check both match exactly.")
        return 0.0
    return hour_data["baseline_delay_2023_min"]


def compute_cis_for_all_rows(df: pd.DataFrame, eta_baselines: dict) -> pd.DataFrame:
    df["baseline_delay_min"] = df.apply(
        lambda r: get_baseline_delay(r["baseline_corridor"], r["hour"], eta_baselines),
        axis=1
    )

    # signal_score already computed in M2 (equal-weight average of 5 signals)
    df["raw_score"] = df["baseline_delay_min"] * df["signal_score"] * 10

    max_possible = df["raw_score"].quantile(0.95)  # LOCKED: 95th percentile, outlier-robust
    print(f"[M4b] max_possible (95th percentile of raw_score): {max_possible:.4f}")

    df["CIS"] = (df["raw_score"] / max_possible * 10).clip(upper=10.0)

    print(f"\n[M4b] CIS distribution:")
    print(df["CIS"].describe())

    return df, max_possible


def validate_cis(df: pd.DataFrame):
    """Single honest validation against the one real outcome we have."""
    mean_closed = df.loc[df["requires_road_closure"] == True, "CIS"].mean()
    mean_not_closed = df.loc[df["requires_road_closure"] == False, "CIS"].mean()
    print(f"\n[M4b] === CIS validation against real closure outcome ===")
    print(f"      Mean CIS (closure=True)  : {mean_closed:.3f}")
    print(f"      Mean CIS (closure=False) : {mean_not_closed:.3f}")
    print(f"      Difference                : {mean_closed - mean_not_closed:+.3f}")
    print(f"      (Honest framing: this is a sanity check, not proof of true severity -")
    print(f"       no continuous severity ground truth exists in this dataset.)")
    return {"mean_cis_closed": mean_closed, "mean_cis_not_closed": mean_not_closed}


# ──────────────────────────────────────────────────────────────────
# STAGE 2 — XGBOOST REGRESSOR
# ──────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "event_cause_code", "corridor_final_code", "police_station_code",
    "event_type_planned", "priority_high", "was_escalated_int",
    "is_weekend_int", "hour", "dow_num", "month",
]


def train_xgb_scorer(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df["CIS"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n[M4b] === XGBoost CIS regressor evaluation (test set, n={len(y_test)}) ===")
    print(f"      MAE  : {mae:.4f}  (on 0-10 scale)")
    print(f"      RMSE : {rmse:.4f}")
    print(f"      R2   : {r2:.4f}")

    importances = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    print(f"\n[M4b] XGBoost feature importances:")
    for feat, imp in importances.items():
        print(f"      {feat:25s}: {imp:.4f}")

    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[M4b] Saved XGBoost model -> {MODEL_OUTPUT_PATH}")

    metrics = {
        "mae": mae, "rmse": rmse, "r2": r2,
        "feature_importances": importances,
        "test_set_size": len(y_test),
    }
    return model, metrics


def run_directional_sanity_checks(df: pd.DataFrame):
    """The 4 sanity checks discussed earlier - directional correctness, not magnitude."""
    print(f"\n[M4b] === Directional sanity checks ===")

    checks = {}

    cause_cis = df.groupby("event_cause")["CIS"].mean()
    construction_cis = cause_cis.get("construction", np.nan)
    breakdown_cis = cause_cis.get("vehicle_breakdown", np.nan)
    checks["construction_gt_breakdown"] = bool(construction_cis > breakdown_cis)
    print(f"      construction ({construction_cis:.2f}) > vehicle_breakdown "
          f"({breakdown_cis:.2f}): {checks['construction_gt_breakdown']}")

    peak_mask = df["hour"].isin([8, 9, 17, 18, 19])
    night_mask = df["hour"].isin([0, 1, 2, 3, 4])
    peak_cis = df.loc[peak_mask, "CIS"].mean()
    night_cis = df.loc[night_mask, "CIS"].mean()
    checks["peak_gt_night"] = bool(peak_cis > night_cis)
    print(f"      peak hours ({peak_cis:.2f}) > night hours ({night_cis:.2f}): "
          f"{checks['peak_gt_night']}")

    closed_cis = df.loc[df["requires_road_closure"] == True, "CIS"].mean()
    not_closed_cis = df.loc[df["requires_road_closure"] == False, "CIS"].mean()
    checks["closure_true_gt_false"] = bool(closed_cis > not_closed_cis)
    print(f"      closure=True ({closed_cis:.2f}) > closure=False ({not_closed_cis:.2f}): "
          f"{checks['closure_true_gt_false']}")

    planned_cis = df.loc[df["event_type_planned"] == 1, "CIS"].mean()
    unplanned_cis = df.loc[df["event_type_planned"] == 0, "CIS"].mean()
    checks["planned_gt_unplanned"] = bool(planned_cis > unplanned_cis)
    print(f"      planned ({planned_cis:.2f}) > unplanned ({unplanned_cis:.2f}): "
          f"{checks['planned_gt_unplanned']}")

    n_passed = sum(checks.values())
    print(f"\n[M4b] Sanity checks passed: {n_passed}/4")

    return checks


def run():
    df, eta_baselines = load_inputs()

    nn_model, y_known = build_nearest_corridor_lookup(df)
    df = assign_non_corridor_baselines(df, nn_model, y_known)

    df, max_possible = compute_cis_for_all_rows(df, eta_baselines)
    validation = validate_cis(df)
    sanity_checks = run_directional_sanity_checks(df)
    xgb_model, xgb_metrics = train_xgb_scorer(df)

    audit_cols = [
        "id", "event_cause", "corridor_final", "baseline_corridor",
        "baseline_source", "nn_distance_km", "hour",
        "signal_score", "baseline_delay_min", "raw_score", "CIS",
        "requires_road_closure",
    ]
    df[audit_cols].to_csv(CIS_OUTPUT_PATH, index=False)
    print(f"\n[M4b] Saved CIS scores + audit trail -> {CIS_OUTPUT_PATH}")

    full_metrics = {
        "max_possible_95th_pct": float(max_possible),
        "validation": validation,
        "sanity_checks": sanity_checks,
        "xgb_regressor": xgb_metrics,
    }
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"[M4b] Saved evaluation metrics -> {METRICS_OUTPUT_PATH}")

    return df, xgb_model, full_metrics


if __name__ == "__main__":
    df, xgb_model, metrics = run()
    print("\n[M4b] Sample CIS output:")
    print(df[["id", "event_cause", "corridor_final", "baseline_source", "CIS"]].head(10).to_string())
