"""
M2 — FEATURE ENGINEERING
==========================
Two jobs, in order:

  PART A — Corridor Reconstruction
    ~38% of rows are labeled "Non-corridor". Two independent models -
    a Random Forest and a KNN (k=15, distance-weighted), both trained on
    lat/long alone - each predict a corridor for these rows.
    LOCKED RULE (updated to match the validated notebook methodology):
    never override a real corridor label. Only fill in "Non-corridor"
    (or missing) rows, and only when RF and KNN AGREE on the predicted
    corridor AND their combined confidence >= 0.80. Disagreement or low
    confidence leaves the row as "Non-corridor" - we trust silence over
    a guess from a single model.
    Produces `corridor_final`, used everywhere downstream instead of
    the raw `corridor` column (interface unchanged from before - only
    the reconstruction logic behind it changed).

  PART B — CIS Signal Tables + ML Feature Matrix
    5 signals (cause, planned, escalation, hour, corridor_final), each a
    real closure-rate lookup from this dataset. Combined via SIMPLE
    AVERAGE (locked decision — no learned/unequal weighting, since no
    correct target exists to learn those weights against without
    borrowing M4a's closure target, which would be circular).

  PART D — Categorical Encoding Maps + Police Station Lookup (for M5)
    pd.factorize() assigns codes by first-appearance order in THIS run -
    not alphabetical, not stable across re-runs. If M5 re-factorized a
    single live event row at inference time, it would almost certainly
    get DIFFERENT codes than training used, silently corrupting every
    downstream prediction. So the exact category->code mapping produced
    here is saved to disk and must be reused verbatim at inference time,
    never recomputed.
    Also trains a NearestNeighbors lookup for police_station (same
    pattern as M4b's corridor fallback) - a live event typically arrives
    as lat/long, not a known police_station name, so M5 needs a way to
    infer the nearest station from coordinates alone.

Input  : clean_incidents.csv (from M1)
Output : feature_matrix.csv
         cis_signal_tables.json
         corridor_reconstruction.csv     (id + raw corridor + locked corridor_final diagnostics)
         corridor_model.pkl              (dict: {"rf":..., "knn":...} - both
                                           needed to reproduce the ensemble
                                           at M5 inference time)
         category_encoding_maps.json     (LOCKED category->code maps, reused by M5)
         police_station_nn.pkl           (nearest-station-from-coords lookup, used by M5)
"""

import pandas as pd
import numpy as np
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

INPUT_PATH = "/content/drive/MyDrive/flipkart/clean_incidents.csv"
FEATURE_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/feature_matrix.csv"
SIGNALS_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/cis_signal_tables.json"
CORRIDOR_RECON_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/corridor_reconstruction.csv"
CORRIDOR_MODEL_PATH = "/content/drive/MyDrive/flipkart/corridor_model.pkl"
ENCODING_MAPS_PATH = "/content/drive/MyDrive/flipkart/category_encoding_maps.json"
POLICE_STATION_NN_PATH = "/content/drive/MyDrive/flipkart/police_station_nn.pkl"

CORRIDOR_CONFIDENCE_THRESHOLD = 0.80  # LOCKED - matches notebook's RF+KNN ensemble.
                                       # Also requires RF/KNN agreement (see apply_corridor_reconstruction).
                                       # Was 0.95 under the old single-RF strategy; not comparable 1:1
                                       # since this threshold now applies to a combined-confidence score
                                       # gated by agreement, not one model's raw confidence.


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[M2] Loaded clean dataset: {df.shape[0]} rows")
    return df


# ──────────────────────────────────────────────────────────────────
# PART A — CORRIDOR RECONSTRUCTION
# ──────────────────────────────────────────────────────────────────

def train_corridor_model(df: pd.DataFrame):
    """
    Train an RF and a KNN, both on lat/long -> corridor, using ONLY rows
    that already have a real (non 'Non-corridor', non-missing) label.
    Both models are later reused at M5 inference time (a live event needs
    both predictions to compute agreement), so both are saved to disk
    together as a dict: {"rf": rf_model, "knn": knn_model}.

    Params match the validated notebook run exactly:
      RF  - RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
      KNN - KNeighborsClassifier(n_neighbors=15, weights="distance")
    """
    labeled = df[df["corridor"].notna() & (df["corridor"] != "Non-corridor")].copy()
    print(f"[M2] Training corridor models on {len(labeled)} labeled rows "
          f"({labeled['corridor'].nunique()} unique corridors)")

    X = labeled[["latitude", "longitude"]].values
    y = labeled["corridor"].values

    rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf_model.fit(X, y)
    rf_train_acc = rf_model.score(X, y)
    print(f"[M2] RF corridor model training accuracy: {rf_train_acc:.4f}")

    knn_model = KNeighborsClassifier(n_neighbors=15, weights="distance")
    knn_model.fit(X, y)
    knn_train_acc = knn_model.score(X, y)
    print(f"[M2] KNN corridor model training accuracy: {knn_train_acc:.4f}")
    # NOTE: both are train accuracy as a sanity check only. For the real
    # held-out evaluation number to quote to judges, run a proper
    # train/test split separately when building the evaluation slide.

    models = {"rf": rf_model, "knn": knn_model}
    with open(CORRIDOR_MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print(f"[M2] Saved corridor models (RF + KNN) -> {CORRIDOR_MODEL_PATH}")

    return models


def apply_corridor_reconstruction(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    LOCKED RULE (RF+KNN ensemble):
      - corridor already real (not 'Non-corridor', not missing) -> keep as-is, untouched
      - corridor missing or 'Non-corridor'                      -> predict with both
                                                                     RF and KNN; accept
                                                                     the RF prediction
                                                                     only if RF and KNN
                                                                     AGREE and their
                                                                     combined confidence
                                                                     >= 0.80, else stays
                                                                     'Non-corridor'
    Output columns are unchanged from the old single-RF version
    (corridor_final, corridor_confidence, corridor_source), so nothing
    downstream (CIS tables, ML feature matrix, M5) needs to change.
    `corridor_agreement` is new - 1 where RF/KNN agree (or original
    label), 0 where they disagree - kept for diagnostics/M5 reuse.
    """
    rf_model, knn_model = models["rf"], models["knn"]

    needs_prediction = (df["corridor"].isna() | (df["corridor"] == "Non-corridor")).values
    n_to_predict = needs_prediction.sum()
    print(f"[M2] Rows needing corridor prediction: {n_to_predict} / {len(df)}")

    corridor_final = df["corridor"].copy()
    corridor_confidence = pd.Series(1.0, index=df.index)   # 1.0 = original/known
    corridor_agreement = pd.Series(1, index=df.index)      # 1 = original/known (n/a -> treated as agreeing)
    corridor_source = pd.Series("original", index=df.index)

    if n_to_predict > 0:
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

        accepted_labels = np.where(accept_mask, rf_pred, "Non-corridor")
        corridor_final.loc[idx_subset] = accepted_labels

        n_accepted = accept_mask.sum()
        print(f"[M2] Predictions accepted (RF/KNN agree AND combined conf >= "
              f"{CORRIDOR_CONFIDENCE_THRESHOLD}): {n_accepted} / {n_to_predict} "
              f"({n_accepted/n_to_predict*100:.1f}%)")
        print(f"[M2] RF/KNN agreement rate on predicted rows: {agreement.mean():.4f}")

    df["corridor_final"] = corridor_final
    df["corridor_confidence"] = corridor_confidence
    df["corridor_agreement"] = corridor_agreement
    df["corridor_source"] = corridor_source

    print("[M2] corridor_source breakdown:")
    print(df["corridor_source"].value_counts().to_string())

    return df


# ──────────────────────────────────────────────────────────────────
# PART B — CIS SIGNAL TABLES (5 signals, EQUAL average — locked decision)
# ──────────────────────────────────────────────────────────────────

def build_cis_signal_tables(df: pd.DataFrame) -> dict:
    """
    5 signals, all real closure-rate lookups from this dataset.
    Combined via simple average in compute_signal_score_per_row().

    LOCKED REASONING (do not revert without re-discussing):
    We considered learning unequal weights via Logistic Regression against
    requires_road_closure, but rejected it - CIS is meant to drive resource/
    barricade/diversion decisions, not predict closure (that's M4a's job,
    trained separately). Borrowing M4a's target to weight CIS's inputs would
    be circular and would conflate two different questions. No correct
    target exists to learn resource-need weights against, so equal
    averaging is the neutral, defensible choice until such data exists.
    """
    w_cause = df.groupby("event_cause")["requires_road_closure"].mean().to_dict()
    w_planned = df.groupby("event_type")["requires_road_closure"].mean().to_dict()

    w_escalation = df.groupby("was_escalated")["requires_road_closure"].mean().to_dict()
    w_escalation = {str(k): v for k, v in w_escalation.items()}

    w_hour = df.groupby("hour")["requires_road_closure"].mean().to_dict()
    w_hour = {str(k): v for k, v in w_hour.items()}

    # uses corridor_final (post-reconstruction), not raw corridor
    w_corridor = df.groupby("corridor_final")["requires_road_closure"].mean().to_dict()

    global_rate = df["requires_road_closure"].mean()

    tables = {
        "w_cause": w_cause,
        "w_planned": w_planned,
        "w_escalation": w_escalation,
        "w_hour": w_hour,
        "w_corridor": w_corridor,
        "fallback": {
            "w_cause": global_rate,
            "w_planned": global_rate,
            "w_escalation": global_rate,
            "w_hour": global_rate,
            "w_corridor": global_rate,
        }
    }

    print("[M2] Built 5 CIS signal tables (using corridor_final for w_corridor):")
    print(f"      w_cause      : {len(w_cause)} causes")
    print(f"      w_planned    : {len(w_planned)} event types")
    print(f"      w_escalation : {len(w_escalation)} states")
    print(f"      w_hour       : {len(w_hour)} hours")
    print(f"      w_corridor   : {len(w_corridor)} corridors")

    return tables


def lookup(table: dict, key, fallback: float):
    return table.get(str(key), fallback)


def compute_signal_score_per_row(df: pd.DataFrame, tables: dict) -> pd.Series:
    """LOCKED: simple (equal-weight) average of the 5 signals. Range 0-1."""
    fb = tables["fallback"]

    w1 = df["event_cause"].map(lambda x: lookup(tables["w_cause"], x, fb["w_cause"]))
    w2 = df["event_type"].map(lambda x: lookup(tables["w_planned"], x, fb["w_planned"]))
    w3 = df["was_escalated"].map(lambda x: lookup(tables["w_escalation"], x, fb["w_escalation"]))
    w4 = df["hour"].map(lambda x: lookup(tables["w_hour"], x, fb["w_hour"]))
    w5 = df["corridor_final"].map(lambda x: lookup(tables["w_corridor"], x, fb["w_corridor"]))

    signal_score = (w1 + w2 + w3 + w4 + w5) / 5
    return signal_score


# ──────────────────────────────────────────────────────────────────
# PART C — ML FEATURE MATRIX (for M4a classifier + M4b regressor)
# ──────────────────────────────────────────────────────────────────

def encode_categoricals_for_ml(df: pd.DataFrame):
    """
    Returns (ml_df, encoding_maps). encoding_maps is the EXACT category->code
    mapping pd.factorize() produced in this run - this is what must be saved
    and reused at inference time (see PART D docstring above for why).
    """
    ml_df = df.copy()
    encoding_maps = {}

    for col in ["event_cause", "corridor_final", "police_station"]:
        codes, uniques = pd.factorize(ml_df[col])
        ml_df[f"{col}_code"] = codes
        # uniques is in first-appearance order = the actual code assignment;
        # build the explicit category -> code dict from it
        encoding_maps[col] = {str(cat): int(i) for i, cat in enumerate(uniques)}

    ml_df["event_type_planned"] = (ml_df["event_type"] == "planned").astype(int)
    ml_df["priority_high"] = (ml_df["priority"] == "HIGH").astype(int)
    ml_df["was_escalated_int"] = ml_df["was_escalated"].astype(int)
    ml_df["is_weekend_int"] = ml_df["is_weekend"].astype(int)

    return ml_df, encoding_maps


def train_police_station_nn(df: pd.DataFrame):
    """
    NearestNeighbors on (latitude, longitude) -> police_station, same
    pattern as M4b's corridor fallback. A live event at M5 inference time
    arrives as coordinates, not a known station name, so this lets M5
    infer the nearest real station rather than requiring it as a manual
    input field (LOCKED decision - see M5 discussion).
    Trained on ALL rows (police_station has 0% missing in this dataset,
    unlike corridor), so there is no "unknown station" case to handle here.
    """
    X = df[["latitude", "longitude"]].values
    y = df["police_station"].values

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X)

    with open(POLICE_STATION_NN_PATH, "wb") as f:
        pickle.dump({"model": nn, "labels": y}, f)
    print(f"[M2] Trained police_station nearest-neighbor lookup on {len(df)} points "
          f"({df['police_station'].nunique()} unique stations) -> {POLICE_STATION_NN_PATH}")

    return nn


def run():
    df = load_clean(INPUT_PATH)

    # ---- PART A: corridor reconstruction ----
    corridor_model = train_corridor_model(df)
    df = apply_corridor_reconstruction(df, corridor_model)

    # ---- PART B: CIS signal tables ----
    tables = build_cis_signal_tables(df)
    df["signal_score"] = compute_signal_score_per_row(df, tables)

    print(f"\n[M2] signal_score stats:")
    print(df["signal_score"].describe())

    # validation: does signal_score separate real closure outcomes?
    mean_true = df.loc[df["requires_road_closure"] == True, "signal_score"].mean()
    mean_false = df.loc[df["requires_road_closure"] == False, "signal_score"].mean()
    corr = df["signal_score"].corr(df["requires_road_closure"].astype(int))
    print(f"\n[M2] Validation - signal_score by closure outcome:")
    print(f"      mean(closure=True)  = {mean_true:.4f}")
    print(f"      mean(closure=False) = {mean_false:.4f}")
    print(f"      correlation         = {corr:.4f}")

    # ---- PART C: ML feature matrix ----
    ml_df, encoding_maps = encode_categoricals_for_ml(df)

    feature_cols = [
        "id", "event_cause_code", "corridor_final_code", "police_station_code",
        "event_type_planned", "priority_high", "was_escalated_int",
        "is_weekend_int", "hour", "dow_num", "month",
        "signal_score",
        "latitude", "longitude",
        "corridor_final", "event_cause", "corridor_confidence",
        "corridor_agreement", "corridor_source",
        "requires_road_closure",  # target for M4a
    ]
    feature_matrix = ml_df[feature_cols].copy()

    feature_matrix.to_csv(FEATURE_OUTPUT_PATH, index=False)
    corridor_recon_cols = [
        "id", "latitude", "longitude", "corridor", "corridor_final",
        "corridor_confidence", "corridor_agreement", "corridor_source",
    ]
    df[[c for c in corridor_recon_cols if c in df.columns]].to_csv(
        CORRIDOR_RECON_OUTPUT_PATH, index=False
    )
    with open(SIGNALS_OUTPUT_PATH, "w") as f:
        json.dump(tables, f, indent=2)

    print(f"\n[M2] Saved feature matrix -> {FEATURE_OUTPUT_PATH}")
    print(f"[M2] Saved locked corridor reconstruction -> {CORRIDOR_RECON_OUTPUT_PATH}")
    print(f"[M2] Saved CIS signal tables -> {SIGNALS_OUTPUT_PATH}")
    print(f"[M2] Feature matrix shape: {feature_matrix.shape}")

    # ---- PART D: encoding maps + police station NN (for M5) ----
    with open(ENCODING_MAPS_PATH, "w") as f:
        json.dump(encoding_maps, f, indent=2)
    print(f"[M2] Saved LOCKED category encoding maps -> {ENCODING_MAPS_PATH}")
    for col, mapping in encoding_maps.items():
        print(f"      {col:18s}: {len(mapping)} categories")

    police_station_nn = train_police_station_nn(df)

    return feature_matrix, tables, corridor_model, encoding_maps, police_station_nn


if __name__ == "__main__":
    feature_matrix, tables, corridor_model, encoding_maps, police_station_nn = run()
    print("\n[M2] Sample of feature matrix:")
    print(feature_matrix.head(5).to_string())
