"""
M1 — DATA INGESTION
====================
Reads the raw incident CSV, cleans it, parses all timestamp columns,
extracts the top corridors, computes per-row helper fields needed by
downstream modules (M2 onwards), and saves a clean dataset.

Input  : Astram_event_data_anonymized.csv
Output : clean_incidents.csv
"""

import pandas as pd
import numpy as np

RAW_PATH = "/content/drive/MyDrive/flipkart/Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv"
OUTPUT_PATH = "/content/drive/MyDrive/flipkart/clean_incidents.csv"

# Bengaluru bounding box - used to sanity check lat/long
LAT_MIN, LAT_MAX = 12.7, 13.2
LON_MIN, LON_MAX = 77.3, 77.9


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[M1] Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse every datetime column we might need. Invalid -> NaT, not dropped."""
    ts_cols = ["start_datetime", "end_datetime", "modified_datetime",
               "closed_datetime", "resolved_datetime", "created_date"]
    for col in ts_cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    print(f"[M1] Parsed {len(ts_cols)} timestamp columns")
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that are unusable for the whole pipeline - missing core fields."""
    before = len(df)

    df = df[df["start_datetime"].notna()]
    df = df[
        df["latitude"].between(LAT_MIN, LAT_MAX) &
        df["longitude"].between(LON_MIN, LON_MAX)
    ]
    df = df[df["event_cause"].notna()]
    df = df[df["priority"].notna()]

    after = len(df)
    print(f"[M1] Dropped {before - after} invalid rows ({before} -> {after})")
    return df


def standardize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize text casing so 'Debris' and 'debris' aren't treated as different causes."""
    df["event_cause"] = df["event_cause"].str.strip().str.lower()
    df["event_type"] = df["event_type"].str.strip().str.lower()
    df["priority"] = df["priority"].str.strip().str.upper()
    df["status"] = df["status"].str.strip().str.lower()
    df["corridor"] = df["corridor"].fillna("Non-corridor").str.strip()
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour / day-of-week / month - needed by M2 feature engineering and CIS signals."""
    df["hour"] = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.day_name()
    df["dow_num"] = df["start_datetime"].dt.dayofweek  # 0=Monday
    df["month"] = df["start_datetime"].dt.month
    df["is_weekend"] = df["dow_num"].isin([5, 6])
    return df


def compute_escalation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Signal 4 from CIS design: was the record modified by someone other than its creator."""
    df["was_escalated"] = (
        df["last_modified_by_id"].notna()
        & df["created_by_id"].notna()
        & (df["last_modified_by_id"] != df["created_by_id"])
    )
    return df


def compute_stretch_km(df: pd.DataFrame) -> pd.DataFrame:
    """
    Signal 3 from CIS design: physical length of road affected.

    DATA QUALITY FINDING: most endlatitude/endlongitude values are placeholder
    coordinates far outside Bengaluru (~8700km away), not real endpoints.
    We only trust a stretch value when:
      1. end coordinates exist, AND
      2. the end point falls inside the Bengaluru bounding box, AND
      3. the computed stretch is plausible for a single incident (<= 30km)
    Anything else -> stretch unknown (NaN), not zero.
    """
    has_end = df["endlatitude"].notna() & df["endlongitude"].notna()
    end_in_bbox = (
        df["endlatitude"].between(LAT_MIN, LAT_MAX) &
        df["endlongitude"].between(LON_MIN, LON_MAX)
    )

    stretch = np.full(len(df), np.nan)
    mask = (has_end & end_in_bbox).values
    lat_diff = (df.loc[mask, "endlatitude"] - df.loc[mask, "latitude"]).values
    lon_diff = (df.loc[mask, "endlongitude"] - df.loc[mask, "longitude"]).values
    raw_stretch = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # deg -> km approx

    plausible = raw_stretch <= 30
    final_stretch = np.where(plausible, raw_stretch, np.nan)
    stretch[mask] = final_stretch

    df["stretch_km"] = stretch
    df["has_valid_stretch"] = ~np.isnan(stretch)

    n_valid = df["has_valid_stretch"].sum()
    print(f"[M1] Valid (plausible, in-bbox) stretch values: {n_valid} / {len(df)} rows "
          f"({n_valid/len(df)*100:.1f}%)")
    return df


def get_top_corridors(df: pd.DataFrame, n: int = 20) -> list:
    """These are the corridors M3 will query MapMyIndia for (cost control)."""
    counts = df["corridor"].value_counts()
    top = counts.head(n).index.tolist()
    print(f"[M1] Top {n} corridors by incident count:")
    for c in top:
        print(f"      {c}: {counts[c]} incidents")
    return top


def run():
    df = load_raw(RAW_PATH)
    df = parse_timestamps(df)
    df = drop_invalid_rows(df)
    df = standardize_categoricals(df)
    df = extract_time_features(df)
    df = compute_escalation_flag(df)
    df = compute_stretch_km(df)

    # NOTE FOR BUDGET FLEXIBILITY:
    # n=20 controls how many corridors M3 will query MapMyIndia for (cost control,
    # since each corridor needs ~6 time-slot API calls for TGCF calibration).
    # If leftover ₹ budget remains after the first pass, increase n here (e.g. n=30
    # or n=40) and re-run M1 -> M3 to extend coverage to more (lower-traffic) corridors.
    # Corridors are already ranked by incident count, so increasing n only adds
    # progressively less critical corridors - the top 20 always stay first.
    top_corridors = get_top_corridors(df, n=20)

    keep_cols = [
        "id", "event_type", "event_cause", "requires_road_closure",
        "priority", "status", "corridor", "police_station", "zone",
        "latitude", "longitude", "junction",
        "start_datetime", "hour", "day_of_week", "dow_num", "month", "is_weekend",
        "was_escalated", "has_valid_stretch", "stretch_km",
        "veh_type",
    ]
    df_clean = df[keep_cols].copy()

    df_clean.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[M1] Saved clean dataset -> {OUTPUT_PATH}")
    print(f"[M1] Final shape: {df_clean.shape}")

    return df_clean, top_corridors


if __name__ == "__main__":
    df_clean, top_corridors = run()
    print("\n[M1] Sample of cleaned data:")
    print(df_clean.head(5).to_string())
