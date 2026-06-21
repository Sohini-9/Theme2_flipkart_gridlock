"""
M4c — INCIDENT FORECASTER (Prophet)
======================================
Forecasts weekly incident VOLUME per corridor (count of incidents/week),
NOT severity and NOT closure probability - those are M4a and M4b's jobs.
This module answers a third, separate question: "how many incidents should
we expect on this corridor next week", which feeds resource pre-positioning
in M6.

One Prophet model per real corridor (uses corridor_final from M2, the
post-reconstruction column - NOT raw corridor). 'Non-corridor' is included
as its own series (it is a real, large bucket - ~38% of incidents - not
something to silently drop), but is clearly flagged as a mixed-geography
aggregate rather than a real road, same honesty standard as M4b's
baseline_source flag.

DATA VOLUME REALITY (checked before building, not assumed):
  Dataset spans only 23 calendar weeks (2023-11-09 to 2024-04-08).
  Per LOCKED decision: every corridor is forecast regardless of volume,
  but corridors with thin weekly counts get a LOW_CONFIDENCE flag rather
  than being silently presented with the same trust as a high-volume
  corridor. See CONFIDENCE_MIN_AVG_WEEKLY / CONFIDENCE_MIN_NONZERO_WEEKS
  below for the exact thresholds and reasoning.

EVALUATION (per Project Summary section 7 - LOCKED):
  Temporal train/test split: train = Nov 2023 - Feb 2024, test = Mar - Apr 2024.
  This is a TIME split, not a random split (random split would leak future
  weeks into training for a time-series problem - same reasoning M4a/M4b
  use stratified random splits for, but those targets are NOT time-ordered
  the way weekly counts are, so the split strategy is deliberately different
  here, not an inconsistency).
  Metrics: MAE, MAPE, actual-vs-predicted per corridor.

STATUS: Prophet is NOT installed in the authoring environment and there is
no network access to install it here, so this module is WRITTEN AND
REVIEWED FOR CORRECTNESS/CONSISTENCY BUT NOT YET EXECUTED. Run it yourself
in an environment with `pip install prophet` and report back actual MAE/MAPE
numbers - do not assume the numbers in any other document until this has
actually run once.

Input  : clean_incidents.csv (M1, for start_datetime)
         feature_matrix.csv  (M2, for corridor_final - post-reconstruction)
Output : forecast.json              (next-4-week forecast per corridor + flags)
         forecaster_eval_metrics.json (MAE/MAPE per corridor, temporal split)
         prophet_models/<corridor>.pkl  (one fitted model per corridor, for M5 reuse)
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
import re

from prophet import Prophet

CLEAN_DATA_PATH = "/content/drive/MyDrive/flipkart/clean_incidents.csv"
FEATURE_MATRIX_PATH = "/content/drive/MyDrive/flipkart/feature_matrix.csv"
FORECAST_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/forecast.json"
METRICS_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/forecaster_eval_metrics.json"
MODEL_DIR = "/content/drive/MyDrive/flipkart/prophet_models/"

# ──────────────────────────────────────────────────────────────────
# LOCKED CONFIG
# ──────────────────────────────────────────────────────────────────

TRAIN_END = pd.Timestamp("2024-02-29", tz="UTC")   # train: start of data -> Feb 2024 inclusive
TEST_START = pd.Timestamp("2024-03-01", tz="UTC")  # test: Mar 2024 -> end of data (Apr 2024)

FORECAST_HORIZON_WEEKS = 4  # how many future weeks M5/M6 get per corridor

# LOCKED (per discussion): forecast every corridor regardless of volume,
# but flag thin ones rather than presenting false confidence.
#   - avg weekly count < this -> LOW_CONFIDENCE
#   - OR fraction of weeks with zero incidents > this -> LOW_CONFIDENCE
CONFIDENCE_MIN_AVG_WEEKLY = 8.0
CONFIDENCE_MAX_ZERO_WEEK_FRACTION = 0.25

MIN_NONZERO_WEEKS_TO_FIT = 6  # below this, Prophet has nothing to learn a
                               # weekly pattern from at all - still attempted,
                               # but expect Prophet to mostly emit the mean.


def safe_corridor_filename(corridor: str) -> str:
    """corridor names contain spaces/parens (e.g. 'IRR(Thanisandra road)') -
    sanitize for use as a filename."""
    return re.sub(r"[^A-Za-z0-9]+", "_", corridor).strip("_")


# ──────────────────────────────────────────────────────────────────
# LOAD + BUILD WEEKLY SERIES PER CORRIDOR
# ──────────────────────────────────────────────────────────────────

def load_inputs():
    clean = pd.read_csv(CLEAN_DATA_PATH)
    feat = pd.read_csv(FEATURE_MATRIX_PATH)

    clean["start_datetime"] = pd.to_datetime(clean["start_datetime"], utc=True)

    # clean_incidents.csv has raw `corridor`, not the post-reconstruction
    # `corridor_final` - pull corridor_final in via id, same join key M3
    # implicitly relies on by re-deriving from feature_matrix.
    df = clean[["id", "start_datetime"]].merge(
        feat[["id", "corridor_final"]], on="id", how="inner"
    )
    n_dropped = len(clean) - len(df)
    if n_dropped > 0:
        print(f"[M4c] WARNING: {n_dropped} rows in clean_incidents.csv had no "
              f"matching id in feature_matrix.csv - dropped from forecasting")

    print(f"[M4c] Loaded {len(df)} incidents with corridor_final + start_datetime")
    print(f"[M4c] Date range: {df['start_datetime'].min()} -> {df['start_datetime'].max()}")
    return df


def build_weekly_series(df: pd.DataFrame, corridor: str) -> pd.DataFrame:
    """
    Prophet requires columns named exactly 'ds' (date) and 'y' (value).
    Builds a complete weekly series (W-MON bins) including zero-incident
    weeks - Prophet needs the gaps explicit, not absent, or it will not
    learn a true seasonal/trend shape.
    """
    sub = df[df["corridor_final"] == corridor].copy()
    sub["week"] = sub["start_datetime"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time

    weekly_counts = sub.groupby("week").size()

    full_range = pd.date_range(
        start=df["start_datetime"].dt.tz_convert(None).min().to_period("W-SUN").start_time,
        end=df["start_datetime"].dt.tz_convert(None).max().to_period("W-SUN").start_time,
        freq="W-MON",
    )
    weekly_counts = weekly_counts.reindex(full_range, fill_value=0)

    series = pd.DataFrame({"ds": weekly_counts.index, "y": weekly_counts.values})
    return series


# ──────────────────────────────────────────────────────────────────
# CONFIDENCE FLAGGING (LOCKED: forecast everything, flag thin series)
# ──────────────────────────────────────────────────────────────────

def assess_confidence(series: pd.DataFrame, corridor: str) -> dict:
    avg_weekly = series["y"].mean()
    zero_weeks = (series["y"] == 0).sum()
    zero_fraction = zero_weeks / len(series)
    nonzero_weeks = (series["y"] > 0).sum()

    is_low_confidence = (
        avg_weekly < CONFIDENCE_MIN_AVG_WEEKLY
        or zero_fraction > CONFIDENCE_MAX_ZERO_WEEK_FRACTION
    )

    reasons = []
    if avg_weekly < CONFIDENCE_MIN_AVG_WEEKLY:
        reasons.append(f"avg_weekly={avg_weekly:.2f} < {CONFIDENCE_MIN_AVG_WEEKLY}")
    if zero_fraction > CONFIDENCE_MAX_ZERO_WEEK_FRACTION:
        reasons.append(f"zero_week_fraction={zero_fraction:.2f} > {CONFIDENCE_MAX_ZERO_WEEK_FRACTION}")

    confidence_label = "LOW_CONFIDENCE" if is_low_confidence else "OK"

    print(f"[M4c]   {corridor:25s} avg/wk={avg_weekly:6.2f}  "
          f"zero_weeks={zero_weeks:2d}/{len(series)}  nonzero_weeks={nonzero_weeks:2d}  "
          f"-> {confidence_label}" + (f" ({'; '.join(reasons)})" if reasons else ""))

    return {
        "avg_weekly_incidents": float(avg_weekly),
        "zero_week_fraction": float(zero_fraction),
        "nonzero_weeks": int(nonzero_weeks),
        "total_weeks": int(len(series)),
        "confidence": confidence_label,
        "reasons": reasons,
    }


# ──────────────────────────────────────────────────────────────────
# TEMPORAL TRAIN/TEST SPLIT + FIT + EVALUATE (one corridor)
# ──────────────────────────────────────────────────────────────────

def temporal_split(series: pd.DataFrame):
    train = series[series["ds"] <= TRAIN_END.tz_localize(None)].copy()
    test = series[series["ds"] >= TEST_START.tz_localize(None)].copy()
    return train, test


def fit_and_evaluate_corridor(corridor: str, series: pd.DataFrame) -> dict:
    train, test = temporal_split(series)

    if len(train) < 2 or len(test) < 1:
        print(f"[M4c]   {corridor:25s} SKIPPED - insufficient weeks for a temporal "
              f"split (train={len(train)}, test={len(test)})")
        return None

    model = Prophet(
        weekly_seasonality=False,  # data is already weekly-aggregated; no sub-week pattern to find
        yearly_seasonality=False,  # only ~5 months of data - not enough to estimate a yearly cycle,
                                    # forcing this on would let Prophet hallucinate a fake annual pattern
        daily_seasonality=False,
        interval_width=0.8,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test), freq="W-MON")
    forecast = model.predict(future)

    # align predictions to the actual test weeks
    pred_test = forecast.set_index("ds").loc[test["ds"], "yhat"].values
    actual_test = test["y"].values

    # Prophet can predict negative counts for low-volume series - counts
    # can't be negative, clip before scoring (NOT before fitting - fitting
    # on raw counts is correct, this clip is only for a fair MAE/MAPE).
    pred_test_clipped = np.clip(pred_test, a_min=0, a_max=None)

    mae = float(np.mean(np.abs(actual_test - pred_test_clipped)))
    # MAPE undefined / explodes when actual=0 (common for low-volume corridors) -
    # report it but flag rows where it's not meaningful rather than hiding the issue
    nonzero_mask = actual_test > 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(
            np.abs(actual_test[nonzero_mask] - pred_test_clipped[nonzero_mask])
            / actual_test[nonzero_mask]
        ) * 100)
        mape_n_weeks = int(nonzero_mask.sum())
    else:
        mape = None
        mape_n_weeks = 0

    print(f"[M4c]   {corridor:25s} MAE={mae:6.2f}  "
          f"MAPE={'n/a' if mape is None else f'{mape:6.1f}%'} "
          f"(on {mape_n_weeks}/{len(test)} nonzero test weeks)")

    return {
        "mae": mae,
        "mape": mape,
        "mape_computed_on_n_weeks": mape_n_weeks,
        "test_weeks": int(len(test)),
        "actual": actual_test.tolist(),
        "predicted": pred_test_clipped.tolist(),
        "test_dates": [d.strftime("%Y-%m-%d") for d in test["ds"]],
    }


def fit_full_and_forecast(corridor: str, series: pd.DataFrame) -> tuple:
    """
    Refit on the FULL series (train+test) for the actual forward-looking
    forecast used by M5/M6 - the temporal split above is for evaluation
    only, the deployed model should use every week of real data available.
    """
    model = Prophet(
        weekly_seasonality=False,
        yearly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
    )
    model.fit(series)

    future = model.make_future_dataframe(periods=FORECAST_HORIZON_WEEKS, freq="W-MON")
    forecast = model.predict(future)

    future_rows = forecast.tail(FORECAST_HORIZON_WEEKS)
    next_weeks = [
        {
            "week_start": row["ds"].strftime("%Y-%m-%d"),
            "predicted_incidents": max(round(float(row["yhat"]), 1), 0.0),
            "yhat_lower": max(round(float(row["yhat_lower"]), 1), 0.0),
            "yhat_upper": max(round(float(row["yhat_upper"]), 1), 0.0),
        }
        for _, row in future_rows.iterrows()
    ]

    return model, next_weeks


# ──────────────────────────────────────────────────────────────────
# RUN ALL CORRIDORS
# ──────────────────────────────────────────────────────────────────

def run():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_inputs()
    corridors = sorted(df["corridor_final"].unique().tolist())
    print(f"\n[M4c] Forecasting {len(corridors)} corridor series "
          f"(including 'Non-corridor' as its own series, per locked decision)")
    print(f"[M4c] Train period: start of data -> {TRAIN_END.date()}")
    print(f"[M4c] Test period : {TEST_START.date()} -> end of data\n")

    print("[M4c] === Confidence assessment per corridor ===")
    confidence_results = {}
    weekly_series_by_corridor = {}
    for corridor in corridors:
        series = build_weekly_series(df, corridor)
        weekly_series_by_corridor[corridor] = series
        confidence_results[corridor] = assess_confidence(series, corridor)

    n_low_conf = sum(1 for v in confidence_results.values() if v["confidence"] == "LOW_CONFIDENCE")
    print(f"\n[M4c] {n_low_conf}/{len(corridors)} corridors flagged LOW_CONFIDENCE "
          f"(forecast anyway, per locked decision - flagged, not skipped)\n")

    print("[M4c] === Per-corridor temporal evaluation (train: pre-Mar 2024, test: Mar-Apr 2024) ===")
    eval_results = {}
    for corridor in corridors:
        series = weekly_series_by_corridor[corridor]
        result = fit_and_evaluate_corridor(corridor, series)
        eval_results[corridor] = result

    valid_mae = [v["mae"] for v in eval_results.values() if v is not None]
    print(f"\n[M4c] Mean MAE across corridors with a valid split: "
          f"{np.mean(valid_mae):.3f}" if valid_mae else "[M4c] No corridors had a valid temporal split")

    print("\n[M4c] === Fitting deployment models (full data) + generating forecasts ===")
    forecast_output = {}
    for corridor in corridors:
        series = weekly_series_by_corridor[corridor]
        model, next_weeks = fit_full_and_forecast(corridor, series)

        fname = safe_corridor_filename(corridor)
        model_path = os.path.join(MODEL_DIR, f"{fname}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        forecast_output[corridor] = {
            "confidence": confidence_results[corridor]["confidence"],
            "confidence_reasons": confidence_results[corridor]["reasons"],
            "avg_weekly_incidents_historical": confidence_results[corridor]["avg_weekly_incidents"],
            "next_weeks_forecast": next_weeks,
            "model_path": model_path,
        }
        print(f"[M4c]   {corridor:25s} -> saved {model_path}")

    forecast_output["_meta"] = {
        "forecast_horizon_weeks": FORECAST_HORIZON_WEEKS,
        "train_end": str(TRAIN_END.date()),
        "test_start": str(TEST_START.date()),
        "total_weeks_in_dataset": int(len(next(iter(weekly_series_by_corridor.values())))),
        "low_confidence_threshold_avg_weekly": CONFIDENCE_MIN_AVG_WEEKLY,
        "low_confidence_threshold_zero_week_fraction": CONFIDENCE_MAX_ZERO_WEEK_FRACTION,
        "note": "'Non-corridor' included as its own series - a real, large bucket "
                "(~38% of incidents), not a road, mixed geography by construction.",
    }

    with open(FORECAST_OUTPUT_PATH, "w") as f:
        json.dump(forecast_output, f, indent=2)
    print(f"\n[M4c] Saved forecast -> {FORECAST_OUTPUT_PATH}")

    metrics_output = {
        "per_corridor": eval_results,
        "confidence_assessment": confidence_results,
        "mean_mae_across_corridors": float(np.mean(valid_mae)) if valid_mae else None,
        "n_low_confidence_corridors": n_low_conf,
        "n_total_corridors": len(corridors),
    }
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics_output, f, indent=2)
    print(f"[M4c] Saved evaluation metrics -> {METRICS_OUTPUT_PATH}")

    return forecast_output, metrics_output


if __name__ == "__main__":
    forecast_output, metrics_output = run()
    print("\n[M4c] Sample forecast (first corridor):")
    first_corridor = next(c for c in forecast_output if c != "_meta")
    print(json.dumps({first_corridor: forecast_output[first_corridor]}, indent=2))
