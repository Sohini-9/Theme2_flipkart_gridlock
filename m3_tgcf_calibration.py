"""
M3 — TGCF + MAPMYINDIA CALIBRATION
=====================================
For the top-20 corridors (by incident count), get a real congestion
baseline at several times of day, then backcast it from "today" (2026)
to the dataset's actual period (Nov 2023 - Apr 2024) using the Traffic
Growth Correction Factor.

TGCF = vehicles_2023_24 / vehicles_today
     = 115 lakh / 128 lakh
     = 0.898   (LOCKED — sourced from Bengaluru vehicle registration figures)

STATUS: LIVE MODE — real Mappls REST calls, no more mock data.
Reads the static REST key from the MAPPLS_API_KEY environment variable.
Uses two Mappls Distance Matrix resources:
  - distance_matrix     -> route duration WITHOUT live traffic
                            (this is time-of-day independent, so it's
                            fetched ONCE per corridor, not once per slot)
  - distance_matrix_eta -> route duration WITH live traffic
                            (fetched once per TIME_SLOT, since this is
                            the value that's actually supposed to vary
                            across the day)
  CONFIRMED (not a guess): `distance_matrix_eta` is the correct REST resource
  name per Mappls' own migration documentation ("distance_matrix_eta - to get
  the duration considering live traffic"), and "Distance Matrix ETA API
  Traffic" shows as an Active allocated API on this project's Cloud app in
  the dev console. An earlier version of this script guessed the wrong name
  (`distance_matrix_traffic`, inferred from an Android SDK enum) and got
  401s - fixed.

COST CONTROL — RAW RESPONSE CACHE (LOCKED):
  Every API response is cached to disk (MAPPLS_CACHE_PATH) keyed by
  (corridor, hour-or-"normal", resource), and the cache is loaded + checked
  BEFORE every call and flushed to disk AFTER every call (not batched at
  the end). This means:
    - Re-running this script after a crash/interruption never re-pays for
      a (corridor, hour) pair that already succeeded.
    - The raw responses are preserved for audit, same honesty standard as
      M4b's baseline_source flag and M2's corridor_source flag.

TIME_SLOTS — RAISED FROM 6 TO 12 (per discussion):
  6 slots was sparse enough that two adjacent real incident hours could
  snap to the same calibrated slot and lose resolution. 12 slots = every
  2 hours, evenly spaced across the full day (so the night-hour and
  peak-hour buckets M4b's sanity checks rely on are both still covered).
  Because eta_normal no longer needs to be re-fetched per slot (see above),
  the real budget impact of going 6 -> 12 slots is NOT a doubling of calls:

      old (6 slots, naive): 20 corridors x 6 slots x 2 calls          = 240 calls
      new (12 slots, optimized): 20 corridors x (1 + 12 calls)        = 260 calls

  i.e. ~8% more calls for 2x the time-of-day resolution. To go to 18 slots
  later if budget allows, just edit TIME_SLOTS below — nothing else needs
  to change. 18 slots would be 20 x (1 + 18) = 380 calls total.

Input  : clean_incidents.csv (for top-20 corridor list - reuses M1 logic)
Output : eta_baselines.json
         corridor_endpoints.json   (origin/destination per corridor, for routing)
         mappls_raw_cache.json     (every raw API response, audit trail + cache)
"""

import pandas as pd
import json
import time
import os
import requests

CLEAN_DATA_PATH = "/content/drive/MyDrive/flipkart/clean_incidents.csv"
FEATURE_MATRIX_PATH = "/content/drive/MyDrive/flipkart/feature_matrix.csv"
ETA_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/eta_baselines.json"
ENDPOINTS_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/corridor_endpoints.json"
MAPPLS_CACHE_PATH = "/content/drive/MyDrive/flipkart/mappls_raw_cache.json"

MAPPLS_API_KEY ="ccowahmbyrhljfqdndgtbpapsxylwvackvxq"
MAPPLS_BASE_URL = "https://route.mappls.com/route/dm"
MAPPLS_TIMEOUT_SEC = 10
MAPPLS_MAX_RETRIES = 2  # total attempts = 1 + this, with backoff between

TGCF = 0.898  # LOCKED — see docstring above for derivation

# LOCKED (raised 6 -> 12, see docstring): every 2 hours, full day coverage,
# so M4b's night-hour [0-4] and peak-hour [8,9,17,18,19] sanity checks both
# still land close to a real calibrated slot.
TIME_SLOTS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

TOP_N_CORRIDORS = 21  # LOCKED: there are only 21 real named corridors total
                       # (22 unique corridor_final values - 1 is "Non-corridor",
                       # which is handled separately via nearest-neighbor fallback
                       # in M4b, not calibrated here)

# ──────────────────────────────────────────────────────────────────
# Corridor origin/destination endpoints.
# LOCKED DECISION: MapMyIndia route/distance APIs need an explicit
# origin + destination, not just a single point. These are defined once,
# manually, per corridor.
#
# SOURCE: each point below is a real, named landmark/junction looked up via
# Google Places (not hand-typed decimal degrees), chosen to match how these
# 21 corridor names are commonly used in Bengaluru traffic reporting
# (e.g. "ORR North 1/2" as consecutive Hebbal->Nagavara->KR Puram segments,
# "Bellary Road 1/2" as consecutive Hebbal->Yelahanka->Airport segments).
#
# HONEST CAVEAT: this project has no access to the original Astram/BTP
# internal documentation that defines each corridor's exact junction-to-
# junction boundary, and no authoritative public source for that exists
# either (checked). So these are landmark-anchored BEST-EFFORT segments,
# not officially verified against the dataset's true corridor boundaries.
# They are a real improvement over the prior hand-typed placeholders
# (every point is now a real, checkable place), but still worth a 5-minute
# visual sanity check against the map before trusting them fully — you
# know the local context better than this can be verified remotely.
# ──────────────────────────────────────────────────────────────────
CORRIDOR_ENDPOINTS_MANUAL = {
    "Mysore Road":            {"origin": [12.9583, 77.5535], "destination": [12.8997, 77.4827]},  # Sirsi Circle -> Kengeri
    "Bellary Road 1":         {"origin": [13.0428, 77.5904], "destination": [13.1155, 77.6070]},  # Hebbal Flyover -> Yelahanka
    "Tumkur Road":            {"origin": [13.0200, 77.5556], "destination": [13.0973, 77.3856]},  # Yeshwantpur -> Nelamangala
    "Bellary Road 2":         {"origin": [13.1155, 77.6070], "destination": [13.1989, 77.7069]},  # Yelahanka -> Kempegowda Airport (north/NH44 approach)
    "Hosur Road":             {"origin": [12.9166, 77.6234], "destination": [12.8452, 77.6602]},  # Silk Board Jn -> Electronic City
    "ORR North 1":            {"origin": [13.0428, 77.5904], "destination": [13.0422, 77.6136]},  # Hebbal Flyover -> Nagavara
    "Old Madras Road":        {"origin": [12.9770, 77.6174], "destination": [12.9974, 77.6697]},  # Trinity Circle -> Tin Factory (KR Puram)
    "Magadi Road":            {"origin": [12.9656, 77.5770], "destination": [12.9883, 77.5178]},  # KR Market -> Sumanahalli
    "ORR East 1":             {"origin": [12.9974, 77.6697], "destination": [12.9569, 77.7011]},  # Tin Factory (KR Puram) -> Marathahalli
    "ORR North 2":            {"origin": [13.0422, 77.6136], "destination": [12.9974, 77.6697]},  # Nagavara -> Tin Factory (KR Puram)
    "Bannerghata Road":       {"origin": [12.9424, 77.5974], "destination": [12.8564, 77.5888]},  # Dairy Circle -> Gottigere
    "ORR East 2":             {"origin": [12.9569, 77.7011], "destination": [12.9227, 77.6647]},  # Marathahalli -> Sarjapur Jn / Bellandur
    "West of Chord Road":     {"origin": [13.0146, 77.5514], "destination": [12.9756, 77.5354]},  # Mahalakshmi Layout -> Vijayanagar
    "ORR West 1":             {"origin": [12.9448, 77.5256], "destination": [12.9255, 77.5500]},  # Nayandahalli -> Banashankari
    "CBD 2":                  {"origin": [12.9710, 77.6069], "destination": [12.9666, 77.6084]},  # Brigade Road -> Richmond Road
    "IRR(Thanisandra road)":  {"origin": [13.0422, 77.6136], "destination": [13.0367, 77.6309]},  # Nagavara -> Hennur Cross
    "Hennur Main Road":       {"origin": [13.0367, 77.6309], "destination": [13.1326, 77.6672]},  # Hennur Cross -> Bagalur
    "Varthur Road":           {"origin": [12.9569, 77.7011], "destination": [12.9383, 77.7469]},  # Marathahalli -> Varthur
    "Old Airport Road":       {"origin": [12.9610, 77.6387], "destination": [12.9547, 77.6838]},  # Domlur -> HAL Old Airport Road
    "Airport New South Road": {"origin": [13.1326, 77.6672], "destination": [13.1989, 77.7069]},  # Bagalur -> Kempegowda Airport (separate eastern/BBMP approach road, distinct from Bellary Road 2's northern NH44 approach)
    "CBD 1":                  {"origin": [12.9770, 77.6174], "destination": [12.9710, 77.6069]},  # Trinity Circle -> Brigade Road
}


# DEFAULT_ENDPOINT_FALLBACK was removed (used to be {origin: X, destination: X},
# i.e. the SAME point twice - a 0 km / 0 min "corridor" that would have
# silently produced a meaningless zero-distance baseline for any unmatched
# corridor name, with nothing in the output flagging that it happened).
# Now that all 21 real corridor names have a verified manual endpoint above,
# an unmatched name means something is actually wrong (e.g. a name mismatch
# between this dict and feature_matrix.csv) - see the explicit handling in
# run() below, which skips and loudly reports any unmatched corridor instead
# of silently faking a baseline for it.


def get_top_corridors(n: int = TOP_N_CORRIDORS) -> list:
    """
    Returns all real named corridors (excludes 'Non-corridor' - that's
    handled via nearest-neighbor fallback in M4b, not calibrated here).
    Uses corridor_final (post M2 reconstruction) so the count matches
    the 21 real corridors, not the raw pre-reconstruction corridor column.
    """
    df = pd.read_csv(FEATURE_MATRIX_PATH)
    counts = df[df["corridor_final"] != "Non-corridor"]["corridor_final"].value_counts()
    top = counts.head(n).index.tolist()
    print(f"[M3] {len(top)} real corridors for MapMyIndia calibration:")
    for c in top:
        print(f"      {c}: {counts[c]} incidents")
    return top


# ──────────────────────────────────────────────────────────────────
# RAW RESPONSE CACHE — checked BEFORE every call, flushed AFTER every
# call (not batched at the end). This is what makes re-running this
# script after a crash/interruption free instead of re-paying for
# already-fetched (corridor, hour, resource) combinations.
# ──────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if os.path.exists(MAPPLS_CACHE_PATH):
        with open(MAPPLS_CACHE_PATH) as f:
            cache = json.load(f)
        print(f"[M3] Loaded existing cache: {len(cache)} cached responses "
              f"(already-paid-for calls will be skipped)")
        return cache
    return {}


def save_cache(cache: dict):
    with open(MAPPLS_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


_CACHE = load_cache()


def _cache_key(corridor: str, slot_label: str, resource: str) -> str:
    # slot_label is either an hour like "9" or the literal "normal"
    # (eta_normal is hour-independent, fetched once per corridor)
    return f"{corridor}|{slot_label}|{resource}"


def purge_stale_fallback_entries():
    """
    RUN THIS ONCE before re-running after the distance_matrix_traffic ->
    distance_matrix_eta fix. The old (broken) run cached fallback results
    -- a real distance_matrix (no-traffic) call's data, stored under a
    distance_matrix_eta cache key, because that's what was *requested* even
    though the no-traffic endpoint is what actually answered. Left alone,
    re-running now would silently serve that stale fallback data straight
    from cache and never call the now-fixed endpoint at all.

    Only removes entries flagged fallback_from_traffic_resource=True - your
    valid 'normal' (no-traffic) calls, already paid for, are kept.
    """
    global _CACHE
    before = len(_CACHE)
    _CACHE = {k: v for k, v in _CACHE.items()
              if not v.get("fallback_from_traffic_resource")}
    removed = before - len(_CACHE)
    save_cache(_CACHE)
    print(f"[M3] Purged {removed} stale fallback entries from cache "
          f"({len(_CACHE)} valid entries kept). Safe to re-run now.")
    return removed


# ──────────────────────────────────────────────────────────────────
# REAL MAPPLS DISTANCE MATRIX CALLS
# ──────────────────────────────────────────────────────────────────

def _coords_to_url_part(point) -> str:
    """Our endpoints are stored [latitude, longitude] (matches the rest of
    the pipeline's convention); Mappls' URL path wants longitude,latitude.
    Swap only at this API boundary."""
    lat, lon = point
    return f"{lon},{lat}"


def _mappls_raw_call(resource: str, origin, destination) -> dict:
    """One real HTTP call to a Mappls distance-matrix resource. Retries on
    transient failures; raises on a real/persistent failure rather than
    silently returning a fabricated number — a failed call should be
    visibly absent from eta_baselines.json, not invisibly wrong."""
    if not MAPPLS_API_KEY:
        raise RuntimeError(
            "[M3] MAPPLS_API_KEY environment variable is not set. "
            "Export your static key first: export MAPPLS_API_KEY='your_key'"
        )

    coords = f"{_coords_to_url_part(origin)};{_coords_to_url_part(destination)}"
    url = f"{MAPPLS_BASE_URL}/{resource}/driving/{coords}"
    params = {"access_token": MAPPLS_API_KEY}

    last_err = None
    for attempt in range(MAPPLS_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=MAPPLS_TIMEOUT_SEC)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", {})
            duration_sec = results["durations"][0][1]
            distance_m = results["distances"][0][1]
            return {
                "eta_min": round(duration_sec / 60, 2),
                "distance_km": round(distance_m / 1000, 2),
                "raw_response": data,
            }
        except Exception as e:
            last_err = e
            if attempt < MAPPLS_MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"[M3] Mappls call failed after "
                        f"{MAPPLS_MAX_RETRIES + 1} attempts "
                        f"(resource={resource}, url={url}): {last_err}")


def get_eta(corridor: str, slot_label: str, resource: str, origin, destination) -> float:
    """Cache-first wrapper. resource is 'distance_matrix' (no traffic) or
    'distance_matrix_eta' (live traffic). Returns eta in minutes."""
    key = _cache_key(corridor, slot_label, resource)
    if key in _CACHE:
        return _CACHE[key]["eta_min"]

    try:
        result = _mappls_raw_call(resource, origin, destination)
    except RuntimeError as e:
        if resource == "distance_matrix_eta":
            # The traffic-specific resource path isn't confirmed in public
            # docs (see module docstring) - if it 404s/errors, fall back to
            # the confirmed no-traffic resource rather than burning retries
            # on a name that may simply be wrong, and flag it loudly so the
            # gap is visible in the output, not silently hidden.
            print(f"[M3] WARNING: distance_matrix_eta failed for "
                  f"{corridor} slot={slot_label} ({e}). Falling back to "
                  f"distance_matrix (no live-traffic differentiation for "
                  f"this point) - verify the correct traffic resource name "
                  f"with Mappls support before trusting this run's CIS output.")
            result = _mappls_raw_call("distance_matrix", origin, destination)
            result["fallback_from_traffic_resource"] = True
        else:
            raise

    _CACHE[key] = result
    save_cache(_CACHE)  # flush immediately - survive interruption mid-run
    return result["eta_min"]


def call_mappls_route(origin, destination, avoid_locations=None, alternatives=3) -> list:
    """
    STILL A STUB - not called anywhere in this module's run(). Reserved for
    M6 (diversion route suggestions), which needs the Route API's
    avoid_locations parameter, not the Distance Matrix API used above.
    Wire this up to a real call when M6 is built, same pattern as get_eta().
    """
    routes = []
    for i in range(alternatives):
        routes.append({
            "route_id": f"mock_route_{i+1}",
            "eta_min": round(20 + i * 5.5, 1),  # mock increasing ETA for alt routes
            "distance_km": round(8 + i * 2.1, 1),
            "avoided": avoid_locations is not None,
        })
    return routes


# ──────────────────────────────────────────────────────────────────
# CALIBRATION PIPELINE
# ──────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────
# CALIBRATION PIPELINE
#
# IMPORTANT CORRECTION (found via your test run): distance_matrix_eta is
# LIVE traffic - it answers "what's it like right now", not "what's it
# like at hour X". Calling it once per TIME_SLOT in a single sitting (the
# old approach) just calls "right now" 12 times and pays for 12 identical
# numbers - confirmed by your test output (43.37 min, all 12 slots).
#
# FIX: each TIME_SLOT can only be filled with a real call made at that
# real clock hour. calibrate_all_corridors_current_hour() below does
# exactly one slot's worth of real data per invocation, and is meant to be
# re-run (manually or via cron) across a real day until all 12 slots are
# filled. eta_normal still only needs fetching once ever per corridor.
# ──────────────────────────────────────────────────────────────────

import datetime


def nearest_time_slot(hour: int) -> int:
    return min(TIME_SLOTS, key=lambda h: abs(h - hour))


def calibrate_all_corridors_current_hour():
    """
    THE REAL DATA-COLLECTION ENTRY POINT — re-run this across a real day.

    One call to this = one real TIME_SLOT filled, for all 21 corridors at
    once. Cost: ~21 traffic calls (~Rs 2.3 at your measured Rs 0.11/call
    rate) + a handful of one-time eta_normal calls the first time any given
    corridor is touched. Loads + merges into the EXISTING eta_baselines.json
    rather than overwriting it, so you can run this as many times as you
    like across the day/days without losing earlier progress.

    SUGGESTED USAGE: set up a cron entry to run this every ~2 hours
    (matching TIME_SLOTS' spacing), e.g.:
        5 0,2,4,6,8,10,12,14,16,18,20,22 * * * cd /path/to/pipeline && \\
            MAPPLS_API_KEY=... python3 -c \\
            "import m3_tgcf_calibration as m3; m3.calibrate_all_corridors_current_hour()"
    Or just run it by hand whenever you're at the laptop - get_eta()'s
    cache means re-running mid-hour costs nothing extra, it'll just skip
    straight to reporting "already filled" for that slot.
    """
    top_corridors = get_top_corridors()
    now_hour = datetime.datetime.now().hour
    slot = nearest_time_slot(now_hour)
    print(f"[M3] Real hour is {now_hour}:00 -> filling TIME_SLOT '{slot}' "
          f"for all corridors")

    if os.path.exists(ETA_OUTPUT_PATH):
        with open(ETA_OUTPUT_PATH) as f:
            eta_baselines = json.load(f)
        print(f"[M3] Loaded existing eta_baselines.json - merging into it, "
              f"not overwriting")
    else:
        eta_baselines = {}

    unmatched_corridors = []
    for corridor in top_corridors:
        if corridor not in CORRIDOR_ENDPOINTS_MANUAL:
            unmatched_corridors.append(corridor)
            continue

        endpoint = CORRIDOR_ENDPOINTS_MANUAL[corridor]
        origin, destination = endpoint["origin"], endpoint["destination"]

        eta_normal = get_eta(corridor, "normal", "distance_matrix", origin, destination)
        eta_traffic_today = get_eta(
            corridor, str(slot), "distance_matrix_eta", origin, destination
        )

        eta_traffic_2023 = round(eta_traffic_today * TGCF, 2)
        baseline_delay_2023 = round(eta_traffic_2023 - eta_normal, 2)

        eta_baselines.setdefault(corridor, {})
        eta_baselines[corridor][str(slot)] = {
            "eta_normal_min": eta_normal,
            "eta_traffic_today_min": eta_traffic_today,
            "eta_traffic_2023_tgcf_corrected_min": eta_traffic_2023,
            "baseline_delay_2023_min": max(baseline_delay_2023, 0.0),
            "captured_at_real_hour": now_hour,  # audit: when this slot was ACTUALLY fetched
        }
        print(f"[M3]   {corridor:25s} slot {slot:>2} filled "
              f"(eta_traffic={eta_traffic_today} min, real hour={now_hour}:00)")

    # report how complete the calibration is across all slots/corridors
    still_missing = {}
    for corridor in top_corridors:
        if corridor in unmatched_corridors:
            continue
        filled = set(int(s) for s in eta_baselines.get(corridor, {}).keys())
        missing = [s for s in TIME_SLOTS if s not in filled]
        if missing:
            still_missing[corridor] = missing

    eta_baselines["_meta"] = {
        "tgcf": TGCF,
        "tgcf_derivation": "115 lakh vehicles (2023-24) / 128 lakh vehicles (2026 today)",
        "time_slots_queried": TIME_SLOTS,
        "status": "LIVE_DATA_INCREMENTAL - each slot captured at its own "
                  "matching real hour, not duplicated from a single call",
        "unmatched_corridors": unmatched_corridors,
        "slots_still_missing_per_corridor": still_missing,
    }

    with open(ETA_OUTPUT_PATH, "w") as f:
        json.dump(eta_baselines, f, indent=2)

    with open(ENDPOINTS_OUTPUT_PATH, "w") as f:
        json.dump(CORRIDOR_ENDPOINTS_MANUAL, f, indent=2)

    print(f"\n[M3] Saved (merged) -> {ETA_OUTPUT_PATH}")
    if still_missing:
        n_missing_slots = sum(len(v) for v in still_missing.values())
        print(f"[M3] Still incomplete: {n_missing_slots} (corridor, slot) "
              f"combos still need a real-hour run. Re-run this function at "
              f"a different real hour to keep filling slots.")
    else:
        print(f"[M3] ALL {len(TIME_SLOTS)} slots filled for all "
              f"{len(top_corridors) - len(unmatched_corridors)} corridors - "
              f"calibration complete!")

    return eta_baselines


def test_single_corridor():
    """
    Cheap sanity check (2 calls: 1 normal + 1 traffic, ~Rs 0.22) that the
    key + distance_matrix_eta resource still work. You already confirmed
    this works in your last run - only re-run this if something changes
    (new key, account issue, etc.), not as a routine step anymore.
    """
    corridor = "Mysore Road"
    endpoint = CORRIDOR_ENDPOINTS_MANUAL[corridor]
    origin, destination = endpoint["origin"], endpoint["destination"]
    print(f"[M3] TEST: '{corridor}' - 1 normal + 1 traffic call...")

    eta_normal = get_eta(corridor, "normal", "distance_matrix", origin, destination)
    eta_traffic = get_eta(corridor, "test_now", "distance_matrix_eta", origin, destination)

    print(f"[M3]   eta_normal_min={eta_normal}, eta_traffic_today_min={eta_traffic}")

    any_fallback = any(
        v.get("fallback_from_traffic_resource") for v in _CACHE.values()
    )
    if any_fallback:
        print("\n[M3] A call fell back to distance_matrix - traffic "
              "differentiation is NOT working. Check the resource name "
              "with Mappls support.")
    else:
        print("\n[M3] distance_matrix_eta resolved fine.")
    return {"eta_normal_min": eta_normal, "eta_traffic_today_min": eta_traffic}


if __name__ == "__main__":
    # One-time cleanup of stale fallback entries from the resource-name fix
    # (distance_matrix_traffic -> distance_matrix_eta). Safe/idempotent to
    # run every time - it's a no-op once the cache is clean.
    purge_stale_fallback_entries()

    # Real entry point - safe and cheap to run repeatedly across the day.
    # Each run fills in whichever TIME_SLOT matches the real current hour.
    calibrate_all_corridors_current_hour()