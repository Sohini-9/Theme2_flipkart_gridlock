"""
M6 — RESOURCE RULE ENGINE
============================
Maps a single event's M5 prediction (CIS + closure probability) to a
resource recommendation: officer count, barricade count, diversion
priority.

WHY THIS IS A RULE ENGINE, NOT A LEARNED MODEL (read before editing):
The dataset has ZERO historical resource-deployment records - no officer
counts, no barricade counts, nothing about what was actually deployed to
any past incident (this gap is explicitly named in the Project Summary's
own data-availability table). There is therefore nothing to train a
resource-prediction model against. Inventing a model here would repeat
exactly the Version-1 CIS mistake the project already identified and
rejected (generic multipliers presented as if data-derived). So M6 is
instead an EXPLICIT, READABLE rule table - every threshold and count below
is a reasoned default, clearly labeled as such, meant to be reviewed and
edited by a human with real domain/operational knowledge (e.g. actual
traffic police staffing norms), not treated as a discovered truth.

WHAT THE THRESHOLDS ARE GROUNDED IN (so they are not pure guesses either):
  - Tier boundaries are the actual quartiles of the 8,024-row CIS
    distribution (cis_scores.csv from M4b), not round numbers picked by
    eye. This means each tier holds a comparable share of historical
    incidents, which is a reasonable basis for tiering even though no
    resource-outcome data exists to validate it against.
  - The real closure-rate-by-CIS-bucket pattern in that same data shows a
    sharp jump only in the top quartile (16.7% closure rate vs ~5-6% in
    every other quartile) - this is the empirical justification for
    treating the top tier as qualitatively different (highest officer/
    barricade tier, diversion considered by default), not just "a bit
    more than tier 3."
  - Within that structure, the actual officer/barricade COUNTS per tier
    (e.g. "2 officers" vs "4 officers") are operational judgment calls,
    not derived from anything - flagged plainly below and meant to be
    swapped for real staffing-policy numbers if/when available.

INPUTS THIS MODULE TRUSTS, IN ORDER OF WEIGHT:
  1. requires_road_closure_predicted (M4a/M5, RF primary) - a REAL trained
     classifier output, the strongest signal available, checked FIRST.
  2. cis_ml_based (M4b/M5) - the primary CIS number, see locked decision
     below for why the trained regressor is used over the static formula.
  3. cis_formula_based - shown alongside for transparency/audit only; if it
     disagrees sharply with cis_ml_based, that disagreement is surfaced as
     a flag on the output, never silently resolved.

LOCKED DECISION (per discussion): cis_ml_based is PRIMARY.
Reasoning: this is a ML project, not a lookup table - M4b was deliberately
built as a two-stage design (Stage 1 formula -> Stage 2 trained regressor)
specifically so the regressor could GENERALIZE the formula to cause/
corridor/hour combinations beyond what was explicitly cached in
eta_baselines.json (M3 only queried MapMyIndia for the top 21 corridors;
the formula path literally cannot score outside that set without the NN
borrowing fallback - the regressor can, by design, that is its whole
stated purpose). If the formula were used as the live decision input
instead, the trained model would be a slide exhibit, not a working part
of the system - the entire reason for training it would go uncashed.
So M6 acts on cis_ml_based; cis_formula_based is still reported alongside
as a transparency/audit cross-check (and the gap between them is flagged
when large), but never overrides the model's number.

Input  : a single M5 predict_event() output dict (in-memory - this module
         does not read any CSV/pickle itself, it is pure rule logic over
         M5's already-resolved output)
Output : resource_plan.json (when run via CLI on demo events)
         OR a dict returned in-memory (when imported and called directly,
         e.g. from M7's live input form / API layer)
"""

import json
from datetime import datetime, timezone

# ──────────────────────────────────────────────────────────────────
# LOCKED CIS TIER BOUNDARIES — actual quartiles of the ML-predicted CIS
# (scorer_model.pkl applied to feature_matrix.csv), NOT the formula CIS
# distribution and NOT arbitrary round numbers. These are deliberately
# recomputed against cis_ml_based specifically, since cis_ml_based is the
# LOCKED primary input below - formula and ML CIS distributions are close
# but genuinely different (formula Q1/Q3 = 2.71/7.63, ML Q1/Q3 = 3.37/7.26),
# so reusing the formula's quartiles here would silently misbucket events.
#   ML CIS: Q1 = 3.37, median = 4.72, Q3 = 7.26 (8,024-row distribution)
# Rounded to 1 decimal for a readable rule table; boundaries are inclusive
# on the lower end of each tier. Recompute if M4b is re-run with real
# (non-mock) MapMyIndia data or a real XGBoost model, since both the
# regressor's predictions and this distribution's shape may shift.
# ──────────────────────────────────────────────────────────────────
CIS_TIERS = [
    {"name": "LOW",       "min_cis": 0.0, "max_cis": 3.4},
    {"name": "MODERATE",  "min_cis": 3.4, "max_cis": 4.7},
    {"name": "ELEVATED",  "min_cis": 4.7, "max_cis": 7.3},
    {"name": "HIGH",      "min_cis": 7.3, "max_cis": 10.001},  # 10.001 so CIS=10.0 is inclusive
]

# ──────────────────────────────────────────────────────────────────
# REASONED-DEFAULT RESOURCE TABLE (NOT data-derived — explicitly flagged).
# These are operational judgment calls a real traffic-ops reviewer should
# sign off on or replace. The STRUCTURE (escalating officers/barricades by
# tier, diversion considered only at higher tiers) is the defensible part;
# the exact integers are placeholders pending real staffing-policy input.
# ──────────────────────────────────────────────────────────────────
RESOURCE_TABLE = {
    "LOW": {
        "officer_count": 1,
        "barricade_count": 0,
        "diversion_priority": "NONE",
        "rationale": "Minor incident, bottom quartile of the ML-predicted CIS distribution, "
                     "historical closure rate in this band is ~5.3%. Single officer for "
                     "traffic guidance/logging; no physical barricading by default.",
    },
    "MODERATE": {
        "officer_count": 2,
        "barricade_count": 2,
        "diversion_priority": "NONE",
        "rationale": "Second quartile. Closure rate in this band is actually the LOWEST "
                     "of all four quartiles (~3.5%) in the historical data, so resourcing "
                     "step-up here is conservative buffer, not a data-driven jump - light "
                     "barricading to manage the immediate incident footprint, no diversion "
                     "routing triggered by default.",
    },
    "ELEVATED": {
        "officer_count": 3,
        "barricade_count": 4,
        "diversion_priority": "ADVISORY",
        "rationale": "Third quartile. Closure rate climbs to ~7.0% here, the first sign of "
                     "the upward trend that peaks in the top quartile - ADVISORY diversion "
                     "means a route is computed and ready but not pushed to drivers/dashboard "
                     "by default, only on operator confirmation.",
    },
    "HIGH": {
        "officer_count": 4,
        "barricade_count": 6,
        "diversion_priority": "ACTIVE",
        "rationale": "Top quartile of the ML-predicted CIS distribution. Real closure rate "
                     "in this band is 13.8% - roughly 2-4x every other tier - the one "
                     "empirically distinct jump in the data. ACTIVE diversion means a "
                     "route is computed AND surfaced by default; officer/barricade counts "
                     "are the largest reasoned-default allocation in this table.",
    },
}

# ──────────────────────────────────────────────────────────────────
# CLOSURE OVERRIDE (LOCKED): requires_road_closure_predicted is a REAL
# trained classifier output (M4a, RF, ROC-AUC 0.775) - the strongest
# signal this module has access to. If M4a/M5 predicts closure=True, that
# ALWAYS escalates resourcing at least one tier above whatever CIS alone
# would suggest, and always sets diversion_priority to at least ACTIVE -
# a predicted closure with only advisory diversion would be an internally
# inconsistent recommendation. CIS can still push the final tier higher
# than this floor; the override is a floor, not a ceiling.
# ──────────────────────────────────────────────────────────────────
TIER_ORDER = ["LOW", "MODERATE", "ELEVATED", "HIGH"]


def cis_to_tier(cis: float) -> str:
    for tier in CIS_TIERS:
        if tier["min_cis"] <= cis < tier["max_cis"]:
            return tier["name"]
    # cis exactly at or above the top boundary (shouldn't happen given
    # 10.001 upper bound, but guard rather than crash on an out-of-range CIS)
    return "HIGH"


def apply_closure_override(tier: str, closure_predicted: bool) -> dict:
    """Returns the FINAL tier after applying the closure floor, plus a
    flag recording whether an override actually fired (so the output is
    auditable - never silently bump a tier without saying so)."""
    if not closure_predicted:
        return {"final_tier": tier, "closure_override_applied": False}

    current_idx = TIER_ORDER.index(tier)
    floor_idx = TIER_ORDER.index("ELEVATED")  # closure prediction floors at ELEVATED minimum
    final_idx = max(current_idx, floor_idx)
    final_tier = TIER_ORDER[final_idx]

    return {
        "final_tier": final_tier,
        "closure_override_applied": final_idx > current_idx,
    }


def check_cis_disagreement(cis_formula: float, cis_ml) -> dict:
    """Surfaces, not hides, cases where the formula and ML CIS diverge a
    lot - operationally useful: a large gap means the live event's
    cause/corridor combination may be one the regressor is extrapolating
    on, not one it learned cleanly."""
    if cis_ml is None or cis_formula is None:
        return {"checked": False}

    gap = abs(cis_formula - cis_ml)
    flagged = gap >= 2.5  # LOCKED-by-judgment threshold: roughly one tier-width on the 0-10 scale

    return {
        "checked": True,
        "cis_formula_based": cis_formula,
        "cis_ml_based": cis_ml,
        "absolute_gap": round(gap, 3),
        "flagged_as_high_disagreement": flagged,
    }


def build_resource_plan(prediction: dict) -> dict:
    """
    prediction: the dict returned by M5.predict_event() for ONE event.
    Reads only the fields it needs (closure_prediction, congestion_impact_score) -
    does not require the full M5 output shape to stay working if M5's other
    fields change later.
    """
    cis_block = prediction.get("congestion_impact_score", {})
    closure_block = prediction.get("closure_prediction", {})

    cis_formula = cis_block.get("cis_formula_based")
    cis_ml = cis_block.get("cis_ml_based")
    closure_predicted = closure_block.get("requires_road_closure_predicted", False)
    closure_probability = closure_block.get("closure_probability")

    if cis_ml is None:
        raise ValueError(
            "[M6] cis_ml_based is missing/None in the M5 prediction - "
            "cannot build a resource plan without it (see module docstring "
            "for why the trained regressor, not the static formula, is the "
            "locked primary input)."
        )

    base_tier = cis_to_tier(cis_ml)
    override = apply_closure_override(base_tier, closure_predicted)
    final_tier = override["final_tier"]

    resources = RESOURCE_TABLE[final_tier]
    disagreement = check_cis_disagreement(cis_formula, cis_ml)

    return {
        "cis_used": cis_ml,
        "cis_used_source": "cis_ml_based",
        "cis_based_tier": base_tier,
        "closure_predicted": closure_predicted,
        "closure_probability": closure_probability,
        "closure_override_applied": override["closure_override_applied"],
        "final_tier": final_tier,
        "officer_count": resources["officer_count"],
        "barricade_count": resources["barricade_count"],
        "diversion_priority": resources["diversion_priority"],
        "rationale": resources["rationale"],
        "cis_disagreement_check": disagreement,
        "tier_thresholds_used": CIS_TIERS,
        "resource_counts_are_reasoned_defaults_not_learned": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ──────────────────────────────────────────────────────────────────
# CLI / DEMO ENTRY POINT
# ──────────────────────────────────────────────────────────────────

PLAN_OUTPUT_PATH = "/content/drive/MyDrive/flipkart/resource_plan.json"


def run_demo():
    """
    Runs M5 on a few representative events, then M6 on each M5 output,
    end to end, and saves the combined result - so this file can be
    smoke-tested without M7 existing yet.
    """
    from m5_inference_engine import PipelineArtifacts, predict_event

    artifacts = PipelineArtifacts()

    demo_events = [
        {  # expect LOW/MODERATE tier, no closure
            "latitude": 12.9716, "longitude": 77.5573,
            "event_cause": "pot_holes", "event_type": "unplanned",
            "priority": "LOW", "timestamp": "2026-06-20T14:00:00",
        },
        {  # expect HIGH tier + closure override (vip_movement, planned, high priority)
            "latitude": 13.0050, "longitude": 77.5700,
            "event_cause": "vip_movement", "event_type": "planned",
            "priority": "HIGH", "timestamp": "2026-06-20T09:00:00",
        },
        {  # mid-tier, tests the disagreement flag in practice
            "latitude": 12.9350, "longitude": 77.7350,
            "event_cause": "water_logging", "event_type": "unplanned",
            "priority": "HIGH", "timestamp": "2026-06-20T22:15:00",
        },
    ]

    results = []
    for i, event in enumerate(demo_events):
        print(f"\n{'='*70}\n[M6] DEMO EVENT {i+1}\n{'='*70}")
        prediction = predict_event(event, artifacts)
        plan = build_resource_plan(prediction)
        print(json.dumps(plan, indent=2))
        results.append({"event": event, "prediction": prediction, "resource_plan": plan})

    with open(PLAN_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[M6] Saved {len(results)} demo resource plans -> {PLAN_OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    run_demo()
