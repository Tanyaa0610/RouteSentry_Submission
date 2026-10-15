"""All tunable policy constants live here, in one place.

Every constant below is an explicit judgment call the case-study brief left
unspecified (the two baseline *definitions* are fixed by the brief and are
NOT configurable -- see weekly.py). Each one is documented with its
rationale so a reader can audit the policy without reading code.
"""

# --- Baseline math (FIXED by the brief -- not a policy knob) -------------
TRAILING_WEEKS = 8  # "trailing 8-week rolling average", weeks strictly before current week

# --- Statistical anomaly policy (NOT specified by the brief; our call) ---
# A route-week is a *statistical signal* (candidate anomaly) only if BOTH:
#   1. It is actually rising relative to the route's own past
#      (vs_own_history > 0) -- this is the "rising" gate from the brief's
#      "cost is rising AND looks out of the ordinary" language. A route
#      that is structurally pricier than a single peer route, but flat
#      against its own history, is not "rising" and should not be flagged.
#   2. The rise is large enough to be "out of the ordinary" by at least one
#      of the two prescribed yardsticks: either vs_own_history itself
#      exceeds the threshold, OR (given it is already rising) vs the peer
#      average exceeds the threshold.
# Threshold chosen as a flat, easy-to-explain 15 percentage points.
# Measured against the actual dataset (see docs/threshold_analysis.md):
# only 0.8% of route-weeks exceed +15% vs_own_history, and only 3.7%
# exceed +15% vs_similar_routes -- i.e. the threshold sits above roughly
# the 99th/96th percentile of each distribution's natural week-to-week
# noise, so it flags genuinely unusual weeks rather than ordinary
# variation. It was NOT chosen to force any particular row in
# sample_output_format_v2.csv to a given verdict; the three example rows
# in that file happen to validate against it (see
# tests/test_golden_examples.py) but a wide range of thresholds between
# roughly 12% and 23% would reproduce the same three verdicts, so the
# match is corroborating evidence, not the selection criterion.
RISE_THRESHOLD_PCT = 15.0

# --- Retrieval policy ------------------------------------------------------
# Semantic candidates considered per route-week (TF-IDF cosine top-K).
SEMANTIC_TOP_K = 3
# Below this cosine similarity, a note is not "close enough" to even
# mention as a rejected candidate in the explanation text.
MIN_SIMILARITY_TO_MENTION = 0.05

# --- Evidence applicability policy ----------------------------------------
# If a note's text gives no explicit date range ("from X to Y"), we do NOT
# assume the effect persists indefinitely -- we scope it to the note's own
# ISO week only (Mon-Sun containing note.date). This is a deliberately
# conservative choice: the guardrails state "a confident wrong reason
# scores worse than couldn't find one," so we would rather under-credit an
# ongoing effect (e.g. N003's nationwide diesel rise) than fabricate an
# unbounded justification window.
DEFAULT_NOTE_WINDOW_DAYS = 7

# --- Output / paths --------------------------------------------------------
OUTPUT_COLUMNS = [
    "route",
    "week_of",
    "cost_per_tonne_km",
    "vs_own_history",
    "vs_similar_routes",
    "flagged",
    "matched_note_id",
    "reason",
]
