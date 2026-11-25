"""Shared plot colors so each turbine keeps one color across every chart.

The four turbine colors are the first four slots of a categorical palette
checked for color-vision-deficiency separation (adjacent pairs stay
distinguishable under protan/deutan/tritan simulation). Each turbine also gets
its own marker, so identity never depends on color alone. Farm-level series use
the primary ink color rather than a turbine color.
"""

# Fixed order: Turbine 1, 2, 3, ... Never re-order by rank or value.
TURBINE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948")
TURBINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
FARM_COLOR = "#0b0b0b"          # farm totals and farm critic
BASELINE_COLOR = "#52514e"      # aligned / yaw-tracking reference (dashed)
# Rate-limited wind-aligned benchmark (evaluate_wind_farm.WIND_ALIGNED_POLICY):
# same neutral family as the baseline, told apart by its dash-dot line style.
WIND_ALIGNED_COLOR = "#8a8984"
WIND_ALIGNED_STYLE = dict(color=WIND_ALIGNED_COLOR, linestyle=(0, (6, 2, 1, 2)), linewidth=1.4,
                          label="Wind-aligned benchmark (±6°/s)")
TEXT_SECONDARY = "#52514e"
ZERO_LINE_COLOR = "#c3c2b7"


def turbine_color(index):
    return TURBINE_COLORS[index % len(TURBINE_COLORS)]


def turbine_marker(index):
    return TURBINE_MARKERS[index % len(TURBINE_MARKERS)]
