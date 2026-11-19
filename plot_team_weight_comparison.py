"""Compare greedy evaluations across runs that differ only in --team-reward-weight.

Usage:
    python plot_team_weight_comparison.py <run> <run> ... [--output comparison.png]

Three stacked panels share the episode axis: farm energy against the aligned
baseline, then the mean yaw of the two upstream turbines (Turbine 1 and
Turbine 2 in the default square layout), whose yaw is what steers wakes away
from the downstream pair. Weight is ordered, so it uses one purple ramp
(light = 0, dark = 1); the turbine hues stay reserved for turbines.
"""

import argparse
import json
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from plot_evaluation_turbines import _direct_labels, _float, _mean_yaw, _rows
from plot_style import BASELINE_COLOR, TEXT_SECONDARY, ZERO_LINE_COLOR

# Checked for colour-vision separation: every pair is >= 20 OKLab dE in normal,
# protan, deutan and tritan vision. The lightest step has low contrast on
# white, so lines also carry markers and direct labels.
WEIGHT_RAMP = LinearSegmentedColormap.from_list("team_weight", ["#a39af0", "#6250d6", "#2a1f7a"])
WEIGHT_MARKERS = ("o", "s", "^", "D", "v")
UPSTREAM_TURBINES = (0, 1)


def _run_summary(run_dir):
    run_dir = Path(run_dir)
    weight = float(json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["team_reward_weight"])
    rows = _rows(run_dir / "evaluation_metrics.csv")
    greedy = sorted((row for row in rows if row.get("policy") == "greedy"), key=lambda row: int(row["episode"]))
    if not greedy:
        raise ValueError(f"{run_dir}: no greedy evaluations are recorded.")
    baseline = next((row for row in rows if row.get("policy") == "aligned_baseline"), None)
    episodes = [int(row["episode"]) for row in greedy]
    yaw = {turbine: [_mean_yaw(run_dir / f"evaluation_episode{episode}_steps.csv", turbine)
                     for episode in episodes] for turbine in UPSTREAM_TURBINES}
    return dict(weight=weight, episodes=episodes, farm=[_float(row["farm_energy_mwh"]) for row in greedy],
                yaw=yaw, baseline=_float(baseline["farm_energy_mwh"]) if baseline else None)


def plot_team_weight_comparison(run_dirs, output):
    runs = sorted((_run_summary(run) for run in run_dirs), key=lambda run: run["weight"])
    figure = Figure(figsize=(10, 9), dpi=160, layout="constrained")
    FigureCanvasAgg(figure)
    energy_axis, *yaw_axes = figure.subplots(3, 1, sharex=True)
    baselines = {run["baseline"] for run in runs if run["baseline"] is not None}
    if len(baselines) == 1:
        energy_axis.axhline(baselines.pop(), color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                            label="Aligned baseline")
    panels = [(energy_axis, "farm", "Farm energy per greedy evaluation", "Energy (MWh)")]
    panels += [(axis, turbine, f"Turbine {turbine + 1} mean yaw (upstream)", "Mean yaw (deg)")
               for axis, turbine in zip(yaw_axes, UPSTREAM_TURBINES)]
    for axis, key, title, ylabel in panels:
        ends = []
        for index, run in enumerate(runs):
            values = run["farm"] if key == "farm" else run["yaw"][key]
            label = f"Weight {run['weight']:g}"
            axis.plot(run["episodes"], values, color=WEIGHT_RAMP(run["weight"]), linewidth=2,
                      marker=WEIGHT_MARKERS[index % len(WEIGHT_MARKERS)], markersize=6,
                      markeredgecolor="white", markeredgewidth=0.8, label=label)
            ends.append((label, values[-1]))
        if key != "farm":
            axis.axhline(0, color=ZERO_LINE_COLOR, linewidth=1)
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        _direct_labels(axis, ends)
    legend = energy_axis.legend(loc="lower right", frameon=False, fontsize=9,
                                title="Team-reward weight\n0 = own turbine, 1 = whole farm")
    legend.get_title().set_fontsize(9)
    legend.get_title().set_color(TEXT_SECONDARY)
    episodes = sorted({episode for run in runs for episode in run["episodes"]})
    if len(episodes) <= 15:
        yaw_axes[-1].set_xticks(episodes)  # one tick per evaluation
    else:
        yaw_axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    yaw_axes[-1].set_xlabel("Training episode of the greedy evaluation")
    figure.suptitle("Team-reward weight comparison", fontsize=15, weight="bold")
    output = Path(output)
    figure.savefig(output)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Run folders that differ only in team weight")
    parser.add_argument("--output", type=Path, default=Path("team_weight_comparison.png"))
    args = parser.parse_args()
    print(plot_team_weight_comparison(args.runs, args.output))
