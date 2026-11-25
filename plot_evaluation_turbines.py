"""Per-turbine view of every greedy evaluation in a run.

Three stacked panels sharing the evaluation-episode axis:
1. Farm energy per evaluation, with the aligned/yaw-tracking baseline (if
   recorded) and the best evaluation marked.
2. Each turbine's energy change against the baseline (or, without a baseline,
   against the episode-0 greedy evaluation). Wake steering shows up here as an
   upstream turbine going negative while the turbine behind it goes positive.
3. Each turbine's mean wind-relative yaw during the evaluation episode.

Reads evaluation_metrics.csv and evaluation_episode*_steps.csv, so it works on
runs with or without video and on runs that are still training.
"""

import argparse
import csv
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from plot_rewards import wind_aligned_reference
from plot_style import (WIND_ALIGNED_STYLE, BASELINE_COLOR, FARM_COLOR, TEXT_SECONDARY, ZERO_LINE_COLOR,
                        turbine_color, turbine_marker)

BASELINE_POLICIES = ("aligned_baseline", "yaw_tracking_baseline")


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean_yaw(steps_path, turbine):
    values = [_float(row.get(f"turbine{turbine}_yaw_deg")) for row in _rows(steps_path)]
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def _direct_labels(axis, series, min_gap=0.05):
    """Label each line at its right end without overlaps.

    Labels closer than min_gap (axes fraction) form a group; each group is
    spread evenly and centered on its lines' mean end value, so labels stay
    next to the lines they name instead of drifting upward.
    """
    low, high = axis.get_ylim()
    span = (high - low) or 1.0
    ends = sorted((((y - low) / span), label) for label, y in series if not math.isnan(y))
    groups = []
    for fraction, label in ends:
        if groups:
            group = groups[-1]
            center = sum(f for f, _ in group) / len(group)
            if center + (len(group) - 1) / 2 * min_gap + min_gap > fraction - 1e-12:
                group.append((fraction, label))
                continue
        groups.append([(fraction, label)])
    for group in groups:
        center = sum(f for f, _ in group) / len(group)
        start = min(max(center - (len(group) - 1) / 2 * min_gap, 0.02),
                    0.98 - (len(group) - 1) * min_gap)
        for k, (_, label) in enumerate(group):
            axis.text(1.01, start + k * min_gap, label, transform=axis.transAxes,
                      va="center", ha="left", fontsize=9, color=TEXT_SECONDARY)


def plot_evaluation_turbines(run_dir):
    run_dir = Path(run_dir)
    rows = _rows(run_dir / "evaluation_metrics.csv")
    greedy = sorted((row for row in rows if row.get("policy") == "greedy"),
                    key=lambda row: int(row["episode"]))
    if not greedy:
        raise ValueError("No greedy evaluations are recorded yet.")
    count = sum(1 for key in greedy[0] if key.startswith("turbine") and key.endswith("_energy_mwh"))
    baseline = next((row for row in rows if row.get("policy") in BASELINE_POLICIES), None)
    reference = baseline or greedy[0]
    reference_name = ("yaw-tracking baseline" if baseline and baseline["policy"] == "yaw_tracking_baseline"
                      else "aligned baseline" if baseline else f"episode-{greedy[0]['episode']} greedy policy")
    episodes = [int(row["episode"]) for row in greedy]
    farm = [_float(row["farm_energy_mwh"]) for row in greedy]

    figure = Figure(figsize=(11, 10.5), dpi=140, layout="constrained")
    FigureCanvasAgg(figure)
    energy_axis, delta_axis, yaw_axis = figure.subplots(3, 1, sharex=True)

    # Panel 1: farm energy against the reference controller.
    energy_axis.plot(episodes, farm, color=FARM_COLOR, linewidth=2, marker="o", markersize=5,
                     label="Greedy policy (farm total)")
    if baseline is not None:
        energy_axis.axhline(_float(baseline["farm_energy_mwh"]), color=BASELINE_COLOR,
                            linestyle="--", linewidth=1.3, label=reference_name.capitalize())
    wind_aligned = wind_aligned_reference(run_dir, "farm_energy_mwh")
    if wind_aligned is not None:
        energy_axis.axhline(wind_aligned, **WIND_ALIGNED_STYLE)
    best = max(range(len(farm)), key=lambda i: -math.inf if math.isnan(farm[i]) else farm[i])
    best_text = f"Best: episode {episodes[best]}, {farm[best]:.3f} MWh"
    if baseline is not None:
        base_energy = _float(baseline["farm_energy_mwh"])
        best_text += f" ({100 * (farm[best] / base_energy - 1):+.1f}% vs baseline)"
    if wind_aligned is not None:
        best_text += f" ({100 * (farm[best] / wind_aligned - 1):+.1f}% vs wind-aligned)"
    energy_axis.plot([episodes[best]], [farm[best]], marker="o", markersize=12, markerfacecolor="none",
                     markeredgecolor=FARM_COLOR, markeredgewidth=1.5, linestyle="none")
    energy_axis.annotate(best_text, xy=(episodes[best], farm[best]), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=9, color=TEXT_SECONDARY)
    energy_axis.set_title("Farm energy per greedy evaluation", loc="left", weight="bold", fontsize=11)
    energy_axis.set_ylabel("Energy per episode (MWh)")
    energy_axis.legend(fontsize=8, loc="lower right")
    energy_axis.margins(y=0.25)

    # Panel 2: who gains and who loses, relative to the reference.
    delta_ends, yaw_ends = [], []
    for i in range(count):
        style = dict(color=turbine_color(i), marker=turbine_marker(i), markersize=6, linewidth=2)
        deltas = [_float(row[f"turbine{i}_energy_mwh"]) - _float(reference[f"turbine{i}_energy_mwh"])
                  for row in greedy]
        delta_axis.plot(episodes, deltas, label=f"Turbine {i + 1}", **style)
        delta_ends.append((f"T{i + 1}", deltas[-1]))
        yaws = [_mean_yaw(run_dir / f"evaluation_episode{episode}_steps.csv", i) for episode in episodes]
        yaw_axis.plot(episodes, yaws, label=f"Turbine {i + 1}", **style)
        yaw_ends.append((f"T{i + 1}", yaws[-1]))
    farm_delta = [value - _float(reference["farm_energy_mwh"]) for value in farm]
    delta_axis.plot(episodes, farm_delta, color=FARM_COLOR, linewidth=1.3, linestyle=":",
                    marker="o", markersize=3, label="Farm total")
    delta_ends.append(("Farm", farm_delta[-1]))
    delta_axis.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.2, zorder=0)
    delta_axis.set_title(f"Energy change per turbine vs {reference_name}", loc="left",
                         weight="bold", fontsize=11)
    delta_axis.set_ylabel("Change (MWh)")
    delta_axis.legend(fontsize=8, ncol=count + 1, loc="upper left")
    delta_axis.margins(y=0.2)

    # Panel 3: what each turbine did.
    yaw_axis.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.2, zorder=0)
    yaw_axis.set_title("Mean wind-relative yaw during the evaluation episode", loc="left",
                       weight="bold", fontsize=11)
    yaw_axis.set_ylabel("Mean yaw (°)")
    yaw_axis.set_xlabel("Completed training episode")
    yaw_axis.margins(y=0.15)

    for axis in (energy_axis, delta_axis, yaw_axis):
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(episodes) == 1:
            axis.set_xlim(episodes[0] - 0.5, episodes[0] + 0.5)
    _direct_labels(delta_axis, delta_ends)
    _direct_labels(yaw_axis, yaw_ends)
    figure.suptitle("Greedy evaluations by turbine", fontsize=15, weight="bold")
    output = run_dir / "evaluation_turbines.png"
    figure.savefig(output)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folder", type=Path)
    print(plot_evaluation_turbines(parser.parse_args().run_folder))
