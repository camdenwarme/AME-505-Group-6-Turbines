"""Plot greedy turbine energy and shared critic loss by episode."""

import argparse
import csv
from pathlib import Path
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np


def _turbine_energy_panels(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = sorted((int(match[1]), key) for key in (reader.fieldnames or ())
                         if (match := re.fullmatch(r"turbine(\d+)_energy_mwh", key)))
        rows = list(reader)

    panels = []
    for turbine, column in columns:
        greedy, benchmarks = {}, {}
        for row in rows:
            policy = row.get("policy")
            if policy not in ("greedy", "wind_aligned") or not row.get(column):
                continue
            energy = float(row[column])
            if not np.isfinite(energy):
                raise ValueError(f"Evaluation energy in {column} must be finite.")
            scenario = row.get("scenario_id")
            if policy == "greedy":
                greedy[int(row["episode"])] = (energy, scenario)
            elif scenario:
                benchmarks[scenario] = energy
        if greedy:
            episodes = sorted(greedy)
            energies, scenarios = zip(*(greedy[episode] for episode in episodes))
            relative = all(scenario in benchmarks for scenario in scenarios)
            values = np.array(energies)
            if relative:
                values -= [benchmarks[scenario] for scenario in scenarios]
            panels.append((turbine, episodes, values, relative))
    return panels


def plot_training_loss(metrics_path):
    metrics_path = Path(metrics_path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "training_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No completed training episodes are available to plot.")
    turbine_panels = _turbine_energy_panels(metrics_path.parent / "evaluation_metrics.csv")
    panel_count = len(turbine_panels) + 1
    figure = Figure(figsize=(10, 2.6 * panel_count), dpi=160, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(panel_count, 1, sharex=True, squeeze=False)[:, 0]

    def draw_panel(axis, episodes, values, title, label, color, relative=False):
        axis.plot(episodes, values, color=color, linewidth=1.5,
                  marker="o", markersize=3)
        if relative:
            axis.axhline(0, color="#777777", linestyle="--", linewidth=1.2)
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        low, high = float(values.min()), float(values.max())
        if relative:
            low, high = min(low, 0), max(high, 0)
        padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 1e-6, 1e-12)
        axis.set_ylim(low - padding, high + padding)

    colors = ("#1768ac", "#b76a12", "#8054a8", "#b34e4e")
    for axis, (turbine, episodes, values, relative) in zip(axes, turbine_panels):
        label = "Δ vs. wind-aligned (MWh)" if relative else "Energy (MWh)"
        draw_panel(axis, episodes, values, f"Turbine {turbine + 1} greedy evaluation",
                   label, colors[turbine % len(colors)], relative)

    loss = np.array([float(row["critic_value_loss"]) for row in rows])
    if not np.isfinite(loss).all():
        raise ValueError("Loss values in critic_value_loss must be finite.")
    draw_panel(axes[-1], [int(row["episode"]) for row in rows], loss,
               "Shared centralized critic", "Value loss", "#16806a")
    axes[-1].set_xlabel("Episode number")
    title = "Greedy turbine energy and critic loss" if turbine_panels else "Shared critic loss"
    figure.suptitle(title, fontsize=15, weight="bold")
    output = metrics_path.parent / "training_loss_by_episode.png"
    figure.savefig(output)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_csv", type=Path, help="Run folder or training_metrics.csv path")
    print(plot_training_loss(parser.parse_args().run_or_csv))
