"""Plot per-episode and running cumulative farm energy from a run's CSV."""

import argparse
import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_training_energy(metrics_path):
    metrics_path = Path(metrics_path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "training_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No completed training episodes are available to plot.")
    episodes = np.array([int(row["episode"]) for row in rows])
    energy = np.array([float(row["episode_energy_mwh"]) for row in rows])
    cumulative = np.array([float(row["cumulative_energy_mwh"]) for row in rows])
    if not np.isfinite(energy).all() or not np.isfinite(cumulative).all():
        raise ValueError("Energy values must be finite.")

    figure = Figure(figsize=(10, 7), dpi=160, layout="constrained")
    FigureCanvasAgg(figure)
    per_episode, running_total = figure.subplots(2, 1, sharex=True)
    for axis, values, title, label, color in (
        (per_episode, energy, "Energy generated during each training episode",
         "Episode energy (MWh)", "#1768ac"),
        (running_total, cumulative, "Cumulative energy across training episodes",
         "Cumulative energy (MWh)", "#16806a"),
    ):
        axis.plot(episodes, values, color=color, linewidth=1.5,
                  marker="o", markersize=3)
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        # Fit the observed range rather than forcing zero. A small fallback
        # keeps single-episode or constant-energy runs from producing zero span.
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 1e-6, 1e-10)
        axis.set_ylim(max(0, low - padding) if low >= 0 else low - padding,
                      high + padding)
    per_episode.text(0.99, 0.96, "Zoomed y-axis", ha="right", va="top",
                     transform=per_episode.transAxes, fontsize=9, color="#555555")
    running_total.set_xlabel("Episode number")
    if len(episodes) == 1:
        running_total.set_xlim(episodes[0] - 0.5, episodes[0] + 0.5)
    figure.suptitle("Wind-farm training energy", fontsize=15, weight="bold")
    output = metrics_path.parent / "cumulative_energy_by_episode.png"
    figure.savefig(output)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_csv", type=Path, help="Run folder or training_metrics.csv path")
    print(plot_training_energy(parser.parse_args().run_or_csv))
