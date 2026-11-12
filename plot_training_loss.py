"""Plot the two actor policy losses and shared critic loss by episode."""

import argparse
import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_training_loss(metrics_path):
    metrics_path = Path(metrics_path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "training_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No completed training episodes are available to plot.")
    episodes = np.array([int(row["episode"]) for row in rows])
    figure = Figure(figsize=(10, 9), dpi=160, layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1, sharex=True)
    for axis, (column, title, label, color) in zip(axes, (
        ("actor0_policy_loss", "Turbine 1 actor", "PPO policy loss", "#1768ac"),
        ("actor1_policy_loss", "Turbine 2 actor", "PPO policy loss", "#b76a12"),
        ("critic_value_loss", "Shared centralized critic", "Value loss", "#16806a"),
    )):
        values = np.array([float(row[column]) for row in rows])
        if not np.isfinite(values).all():
            raise ValueError(f"Loss values in {column} must be finite.")
        axis.plot(episodes, values, color=color, linewidth=1.5,
                  marker="o", markersize=3)
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useOffset=False)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        low, high = float(values.min()), float(values.max())
        padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 1e-6, 1e-12)
        axis.set_ylim(low - padding, high + padding)
    axes[-1].set_xlabel("Episode number")
    if len(episodes) == 1:
        axes[-1].set_xlim(episodes[0] - 0.5, episodes[0] + 0.5)
    figure.suptitle("Wind-farm training loss", fontsize=15, weight="bold")
    output = metrics_path.parent / "training_loss_by_episode.png"
    figure.savefig(output)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_csv", type=Path, help="Run folder or training_metrics.csv path")
    print(plot_training_loss(parser.parse_args().run_or_csv))
