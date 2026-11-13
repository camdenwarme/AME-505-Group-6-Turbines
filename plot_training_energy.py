"""Plot the final energy captured within each episode from a run's CSV."""

import argparse
import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
from plot_rewards import evaluation_series


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
    if not np.isfinite(energy).all():
        raise ValueError("Energy values must be finite.")

    figure = Figure(figsize=(10, 5), dpi=160, layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(episodes, energy, color="#1768ac", linewidth=1.5,
              marker="o", markersize=3, label="Training (exploration)")
    values, all_episodes = list(energy), list(episodes)
    evaluations, baseline = evaluation_series(metrics_path.parent, "farm_energy_mwh")
    if evaluations:
        eval_episodes, eval_energy = zip(*evaluations)
        axis.plot(eval_episodes, eval_energy, color="#b76a12", linewidth=1.5,
                  marker="o", markersize=3, label="Evaluation (greedy policy)")
        values.extend(eval_energy)
        all_episodes.extend(eval_episodes)
    if baseline is not None:
        axis.axhline(baseline, color="#536276", linestyle="--", linewidth=1.2,
                     label="Aligned baseline")
        values.append(baseline)
    axis.set_title("Farm energy within each episode", loc="left",
                   fontsize=12, weight="bold")
    axis.set_ylabel("Energy captured at episode end (MWh)")
    axis.set_xlabel("Episode number")
    axis.grid(alpha=0.25)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend(fontsize=9)
    # Fit the observed range rather than forcing zero. A small fallback keeps
    # single-episode or constant-energy runs from producing zero span.
    low, high = float(min(values)), float(max(values))
    padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 1e-6, 1e-10)
    axis.set_ylim(max(0, low - padding) if low >= 0 else low - padding, high + padding)
    axis.text(0.99, 0.96, "Zoomed y-axis", ha="right", va="top",
              transform=axis.transAxes, fontsize=9, color="#555555")
    if min(all_episodes) == max(all_episodes):
        axis.set_xlim(all_episodes[0] - 0.5, all_episodes[0] + 0.5)
    figure.suptitle("Wind-farm training energy", fontsize=15, weight="bold")
    output = metrics_path.parent / "cumulative_energy_by_episode.png"
    figure.savefig(output)
    figure.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_csv", type=Path, help="Run folder or training_metrics.csv path")
    print(plot_training_energy(parser.parse_args().run_or_csv))
