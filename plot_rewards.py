"""Plot reward histories and training/evaluation reward summaries."""

import argparse
import csv
from pathlib import Path
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np


def _style(axis, values):
    low, high = float(np.min(values)), float(np.max(values))
    padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 0.001, 1e-6)
    axis.set_ylim(low - padding, high + padding)
    axis.grid(alpha=0.25)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)


def plot_evaluation_rewards(times, rewards, output_dir, episode):
    times, rewards = np.asarray(times, float), np.asarray(rewards, float)
    if (times.ndim != 1 or times.shape != rewards.shape or len(times) == 0
            or not np.isfinite(times).all() or not np.isfinite(rewards).all()):
        raise ValueError("Expected matching nonempty finite reward/time histories.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Raw values allow report regeneration without replaying old policies.
    np.savez_compressed(output_dir / f"evaluation_rewards_episode{episode}.npz",
                        time_s=times, reward=rewards)
    fig = Figure(figsize=(10, 4.5), dpi=160, layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    axis.plot(times, rewards, color="#1768ac", linewidth=1.5)
    axis.set_title(f"Shared farm reward — episode {episode}", loc="left", weight="bold")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Reward per control step")
    _style(axis, rewards)
    if times[-1] > times[0]:
        axis.set_xlim(times[0], times[-1])
    output = output_dir / f"reward_history_episode{episode}.png"
    fig.savefig(output)
    fig.clear()
    return output


def plot_reward_summary(run_dir):
    run_dir = Path(run_dir)
    series = []
    csv_path = run_dir / "training_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = [row for row in csv.DictReader(stream)
                    if row.get("episode_return")]
        if rows:
            series.append(("Training (exploration)", "#1768ac",
                           [int(r["episode"]) for r in rows],
                           [float(r["episode_return"]) for r in rows]))
    evaluations = []
    for path in run_dir.glob("evaluation_rewards_episode*.npz"):
        match = re.fullmatch(r"evaluation_rewards_episode(\d+)\.npz", path.name)
        if match:
            with np.load(path, allow_pickle=False) as data:
                rewards = data["reward"]
                evaluations.append((int(match[1]), float(rewards.sum())))
    if evaluations:
        episodes, returns = zip(*sorted(evaluations))
        series.append(("Evaluation (greedy policy)", "#b76a12", episodes, returns))
    if not series:
        raise ValueError("No recorded episode reward data are available to plot.")
    fig = Figure(figsize=(10, 4.5), dpi=160, layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    all_values = []
    for name, color, episodes, returns in series:
        axis.plot(episodes, returns, label=name, color=color, marker="o", markersize=3, linewidth=1.3)
        all_values.extend(returns)
    _style(axis, all_values)
    axis.set_title("Total reward earned within each episode", loc="left", fontsize=12, weight="bold")
    axis.set_ylabel("Episode return (sum of rewards)")
    axis.legend(fontsize=9)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("Episode number (completed training episodes)")
    fig.suptitle("Wind-farm reward by episode", fontsize=15, weight="bold")
    output = run_dir / "reward_by_episode.png"
    fig.savefig(output)
    fig.clear()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    print(plot_reward_summary(parser.parse_args().run_folder))
