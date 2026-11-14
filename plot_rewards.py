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


def evaluation_series(run_dir, field):
    """Get per-episode greedy values and optional aligned value for a plot.

    Numeric evaluation CSV values take precedence. Legacy NPZ evaluation
    histories fill missing episodes, so a rendered evaluation is never plotted
    twice when both artifacts exist.
    """
    if field not in ("team_return", "farm_energy_mwh"):
        raise ValueError(f"Unsupported evaluation field: {field}")
    run_dir = Path(run_dir)
    greedy, baseline = {}, None
    path = run_dir / "evaluation_metrics.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if not row.get(field):
                    continue
                value = float(row[field])
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite {field} in {path}")
                if row.get("policy") == "greedy":
                    greedy[int(row["episode"])] = value
                elif row.get("policy") in ("aligned_baseline", "yaw_tracking_baseline"):
                    baseline = value

    pattern = ("evaluation_rewards_episode*.npz" if field == "team_return"
               else "evaluation_energy_episode*.npz")
    regex = (r"evaluation_rewards_episode(\d+)\.npz" if field == "team_return"
             else r"evaluation_energy_episode(\d+)\.npz")
    for path in run_dir.glob(pattern):
        match = re.fullmatch(regex, path.name)
        if not match or int(match[1]) in greedy:
            continue
        with np.load(path, allow_pickle=False) as data:
            values = np.asarray(data["reward" if field == "team_return" else "energy_j"], float)
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"Invalid evaluation history in {path}")
            value = float(values.sum() if field == "team_return" else values[-1] / 3.6e9)
        greedy[int(match[1])] = value
    return sorted(greedy.items()), baseline


def evaluation_baseline_label(run_dir):
    path = Path(run_dir) / "evaluation_metrics.csv"
    if path.exists():
        with path.open(newline="", encoding="utf-8") as stream:
            if any(row.get("policy") == "yaw_tracking_baseline" for row in csv.DictReader(stream)):
                return "Yaw-tracking baseline"
    return "Aligned baseline"


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
    evaluations, baseline = evaluation_series(run_dir, "team_return")
    if evaluations:
        episodes, returns = zip(*sorted(evaluations))
        series.append(("Evaluation (greedy policy)", "#b76a12", episodes, returns))
    if not series and baseline is None:
        raise ValueError("No recorded episode reward data are available to plot.")
    fig = Figure(figsize=(10, 4.5), dpi=160, layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    all_values = []
    for name, color, episodes, returns in series:
        axis.plot(episodes, returns, label=name, color=color, marker="o", markersize=3, linewidth=1.3)
        all_values.extend(returns)
    if baseline is not None:
        axis.axhline(baseline, color="#536276", linestyle="--", linewidth=1.2,
                     label=evaluation_baseline_label(run_dir))
        all_values.append(baseline)
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
