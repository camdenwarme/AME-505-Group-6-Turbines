"""Plot cumulative farm energy within individual evaluation episodes."""

import argparse
import csv
import json
from pathlib import Path
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def plot_evaluation_energy(times, energy_j, output_dir, episode, source="simulator"):
    times, energy_j = np.asarray(times, float), np.asarray(energy_j, float)
    if (times.ndim != 1 or times.shape != energy_j.shape or len(times) < 2
            or not np.isfinite(times).all() or not np.isfinite(energy_j).all()):
        raise ValueError("Expected matching finite time and cumulative-energy histories.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / f"evaluation_energy_episode{episode}.npz",
                        time_s=times, energy_j=energy_j, source=source)
    fig = Figure(figsize=(10, 4.5), dpi=160, layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    energy_mwh = energy_j / 3.6e9
    axis.plot(times, energy_mwh, color="#16806a", linewidth=1.5)
    axis.set_title(f"Cumulative farm energy — evaluation episode {episode}",
                   loc="left", weight="bold")
    axis.set_xlabel("Simulation time (s)")
    axis.set_ylabel("Cumulative energy generated (MWh)")
    axis.set_xlim(times[0], times[-1])
    low, high = float(energy_mwh.min()), float(energy_mwh.max())
    padding = max((high - low) * 0.08, 1e-9)
    axis.set_ylim(max(0, low - padding), high + padding)
    axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axis.grid(alpha=0.25)
    output = output_dir / f"farm_energy_history_episode{episode}.png"
    fig.savefig(output)
    fig.clear()
    return output


def regenerate_evaluation_energy(run_dir):
    """Use saved energy, or invert this environment's reward normalization.

    Reward-only reconstruction uses the recorded farm rating (historically
    20 MW for two turbines). It has the saved float32 reward rounding error.
    """
    run_dir = Path(run_dir)
    outputs = []
    for path in sorted(run_dir.glob("evaluation_rewards_episode*.npz")):
        match = re.fullmatch(r"evaluation_rewards_episode(\d+)\.npz", path.name)
        if not match:
            continue
        episode = int(match[1])
        energy_path = run_dir / f"evaluation_energy_episode{episode}.npz"
        if energy_path.exists():
            with np.load(energy_path, allow_pickle=False) as data:
                times, energy = data["time_s"], data["energy_j"]
                source = str(data["source"])
        else:
            with np.load(path, allow_pickle=False) as data:
                times = np.concatenate(([0.0], data["time_s"]))
                rating = _farm_rating(run_dir, episode)
                energy = np.concatenate(([0.0], np.cumsum(data["reward"] * rating * np.diff(times))))
                source = "recorded_rewards"
        outputs.append(plot_evaluation_energy(times, energy, run_dir, episode, source))
    return outputs


def _farm_rating(run_dir, episode):
    """Recover normalization from recorded metadata, never assume two turbines."""
    metrics = run_dir / "evaluation_metrics.csv"
    if metrics.exists():
        with metrics.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if int(row["episode"]) == episode and row.get("farm_rated_power_w"):
                    return float(row["farm_rated_power_w"])
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if "total_rated_power_w" in config:
            return float(config["total_rated_power_w"])
        if "farm_rated_power_w" in config:
            return float(config["farm_rated_power_w"])
        if "num_agents" in config:
            return int(config["num_agents"]) * float(config.get("turbine_rated_power_w", 10e6))
    raise ValueError("Reward-only energy reconstruction requires recorded farm rating or agent count.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    for output in regenerate_evaluation_energy(parser.parse_args().run_folder):
        print(output)
