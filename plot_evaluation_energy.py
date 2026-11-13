"""Plot cumulative farm energy within individual evaluation episodes."""

import argparse
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

    Older reward-only evaluations used reward = interval_energy / (20 MW * dt).
    Reconstruction has only the rounding error in the saved float32 rewards.
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
                energy = np.concatenate(([0.0], np.cumsum(data["reward"] * 20e6 * np.diff(times))))
                source = "recorded_rewards"
        outputs.append(plot_evaluation_energy(times, energy, run_dir, episode, source))
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path)
    for output in regenerate_evaluation_energy(parser.parse_args().run_folder):
        print(output)
