"""Plot each turbine's actor observations from an evaluation (greedy policy)."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


def plot_evaluation_states(times, observations, output_dir, episode=None):
    times = np.asarray(times, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if (times.ndim != 1 or observations.ndim != 3
            or observations.shape[0] != len(times) or observations.shape[2] != 3
            or len(times) == 0):
        raise ValueError("Expected times (T,) and turbine observations (T, agents, 3).")
    if not np.isfinite(times).all() or not np.isfinite(observations).all():
        raise ValueError("State histories must contain finite times and observations.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_episode{episode}" if episode is not None else ""
    outputs = []
    for agent in range(observations.shape[1]):
        figure = Figure(figsize=(10, 8), dpi=160, layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(3, 1, sharex=True)
        for state, (axis, title, label, color) in enumerate(zip(
            axes,
            ("Local wind speed", "Wind direction (meteorological, from)", "Wind-relative turbine yaw"),
            ("Wind speed (m/s)", "Wind direction (°)", "Relative yaw (°)"),
            ("#1768ac", "#16806a", "#b76a12"),
        )):
            values = observations[:, agent, state]
            axis.plot(times, values, color=color, linewidth=1.5)
            axis.set_title(title, loc="left", fontsize=11, weight="bold")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
            low, high = float(values.min()), float(values.max())
            padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 0.001, 0.001)
            axis.set_ylim(low - padding, high + padding)
        axes[-1].set_xlabel("Simulation time (s)")
        if times[-1] > times[0]:
            axes[-1].set_xlim(times[0], times[-1])
        title = f"Turbine {agent + 1} state history — evaluation"
        if episode is not None:
            title += f" — episode {episode}"
        figure.suptitle(title, fontsize=13, weight="bold")
        output = output_dir / f"turbine{agent + 1}_states{suffix}.png"
        figure.savefig(output)
        figure.clear()
        outputs.append(output)
    return outputs
