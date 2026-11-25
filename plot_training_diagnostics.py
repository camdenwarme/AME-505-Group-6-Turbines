"""Summarize critic error, PPO clipping, yaw safety, and greedy farm energy."""

import argparse
import csv
import json
import math
from pathlib import Path
import re

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from plot_rewards import draw_wind_aligned, evaluation_baseline_label, wind_aligned_reference
from plot_style import FARM_COLOR, TURBINE_COLORS


def _rows(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _episode(value):
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _series(rows, field, *, policy=None):
    points = {}
    for row in rows:
        if policy is not None and row.get("policy") != policy:
            continue
        episode, value = _episode(row.get("episode")), _number(row.get(field))
        if episode is not None and value is not None:
            points[episode] = value
    return sorted(points.items())


def _stats(series):
    values = [value for _, value in series]
    if not values:
        return {"count": 0, "first": None, "last": None, "mean": None,
                "min": None, "max": None}
    return {"count": len(values), "first": values[0], "last": values[-1],
            "mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _yaw_series(rows, agent_ids):
    """Return per-episode and pooled fractions of finite yaw samples >= 90°."""
    result = []
    for turbine in agent_ids:
        counts = {}
        for row in rows:
            episode = _episode(row.get("episode"))
            yaw = _number(row.get(f"actor{turbine}_yaw_deg"))
            if episode is None or yaw is None:
                continue
            outside, total = counts.get(episode, (0, 0))
            counts[episode] = (outside + int(abs(yaw) >= 90), total + 1)
        series = [(episode, outside / total) for episode, (outside, total) in sorted(counts.items())]
        outside = sum(pair[0] for pair in counts.values())
        total = sum(pair[1] for pair in counts.values())
        result.append((series, {"valid_samples": total, "samples_abs_yaw_ge_90": outside,
                                "pooled_fraction": outside / total if total else None}))
    return result


def make_diagnostics(run_dir, output_dir=None):
    """Write a four-panel PNG and a JSON summary, omitting missing/nonfinite data."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    training = _rows(run_dir / "training_metrics.csv")
    evaluation = _rows(run_dir / "evaluation_metrics.csv")
    rollout = _rows(run_dir / "rollout_steps.csv")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    found = {int(match[1]) for row in training[:1] + rollout[:1] for key in row
             if (match := re.match(r"actor(\d+)_", key))}
    agent_ids = list(range(int(config["num_agents"]))) if "num_agents" in config else sorted(found)
    if not agent_ids:  # Preserve empty legacy-report behavior without assuming new runs are two-agent.
        agent_ids = [0, 1]
    colors = TURBINE_COLORS
    rmse = [("Farm critic", FARM_COLOR, _series(training, "critic_return_rmse"))]
    rmse.extend((f"Turbine {i + 1} critic", colors[j % len(colors)],
                 _series(training, f"actor{i}_individual_value_rmse")) for j, i in enumerate(agent_ids))
    clips = [(f"Turbine {i + 1} actor", colors[j % len(colors)],
              _series(training, f"actor{i}_ppo_clip_fraction")) for j, i in enumerate(agent_ids)]
    yaw = _yaw_series(rollout, agent_ids)
    yaw_lines = [(f"Turbine {i + 1}", colors[j % len(colors)], yaw[j][0])
                 for j, i in enumerate(agent_ids)]
    greedy = _series(evaluation, "farm_energy_mwh", policy="greedy")
    baseline_points = (_series(evaluation, "farm_energy_mwh", policy="aligned_baseline") or
                       _series(evaluation, "farm_energy_mwh", policy="yaw_tracking_baseline"))
    baseline = baseline_points[-1][1] if baseline_points else None
    wind_aligned = wind_aligned_reference(run_dir, "farm_energy_mwh")

    fig = Figure(figsize=(12, 8.4), dpi=140, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2).reshape(-1)

    def panel(axis, lines, title, ylabel, baseline=None, references=False):
        drawn = False
        episodes = []
        for label, color, points in lines:
            if points:
                xs, ys = zip(*points)
                episodes.extend(xs)
                axis.plot(xs, ys, marker="o", markersize=3, linewidth=1.3,
                          label=label, color=color)
                drawn = True
        if baseline is not None:
            axis.axhline(baseline, color="#536276", linestyle="--", linewidth=1.2,
                         label=evaluation_baseline_label(run_dir))
            drawn = True
        if references and draw_wind_aligned(axis, run_dir, "farm_energy_mwh") is not None:
            drawn = True
        axis.set_title(title, loc="left", weight="bold", fontsize=11)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Completed training episode")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if episodes and min(episodes) == max(episodes):
            axis.set_xlim(episodes[0] - 0.5, episodes[0] + 0.5)
        axis.grid(alpha=0.25)
        if drawn:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No data recorded", ha="center", va="center",
                      transform=axis.transAxes, color="#536276")
        return drawn

    panel(axes[0], rmse, "Critic prediction error", "Return RMSE (reward units)")
    panel(axes[1], clips, "PPO ratios outside clip range", "Fraction of samples")
    panel(axes[2], yaw_lines, "Exploratory yaw magnitude ≥ 90°", "Fraction of finite samples")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.02)
    panel(axes[3], [("Greedy evaluation", "#1768ac", greedy)],
          "Greedy farm energy", "Energy per episode (MWh)", baseline=baseline,
          references=True)
    fig.suptitle("Wind-farm training diagnostics", fontsize=16, weight="bold")
    plot_path = output_dir / "training_diagnostics.png"
    fig.savefig(plot_path)
    fig.clear()

    summary = {"source_run": str(run_dir.resolve()),
               "training_metric_rows": len(training), "evaluation_rows": len(evaluation),
               "rollout_rows": len(rollout),
               "agent_ids": agent_ids,
               "team_value_rmse": _stats(rmse[0][2]),
               "individual_value_rmse": [_stats(item[2]) for item in rmse[1:]],
               "ppo_clip_fraction": [_stats(item[2]) for item in clips],
               "yaw_abs_ge_90": [item[1] for item in yaw],
               "greedy_farm_energy_mwh": _stats(greedy),
               "aligned_baseline_energy_mwh": baseline,
               "wind_aligned_energy_mwh": wind_aligned,
               "note": "Critic errors and yaw fractions use exploratory training rollouts; energy uses greedy evaluation."}
    json_path = output_dir / "training_diagnostics.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return plot_path, json_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    for path in make_diagnostics(args.run_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
