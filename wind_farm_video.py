"""Render a deterministic wind-farm evaluation rollout to MP4."""

from pathlib import Path

from plot_evaluation_states import plot_evaluation_states
from plot_evaluation_energy import plot_evaluation_energy
from plot_rewards import plot_evaluation_rewards
from training_report import write_evaluation_state_report
from evaluate_wind_farm import evaluation_controls, evaluation_snapshot, write_evaluation_trajectory, scenario_metadata

import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
import numpy as np
import torch


BACKGROUND = "#101b2b"
FOREGROUND = "#e6edf5"
COLORS = ("#69b9ff", "#ffc46b", "#dd95ff", "#ff8f98", "#96d885", "#78ded5")
WIND = "#5ce0c5"
WAKE_COLORS = LinearSegmentedColormap.from_list("wake_deficit", (BACKGROUND, "#244888", "#4698cf", "#c6f2f8"))


class WindFarmVideo:
    def __init__(self, fps, episode=None, policy="greedy", *, wake_overlay=True):
        self.fps = fps
        self.episode = episode
        self.policy = policy.replace("wind_aligned", "wind-aligned").replace("_", " ")
        self.wake_overlay = wake_overlay
        self.figure = Figure(figsize=(12.8, 8), dpi=100, facecolor=BACKGROUND)
        self.canvas = FigureCanvasAgg(self.figure)

    def frame(self, env, obs, rates, times, powers):
        fig = self.figure
        fig.clear()
        turbines = env.interface.wind_farm.turbines
        count = len(turbines)
        positions = np.array([t.base_location[:2] for t in turbines])
        headings = np.array([t.get_yaw_orientation() for t in turbines])
        title = "WIND FARM  /  POLICY EVALUATION"
        if self.episode is not None:
            title += f"  /  EPISODE {self.episode}"
        fig.text(0.055, 0.94, title,
                 color=FOREGROUND, fontsize=19, weight="bold")
        fig.text(0.055, 0.90,
                 f"Time {times[-1]:.0f} / {env.episode_length * env.dt:.0f} s"
                 f"     Farm power {powers[-1] / 1e6:.2f} MW"
                 f"     Energy {env.energy_j / 3.6e9:.3f} MWh",
                 color=FOREGROUND, fontsize=12)
        scenario_id = scenario_metadata(env)["scenario_id"]
        if scenario_id:
            fig.text(0.055, 0.865, f"Wind scenario: {scenario_id}",
                     color="#a5b4c8", fontsize=8)

        sample_wake = getattr(env, "sample_wake_deficit", None) if self.wake_overlay else None
        farm_bounds = [0.065, 0.47, 0.65, 0.37] if sample_wake else [0.065, 0.40, 0.65, 0.44]
        farm = fig.add_axes(farm_bounds, facecolor=BACKGROUND)
        padding = 3 * max(turbine.diameter for turbine in turbines) if sample_wake else 300
        farm.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
        farm.set_ylim(positions[:, 1].min() - (padding if sample_wake else 290),
                      positions[:, 1].max() + max(padding, 370))
        farm.set_aspect("equal", adjustable="box")
        farm.set_xlabel("East / x (m)", color=FOREGROUND)
        farm.set_ylabel("North / y (m)", color=FOREGROUND)
        farm.tick_params(colors=FOREGROUND, labelsize=9)
        for spine in farm.spines.values():
            spine.set_color("#35465b")
        farm.grid(color="#26364b", linewidth=0.6)

        if sample_wake:
            x = np.linspace(*farm.get_xlim(), 81)
            y = np.linspace(*farm.get_ylim(), 61)
            deficit = sample_wake(x, y)
            mesh = farm.pcolormesh(x, y, deficit, shading="gouraud", cmap=WAKE_COLORS,
                                   vmin=0, vmax=100, zorder=0)
            legend_axis = fig.add_axes([0.17, 0.36, 0.44, 0.012])
            legend = fig.colorbar(mesh, cax=legend_axis, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
            legend.ax.set_title("Modeled wind-speed deficit (%)", color=FOREGROUND, fontsize=8)
            legend.ax.tick_params(colors=FOREGROUND, labelsize=8, length=2)
            legend.outline.set_edgecolor("#35465b")

        # Meteorological angles point toward the source; flow points opposite.
        ambient_radians = np.deg2rad(env.wind_direction)
        flow = -np.array([np.sin(ambient_radians), np.cos(ambient_radians)])
        anchor = np.array([positions[:, 0].mean(), positions[:, 1].max() + 250])
        farm.annotate("", xy=anchor + flow * 135, xytext=anchor - flow * 135,
                      arrowprops={"arrowstyle": "->", "color": WIND, "lw": 3})
        farm.text(anchor[0], anchor[1] + 60,
                  f"Ambient wind: {env.wind_speed:.1f} m/s from {env.wind_direction:.1f}°",
                  ha="center", color=WIND, fontsize=11)

        for i, (position, heading, turbine) in enumerate(zip(positions, headings, turbines)):
            angle = np.deg2rad(heading)
            front = np.array([np.sin(angle), np.cos(angle)])
            span = np.array([np.cos(angle), -np.sin(angle)]) * turbine.diameter / 2
            # Top-down rotor disk projected as a line, with an upstream-facing
            # arrow to distinguish front from back even after a half turn.
            farm.plot([position[0] - span[0], position[0] + span[0]],
                      [position[1] - span[1], position[1] + span[1]],
                      color=COLORS[i % len(COLORS)], linewidth=6, solid_capstyle="round")
            farm.scatter(*position, s=70, color=FOREGROUND, zorder=5)
            farm.annotate("", xy=position + front * 135, xytext=position,
                          arrowprops={"arrowstyle": "->", "color": COLORS[i % len(COLORS)], "lw": 2})
            local_angle = np.deg2rad(obs[i, 1])
            local_flow = -np.array([np.sin(local_angle), np.cos(local_angle)])
            start = position + np.array([0, -170])
            farm.annotate("", xy=start + local_flow * obs[i, 0] * 12, xytext=start,
                          arrowprops={"arrowstyle": "->", "color": WIND, "lw": 2})
            farm.text(position[0], position[1] - 240,
                      (f"Turbine {i + 1}\nlocal wind {obs[i, 0]:.2f} m/s" if count > 2 else
                       f"Turbine {i + 1}  |  local wind {obs[i, 0]:.2f} m/s"),
                      color=COLORS[i % len(COLORS)], ha="center", fontsize=8.5 if count > 2 else 10,
                      va="center", linespacing=1.2)

        panel = fig.add_axes([0.765, 0.12, 0.22, 0.72])
        panel.axis("off")
        compact = count > 2
        for i in range(count):
            y = 0.97 - i * (0.86 / count)
            panel.text(0, y, f"TURBINE {i + 1}", color=COLORS[i % len(COLORS)],
                       fontsize=12 if compact else 14, weight="bold")
            if count > 4:
                readings = (f"Heading {headings[i]:.1f}°   Yaw {obs[i, 2]:+.1f}°\n"
                            f"Yaw rate {rates[i]:+.2f}°/s\n"
                            f"Wind {obs[i, 0]:.2f} m/s from {obs[i, 1]:.1f}°")
            else:
                readings = (f"Heading      {headings[i]:7.1f}°\n"
                            f"Relative yaw {obs[i, 2]:+7.1f}°\n"
                            f"Yaw rate     {rates[i]:+7.2f}°/s\n"
                            f"Local wind   {obs[i, 0]:7.2f} m/s\n"
                            f"Wind from    {obs[i, 1]:7.1f}°")
            panel.text(0, y - (0.035 if compact else 0.055), readings,
                       color=FOREGROUND, fontsize=8.5 if compact else 11, family="monospace",
                       va="top", linespacing=1.3 if compact else 1.8)
        panel.text(0, 0.0 if compact else 0.08, "Rotor bar + front-facing arrow\nGreen arrows: wind travel direction\n"
                   "Headings: clockwise from north\nYaw = wind direction - heading",
                   color="#a5b4c8", fontsize=7.5 if compact else 9, linespacing=1.3 if compact else 1.7)

        power = fig.add_axes([0.08, 0.11, 0.62, 0.21], facecolor=BACKGROUND)
        power_mw = np.asarray(powers) / 1e6
        power.plot(times, power_mw, color=WIND, linewidth=2)
        power.set_xlim(0, env.episode_length * env.dt)
        # Zoom around the observed power history rather than the full farm rating.
        # Keep a small finite range for the initial frame or constant output.
        low, high = float(power_mw.min()), float(power_mw.max())
        padding = max((high - low) * 0.12, max(abs(low), abs(high)) * 0.001, 0.001)
        power.set_ylim(max(0, low - padding), high + padding)
        power.ticklabel_format(axis="y", style="plain", useOffset=False)
        power.set_xlabel("Simulation time (s)", color=FOREGROUND)
        power.set_ylabel("Farm power (MW)", color=FOREGROUND)
        power.tick_params(colors=FOREGROUND, labelsize=9)
        power.grid(color="#26364b", linewidth=0.6)
        for spine in power.spines.values():
            spine.set_color("#35465b")
        fig.text(0.055, 0.025,
                 f"Top view • {self.policy} policy evaluation • {env.dt * self.fps:g}× simulation speed",
                 color="#a5b4c8", fontsize=9)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()


@torch.no_grad()
def record_policy_video(env, policies, output_path, fps=None, episode=None, *, scenario=None,
                        wind_aligned=False, wake_overlay=True):
    """Record a policy's video and per-turbine state-history plots.

    By default, each frame lasts dt / 10 seconds (10x simulated-time playback).
    Uses the raw environment so the last frame is the terminal state, rather
    than DummyVecEnv's automatic reset. Training weights are never updated.
    """
    if fps is None:
        fps = 10.0 / env.dt
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video FPS must be positive.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with evaluation_controls(env, policies, scenario, wind_aligned=wind_aligned) as (obs, control, kind):
        observations = [obs.copy()]
        times, powers = [0.0], [env._power]
        reward_history = []
        energy_history = [0.0]
        initial_rates = np.zeros(len(env.action_space))
        snapshots = [evaluation_snapshot(env, obs, 0, initial_rates)]
        renderer = WindFarmVideo(fps, episode, kind, wake_overlay=wake_overlay)
        # ImageIO supplies the input FPS explicitly, so FFmpeg can skip rate probing.
        with imageio.get_writer(output_path, format="FFMPEG", fps=fps,
                                codec="libx264", quality=8, macro_block_size=16,
                                input_params=["-fpsprobesize", "0"]) as writer:
            writer.append_data(renderer.frame(env, obs, initial_rates, times, powers))
            for step in range(env.episode_length):
                actions = None if control is None else control(obs)
                obs, _, dones, infos = env.step(actions)
                # The environment now returns turbine-specific rewards; plots
                # labeled as farm reward must use the separately reported team value.
                reward_history.append(float(infos[0]["team_reward"]))
                energy_history.append(float(infos[0]["episode_energy_j"]))
                observations.append(obs.copy())
                times.append((step + 1) * env.dt)
                powers.append(infos[0]["farm_power_w"])
                rates = [item["yaw_rate_deg_s"] for item in infos]
                snapshots.append(evaluation_snapshot(env, obs, step + 1, rates, infos, actions))
                writer.append_data(renderer.frame(env, obs, rates, times, powers))
                if np.all(dones):
                    break
        write_evaluation_trajectory(output_path.with_name(output_path.stem + "_steps.csv"), snapshots)
        plot_evaluation_states(times, observations, output_path.parent, episode)
        if episode is not None:
            plot_evaluation_rewards(times[1:], reward_history, output_path.parent, episode)
            plot_evaluation_energy(times, energy_history, output_path.parent, episode)
            write_evaluation_state_report(output_path.parent)
        return output_path
