"""Render an evaluation rollout using the actors' greedy policy to MP4."""

from pathlib import Path

from plot_evaluation_states import plot_evaluation_states
from plot_evaluation_energy import plot_evaluation_energy
from plot_rewards import plot_evaluation_rewards
from training_report import write_evaluation_state_report

import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import torch


BACKGROUND = "#101b2b"
FOREGROUND = "#e6edf5"
COLORS = ("#69b9ff", "#ffc46b")
WIND = "#5ce0c5"


class WindFarmVideo:
    def __init__(self, fps, episode=None):
        self.fps = fps
        self.episode = episode
        self.figure = Figure(figsize=(12.8, 8), dpi=100, facecolor=BACKGROUND)
        self.canvas = FigureCanvasAgg(self.figure)

    def frame(self, env, obs, rates, times, powers):
        fig = self.figure
        fig.clear()
        turbines = env.interface.wind_farm.turbines
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

        farm = fig.add_axes([0.065, 0.40, 0.65, 0.44], facecolor=BACKGROUND)
        farm.set_xlim(positions[:, 0].min() - 300, positions[:, 0].max() + 300)
        farm.set_ylim(positions[:, 1].min() - 290, positions[:, 1].max() + 370)
        farm.set_aspect("equal", adjustable="box")
        farm.set_xlabel("East / x (m)", color=FOREGROUND)
        farm.set_ylabel("North / y (m)", color=FOREGROUND)
        farm.tick_params(colors=FOREGROUND, labelsize=9)
        for spine in farm.spines.values():
            spine.set_color("#35465b")
        farm.grid(color="#26364b", linewidth=0.6)

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
                      color=COLORS[i], linewidth=6, solid_capstyle="round")
            farm.scatter(*position, s=70, color=FOREGROUND, zorder=5)
            farm.annotate("", xy=position + front * 135, xytext=position,
                          arrowprops={"arrowstyle": "->", "color": COLORS[i], "lw": 2})
            local_angle = np.deg2rad(obs[i, 1])
            local_flow = -np.array([np.sin(local_angle), np.cos(local_angle)])
            start = position + np.array([0, -170])
            farm.annotate("", xy=start + local_flow * obs[i, 0] * 12, xytext=start,
                          arrowprops={"arrowstyle": "->", "color": WIND, "lw": 2})
            farm.text(position[0], position[1] - 240,
                      f"Turbine {i + 1}  |  local wind {obs[i, 0]:.2f} m/s",
                      color=COLORS[i], ha="center", fontsize=10)

        panel = fig.add_axes([0.765, 0.12, 0.22, 0.72])
        panel.axis("off")
        for i in range(2):
            y = 0.94 - i * 0.42
            panel.text(0, y, f"TURBINE {i + 1}", color=COLORS[i], fontsize=14, weight="bold")
            panel.text(0, y - 0.055,
                       f"Heading      {headings[i]:7.1f}°\n"
                       f"Relative yaw {obs[i, 2]:+7.1f}°\n"
                       f"Yaw rate     {rates[i]:+7.2f}°/s\n"
                       f"Local wind   {obs[i, 0]:7.2f} m/s\n"
                       f"Wind from    {obs[i, 1]:7.1f}°",
                       color=FOREGROUND, fontsize=11, family="monospace",
                       va="top", linespacing=1.8)
        panel.text(0, 0.08, "Rotor bar + front-facing arrow\nGreen arrows: wind travel direction\n"
                   "Headings: clockwise from north\nYaw = wind direction - heading",
                   color="#a5b4c8", fontsize=9, linespacing=1.7)

        power = fig.add_axes([0.08, 0.11, 0.62, 0.21], facecolor=BACKGROUND)
        power_mw = np.asarray(powers) / 1e6
        power.plot(times, power_mw, color=WIND, linewidth=2)
        power.set_xlim(0, env.episode_length * env.dt)
        # Zoom around the observed power history instead of the 20 MW rating.
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
                 f"Top view • evaluation (greedy policy, no exploration) • {env.dt * self.fps:g}× simulation speed • "
                 "wake field not drawn", color="#a5b4c8", fontsize=9)
        self.canvas.draw()
        return np.asarray(self.canvas.buffer_rgba())[:, :, :3].copy()


@torch.no_grad()
def record_policy_video(env, policies, output_path, fps=None, episode=None):
    """Use actor means, recording video and per-turbine state-history plots.

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
    training_modes = [policy.actor.training for policy in policies]
    try:
        for policy in policies:
            policy.actor.eval()
        obs = env.reset()
        observations = [obs.copy()]
        states = [np.zeros((1, policy.actor._recurrent_N, policy.actor.hidden_size), np.float32)
                  for policy in policies]
        mask = np.ones((1, 1), np.float32)
        times, powers = [0.0], [env._power]
        reward_history = []
        energy_history = [0.0]
        renderer = WindFarmVideo(fps, episode)
        with imageio.get_writer(output_path, format="FFMPEG", fps=fps,
                                codec="libx264", quality=8, macro_block_size=16) as writer:
            writer.append_data(renderer.frame(env, obs, np.zeros(2), times, powers))
            for step in range(env.episode_length):
                actions = []
                for i, policy in enumerate(policies):
                    action, states[i] = policy.act(obs[i:i + 1], states[i], mask, deterministic=True)
                    actions.append(action.detach().cpu().numpy()[0])
                obs, rewards, dones, infos = env.step(np.stack(actions))
                reward_history.append(float(rewards[0, 0]))
                energy_history.append(float(infos[0]["episode_energy_j"]))
                observations.append(obs.copy())
                times.append((step + 1) * env.dt)
                powers.append(infos[0]["farm_power_w"])
                rates = [item["yaw_rate_deg_s"] for item in infos]
                writer.append_data(renderer.frame(env, obs, rates, times, powers))
                if np.all(dones):
                    break
        plot_evaluation_states(times, observations, output_path.parent, episode)
        if episode is not None:
            plot_evaluation_rewards(times[1:], reward_history, output_path.parent, episode)
            plot_evaluation_energy(times, energy_history, output_path.parent, episode)
            write_evaluation_state_report(output_path.parent)
        return output_path
    finally:
        for policy, training in zip(policies, training_modes):
            policy.actor.train(training)
