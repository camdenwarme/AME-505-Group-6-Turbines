"""Numerically evaluate arbitrary-size farms without changing training state."""

import csv
from pathlib import Path

import numpy as np
import torch


def scenario_metadata(env):
    """Return stable identifiers for the wind scenario being replayed."""
    scenario = getattr(env, "current_scenario", getattr(env, "wind_scenario", None))
    data = scenario.to_dict() if hasattr(scenario, "to_dict") else (scenario or {})
    return {key: data.get(key, "") for key in
            ("scenario_id", "episode_id", "episode_seed")}


def evaluation_snapshot(env, obs, tick, rates, infos=None, commands=None):
    """Measured state for numerical/video audit logs; rates end at this tick."""
    count = len(env.action_space)
    farm = getattr(getattr(env, "interface", None), "wind_farm", None)
    turbines = [] if farm is None else farm.turbines
    metadata = scenario_metadata(env)
    row = {"tick": tick, "time_s": tick * env.dt,
           "scenario_id": metadata["scenario_id"],
           "wind_episode_id": metadata["episode_id"],
           "wind_episode_seed": metadata["episode_seed"],
           "ambient_wind_speed_m_s": env.wind_speed,
           "ambient_wind_direction_deg": env.wind_direction,
           "farm_power_w": env._power, "farm_energy_j": env.energy_j,
           "team_reward": "" if infos is None else infos[0]["team_reward"]}
    powers = getattr(env, "_turbine_power", np.full(count, np.nan))
    energies = getattr(env, "turbine_energy_j", np.zeros(count))
    commands = np.asarray(rates if commands is None else commands).reshape(count)
    for i in range(count):
        turbine = turbines[i] if turbines else None
        position = turbine.base_location[:2] if turbine is not None else (np.nan, np.nan)
        heading = turbine.get_yaw_orientation() if turbine is not None else np.nan
        values = (position[0], position[1], heading, *obs[i], float(commands[i]), float(rates[i]),
                  float(powers[i]), float(energies[i]))
        for field, value in zip(("x_m", "y_m", "heading_deg", "local_wind_speed_m_s",
                                 "local_wind_direction_deg", "yaw_deg", "commanded_rate_deg_s", "applied_rate_deg_s",
                                 "power_w", "energy_j"), values):
            row[f"turbine{i}_{field}"] = value
    return row


def write_evaluation_trajectory(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@torch.no_grad()
def evaluate_episode(env, policies=None, *, scenario=None, trajectory_path=None):
    """Evaluate a policy or initially aligned fixed headings on one scenario.

    An explicit scenario prevents baseline/policy/video resets from sampling
    different winds. In dynamic wind modes the baseline tracks measured local
    wind with the same actuator bounds; fixed wind retains initially aligned
    headings for compatibility with the historical reference.
    """
    modes = [] if policies is None else [p.actor.training for p in policies]
    count = len(env.action_space)
    if policies is not None and len(policies) != count:
        raise ValueError("Expected one policy per turbine.")
    try:
        if policies is not None:
            for policy in policies:
                policy.actor.eval()
            states = [np.zeros((1, p.actor._recurrent_N, p.actor.hidden_size), np.float32)
                      for p in policies]
        obs = env.reset() if scenario is None else env.reset(scenario=scenario)
        active_scenario = getattr(env, "current_scenario", getattr(env, "wind_scenario", None))
        mode = getattr(active_scenario, "mode", "off")
        if isinstance(active_scenario, dict):
            mode = active_scenario.get("mode", "off")
        baseline_kind = "yaw_tracking_baseline" if mode != "off" else "aligned_baseline"
        team_return = 0.0
        individual_returns = np.zeros(count, dtype=float)
        mask = np.ones((1, 1), np.float32)
        snapshots = [evaluation_snapshot(env, obs, 0, np.zeros(count))] if trajectory_path else []
        for tick in range(1, env.episode_length + 1):
            if policies is None:
                actions = np.zeros((count, 1), dtype=np.float32)
                if mode != "off":
                    actions[:, 0] = obs[:, 2] / env.dt
                    for i, space in enumerate(env.action_space):
                        actions[i] = np.clip(actions[i], space.low, space.high)
            else:
                actions = []
                for i, policy in enumerate(policies):
                    action, states[i] = policy.act(obs[i:i + 1], states[i], mask,
                                                   deterministic=True)
                    actions.append(action.detach().cpu().numpy()[0])
                actions = np.stack(actions)
            obs, rewards, dones, infos = env.step(actions)
            team_return += float(infos[0]["team_reward"])
            individual_returns += rewards[:, 0]
            if trajectory_path:
                snapshots.append(evaluation_snapshot(env, obs, tick,
                    [info["yaw_rate_deg_s"] for info in infos], infos, actions))
            if np.all(dones):
                break
        if trajectory_path:
            write_evaluation_trajectory(trajectory_path, snapshots)
        metadata = scenario_metadata(env)
        result = {"scenario_id": metadata["scenario_id"],
                  "baseline_kind": baseline_kind if policies is None else "",
                  "wind_episode_id": metadata["episode_id"],
                  "wind_episode_seed": metadata["episode_seed"],
                  "num_agents": count,
                  "farm_rated_power_w": float(getattr(env, "farm_rated_power_w",
                                                      count * env.turbine_rated_power_w)),
                  "farm_energy_mwh": float(infos[0]["episode_energy_j"]) / 3.6e9,
                  "team_return": team_return}
        for i in range(count):
            result[f"turbine{i}_energy_mwh"] = float(infos[i]["turbine_episode_energy_j"]) / 3.6e9
            result[f"turbine{i}_return"] = float(individual_returns[i])
        return result
    finally:
        if policies is not None:
            for policy, mode in zip(policies, modes):
                policy.actor.train(mode)
