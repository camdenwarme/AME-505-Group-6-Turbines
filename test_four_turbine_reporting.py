"""Four-agent evaluation, replay-audit logs and reporting checks."""

import csv
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from gym import spaces
import numpy as np
import torch

from evaluate_wind_farm import evaluate_episode
from plot_evaluation_energy import regenerate_evaluation_energy
from plot_training_diagnostics import make_diagnostics
from plot_training_loss import plot_training_loss
from select_team_weight import read_pilot, select
from wind_farm_video import record_policy_video, WindFarmVideo


class _Policy:
    def __init__(self):
        self.actor = torch.nn.Linear(1, 1)
        self.actor._recurrent_N, self.actor.hidden_size = 1, 1

    def act(self, obs, states, masks, deterministic=False):
        assert deterministic
        return torch.zeros((len(obs), 1)), states


class _Env:
    episode_length, dt, turbine_rated_power_w, farm_rated_power_w = 2, 4.0, 10e6, 40e6
    action_space = [spaces.Box(-6, 6, (1,), np.float32) for _ in range(4)]

    def reset(self, scenario=None):
        self.current_scenario = scenario
        self.tick, self.wind_speed, self.wind_direction = 0, 8.0, 270.0
        self.headings = np.full(4, 270.0)
        self._turbine_power = np.arange(1, 5) * 1e6
        self._power = self._turbine_power.sum()
        self.turbine_energy_j, self.energy_j = np.zeros(4), 0.0
        self.commands = []
        self.interface = SimpleNamespace(wind_farm=SimpleNamespace(turbines=[
            SimpleNamespace(base_location=np.array([i % 2 * 990, i // 2 * 990]), diameter=198,
                            get_yaw_orientation=lambda i=i: self.headings[i]) for i in range(4)]))
        return self._obs()

    def _obs(self):
        return np.column_stack((np.full(4, 8.0), np.full(4, self.wind_direction),
                                (self.wind_direction - self.headings + 180) % 360 - 180)).astype(np.float32)

    def step(self, actions):
        self.commands.append(np.asarray(actions).copy())
        self.headings += np.asarray(actions)[:, 0] * self.dt
        self.tick += 1
        self.wind_direction += 8
        self.turbine_energy_j += self._turbine_power * self.dt
        self.energy_j = float(self.turbine_energy_j.sum())
        rewards = (self._turbine_power / 10e6)[:, None].astype(np.float32)
        infos = [{"team_reward": float(rewards.mean()), "episode_energy_j": self.energy_j,
                  "turbine_episode_energy_j": self.turbine_energy_j[i],
                  "farm_power_w": self._power, "yaw_rate_deg_s": float(actions[i, 0])}
                 for i in range(4)]
        return self._obs(), rewards, np.full(4, self.tick == 2), infos


def _scenario():
    data = {"scenario_id": "wind-42", "episode_id": 7, "episode_seed": 42, "mode": "both"}
    return SimpleNamespace(**data, to_dict=lambda: data)


class FourTurbineReportingTests(unittest.TestCase):
    def test_numeric_and_video_replay_write_matching_all_tick_records(self):
        policies, scenario = [_Policy() for _ in range(4)], _scenario()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            metrics = evaluate_episode(_Env(), policies, scenario=scenario,
                                       trajectory_path=root / "numeric.csv")
            self.assertEqual(metrics["num_agents"], 4)
            self.assertEqual(metrics["scenario_id"], "wind-42")
            self.assertAlmostEqual(sum(metrics[f"turbine{i}_energy_mwh"] for i in range(4)),
                                   metrics["farm_energy_mwh"])
            class Writer:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def append_data(self, data): pass
            with patch("wind_farm_video.imageio.get_writer", return_value=Writer()), \
                 patch.object(WindFarmVideo, "frame", return_value=np.zeros((16, 16, 3), np.uint8)), \
                 patch("wind_farm_video.plot_evaluation_states"), \
                 patch("wind_farm_video.plot_evaluation_rewards"), \
                 patch("wind_farm_video.plot_evaluation_energy"), \
                 patch("wind_farm_video.write_evaluation_state_report"):
                record_policy_video(_Env(), policies, root / "video.mp4", episode=7, scenario=scenario)
            with (root / "numeric.csv").open() as stream:
                numeric = list(csv.DictReader(stream))
            with (root / "video_steps.csv").open() as stream:
                video = list(csv.DictReader(stream))
            self.assertEqual(numeric, video)
            self.assertEqual(len(video), 3)
            self.assertEqual(video[0]["team_reward"], "")
            self.assertEqual(video[-1]["time_s"], "8.0")
            self.assertIn("turbine3_heading_deg", video[-1])

    def test_dynamic_baseline_tracks_with_actuator_limits(self):
        env = _Env()
        result = evaluate_episode(env, scenario=_scenario())
        self.assertEqual(result["baseline_kind"], "yaw_tracking_baseline")
        np.testing.assert_array_equal(env.commands[0], np.zeros((4, 1)))
        np.testing.assert_allclose(env.commands[1], np.full((4, 1), 2.0))

    def test_four_turbine_frame_and_plots_include_all_agents(self):
        env = _Env()
        obs = env.reset(_scenario())
        renderer = WindFarmVideo(2.5, 0)
        frame = renderer.frame(env, obs, np.zeros(4), [0], [env._power])
        self.assertEqual(frame.shape, (800, 1280, 3))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = ["episode", "critic_value_loss", "critic_return_rmse"]
            fields += [f"actor{i}_{kind}" for i in range(4) for kind in
                       ("policy_loss", "individual_value_rmse", "ppo_clip_fraction")]
            row = {key: 0.1 for key in fields}
            row["episode"] = 1
            with (root / "training_metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerow(row)
            self.assertTrue(plot_training_loss(root).exists())
            _, summary = make_diagnostics(root)
            data = json.loads(summary.read_text())
            self.assertEqual(data["agent_ids"], [0, 1, 2, 3])
            self.assertEqual(len(data["individual_value_rmse"]), 4)

    def test_reward_energy_reconstruction_uses_recorded_four_turbine_rating(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps({"num_agents": 4}))
            np.savez(root / "evaluation_rewards_episode1.npz", time_s=[4, 8], reward=[0.5, 0.5])
            regenerate_evaluation_energy(root)
            with np.load(root / "evaluation_energy_episode1.npz") as data:
                self.assertEqual(data["energy_j"][-1], 160e6)

    def test_weight_selection_records_four_turbines_and_checks_scenario_match(self):
        with tempfile.TemporaryDirectory() as temp:
            pilots = []
            for weight, energy in ((0.25, 1.0), (0.75, 1.2)):
                root = Path(temp) / str(weight); root.mkdir()
                (root / "config.json").write_text(json.dumps({"num_agents": 4, "seed": 1,
                    "num_env_steps": 2, "episode_length": 2, "n_rollout_threads": 1,
                    "team_reward_weight": weight}))
                with (root / "training_metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, ["episode", "environment_steps"])
                    writer.writeheader(); writer.writerow({"episode": 1, "environment_steps": 2})
                fields = ["episode", "policy", "scenario_id", "farm_energy_mwh"] + [
                    f"turbine{i}_energy_mwh" for i in range(4)]
                with (root / "evaluation_metrics.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fields); writer.writeheader()
                    for episode in (0, 1):
                        row = {"episode": episode, "policy": "greedy", "scenario_id": "same-wind",
                               "farm_energy_mwh": energy if episode else 1.0}
                        row.update({f"turbine{i}_energy_mwh": row["farm_energy_mwh"] / 4 for i in range(4)})
                        writer.writerow(row)
                pilots.append(read_pilot(root))
            self.assertEqual(select(pilots)["selected_weight"], 0.75)
            self.assertIn("turbine3_energy_mwh", pilots[1]["final_greedy"])
            pilots[1]["evaluation_scenario_id"] = "different-wind"
            with self.assertRaisesRegex(ValueError, "different evaluation wind"):
                select(pilots)

    def test_weight_selection_matches_nested_wind_seeds_within_each_model_seed(self):
        pilots = []
        for seed in (1, 2):
            for weight in (0.25, 0.75):
                config = {"seed": seed, "wind_seed": seed + 10, "eval_wind_seed": seed + 20,
                          "team_reward_weight": weight, "experiment_name": f"{seed}-{weight}",
                          "num_agents": 4, "lr": 0.00035,
                          "training_wind_config": {"seed": seed + 10, "mode": "both", "speed_min": 6},
                          "evaluation_wind_config": {"seed": seed + 20, "mode": "both", "speed_min": 6}}
                pilots.append({"path": config["experiment_name"], "config": config,
                    "seed": seed, "weight": weight, "evaluation_scenario_id": f"heldout-{seed}",
                    "training_episodes": 5,
                    "initial_greedy": {"farm_energy_mwh": 1.0},
                    "final_greedy": {"farm_energy_mwh": 1.0 + weight}})
        result = select(pilots)
        self.assertEqual(result["selected_weight"], 0.75)
        self.assertEqual(result["seeds_per_candidate"], 2)
        mismatched_seed = deepcopy(pilots)
        mismatched_seed[1]["config"]["training_wind_config"]["seed"] = 999
        with self.assertRaisesRegex(ValueError, "training_wind_config"):
            select(mismatched_seed)
        mismatched_distribution = deepcopy(pilots)
        for pilot in mismatched_distribution[2:]:
            pilot["config"]["training_wind_config"]["speed_min"] = 5
        with self.assertRaisesRegex(ValueError, "training_wind_config"):
            select(mismatched_distribution)


if __name__ == "__main__":
    unittest.main()
