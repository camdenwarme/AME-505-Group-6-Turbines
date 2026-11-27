"""Integration checks: python -m unittest test_wind_farm_env -v."""

from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace

import numpy as np

from fdynsim import FDynSim, REPO
from wind_farm_env import WindFarmEnv
from wind_profiles import WindProfileConfig, WindScenario


def two_env(*args, **kwargs):
    return WindFarmEnv(*args, layout=((0, 0), (5, 0)), **kwargs)


class WindFarmEnvTests(unittest.TestCase):
    def test_four_turbine_square_shapes_energy_and_metadata(self):
        env = WindFarmEnv(episode_length=2)
        try:
            obs = env.reset()
            self.assertEqual(env.num_agents, 4)
            self.assertEqual(env.remaining_time_index, 12)
            self.assertEqual(env.share_observation_space[0].shape, (37,))
            self.assertEqual(obs.shape, (4, 3))
            positions = np.array([t.base_location[:2] for t in env.interface.wind_farm.turbines])
            np.testing.assert_allclose(positions, [[0, 0], [0, 990], [990, 0], [990, 990]])
            initial_power = env._turbine_power.copy()
            obs, rewards, dones, infos = env.step(np.zeros((4, 1)))
            expected = .5 * (initial_power + env._turbine_power) * env.dt
            self.assertEqual(rewards.shape, (4, 1))
            self.assertEqual(dones.shape, (4,))
            self.assertFalse(dones.any())
            np.testing.assert_allclose(rewards[:, 0], expected / (10e6 * env.dt), rtol=1e-6)
            self.assertAlmostEqual(infos[0]["team_reward"], expected.sum() / (40e6 * env.dt))
            self.assertAlmostEqual(infos[0]["team_reward"], float(rewards.mean()), places=7)
            self.assertEqual(len(infos), 4)
            self.assertEqual(infos[0]["scenario_id"], env.wind_scenario.scenario_id)
            self.assertNotIn("wind_scenario", infos[0])
            _, _, dones, infos = env.step(np.zeros((4, 1)))
            self.assertTrue(dones.all())
            self.assertIn("wind_scenario", infos[0])
            self.assertAlmostEqual(env.energy_j, env.turbine_energy_j.sum())
        finally:
            env.close()

    def test_dynamic_wind_has_correct_ticks_and_no_uncommanded_heading_motion(self):
        scenario = WindScenario([0, 4, 8, 12], [8, 9, 10, 8], [359, 361, 363, 365],
                                mode="both", episode_seed=7)
        env = two_env(episode_length=3)
        try:
            initial = env.reset(scenario=scenario)
            headings = np.array([t.get_yaw_orientation() for t in env.interface.wind_farm.turbines])
            np.testing.assert_allclose(headings, [359, 359])
            self.assertEqual(env.wind_direction, 359)
            for tick in range(1, 4):
                obs, _, _, infos = env.step(np.zeros((2, 1)))
                expected_speed, expected_direction = scenario.sample(tick * env.dt)
                self.assertEqual(infos[0]["time_s"], tick * env.dt)
                self.assertAlmostEqual(env.wind_speed, expected_speed)
                self.assertAlmostEqual(env.wind_direction, expected_direction)
                for turbine in env.interface.wind_farm.turbines:
                    self.assertAlmostEqual(turbine.get_yaw_orientation(), 359)
                    self.assertAlmostEqual(turbine.turbine_states.get_current_yaw(), tick * 2)
                    ambient = turbine.ambient_states.get_all_states()
                    self.assertAlmostEqual(ambient[0, 0], expected_speed)
                    self.assertAlmostEqual(ambient[0, 1], 359 + tick * 2)
                    self.assertAlmostEqual(ambient[1, 0], scenario.sample((tick - 1) * env.dt)[0])
                    self.assertAlmostEqual(ambient[1, 1], 359 + (tick - 1) * 2)
                self.assertTrue(np.isfinite(obs).all())
            np.testing.assert_allclose(env.reset(scenario=scenario.to_dict()), initial)
            self.assertEqual(env.wind_scenario.scenario_id, scenario.scenario_id)
        finally:
            env.close()

    def test_dynamic_scenarios_are_seeded_and_explicit_replay_does_not_consume(self):
        config = WindProfileConfig(mode="both", seed=91, knot_interval_s=4)
        env = two_env(episode_length=2, wind_config=config)
        other = two_env(episode_length=2, wind_config=config)
        try:
            first = env.make_scenario()
            env.reset(scenario=first)
            self.assertEqual(env.make_scenario().scenario_id, first.scenario_id)
            env.reset()
            self.assertEqual(env.wind_scenario.scenario_id, first.scenario_id)
            second = env.make_scenario()
            self.assertNotEqual(first.scenario_id, second.scenario_id)
            self.assertEqual(other.make_scenario(1).scenario_id, second.scenario_id)
        finally:
            env.close()
            other.close()

    def test_wake_overlay_preserves_downstream_history_after_wind_switch(self):
        scenario = WindScenario([0, 4, 8], [8, 8, 8], [270, 360, 360],
                                mode="direction", direction_interpolation="step_hold")
        env = WindFarmEnv(episode_length=2, layout=[(0, 0)])
        x = np.linspace(-600, 900, 31)
        y = np.linspace(-600, 600, 25)
        try:
            env.reset(scenario=scenario)
            field = env.sample_wake_deficit(x, y)
            self.assertEqual(field.shape, (len(y), len(x)))
            self.assertTrue(np.all((field >= 0) & (field <= 100)))
            east_wake = field[12, 24]
            self.assertGreater(east_wake, 10)
            self.assertLess(field[12, 4], 1)

            env.step(np.zeros((1, 1)))
            self.assertEqual(env.wind_direction, 0)
            switched = env.sample_wake_deficit(x, y)
            self.assertAlmostEqual(switched[12, 24], east_wake, delta=0.1)
            self.assertGreater(np.max(np.abs(switched - field)), 1)
        finally:
            env.close()

    def test_wind_aligned_policy_tracks_local_wind_at_limited_rate(self):
        scenario = WindScenario([0, 4, 8, 24, 36], [8] * 5, [359, 361, 405, 361, 361],
                                mode="direction", direction_interpolation="step_hold")
        env = WindFarmEnv(episode_length=9, layout=[(0, 0)])
        try:
            obs = env.reset(scenario=scenario, wind_aligned=True)
            turbine = env.interface.wind_farm.turbines[0]
            rates, yaws, powers = [], [], []
            for _ in range(env.episode_length):
                expected_rate = np.clip(float(obs[0, 2]) / env.dt, -6, 6)
                previous_heading = turbine.get_yaw_orientation()
                obs, _, _, infos = env.step()
                rate = infos[0]["yaw_rate_deg_s"]
                heading = turbine.get_yaw_orientation()
                self.assertAlmostEqual(rate, expected_rate, places=6)
                self.assertAlmostEqual(heading, (previous_heading + rate * env.dt) % 360)
                ambient_yaw = (env.wind_direction - heading + 180) % 360 - 180
                self.assertAlmostEqual(turbine.turbine_states.get_current_yaw(), ambient_yaw)
                rates.append(rate)
                yaws.append(float(obs[0, 2]))
                powers.append(env._power)

            self.assertEqual(max(rates), 6.0)
            self.assertEqual(min(rates), -6.0)
            self.assertGreater(yaws[1], 24.0)
            self.assertLess(abs(yaws[4]), 1.0)
            self.assertLess(yaws[5], -24.0)
            self.assertLess(abs(yaws[-1]), 1.0)
            self.assertLess(powers[1], powers[4])
            self.assertLess(powers[5], powers[-1])
        finally:
            env.close()

    def test_configured_wind_changes_aligned_reference(self):
        reference = []
        for speed in (8.0, 10.0):
            env = two_env(episode_length=4, wind_speed=speed)
            try:
                env.reset()
                reference.append(env.get_individual_baseline_values(1.0))
            finally:
                env.close()
        self.assertGreater(reference[1][0], reference[0][0])
        self.assertLess(reference[0][1], reference[0][0])  # downstream wake

    def test_aligned_reference_uses_local_wind_and_remaining_horizon(self):
        # No OFF simulation is started: values are read from FLORIS's turbine
        # curve and the critic's current local measurements.
        env = two_env(episode_length=4, wind_speed=8.0)
        env._iterator = object()
        env._steps = 0
        env._obs = np.array([[8.0, 270.0, 0.0], [6.0, 270.0, 0.0]], np.float32)
        # FLORIS's IEA-10MW table reports 4.44026484 MW and 1.85502326 MW.
        rates = np.array([4.44026484, 1.85502326]) / 10.0
        np.testing.assert_allclose(env.get_individual_baseline_values(1.0), 4 * rates, rtol=1e-6)
        np.testing.assert_allclose(env.get_individual_baseline_values(0.5),
                                   (1 + 0.5 + 0.25 + 0.125) * rates, rtol=1e-6)
        env._steps = 3
        np.testing.assert_allclose(env.get_individual_baseline_values(1.0), rates, rtol=1e-6)
        env._obs[:, 0] = [10.0, 8.0]  # changing wind changes each reference
        changed = env.get_individual_baseline_values(1.0)
        self.assertGreater(changed[0], rates[0])
        self.assertGreater(changed[1], rates[1])
        env._steps = 4
        np.testing.assert_array_equal(env.get_individual_baseline_values(1.0), [0, 0])
        with self.assertRaises(ValueError):
            env.get_individual_baseline_values(1.1)
        env._iterator = None

    def test_feedback_matches_prescribed_rollout(self):
        """Catch action timing errors by comparing to the existing batch wrapper."""
        env = two_env(episode_length=3)
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (2, 3))
            np.testing.assert_allclose(obs[:, 2], obs[:, 1].astype(float) - 270, atol=3e-5)
            self.assertEqual(env.share_observation_space[0].shape, (19,))
            initial_central = env.get_centralized_obs()
            np.testing.assert_allclose(initial_central[:6], (obs / [10, 360, 180]).reshape(-1), atol=1e-7)
            self.assertEqual(initial_central[6], 1.0)
            self.assertTrue(env.share_observation_space[0].contains(initial_central))
            self.assertLess(obs[1, 0], obs[0, 0])  # downstream wake is observed
            case = Path(env._temporary.name) / "case.yaml"
            with tempfile.TemporaryDirectory() as work:
                batch = FDynSim([lambda *_: 1.0, lambda *_: -1.0],
                                np.full(4, 8.0), np.full(4, 270.0),
                                case=case, dt=4, work_dir=work)
            rewards = []
            team_rewards = []
            batch_turbine_power = batch["measurements"].pivot(index="time", columns="t_idx", values="Power_FLORIS").sort_index().to_numpy()
            for step in range(3):
                obs, reward, done, info = env.step([[1.0], [-1.0]])
                self.assertEqual(obs.shape, (2, 3))
                headings = np.array([t.get_yaw_orientation() for t in env.interface.wind_farm.turbines])
                expected = (obs[:, 1].astype(float) - headings + 180) % 360 - 180
                np.testing.assert_allclose(obs[:, 2], expected, atol=3e-5)
                for i in range(2):
                    self.assertTrue(env.observation_space[i].contains(obs[i]))
                self.assertEqual(reward.shape, (2, 1))
                expected_turbine_energy = 0.5 * (batch_turbine_power[step] + batch_turbine_power[step + 1]) * env.dt
                np.testing.assert_allclose(reward[:, 0], expected_turbine_energy / (10e6 * env.dt), rtol=1e-6)
                self.assertAlmostEqual(info[0]["team_reward"], float(reward.mean()), places=7)
                np.testing.assert_allclose([item["turbine_interval_energy_j"] for item in info], expected_turbine_energy, rtol=1e-10)
                self.assertAlmostEqual(env.get_centralized_obs()[6], 1 - (step + 1) / 3, places=6)
                self.assertTrue(np.all(done == (step == 2)))
                self.assertAlmostEqual(info[0]["farm_power_w"], batch["farm_power"][step + 1], delta=1e-5)
                rewards.append(reward[:, 0].copy())
                team_rewards.append(info[0]["team_reward"])
            self.assertAlmostEqual(env.energy_j, batch["energy"], delta=1e-3)
            self.assertAlmostEqual(sum(team_rewards) * 20e6 * env.dt, env.energy_j, delta=1e-3)
            np.testing.assert_allclose(np.sum(rewards, axis=0) * 10e6 * env.dt, env.turbine_energy_j, rtol=1e-6)
            self.assertAlmostEqual(sum(item["turbine_episode_energy_j"] for item in info), env.energy_j, delta=1e-3)
            with self.assertRaises(RuntimeError):
                env.step([[0], [0]])
        finally:
            env.close()

    def test_rate_saturation_full_rotation_zero_power_and_reset(self):
        env = two_env(episode_length=16, wind_direction=359)
        try:
            initial = env.reset()
            for step in range(16):
                obs, _, _, info = env.step([[100], [-100]])
                self.assertTrue(np.isfinite(obs).all())
                for item in info:
                    self.assertLessEqual(abs(item["yaw_rate_deg_s"]), 6.0)
                for i, turbine in enumerate(env.interface.wind_farm.turbines):
                    expected = (359 + (1 if i == 0 else -1) * 24 * (step + 1)) % 360
                    self.assertAlmostEqual(turbine.get_yaw_orientation(), expected)
                    self.assertEqual(abs(turbine.turbine_states.get_current_yaw()), 24 * (step + 1))
                    self.assertEqual(abs(info[i]["yaw_rate_deg_s"]), 6)
                    yaw = (float(obs[i, 1]) - expected + 180) % 360 - 180
                    self.assertAlmostEqual(float(obs[i, 2]), yaw, delta=3e-5)
                    if 90 < 24 * (step + 1) < 270:
                        self.assertEqual(turbine.calc_power(8.0), 0)
                if 90 < 24 * (step + 1) < 270:
                    self.assertEqual(info[0]["farm_power_w"], 0)
                if step == 14:  # 360 degrees: aligned again
                    self.assertGreater(info[0]["farm_power_w"], 0)
            np.testing.assert_allclose(env.reset(), initial)
            self.assertEqual(env.energy_j, 0)
        finally:
            env.close()

    def test_exact_sideways_angles_and_one_turbine_still_generating(self):
        env = two_env(episode_length=4, dt=15.0)
        try:
            env.reset()
            for step in range(4):
                obs, reward, _, info = env.step([[6], [0]])
                self.assertTrue(np.isfinite(obs).all())
                self.assertTrue(np.isfinite(reward).all())
                self.assertGreater(info[0]["farm_power_w"], 0)
                upstream = env.interface.wind_farm.turbines[0]
                if step < 3:  # 90, 180, 270 degrees are sideways/backward.
                    self.assertEqual(upstream.calc_power(8.0), 0)
                else:  # 360 degrees: normal power resumes.
                    self.assertGreater(upstream.calc_power(8.0), 0)
        finally:
            env.close()

    def test_invalid_actions_do_not_advance(self):
        env = two_env(episode_length=1)
        try:
            env.reset()
            for invalid in ([1, 2], [[np.nan], [0]]):
                with self.assertRaises(ValueError):
                    env.step(invalid)
            self.assertEqual(env._steps, 0)
        finally:
            env.close()

    def test_rotor_vector_averaging_and_world_rotation(self):
        sys.path.insert(0, str(REPO / "03_Code"))
        from off.wake_model import Floris4Wake
        wake = Floris4Wake.__new__(Floris4Wake)
        # Two turbines in original order. Signed lateral samples must be
        # averaged linearly, not converted to a speed or a cubic mean.
        u = np.array([[[[8.0], [8.0]], [[4.0], [4.0]]]])
        v = np.array([[[[3.0], [-1.0]], [[-3.0], [1.0]]]])
        flow = SimpleNamespace(u=u, v=v, wind_directions=np.array([270.0]))
        grid = SimpleNamespace(cubature_weights=None)
        wake.fmodel = SimpleNamespace(core=SimpleNamespace(flow_field=flow, grid=grid))
        np.testing.assert_allclose(wake.get_rotor_velocities_world()[0], [[8, 1], [4, -1]])
        flow.wind_directions[:] = 180
        np.testing.assert_allclose(wake.get_rotor_velocities_world()[0], [[-1, 8], [1, 4]], atol=1e-12)
        flow.wind_directions[:] = 0
        np.testing.assert_allclose(wake.get_rotor_velocities_world()[0], [[1, -8], [-1, -4]], atol=1e-12)
        flow.wind_directions[:] = 270
        grid.cubature_weights = np.array([0.75, 0.25])
        np.testing.assert_allclose(wake.get_rotor_velocities_world()[0], [[8, 2], [4, -2]])

    def test_observation_yaw_uses_local_velocity_not_ambient_direction(self):
        import pandas as pd
        env = WindFarmEnv.__new__(WindFarmEnv)
        env.num_agents = env.num_agent = 2
        env.controller = None
        turbines = [SimpleNamespace(get_yaw_orientation=lambda: 270.0) for _ in range(2)]
        env.interface = SimpleNamespace(wind_farm=SimpleNamespace(turbines=turbines))
        env._iterator = iter([(pd.DataFrame({"Power_FLORIS": [1.0, 2.0]}), None,
                               np.array([[8.0, 0.0], [4.0, 1.0]]))])
        obs, _ = env._measure()
        np.testing.assert_allclose(obs[0], [8, 270, 0])
        direction = 270 - np.degrees(np.arctan2(1, 4))
        np.testing.assert_allclose(obs[1], [np.sqrt(17), direction, direction - 270], atol=2e-5)

    def test_critic_observation_includes_time_power_and_yaw_history(self):
        env = two_env(episode_length=10)
        env._iterator = iter(())  # get_centralized_obs only checks an active episode
        env._steps = 3
        env._obs = np.array([[8, 270, 3], [6, 268, -2]], dtype=np.float32)
        env._turbine_power = np.array([8e6, 4e6])
        histories = [np.arange(41, dtype=float), np.full(3, 270.0)]
        turbines = [SimpleNamespace(turbine_states=SimpleNamespace(get_all_yaw=lambda h=h: h))
                    for h in histories]
        env.interface = SimpleNamespace(wind_farm=SimpleNamespace(turbines=turbines))
        central = env.get_centralized_obs()
        np.testing.assert_allclose(central[:6], (env._obs / [10, 360, 180]).flatten(), atol=1e-7)
        np.testing.assert_allclose(central[6:9], [0.7, 0.8, 0.4])
        np.testing.assert_allclose(central[9:14], np.array([5, 10, 20, 30, 40]) / 180)
        np.testing.assert_allclose(central[14:], np.full(5, -0.5))
        self.assertTrue(env.share_observation_space[0].contains(central))

    def test_yawed_wake_changes_downstream_observed_direction_after_travel(self):
        env = two_env(episode_length=42)
        try:
            initial = env.reset()
            early = None
            for step in range(42):
                # Reach +30 ambient-relative yaw, then hold that orientation.
                rate = -6 if step == 0 else (-1.5 if step == 1 else 0)
                obs, _, _, _ = env.step([[rate], [0]])
                if step == 20:
                    early = obs.copy()
            np.testing.assert_allclose(early[1], initial[1], atol=1e-4)
            self.assertGreater(abs(float(obs[1, 1] - initial[1, 1])), 0.1)
            self.assertAlmostEqual(env.interface.wind_farm.turbines[1].get_yaw_orientation(), 270)
            self.assertAlmostEqual(float(obs[1, 2]), float(obs[1, 1]) - 270, delta=3e-5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
