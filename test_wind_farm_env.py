"""Integration checks: python -m unittest test_wind_farm_env -v."""

from pathlib import Path
import tempfile
import unittest
import sys
from types import SimpleNamespace

import numpy as np

from fdynsim import FDynSim, REPO
from wind_farm_env import WindFarmEnv


class WindFarmEnvTests(unittest.TestCase):
    def test_feedback_matches_prescribed_rollout(self):
        """Catch action timing errors by comparing to the existing batch wrapper."""
        env = WindFarmEnv(episode_length=3)
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (2, 3))
            np.testing.assert_allclose(obs[:, 2], obs[:, 1].astype(float) - 270, atol=3e-5)
            self.assertEqual(env.share_observation_space[0].shape, (6,))
            self.assertLess(obs[1, 0], obs[0, 0])  # downstream wake is observed
            case = Path(env._temporary.name) / "case.yaml"
            with tempfile.TemporaryDirectory() as work:
                batch = FDynSim([lambda *_: 1.0, lambda *_: -1.0],
                                np.full(4, 8.0), np.full(4, 270.0),
                                case=case, dt=4, work_dir=work)
            rewards = []
            for step in range(3):
                obs, reward, done, info = env.step([[1.0], [-1.0]])
                self.assertEqual(obs.shape, (2, 3))
                headings = np.array([t.get_yaw_orientation() for t in env.interface.wind_farm.turbines])
                expected = (obs[:, 1].astype(float) - headings + 180) % 360 - 180
                np.testing.assert_allclose(obs[:, 2], expected, atol=3e-5)
                for i in range(2):
                    self.assertTrue(env.observation_space[i].contains(obs[i]))
                self.assertEqual(reward.shape, (2, 1))
                self.assertEqual(reward[0, 0], reward[1, 0])
                self.assertTrue(np.all(done == (step == 2)))
                self.assertAlmostEqual(info[0]["farm_power_w"], batch["farm_power"][step + 1], delta=1e-5)
                rewards.append(float(reward[0, 0]))
            self.assertAlmostEqual(env.energy_j, batch["energy"], delta=1e-3)
            self.assertAlmostEqual(sum(rewards) * 20e6 * env.dt, env.energy_j, delta=20)
            with self.assertRaises(RuntimeError):
                env.step([[0], [0]])
        finally:
            env.close()

    def test_rate_saturation_full_rotation_zero_power_and_reset(self):
        env = WindFarmEnv(episode_length=16, wind_direction=359)
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
        env = WindFarmEnv(episode_length=4, dt=15.0)
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
        env = WindFarmEnv(episode_length=1)
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
        turbines = [SimpleNamespace(get_yaw_orientation=lambda: 270.0) for _ in range(2)]
        env.interface = SimpleNamespace(wind_farm=SimpleNamespace(turbines=turbines))
        env._iterator = iter([(pd.DataFrame({"Power_FLORIS": [1.0, 2.0]}), None,
                               np.array([[8.0, 0.0], [4.0, 1.0]]))])
        obs, _ = env._measure()
        np.testing.assert_allclose(obs[0], [8, 270, 0])
        direction = 270 - np.degrees(np.arctan2(1, 4))
        np.testing.assert_allclose(obs[1], [np.sqrt(17), direction, direction - 270], atol=2e-5)

    def test_yawed_wake_changes_downstream_observed_direction_after_travel(self):
        env = WindFarmEnv(episode_length=42)
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
