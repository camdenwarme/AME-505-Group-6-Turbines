"""Integration checks: python -m unittest test_wind_farm_env -v."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from fdynsim import FDynSim
from wind_farm_env import WindFarmEnv


class WindFarmEnvTests(unittest.TestCase):
    def test_feedback_matches_prescribed_rollout(self):
        """Catch action timing errors by comparing to the existing batch wrapper."""
        env = WindFarmEnv(episode_length=3)
        try:
            obs = env.reset()
            self.assertEqual(obs.shape, (2, 3))
            np.testing.assert_allclose(obs[:, 2], 0)
            self.assertEqual(env.share_observation_space[0].shape, (6,))
            self.assertTrue(np.allclose(obs[:, 1], 270))
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
                np.testing.assert_allclose(obs[:, 2], [-4 * (step + 1), 4 * (step + 1)])
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
                    yaw = ((-1 if i == 0 else 1) * 24 * (step + 1) + 180) % 360 - 180
                    self.assertAlmostEqual(obs[i, 2], yaw)
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


if __name__ == "__main__":
    unittest.main()
