"""Wind reproducibility, direction switches, and replay checks without a simulator."""

import json
import unittest

import numpy as np

from wind_profiles import WindProfileConfig, WindScenario, make_wind_scenario


class WindProfileTests(unittest.TestCase):
    def generate(self, mode="both", **kwargs):
        return make_wind_scenario(WindProfileConfig(mode=mode, seed=120, knot_interval_s=30),
                                  95, 8, 359, **({"episode_id": 2} | kwargs))

    def test_episode_and_stream_ids_are_reproducible_and_independent(self):
        first = self.generate()
        self.generate(episode_id=100, stream_id=1)
        self.assertEqual(first.to_dict(), self.generate().to_dict())
        for kw in ({"episode_id": 3}, {"stream_id": 1}, {"environment_id": 1}):
            other = self.generate(**kw)
            self.assertNotEqual(first.episode_seed, other.episode_seed)
            self.assertNotEqual(first.scenario_id, other.scenario_id)
            self.assertFalse(np.array_equal(first.speeds_m_s, other.speeds_m_s))

    def test_generation_does_not_consume_global_numpy_randomness(self):
        np.random.seed(17)
        expected = np.random.random(5)
        np.random.seed(17)
        self.generate()
        np.testing.assert_array_equal(expected, np.random.random(5))

    def test_toggling_one_quantity_keeps_the_other_realization(self):
        combined = self.generate("both")
        np.testing.assert_array_equal(combined.speeds_m_s, self.generate("speed").speeds_m_s)
        np.testing.assert_array_equal(combined.directions_deg, self.generate("direction").directions_deg)

    def test_short_final_interval_preserves_speed_transition_rate(self):
        config = WindProfileConfig(mode="both", seed=31, knot_interval_s=30,
                                   speed_min=6, speed_max=10, direction_half_width=15)
        profile = make_wind_scenario(config, 91, 8, 359, 0)
        delta = np.diff(profile.times_s)
        self.assertTrue(np.all(np.abs(np.diff(profile.speeds_m_s)) / delta <= 4 / 30 + 1e-12))

    def test_direction_switches_at_each_knot_and_holds_through_partial_interval(self):
        config = WindProfileConfig(mode="both", seed=31)
        profile = make_wind_scenario(config, 251, 8, 359, 0)
        np.testing.assert_array_equal(profile.times_s, [0, 120, 240, 251])
        first, second, third, final = profile.directions_deg % 360
        self.assertNotEqual(first, second)
        self.assertNotEqual(second, third)
        self.assertEqual(third, final)
        times = [-1, 0, 60, 119.999, 120, 180, 239.999, 240, 250, 251, 260]
        speeds, directions = profile.sample(times)
        np.testing.assert_array_equal(directions,
                                      [first] * 4 + [second] * 3 + [third] * 4)
        self.assertAlmostEqual(speeds[2], (profile.speeds_m_s[0] + profile.speeds_m_s[1]) / 2)
        exact_end = make_wind_scenario(config, 240, 8, 359, 0)
        self.assertEqual(exact_end.sample(240)[1], third)

    def test_modes_vary_only_requested_quantities_and_cover_exact_duration(self):
        for mode in ("off", "speed", "direction", "both"):
            with self.subTest(mode=mode):
                profile = self.generate(mode)
                self.assertEqual(profile.times_s[0], 0)
                self.assertEqual(profile.times_s[-1], 95)
                self.assertTrue(np.all(np.diff(profile.times_s) > 0))
                speeds, directions = profile.sample(np.linspace(0, 95, 101))
                if mode in ("speed", "both"):
                    self.assertGreater(np.ptp(speeds), 0)
                    self.assertGreaterEqual(speeds.min(), 6)
                    self.assertLessEqual(speeds.max(), 10)
                else:
                    np.testing.assert_array_equal(speeds, np.full(101, 8))
                if mode in ("direction", "both"):
                    self.assertGreater(np.ptp(profile.directions_deg), 0)
                    self.assertTrue(np.all(np.abs(np.diff(profile.directions_deg)) <= 180 + 1e-10))
                else:
                    np.testing.assert_array_equal(directions, np.full(101, 359))

    def test_north_crossing_and_scalar_vector_samples(self):
        profile = WindScenario([0, 10], [6, 10], [359, 361], mode="both")
        replay = WindScenario.from_dict(profile.to_dict())
        self.assertEqual(replay.sample(5), (8, 0))
        self.assertEqual(profile.sample(5), (8, 0))
        np.testing.assert_allclose(profile.sample([0, 5, 10]), [[6, 8, 10], [359, 0, 1]])
        self.assertEqual(profile.direction_unwrapped(5), 360)
        self.assertEqual(profile.sample(-1), (6, 359))
        self.assertEqual(profile.sample(20), (10, 1))
        with self.assertRaises(ValueError):
            profile.sample(float("nan"))

    def test_json_roundtrip_keeps_realized_profile_and_metadata(self):
        profile = self.generate()
        replay = WindScenario.from_dict(json.loads(json.dumps(profile.to_dict())))
        self.assertEqual(replay.to_dict(), profile.to_dict())
        np.testing.assert_array_equal(replay.sample(np.arange(96)), profile.sample(np.arange(96)))
        with self.assertRaises(ValueError):
            replay.speeds_m_s[0] = 123
        config = WindProfileConfig(mode="speed", seed=9)
        self.assertEqual(config, WindProfileConfig.from_dict(json.loads(json.dumps(config.to_dict()))))

    def test_short_episode_and_input_validation(self):
        scenario = make_wind_scenario(WindProfileConfig(mode="speed"), 1, 8, 270, 0)
        np.testing.assert_array_equal(scenario.times_s, [0, 1])
        for kwargs in ({"mode": "bad"}, {"speed_min": 0}, {"speed_min": 12},
                       {"direction_half_width": 181}, {"knot_interval_s": 0}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                WindProfileConfig(**kwargs)
        for args in (([0, 0], [8, 8], [270, 270]),
                     ([1, 2], [8, 8], [270, 270]),
                     ([0, 1], [0, 8], [270, 270]),
                     ([0, 1], [8, 8, 8], [270, 270])):
            with self.subTest(args=args), self.assertRaises(ValueError):
                WindScenario(*args)
        with self.assertRaises(ValueError):
            self.generate(episode_id=-1)


if __name__ == "__main__":
    unittest.main()
