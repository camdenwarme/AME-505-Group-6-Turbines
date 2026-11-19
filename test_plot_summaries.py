"""Check numeric evaluations and legacy plots without rendering pixels."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import numpy as np

from plot_evaluation_turbines import plot_evaluation_turbines
from plot_rewards import plot_reward_summary
from plot_training_energy import plot_training_energy


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _capture_plot(call):
    captured = {}

    def savefig(figure, *_args, **_kwargs):
        axis = figure.axes[0]
        captured["lines"] = {line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
                             for line in axis.get_lines()}
        captured["xlim"] = axis.get_xlim()
        captured["ylim"] = axis.get_ylim()

    with patch.object(Figure, "savefig", savefig):
        call()
    return captured


class PlotSummaryTests(unittest.TestCase):
    def test_numeric_csv_takes_precedence_and_legacy_fills_gaps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_csv(root / "training_metrics.csv", ["episode", "episode_energy_mwh", "episode_return"],
                       [{"episode": 1, "episode_energy_mwh": 8, "episode_return": 80}])
            fields = ["episode", "policy", "farm_energy_mwh", "team_return"]
            _write_csv(root / "evaluation_metrics.csv", fields, [
                dict(episode=0, policy="aligned_baseline", farm_energy_mwh=10, team_return=100),
                dict(episode=0, policy="greedy", farm_energy_mwh=9, team_return=90),
                dict(episode=1, policy="greedy", farm_energy_mwh=9.5, team_return=95),
            ])
            # Conflicting duplicate at episode 1 must never create a second
            # point or override the numeric evaluation; episode 2 is legacy.
            np.savez(root / "evaluation_energy_episode1.npz", energy_j=[0, 99 * 3.6e9])
            np.savez(root / "evaluation_rewards_episode1.npz", reward=[990])
            np.savez(root / "evaluation_energy_episode2.npz", energy_j=[0, 8.2 * 3.6e9])
            np.savez(root / "evaluation_rewards_episode2.npz", reward=[82])

            energy = _capture_plot(lambda: plot_training_energy(root))
            reward = _capture_plot(lambda: plot_reward_summary(root))
            for result, expected in ((energy, [9, 9.5, 8.2]), (reward, [90, 95, 82])):
                xs, ys = result["lines"]["Evaluation (greedy policy)"]
                np.testing.assert_allclose(xs, [0, 1, 2])
                np.testing.assert_allclose(ys, expected)
                self.assertIn("Training (exploration)", result["lines"])
                self.assertIn("Aligned baseline", result["lines"])
            self.assertLessEqual(energy["ylim"][0], 8)
            self.assertGreaterEqual(energy["ylim"][1], 10)
            self.assertGreaterEqual(reward["ylim"][1], 100)

    def test_single_training_point_and_csv_only_greedy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_csv(root / "training_metrics.csv", ["episode", "episode_energy_mwh", "episode_return"],
                       [{"episode": 1, "episode_energy_mwh": 8, "episode_return": 80}])
            single = _capture_plot(lambda: plot_training_energy(root))
            self.assertLess(single["xlim"][0], 1)
            self.assertGreater(single["xlim"][1], 1)
            _write_csv(root / "evaluation_metrics.csv", ["episode", "policy", "farm_energy_mwh", "team_return"],
                       [dict(episode=0, policy="greedy", farm_energy_mwh=9, team_return=90)])
            energy = _capture_plot(lambda: plot_training_energy(root))
            reward = _capture_plot(lambda: plot_reward_summary(root))
            self.assertEqual(energy["lines"]["Evaluation (greedy policy)"], ([0], [9.0]))
            self.assertEqual(reward["lines"]["Evaluation (greedy policy)"], ([0], [90.0]))
            self.assertNotIn("Aligned baseline", energy["lines"])

    def test_evaluation_turbines_uses_baseline_and_mean_yaw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fields = ["episode", "policy", "farm_energy_mwh", "turbine0_energy_mwh", "turbine1_energy_mwh"]
            _write_csv(root / "evaluation_metrics.csv", fields, [
                dict(episode=0, policy="aligned_baseline", farm_energy_mwh=1.2,
                     turbine0_energy_mwh=1.0, turbine1_energy_mwh=0.2),
                dict(episode=0, policy="greedy", farm_energy_mwh=1.2,
                     turbine0_energy_mwh=1.0, turbine1_energy_mwh=0.2),
                dict(episode=10, policy="greedy", farm_energy_mwh=1.3,
                     turbine0_energy_mwh=0.95, turbine1_energy_mwh=0.35),
            ])
            for episode, yaw in ((0, 0.0), (10, 15.0)):
                _write_csv(root / f"evaluation_episode{episode}_steps.csv",
                           ["tick", "turbine0_yaw_deg", "turbine1_yaw_deg"],
                           [dict(tick=t, turbine0_yaw_deg=yaw, turbine1_yaw_deg=-1.0) for t in range(3)])
            captured = {}

            def savefig(figure, *_args, **_kwargs):
                captured["axes"] = [{line.get_label(): list(line.get_ydata()) for line in axis.get_lines()}
                                    for axis in figure.axes]

            with patch.object(Figure, "savefig", savefig):
                plot_evaluation_turbines(root)
            energy, delta, yaw = captured["axes"]
            np.testing.assert_allclose(energy["Greedy policy (farm total)"], [1.2, 1.3])
            self.assertIn("Aligned baseline", energy)
            np.testing.assert_allclose(delta["Turbine 1"], [0.0, -0.05], atol=1e-12)
            np.testing.assert_allclose(delta["Turbine 2"], [0.0, 0.15], atol=1e-12)
            np.testing.assert_allclose(delta["Farm total"], [0.0, 0.1], atol=1e-12)
            np.testing.assert_allclose(yaw["Turbine 1"], [0.0, 15.0])
            np.testing.assert_allclose(yaw["Turbine 2"], [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
