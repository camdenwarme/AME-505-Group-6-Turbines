"""Check numeric evaluations and legacy plots without rendering pixels."""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure
import numpy as np

from plot_rewards import plot_reward_summary
from plot_training_diagnostics import make_diagnostics
from plot_training_energy import plot_training_energy
from plot_training_loss import plot_training_loss


def _write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _capture_plot(call, axis_index=0):
    captured = {}

    def savefig(figure, *_args, **_kwargs):
        axis = figure.axes[axis_index]
        captured["lines"] = {line.get_label(): (list(line.get_xdata()), list(line.get_ydata()))
                             for line in axis.get_lines()}
        captured["styles"] = {line.get_label(): (line.get_color(), line.get_linestyle())
                              for line in axis.get_lines()}
        captured["xlim"] = axis.get_xlim()
        captured["ylim"] = axis.get_ylim()
        captured["ylabel"] = axis.get_ylabel()
        captured["title"] = axis.get_title(loc="left")

    with patch.object(Figure, "savefig", savefig):
        call()
    return captured


class PlotSummaryTests(unittest.TestCase):
    def test_turbine_energy_replaces_actor_losses_and_uses_matching_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_csv(root / "training_metrics.csv", ["episode", "critic_value_loss", "actor0_policy_loss"],
                       [dict(episode=1, critic_value_loss=0.25, actor0_policy_loss=-0.01)])
            fields = ["episode", "policy", "scenario_id", "turbine0_energy_mwh", "turbine2_energy_mwh"]
            rows = [dict(episode=3, policy="greedy", scenario_id="test", turbine0_energy_mwh=6,
                         turbine2_energy_mwh=2),
                    dict(episode=0, policy="greedy", scenario_id="test", turbine0_energy_mwh=3,
                         turbine2_energy_mwh=3),
                    dict(episode=0, policy="wind_aligned", scenario_id="test", turbine0_energy_mwh=4,
                         turbine2_energy_mwh=1)]
            _write_csv(root / "evaluation_metrics.csv", fields, rows)
            first = _capture_plot(lambda: plot_training_loss(root))
            third = _capture_plot(lambda: plot_training_loss(root), axis_index=1)
            critic = _capture_plot(lambda: plot_training_loss(root), axis_index=2)
            self.assertEqual(first["title"], "Turbine 1 greedy evaluation")
            self.assertEqual(third["title"], "Turbine 3 greedy evaluation")
            self.assertEqual(first["ylabel"], "Δ vs. wind-aligned (MWh)")
            self.assertEqual(list(first["lines"].values()), [([0, 3], [-1, 2]), ([0, 1], [0, 0])])
            self.assertEqual(list(third["lines"].values())[0], ([0, 3], [2, 1]))
            self.assertEqual(critic["title"], "Shared centralized critic")
            self.assertEqual(list(critic["lines"].values()), [([1], [0.25])])

            rows[-1]["scenario_id"] = "other-wind"
            _write_csv(root / "evaluation_metrics.csv", fields, rows)
            absolute = _capture_plot(lambda: plot_training_loss(root))
            self.assertEqual(absolute["ylabel"], "Energy (MWh)")
            self.assertEqual(list(absolute["lines"].values()), [([0, 3], [3, 6])])

    def test_wind_aligned_energy_is_a_reference_on_both_plots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _write_csv(root / "training_metrics.csv", ["episode", "episode_energy_mwh"],
                       [dict(episode=1, episode_energy_mwh=8)])
            _write_csv(root / "evaluation_metrics.csv", ["episode", "policy", "farm_energy_mwh"], [
                dict(episode=0, policy="greedy", farm_energy_mwh=9),
                dict(episode=0, policy="wind_aligned", farm_energy_mwh=12),
                dict(episode=0, policy="aligned_baseline", farm_energy_mwh=10),
                dict(episode=1, policy="greedy", farm_energy_mwh=9.5),
            ])
            energy = _capture_plot(lambda: plot_training_energy(root))
            diagnostics = _capture_plot(lambda: make_diagnostics(root), axis_index=3)
            for result, greedy_label in ((energy, "Evaluation (greedy policy)"),
                                         (diagnostics, "Greedy evaluation")):
                self.assertEqual(result["lines"][greedy_label], ([0, 1], [9, 9.5]))
                self.assertEqual(result["lines"]["Wind-aligned benchmark"], ([0, 1], [12, 12]))
                self.assertEqual(result["styles"]["Wind-aligned benchmark"], ("#777777", "--"))
                self.assertIn("Aligned baseline", result["lines"])
                self.assertGreaterEqual(result["ylim"][1], 12)

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
                dict(episode=3, policy="wind_aligned", farm_energy_mwh=11, team_return=500),
            ])
            # Conflicting duplicate at episode 1 must never create a second
            # point or override the numeric evaluation; episode 2 is legacy.
            np.savez(root / "evaluation_energy_episode1.npz", energy_j=[0, 99 * 3.6e9])
            np.savez(root / "evaluation_rewards_episode1.npz", reward=[990])
            np.savez(root / "evaluation_energy_episode2.npz", energy_j=[0, 8.2 * 3.6e9])
            np.savez(root / "evaluation_rewards_episode2.npz", reward=[82])
            np.savez(root / "evaluation_rewards_episode3.npz", reward=[500])

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
            self.assertIn("Wind-aligned benchmark", energy["lines"])
            self.assertNotIn("Evaluation (wind-aligned policy)", reward["lines"])
            self.assertLess(reward["ylim"][1], 500)

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


if __name__ == "__main__":
    unittest.main()
