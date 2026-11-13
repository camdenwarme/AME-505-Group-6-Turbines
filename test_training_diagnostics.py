"""Verify standalone plots on incomplete and nonfinite run records."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from plot_training_diagnostics import make_diagnostics


def _csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TrainingDiagnosticTests(unittest.TestCase):
    def test_reports_finite_metrics_and_pooled_yaw(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = ["episode", "critic_return_rmse", "actor0_individual_value_rmse",
                        "actor1_individual_value_rmse", "actor0_ppo_clip_fraction", "actor1_ppo_clip_fraction"]
            _csv(root / "training_metrics.csv", training, [
                dict(episode=1, critic_return_rmse=5, actor0_individual_value_rmse=2,
                     actor1_individual_value_rmse="nan", actor0_ppo_clip_fraction=0.1,
                     actor1_ppo_clip_fraction=0.2),
                dict(episode=2, critic_return_rmse="nan", actor0_individual_value_rmse=4,
                     actor1_individual_value_rmse=3, actor0_ppo_clip_fraction="", actor1_ppo_clip_fraction=0.3),
            ])
            _csv(root / "evaluation_metrics.csv", ["episode", "policy", "farm_energy_mwh"], [
                dict(episode=0, policy="aligned_baseline", farm_energy_mwh=1.1),
                dict(episode=0, policy="greedy", farm_energy_mwh=1.2),
                dict(episode=2, policy="greedy", farm_energy_mwh=0.9),
                dict(episode=3, policy="greedy", farm_energy_mwh="nan"),
            ])
            _csv(root / "rollout_steps.csv", ["episode", "actor0_yaw_deg", "actor1_yaw_deg"], [
                dict(episode=1, actor0_yaw_deg=0, actor1_yaw_deg=90),
                dict(episode=1, actor0_yaw_deg=-90, actor1_yaw_deg=0),
                dict(episode=2, actor0_yaw_deg=180, actor1_yaw_deg="nan"),
                dict(episode=2, actor0_yaw_deg="nan", actor1_yaw_deg=-100),
            ])
            plot, record = make_diagnostics(root)
            self.assertGreater(plot.stat().st_size, 10000)
            data = json.loads(record.read_text())
            self.assertEqual(data["team_value_rmse"]["count"], 1)
            self.assertEqual(data["individual_value_rmse"][0]["mean"], 3)
            self.assertEqual(data["ppo_clip_fraction"][0]["count"], 1)
            for turbine in data["yaw_abs_ge_90"]:
                self.assertEqual(turbine["valid_samples"], 3)
                self.assertEqual(turbine["samples_abs_yaw_ge_90"], 2)
                self.assertAlmostEqual(turbine["pooled_fraction"], 2 / 3)
            self.assertEqual(data["greedy_farm_energy_mwh"]["count"], 2)
            self.assertEqual(data["greedy_farm_energy_mwh"]["last"], 0.9)
            self.assertEqual(data["aligned_baseline_energy_mwh"], 1.1)

    def test_missing_records_produce_empty_valid_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            plot, record = make_diagnostics(Path(temp))
            self.assertGreater(plot.stat().st_size, 10000)
            data = json.loads(record.read_text())
            self.assertEqual(data["training_metric_rows"], 0)
            self.assertEqual(data["greedy_farm_energy_mwh"]["count"], 0)
            self.assertIsNone(data["yaw_abs_ge_90"][0]["pooled_fraction"])


if __name__ == "__main__":
    unittest.main()
