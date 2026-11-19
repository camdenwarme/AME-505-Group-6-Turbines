"""Exercise real spawned processes and policy-update barriers without FLORIS."""

import csv
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from parallel_env import ProcessVecEnv


@dataclass
class FarmFactory:
    rank: int
    barrier: object = None
    fail: bool = False

    def __call__(self):
        return Farm(self.rank, self.barrier, self.fail)


class Farm:
    def __init__(self, rank, barrier, fail):
        import numpy as np
        from gym import spaces
        self.rank, self.barrier, self.fail = rank, barrier, fail
        self.dt, self.remaining_time_index = 4.0, 6
        self.layout_d = np.array([[0, 0, 0], [0, 5, 0]])
        self.total_rated_power_w = 20e6
        self.observation_space = [spaces.Box(-np.inf, np.inf, (3,), np.float32) for _ in range(2)]
        self.share_observation_space = [spaces.Box(-np.inf, np.inf, (7,), np.float32) for _ in range(2)]
        self.action_space = [spaces.Box(-6, 6, (1,), np.float32) for _ in range(2)]
        self.episode = -1

    def reset(self):
        from wind_profiles import WindScenario
        self.t = 0
        self.episode += 1
        self.wind_scenario = WindScenario([0, 16], [8, 8], [270, 270],
            episode_id=self.episode, environment_id=self.rank, episode_seed=100 * self.rank + self.episode)
        return self.obs()

    def obs(self):
        import numpy as np
        return np.array([[8, 270, -self.t], [6, 270, self.t]], dtype=np.float32)

    def get_centralized_obs(self):
        import numpy as np
        return np.r_[self.obs().ravel(), 1 - self.t / 4].astype(np.float32)

    def get_individual_baseline_values(self, gamma):
        import numpy as np
        return np.array([.2, .8], np.float32) * sum(gamma ** n for n in range(4 - self.t))

    def step(self, actions):
        import numpy as np
        if self.fail:
            raise ValueError("deliberate worker failure")
        if self.barrier is not None:
            # A sequential dispatcher would deadlock here instead of returning.
            self.barrier.wait(timeout=15)
        self.t += 1
        infos = [{"episode_energy_j": self.t * (self.rank + 1) * 3.6e6,
                  "team_reward": .5, "farm_power_w": 10,
                  "turbine_power_w": 5, "yaw_rate_deg_s": float(actions[i][0]),
                  "scenario_id": self.wind_scenario.scenario_id,
                  "episode_seed": self.wind_scenario.episode_seed, "wind_mode": "off"}
                 for i in range(2)]
        if self.t == 4:
            infos[0]["wind_scenario"] = self.wind_scenario.to_dict()
        return self.obs(), np.array([[.2], [.8]], np.float32), np.full(2, self.t == 4), infos

    def close(self):
        pass


class ParallelTrainingTests(unittest.TestCase):
    def test_distinct_processes_run_concurrently_and_wait_for_explicit_reset(self):
        import numpy as np
        barrier = mp.get_context("spawn").Barrier(2)
        envs = ProcessVecEnv([FarmFactory(i, barrier) for i in range(2)], timeout=30)
        try:
            self.assertEqual(len({env.pid for env in envs.envs}), 2)
            self.assertNotIn(os.getpid(), {env.pid for env in envs.envs})
            envs.reset()
            for _ in range(4):
                _, _, dones, _ = envs.step(np.zeros((2, 2, 1)))
            self.assertTrue(dones.all())
            self.assertTrue(all(env.central[-1] == 0 for env in envs.envs))
            self.assertTrue(all(env.wind_scenario.to_dict()["episode_id"] == 0 for env in envs.envs))
            envs.reset()
            self.assertTrue(all(env.wind_scenario.to_dict()["episode_id"] == 1 for env in envs.envs))
        finally:
            envs.close()
        self.assertTrue(all(not process.is_alive() for process in envs.processes))

    def test_worker_failure_is_reported_and_processes_close(self):
        envs = ProcessVecEnv([FarmFactory(0, fail=True), FarmFactory(1)], timeout=30)
        try:
            envs.reset()
            with self.assertRaisesRegex(RuntimeError, "deliberate worker failure"):
                envs.step([[[0], [0]], [[0], [0]]])
        finally:
            envs.close()
        self.assertTrue(all(not process.is_alive() for process in envs.processes))

    def test_updates_follow_complete_rounds_independently_of_training_passes(self):
        import numpy as np
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "light_mappo"))
        from config import get_config
        from shared_critic_runner import SharedCriticRunner
        torch.set_num_threads(1)
        for passes in (1, 3):
            with self.subTest(training_passes=passes), tempfile.TemporaryDirectory() as directory:
                args = get_config().parse_args([])
                args.use_valuenorm = args.use_feature_normalization = False
                args.episode_length, args.num_env_steps, args.n_rollout_threads = 4, 24, 2
                args.ppo_epoch, args.num_mini_batch = passes, 1
                args.advantage_weight_schedule, args.advantage_decay_beta = "exponential", .1
                args.scenario_name = "parallel-test"
                envs = ProcessVecEnv([FarmFactory(i) for i in range(2)])
                runner = None
                try:
                    runner = SharedCriticRunner(dict(all_args=args, envs=envs, eval_envs=None,
                        num_agents=2, device=torch.device("cpu"), run_dir=Path(directory)))
                    collect, train = runner.collect, runner.train
                    snapshots, calls = [], []

                    def checked_collect(step):
                        self.assertAlmostEqual(runner.individual_advantage_weight,
                                               .9 ** runner._completed_rounds)
                        parameters = [p.detach().clone() for policy in runner.policy for p in policy.actor.parameters()]
                        if step == 0:
                            snapshots[:] = parameters
                        else:
                            self.assertTrue(all(torch.equal(a, b) for a, b in zip(parameters, snapshots)))
                        return collect(step)

                    def checked_train():
                        self.assertTrue(all(env.central[-1] == 0 for env in envs.envs))
                        calls.append([env.wind_scenario.to_dict()["episode_id"] for env in envs.envs])
                        expected_weight = runner.team_reward_weight
                        # Every minibatch and PPO pass in this round uses one weight.
                        def check_weight(sample, update_actor=True):
                            self.assertAlmostEqual(runner.team_reward_weight, expected_weight)
                            return original_update(sample, update_actor)
                        original_update = runner.trainer[0].ppo_update
                        with patch.object(runner.trainer[0], "ppo_update", side_effect=check_weight):
                            return train()

                    with patch.object(runner, "collect", side_effect=checked_collect), \
                         patch.object(runner, "train", side_effect=checked_train):
                        runner.run()
                    self.assertEqual(calls, [[0, 0], [1, 1], [2, 2]])
                    self.assertEqual(runner._completed_episodes, 6)
                    with runner.metrics_path.open() as stream:
                        metrics = list(csv.DictReader(stream))
                    self.assertEqual([int(row["episode"]) for row in metrics], [1, 2, 3, 4, 5, 6])
                    self.assertEqual([int(row["update_round"]) for row in metrics], [1, 1, 2, 2, 3, 3])
                    np.testing.assert_allclose([float(row["individual_advantage_weight"]) for row in metrics],
                                               [1, 1, .9, .9, .81, .81])
                    np.testing.assert_allclose([float(row["farm_advantage_weight"]) for row in metrics],
                                               [0, 0, .1, .1, .19, .19])
                    np.testing.assert_allclose([float(row["episode_energy_mwh"]) for row in metrics],
                                               [.004, .008] * 3)
                    self.assertAlmostEqual(float(metrics[-1]["cumulative_energy_mwh"]), .036)
                    with runner.rollout_path.open() as stream:
                        steps = list(csv.DictReader(stream))
                    self.assertEqual(len(steps), 24)
                    self.assertEqual({int(row["episode"]) for row in steps}, {1, 2, 3, 4, 5, 6})
                    for round_number in (1, 2, 3):
                        rows = [row for row in steps if int(row["update_round"]) == round_number]
                        individual = np.array([float(row["actor0_individual_gae_advantage"]) for row in rows], np.float32)
                        farm = np.array([float(row["team_gae_advantage"]) for row in rows], np.float32)
                        weight = .9 ** (round_number - 1)
                        expected = (weight * (individual - individual.mean()) / (individual.std() + 1e-5)
                                    + (1 - weight) * (farm - farm.mean()) / (farm.std() + 1e-5))
                        np.testing.assert_allclose([float(row["actor0_mixed_advantage"]) for row in rows],
                                                   expected, atol=1e-6)
                finally:
                    envs.close()
                    if runner is not None:
                        runner.writter.close()

    def test_decay_endpoints_and_constant_schedule(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "light_mappo"))
        from shared_critic_runner import SharedCriticRunner
        runner = object.__new__(SharedCriticRunner)
        runner.advantage_weight_schedule = "exponential"
        for beta, expected in ((0, [1, 1, 1]), (1, [1, 0, 0])):
            runner.advantage_decay_beta = beta
            for round_index, weight in enumerate(expected):
                runner._completed_rounds = round_index
                runner._set_advantage_weights()
                self.assertEqual(runner.individual_advantage_weight, weight)
                self.assertEqual(runner.team_reward_weight, 1 - weight)
        runner.advantage_weight_schedule = "constant"
        runner._constant_team_reward_weight = .5
        runner._completed_rounds = 24
        runner._set_advantage_weights()
        self.assertEqual(runner.team_reward_weight, .5)
        self.assertEqual(runner.individual_advantage_weight, .5)


if __name__ == "__main__":
    unittest.main()
