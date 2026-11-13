"""Fast training/serialization checks without running the wind simulator."""

from pathlib import Path
import csv
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gym import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "light_mappo"))
from config import get_config
from envs.env_wrappers import DummyVecEnv
from shared_critic_runner import SharedCriticRunner


class _Farm:
    observation_space = [spaces.Box(-np.inf, np.inf, (3,), np.float32) for _ in range(2)]
    share_observation_space = [spaces.Box(-np.inf, np.inf, (6,), np.float32) for _ in range(2)]
    action_space = [spaces.Box(-6, 6, (1,), np.float32) for _ in range(2)]

    def reset(self):
        self.t = 0
        return np.array([[8, 270, 0], [6, 270, 0]], dtype=np.float32)

    def step(self, actions):
        self.t += 1
        obs = np.array([[8, 270, -self.t], [6, 270, self.t]], dtype=np.float32)
        # Deliberately independent of the normalized reward, so tests catch
        # energy reconstructed incorrectly from rewards or double-counted agents.
        infos = [{"episode_energy_j": self.t * 3.6e6} for _ in range(2)]
        return obs, np.full((2, 1), 0.5, np.float32), np.full(2, self.t == 4), infos

    def close(self):
        pass


class SharedCriticTests(unittest.TestCase):
    def test_evaluation_schedule_does_not_change_training(self):
        torch.set_num_threads(1)
        for total_episodes in (10, 12):
            with self.subTest(total_episodes=total_episodes), tempfile.TemporaryDirectory() as directory:
                outcomes = []
                for evaluate in (False, True):
                    torch.manual_seed(11)
                    np.random.seed(11)
                    random.seed(11)
                    args = get_config().parse_args([])
                    args.episode_length, args.n_rollout_threads = 4, 1
                    args.num_env_steps, args.ppo_epoch = 4 * total_episodes, 1
                    args.log_interval = args.save_interval = 100
                    args.scenario_name = "test"
                    args.use_feature_normalization = False
                    evaluated = []

                    def callback(episode, policies):
                        evaluated.append(episode)
                        # The row must already be visible before evaluation starts.
                        with (config["run_dir"] / "training_metrics.csv").open(newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual(len(rows), episode)
                        # Deliberately consume all training RNGs to verify isolation.
                        torch.rand(7)
                        np.random.random(7)
                        random.random()

                    envs = DummyVecEnv([_Farm])
                    config = {"all_args": args, "envs": envs, "eval_envs": None,
                              "num_agents": 2, "device": torch.device("cpu"),
                              "run_dir": Path(directory) / str(evaluate)}
                    if evaluate:
                        config["episode_callback"] = callback
                    runner = SharedCriticRunner(config)
                    try:
                        with patch.object(envs.envs[0], "step", wraps=envs.envs[0].step) as step:
                            runner.run()
                            self.assertEqual(step.call_count, total_episodes * 4)
                        expected = [0, 10] + ([12] if total_episodes == 12 else [])
                        self.assertEqual(evaluated, expected if evaluate else [])
                        with runner.metrics_path.open(newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual(len(rows), total_episodes)
                        for i, row in enumerate(rows, 1):
                            self.assertEqual(int(row["episode"]), i)
                            self.assertEqual(int(row["environment_steps"]), i * 4)
                            self.assertEqual(float(row["episode_energy_j"]), 4 * 3.6e6)
                            self.assertAlmostEqual(float(row["episode_energy_mwh"]), 0.004)
                            self.assertAlmostEqual(float(row["cumulative_energy_mwh"]), i * 0.004)
                            self.assertEqual(float(row["episode_return"]), 2.0)
                            self.assertEqual(float(row["episode_end_reward"]), 0.5)
                            for key in ("actor0_policy_loss", "actor1_policy_loss", "critic_value_loss"):
                                self.assertTrue(np.isfinite(float(row[key])))
                        modules = [p.actor for p in runner.policy] + [runner.critic,
                                   runner.critic_trainer.value_normalizer]
                        outcomes.append([{k: v.clone() for k, v in m.state_dict().items()}
                                         for m in modules])
                    finally:
                        runner.writter.close()
                        envs.close()
                for before, after in zip(*outcomes):
                    for key in before:
                        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_single_critic_updates_and_checkpoint_roundtrip(self):
        torch.set_num_threads(1)
        for mode in ("feedforward", "recurrent", "naive_recurrent"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(7)
                args = get_config().parse_args([])
                args.episode_length, args.n_rollout_threads = 4, 1
                args.ppo_epoch = 2
                args.num_mini_batch = 2 if mode == "feedforward" else 1
                args.use_recurrent_policy = mode == "recurrent"
                args.use_naive_recurrent_policy = mode == "naive_recurrent"
                args.data_chunk_length = 2
                args.use_feature_normalization = False
                envs = DummyVecEnv([_Farm])
                config = {"all_args": args, "envs": envs, "eval_envs": None,
                          "num_agents": 2, "device": torch.device("cpu"),
                          "run_dir": Path(directory)}
                runner = SharedCriticRunner(config)
                try:
                    self.assertIsNot(runner.policy[0].actor, runner.policy[1].actor)
                    self.assertIs(runner.policy[0].critic, runner.policy[1].critic)
                    self.assertIs(runner.policy[0].critic_optimizer, runner.policy[1].critic_optimizer)
                    self.assertIs(runner.trainer[0].value_normalizer, runner.trainer[1].value_normalizer)
                    runner.warmup()
                    for step in range(4):
                        values, actions, log_probs, states, critic_states, commands = runner.collect(step)
                        obs, rewards, dones, infos = envs.step(commands)
                        runner.insert((obs, rewards, dones, infos, values, actions,
                                       log_probs, states, critic_states))
                    runner.compute()
                    np.testing.assert_allclose(runner.buffer[0].value_preds, runner.buffer[1].value_preds)
                    np.testing.assert_allclose(runner.buffer[0].returns, runner.buffer[1].returns)
                    actors_before = [[p.detach().clone() for p in policy.actor.parameters()]
                                     for policy in runner.policy]
                    critic_before = [p.detach().clone() for p in runner.critic.parameters()]
                    with patch.object(runner.critic_optimizer, "step",
                                      wraps=runner.critic_optimizer.step) as critic_step, \
                         patch.object(runner.critic_trainer.value_normalizer, "update",
                                      wraps=runner.critic_trainer.value_normalizer.update) as normalize:
                        train_infos = runner.train()
                        # One critic/normalizer update per epoch/minibatch,
                        # not per turbine. Actor training changes neither.
                        expected_updates = args.ppo_epoch * args.num_mini_batch
                        self.assertEqual(critic_step.call_count, expected_updates)
                        self.assertEqual(normalize.call_count, expected_updates)
                    for policy, before in zip(runner.policy, actors_before):
                        self.assertTrue(any(not torch.equal(a, b)
                                            for a, b in zip(before, policy.actor.parameters())))
                    self.assertTrue(any(not torch.equal(a, b)
                                        for a, b in zip(critic_before, runner.critic.parameters())))
                    runner.log_train(train_infos, 4)
                    runner.save()
                    models = Path(runner.save_dir)
                    self.assertEqual({p.name for p in models.iterdir()},
                                     {"actor_agent0.pt", "actor_agent1.pt", "critic.pt", "value_normalizer.pt"})
                    saved = {k: v.clone() for k, v in runner.critic.state_dict().items()}
                    saved_normalizer = {k: v.clone() for k, v in runner.critic_trainer.value_normalizer.state_dict().items()}
                    with torch.no_grad():
                        for parameter in runner.critic.parameters():
                            parameter.zero_()
                    runner.critic_trainer.value_normalizer.reset_parameters()
                    runner.model_dir = models
                    runner.restore()
                    for key, value in runner.critic.state_dict().items():
                        torch.testing.assert_close(value, saved[key])
                    for key, value in runner.critic_trainer.value_normalizer.state_dict().items():
                        torch.testing.assert_close(value, saved_normalizer[key])
                    # Constructor restoration must also load only one critic.
                    args.model_dir = models
                    restored = SharedCriticRunner(config)
                    try:
                        self.assertIs(restored.policy[0].critic, restored.policy[1].critic)
                        for key, value in restored.critic.state_dict().items():
                            torch.testing.assert_close(value, saved[key])
                    finally:
                        restored.writter.close()
                finally:
                    runner.writter.close()
                    envs.close()


if __name__ == "__main__":
    unittest.main()
