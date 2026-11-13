"""Fast training/serialization checks without running the wind simulator."""

from pathlib import Path
import csv
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gym import spaces

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "light_mappo"))
from config import get_config
from envs.env_wrappers import DummyVecEnv
from shared_critic_runner import (SharedCriticRunner, _critic_return_diagnostics,
                                  _discounted_horizons, _individual_gae, _individual_values)


class _Farm:
    dt = 4.0
    observation_space = [spaces.Box(-np.inf, np.inf, (3,), np.float32) for _ in range(2)]
    share_observation_space = [spaces.Box(-np.inf, np.inf, (7,), np.float32) for _ in range(2)]
    action_space = [spaces.Box(-6, 6, (1,), np.float32) for _ in range(2)]

    def reset(self):
        self.t = 0
        return np.array([[8, 270, 0], [6, 270, 0]], dtype=np.float32)

    def get_centralized_obs(self):
        return np.array([8, 270, -self.t, 6, 270, self.t, 1 - self.t / 4], dtype=np.float32)

    def get_individual_baseline_values(self, gamma):
        return np.array([0.2, 0.8], np.float32) * sum(gamma ** n for n in range(4 - self.t))

    def step(self, actions):
        self.t += 1
        obs = np.array([[8, 270, -self.t], [6, 270, self.t]], dtype=np.float32)
        # Deliberately independent of the normalized reward, so tests catch
        # energy reconstructed incorrectly from rewards or double-counted agents.
        infos = [{"episode_energy_j": self.t * 3.6e6, "team_reward": 0.5,
                  "farm_power_w": 10, "turbine_power_w": 4 if i == 0 else 6,
                  "yaw_rate_deg_s": float(actions[i][0])} for i in range(2)]
        rewards = np.array([[0.2], [0.8]], np.float32)
        return obs, rewards, np.full(2, self.t == 4), infos

    def close(self):
        pass


class SharedCriticTests(unittest.TestCase):
    def test_individual_value_residual_respects_remaining_horizon(self):
        horizons = _discounted_horizons(3, 0.5)
        np.testing.assert_allclose(horizons.reshape(-1), [1.75, 1.5, 1, 0])
        self.assertEqual(_discounted_horizons(200, 1.0)[199, 0, 0], 1)
        self.assertEqual(_discounted_horizons(200, 1.0)[200, 0, 0], 0)
        reference = np.array([8, 5, 1, 8], dtype=np.float32).reshape(4, 1, 1)
        residual = np.full_like(reference, 2)
        # The initial mask can be zero after reset; only subsequent done slots
        # must discard the automatically reset next-episode state.
        masks = np.array([0, 1, 1, 0], dtype=np.float32).reshape(4, 1, 1)
        values = _individual_values(reference, residual, horizons, masks)
        np.testing.assert_allclose(values.reshape(-1), [11.5, 8, 3, 0])

    def test_individual_gae_terminal_mask_and_reference(self):
        rewards = np.array([1, 2, 3], dtype=np.float32).reshape(3, 1, 1)
        # The final value is intentionally large: done must suppress it.
        values = np.array([2, 1, 0.5, 900], dtype=np.float32).reshape(4, 1, 1)
        masks = np.array([1, 1, 1, 0], dtype=np.float32).reshape(4, 1, 1)
        actual, targets = _individual_gae(rewards, values, masks, .9, .8)
        expected = np.array([2.24, 3.25, 2.5], dtype=np.float32)
        np.testing.assert_allclose(actual.reshape(-1), expected, atol=1e-5)
        np.testing.assert_allclose(targets.reshape(-1), expected + values[:-1].reshape(-1), atol=1e-5)

    def test_critic_diagnostics_use_raw_full_returns(self):
        buffer = SimpleNamespace(
            rewards=np.array([1, 2, 3], dtype=np.float32).reshape(3, 1, 1),
            value_preds=np.array([1, 1.5, 2, 0], dtype=np.float32).reshape(4, 1, 1),
            masks=np.array([1, 1, 1, 0], dtype=np.float32).reshape(4, 1, 1),
            gamma=0.5)
        diagnostic = _critic_return_diagnostics(buffer, None)
        actual_returns = np.array([2.75, 3.5, 3])
        errors = np.array([1, 1.5, 2]) - actual_returns
        self.assertAlmostEqual(diagnostic["return_rmse"], np.sqrt(np.mean(errors ** 2)))
        self.assertAlmostEqual(diagnostic["explained_variance"],
                               1 - np.var(errors) / np.var(actual_returns))
        buffer.rewards.fill(0)
        self.assertTrue(np.isnan(_critic_return_diagnostics(buffer, None)["explained_variance"]))

    def test_individual_returns_are_distinct_and_do_not_change_team_targets(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            args = get_config().parse_args([])
            args.use_valuenorm = False
            args.episode_length, args.n_rollout_threads = 4, 1
            args.ppo_epoch, args.num_mini_batch = 1, 1
            args.gamma, args.team_reward_weight = 0.9, 0.0
            args.gae_lambda = 1.0
            envs = DummyVecEnv([_Farm])
            runner = SharedCriticRunner({"all_args": args, "envs": envs, "eval_envs": None,
                                         "num_agents": 2, "device": torch.device("cpu"),
                                         "run_dir": Path(directory)})
            try:
                runner.warmup()
                self.assertEqual(runner.buffer[0].share_obs[0, 0, -1], 1)
                for step in range(4):
                    values, actions, log_probs, states, critic_states, commands = runner.collect(step)
                    obs, rewards, dones, infos = envs.step(commands)
                    runner.insert((obs, rewards, dones, infos, values, actions,
                                   log_probs, states, critic_states))
                self.assertEqual(runner.buffer[0].masks[-1, 0, 0], 0)
                # Auto-reset supplies the next episode's first critic state.
                self.assertEqual(runner.buffer[0].share_obs[-1, 0, -1], 1)
                runner.individual_rewards[:, 0, 0, 0] = [1, 1, 0, 0]
                runner.individual_rewards[:, 0, 1, 0] = [0, 0, 1, 1]
                runner.compute()
                farm_targets = runner.buffer[0].returns.copy()
                captured = []

                def capture(buffer):
                    baseline = buffer.value_preds[:-1]
                    captured.append((buffer.returns[:-1] - baseline).copy().reshape(-1))
                    return {"value_loss": 0, "critic_grad_norm": 0, "policy_loss": 0}

                with patch.object(runner.trainer[0], "train", side_effect=capture), \
                     patch.object(runner.trainer[1], "train", side_effect=capture):
                    runner.train()
                for i, (actual, returns) in enumerate(zip(captured, ([1.9, 1, 0, 0], [1.539, 1.71, 1.9, 1]))):
                    reference = runner.individual_reference[:4, 0, i, 0]
                    raw = np.asarray(returns) - reference
                    raw = np.asarray(raw)
                    expected = (raw - raw.mean()) / (raw.std() + 1e-5)
                    np.testing.assert_allclose(actual, expected, atol=1e-5)
                np.testing.assert_allclose(runner.buffer[0].returns, farm_targets)
                with runner.rollout_path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 4)
                self.assertEqual(rows[0]["episode"], "1")
                self.assertEqual(rows[-1]["tick"], "3")
                self.assertEqual(float(rows[0]["time_start_s"]), 0)
                self.assertEqual(float(rows[-1]["time_end_s"]), 16)
                self.assertEqual(float(rows[0]["farm_power_w"]), 10)
                self.assertEqual(float(rows[0]["actor1_power_w"]), 6)
                self.assertEqual(float(rows[0]["actor0_yaw_deg"]), 0)
                self.assertAlmostEqual(float(rows[0]["team_return_to_go"]), 1.7195, places=6)
                self.assertAlmostEqual(float(rows[0]["actor0_return_to_go"]), 1.9, places=6)
                self.assertAlmostEqual(float(rows[0]["actor1_return_to_go"]), 1.539, places=6)
                self.assertAlmostEqual(float(rows[0]["actor0_individual_value"]),
                                       float(rows[0]["actor0_aligned_reference_value"]), places=6)
                self.assertAlmostEqual(float(rows[0]["actor0_individual_gae_advantage"]),
                                       1.9 - float(rows[0]["actor0_individual_value"]), places=6)
                self.assertAlmostEqual(float(rows[0]["actor0_mixed_advantage"]), captured[0][0], places=6)
                self.assertAlmostEqual(float(rows[0]["actor0_sampled_rate_deg_s"]),
                                       float(rows[0]["actor0_applied_rate_deg_s"]), places=6)
            finally:
                runner.writter.close()
                envs.close()

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
                    args.use_valuenorm = False
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
                            self.assertAlmostEqual(float(row["actor0_individual_return"]), 0.8, places=6)
                            self.assertAlmostEqual(float(row["actor1_individual_return"]), 3.2, places=6)
                            for key in ("actor0_policy_loss", "actor1_policy_loss", "critic_value_loss",
                                        "critic_return_rmse", "critic_explained_variance",
                                        "actor0_ppo_clip_fraction", "actor1_ppo_clip_fraction"):
                                self.assertTrue(np.isfinite(float(row[key])))
                            self.assertTrue(0 <= float(row["actor0_ppo_clip_fraction"]) <= 1)
                            self.assertTrue(0 <= float(row["actor1_ppo_clip_fraction"]) <= 1)
                        modules = [p.actor for p in runner.policy] + [runner.critic, *runner.individual_critics]
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
                args.use_valuenorm = False
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
                    self.assertIsNone(runner.critic_trainer.value_normalizer)
                    runner.warmup()
                    for step in range(4):
                        values, actions, log_probs, states, critic_states, commands = runner.collect(step)
                        obs, rewards, dones, infos = envs.step(commands)
                        runner.insert((obs, rewards, dones, infos, values, actions,
                                       log_probs, states, critic_states))
                    # Deliberately depart from the aligned reference so the
                    # residual critics have a nonzero learning target.
                    runner.individual_rewards[0, 0, :, 0] += 0.3
                    runner.compute()
                    np.testing.assert_allclose(runner.buffer[0].value_preds, runner.buffer[1].value_preds)
                    np.testing.assert_allclose(runner.buffer[0].returns, runner.buffer[1].returns)
                    actors_before = [[p.detach().clone() for p in policy.actor.parameters()]
                                     for policy in runner.policy]
                    critic_before = [p.detach().clone() for p in runner.critic.parameters()]
                    individual_before = [[p.detach().clone() for p in critic.parameters()]
                                         for critic in runner.individual_critics]
                    with patch.object(runner.critic_optimizer, "step",
                                      wraps=runner.critic_optimizer.step) as critic_step:
                        train_infos = runner.train()
                        # One farm critic update per epoch/minibatch, not per turbine.
                        expected_updates = args.ppo_epoch * args.num_mini_batch
                        self.assertEqual(critic_step.call_count, expected_updates)
                    for policy, before in zip(runner.policy, actors_before):
                        self.assertTrue(any(not torch.equal(a, b)
                                            for a, b in zip(before, policy.actor.parameters())))
                    self.assertTrue(any(not torch.equal(a, b)
                                        for a, b in zip(critic_before, runner.critic.parameters())))
                    for critic, before in zip(runner.individual_critics, individual_before):
                        self.assertTrue(any(not torch.equal(a, b)
                                            for a, b in zip(before, critic.parameters())))
                    runner.log_train(train_infos, 4)
                    runner.save()
                    models = Path(runner.save_dir)
                    self.assertEqual({p.name for p in models.iterdir()},
                                     {"actor_agent0.pt", "actor_agent1.pt", "critic.pt",
                                      "checkpoint_metadata.json", "individual_critic_agent0.pt",
                                      "individual_critic_agent1.pt"})
                    saved = {k: v.clone() for k, v in runner.critic.state_dict().items()}
                    individual_saved = [{k: v.clone() for k, v in critic.state_dict().items()}
                                        for critic in runner.individual_critics]
                    with torch.no_grad():
                        for parameter in runner.critic.parameters():
                            parameter.zero_()
                        for critic in runner.individual_critics:
                            for parameter in critic.parameters():
                                parameter.zero_()
                    runner.model_dir = models
                    runner.restore()
                    for key, value in runner.critic.state_dict().items():
                        torch.testing.assert_close(value, saved[key])
                    for critic, saved_state in zip(runner.individual_critics, individual_saved):
                        for key, value in critic.state_dict().items():
                            torch.testing.assert_close(value, saved_state[key])
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
