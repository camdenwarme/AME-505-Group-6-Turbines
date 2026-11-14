"""Separate turbine actors with one centralized critic, using light_mappo."""

import copy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from algorithms.algorithm.r_mappo import RMAPPO
from algorithms.algorithm.r_actor_critic import R_Critic
from bounded_action_head import install_bounded_action_head
from horizon_critic import HorizonScaledCritic
from runner.separated.env_runner import EnvRunner as SeparatedRunner
from utils.util import get_gard_norm


def _discounted_returns(rewards, masks, gamma):
    returns = np.zeros_like(rewards)
    running = np.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        running = rewards[step] + gamma * masks[step + 1] * running
        returns[step] = running
    return returns


def _discounted_horizons(steps, gamma):
    """Discounted count of rewards remaining at each tick, including terminal."""
    horizons = np.zeros((steps + 1, 1, 1), dtype=np.float32)
    for step in reversed(range(steps)):
        horizons[step] = 1 + gamma * horizons[step + 1]
    return horizons


def _individual_values(reference, residual, horizons, masks):
    """Combine aligned return with a per-step residual; reset states don't bootstrap."""
    values = reference + horizons * residual
    # Auto-reset puts the next episode's initial observation at a done slot.
    # Its baseline is needed at index zero of the next rollout, but it is not
    # the value of the preceding episode's terminal state.
    values[1:] = np.where(masks[1:] > 0, values[1:], 0.0)
    return values


def _individual_gae(rewards, values, masks, gamma, gae_lambda, use_gae=True):
    """Individual GAE and targets; values include the aligned reference."""
    if not use_gae:
        targets = _discounted_returns(rewards, masks, gamma)
        return targets - values[:-1], targets
    advantages = np.zeros_like(rewards)
    gae = np.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        mask = masks[step + 1]
        delta = rewards[step] + gamma * mask * values[step + 1] - values[step]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[step] = gae
    return advantages, advantages + values[:-1]


def _critic_return_diagnostics(buffer, value_normalizer):
    """Compare rollout-time farm values to realized discounted team returns.

    This integration collects complete, terminating episodes. These empirical
    returns are noisy samples, so diagnostics are not held-out model scores.
    """
    observed = _discounted_returns(buffer.rewards, buffer.masks, buffer.gamma)
    predictions = buffer.value_preds[:-1]
    if value_normalizer is not None:
        predictions = value_normalizer.denormalize(predictions)
    errors = np.asarray(predictions, dtype=np.float64) - observed
    target_variance = float(np.var(observed, dtype=np.float64))
    return {
        "return_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "explained_variance": (1.0 - float(np.var(errors)) / target_variance
                               if target_variance > 1e-12 else float("nan")),
    }


class _ActorTrainer(RMAPPO):
    """Reuse MAPPO's advantage calculation/minibatching without updating values."""

    def ppo_update(self, sample, update_actor=True):
        (_, obs, states, _, actions, _, _, masks, active_masks,
         old_log_probs, advantages, available_actions) = sample
        active_masks = torch.as_tensor(active_masks, **self.tpdv)
        old_log_probs = torch.as_tensor(old_log_probs, **self.tpdv)
        advantages = torch.as_tensor(advantages, **self.tpdv)
        log_probs, entropy = self.policy.actor.evaluate_actions(
            obs, states, actions, masks, available_actions, active_masks)
        ratios = torch.exp(log_probs - old_log_probs)
        clipped = (torch.abs(ratios.detach() - 1) > self.clip_param).float()
        if self._use_policy_active_masks:
            clip_fraction = (clipped * active_masks).sum() / active_masks.sum().clamp_min(1)
        else:
            clip_fraction = clipped.mean()
        self._clip_fractions.append(float(clip_fraction))
        surrogate = torch.min(ratios * advantages,
                              ratios.clamp(1 - self.clip_param, 1 + self.clip_param) * advantages)
        loss = -surrogate.sum(dim=-1, keepdim=True)
        if self._use_policy_active_masks:
            loss = (loss * active_masks).sum() / active_masks.sum()
        else:
            loss = loss.mean()
        optimizer = self.policy.actor_optimizer
        optimizer.zero_grad()
        if update_actor:
            (loss - self.entropy_coef * entropy).backward()
        grad_norm = (nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                     if self._use_max_grad_norm else get_gard_norm(self.policy.actor.parameters()))
        if update_actor:
            optimizer.step()
        zero = torch.zeros((), **self.tpdv)
        return zero, zero, loss, entropy, grad_norm, ratios


class SharedCriticRunner(SeparatedRunner):
    """Farm critic and individual value models supply independent GAE signals.

    The central critic predicts team return only.  Each actor combines its own
    GAE with team GAE. Each individual critic learns a residual around the
    wind-dependent aligned baseline, so its TD target matches its own reward.
    """

    def __init__(self, config):
        if not config["all_args"].use_centralized_V:
            raise ValueError("SharedCriticRunner requires the centralized value function.")
        if any(len(getattr(config["envs"], name)) != config["num_agents"] for name in
               ("action_space", "observation_space", "share_observation_space")):
            raise ValueError("Runner agent count must match the environment spaces.")
        # Delay restoration until the original runner's policies are connected
        # to the shared critic and its single optimizer/normalizer.
        model_dir = config["all_args"].model_dir
        args = copy.copy(config["all_args"])
        args.model_dir = None
        if args.use_valuenorm or args.use_popart:
            raise ValueError("The horizon-scaled farm critic requires ValueNorm and PopArt to be disabled.")
        super().__init__({**config, "all_args": args})
        self.model_dir = model_dir
        for i, policy in enumerate(self.policy):
            install_bounded_action_head(policy, self.envs.action_space[i])
        self._episode_callback = config.get("episode_callback")
        self._completed_episodes = 0
        self._episode_energy_j = None
        self._cumulative_energy_j = 0.0
        self._started_at = time.perf_counter()
        self.team_reward_weight = float(getattr(args, "team_reward_weight", 0.5))
        if not 0.0 <= self.team_reward_weight <= 1.0:
            raise ValueError("team_reward_weight must be between 0 and 1.")
        self.individual_rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.individual_reference = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.individual_horizons = _discounted_horizons(self.episode_length, self.buffer[0].gamma)
        self.numeric_eval_interval = int(getattr(args, "numeric_eval_interval", 10))
        if self.numeric_eval_interval < 1:
            raise ValueError("numeric_eval_interval must be positive.")
        self._rollout_physical = np.full(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 2), np.nan, dtype=np.float64)
        self._rollout_farm_power = np.full(
            (self.episode_length, self.n_rollout_threads), np.nan, dtype=np.float64)
        self._rollout_conditions = [[{} for _ in range(self.n_rollout_threads)]
                                    for _ in range(self.episode_length)]
        self._episode_conditions = {}
        self._scenario_events = set()
        self.scenarios_path = Path(self.run_dir) / "wind_scenarios.jsonl"
        self.metrics_path = Path(self.run_dir) / "training_metrics.csv"
        self.rollout_path = Path(self.run_dir) / "rollout_steps.csv"
        self._csv_columns = ("episode", "environment_steps", "elapsed_seconds",
                             *(f"actor{i}_policy_loss" for i in range(self.num_agents)),
                             *(f"actor{i}_ppo_clip_fraction" for i in range(self.num_agents)),
                             "critic_value_loss", "critic_return_rmse", "critic_explained_variance",
                             "episode_energy_j", "episode_energy_mwh",
                             "cumulative_energy_j", "cumulative_energy_mwh",
                             "episode_return", "episode_end_reward", "scenario_id", "episode_seed",
                             "wind_mode", "end_ambient_wind_speed_m_s", "end_ambient_wind_direction_deg",
                             *(f"actor{i}_individual_return" for i in range(self.num_agents)),
                             *(f"actor{i}_individual_value_loss" for i in range(self.num_agents)),
                             *(f"actor{i}_individual_value_rmse" for i in range(self.num_agents)))
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=self._csv_columns).writeheader()
        # Return raw team values for GAE, but fit the network's per-remaining-
        # step output on normalized targets. The existing Adam retains the same
        # underlying parameters after wrapping.
        remaining_indices = {getattr(env, "remaining_time_index", 3 * self.num_agents)
                             for env in self.envs.envs}
        if len(remaining_indices) != 1:
            raise ValueError("All rollout environments must use the same critic feature order.")
        self.remaining_time_index = remaining_indices.pop()
        self.critic = HorizonScaledCritic(self.policy[0].critic, self.episode_length, args.gamma,
                                           remaining_feature=self.remaining_time_index)
        self.critic_optimizer = self.policy[0].critic_optimizer
        for policy in self.policy:
            policy.critic = self.critic
            policy.critic_optimizer = self.critic_optimizer
        self.critic_trainer = RMAPPO(args, self.policy[0], device=self.device)
        self.trainer = [_ActorTrainer(args, policy, device=self.device) for policy in self.policy]
        for trainer in self.trainer:
            trainer.value_normalizer = self.critic_trainer.value_normalizer
        # The local value networks use the centralized, time-aware state. They
        # are feedforward even when actors use RNNs. Each predicts the average
        # per-step residual, scaled by the discounted remaining horizon.
        local_args = copy.copy(args)
        local_args.use_recurrent_policy = False
        local_args.use_naive_recurrent_policy = False
        local_args.use_popart = False
        self.individual_critics = [R_Critic(local_args, self.envs.share_observation_space[i],
                                             device=self.device) for i in range(self.num_agents)]
        for critic in self.individual_critics:
            nn.init.zeros_(critic.v_out.weight)
            nn.init.zeros_(critic.v_out.bias)
        self.individual_optimizers = [torch.optim.Adam(critic.parameters(), lr=args.critic_lr,
                                                       eps=args.opti_eps, weight_decay=args.weight_decay)
                                      for critic in self.individual_critics]
        self._individual_value_info = []
        if self.model_dir is not None:
            self.restore()

    def _evaluate_episode(self, episode):
        if self._episode_callback is None:
            return
        # Evaluation must not change subsequent training exploration, even if
        # simulator initialization or video rendering consumes random numbers.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                self._episode_callback(episode, self.policy)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    def run(self):
        self._completed_episodes = 0
        self._cumulative_energy_j = 0.0
        self._started_at = time.perf_counter()
        self._evaluate_episode(0)
        super().run()

    def _central_observations(self):
        # This integration uses DummyVecEnv and one sequential simulator.  Read
        # the full critic-only snapshot, not just concatenated actor inputs.
        return np.asarray([env.get_centralized_obs() for env in self.envs.envs], dtype=np.float32)

    def _individual_references(self):
        references = np.asarray([env.get_individual_baseline_values(self.buffer[0].gamma)
                                 for env in self.envs.envs], dtype=np.float32)
        if references.shape != (self.n_rollout_threads, self.num_agents) or not np.isfinite(references).all():
            raise ValueError("Expected one finite aligned reference per environment and turbine.")
        return references[..., None]

    def _individual_critic_residual(self, critic, central_obs):
        """Predict the average per-step deviation from the aligned reference."""
        count = central_obs.shape[0]
        states = np.zeros((count, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((count, 1), dtype=np.float32)
        residual, _ = critic(central_obs, states, masks)
        return residual

    def warmup(self):
        obs = self.envs.reset()
        for i, env in enumerate(self.envs.envs):
            if getattr(env, "wind_scenario", None) is not None:
                self._register_scenario(env.wind_scenario.to_dict(), i, "reset")
        central = self._central_observations()
        self.individual_reference[0] = self._individual_references()
        for i, buffer in enumerate(self.buffer):
            buffer.share_obs[0] = central.copy()
            buffer.obs[0] = obs[:, i].copy()

    def _register_scenario(self, scenario, environment_index, event, training_episode=None):
        """Preserve full wind knots and seeds, including pending auto-resets."""
        key = (scenario["scenario_id"], environment_index, event)
        if key in self._scenario_events:
            return
        record = {"event": event, "environment_index": environment_index,
                  "training_episode": (self._completed_episodes + 1 if training_episode is None
                                       else training_episode), "scenario": scenario}
        with self.scenarios_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._scenario_events.add(key)

    def insert(self, data):
        # DummyVecEnv resets completed environments before returning. The info
        # snapshot still contains the terminal episode energy before that reset.
        # Count farm energy once, not once per turbine sharing the same reward.
        self._episode_energy_j = float(np.mean([
            env_infos[0]["episode_energy_j"] for env_infos in data[3]]))
        (obs, rewards, dones, infos, values, actions, log_probs,
         states, critic_states) = data
        # DummyVecEnv has reset completed environments, so central observations
        # are the next episode's initial state and terminal masks prevent bootstrap.
        central = self._central_observations()
        self.individual_rewards[self.buffer[0].step] = rewards
        step = self.buffer[0].step
        # A done environment was automatically reset: its reference is the
        # next episode's state, and the terminal mask suppresses bootstrapping.
        self.individual_reference[step + 1] = self._individual_references()
        for env_index, env_infos in enumerate(infos):
            self._rollout_farm_power[step, env_index] = env_infos[0].get("farm_power_w", np.nan)
            self._rollout_conditions[step][env_index] = {
                key: env_infos[0].get(key, "") for key in
                ("scenario_id", "episode_seed", "wind_mode", "ambient_wind_speed_m_s",
                 "ambient_wind_direction_deg")}
            self._episode_conditions = self._rollout_conditions[step][env_index].copy()
            if "wind_scenario" in env_infos[0]:
                self._register_scenario(env_infos[0]["wind_scenario"], env_index, "completed")
                env = self.envs.envs[env_index]
                if getattr(env, "wind_scenario", None) is not None:
                    self._register_scenario(env.wind_scenario.to_dict(), env_index, "reset",
                                            training_episode=self._completed_episodes + 2)
            for i, info in enumerate(env_infos):
                self._rollout_physical[step, env_index, i] = (
                    info.get("yaw_rate_deg_s", np.nan), info.get("turbine_power_w", np.nan))
        team_rewards = np.asarray([[env_infos[0]["team_reward"]] for env_infos in infos],
                                  dtype=np.float32)
        states[dones] = 0
        critic_states[dones] = 0
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones] = 0
        for i, buffer in enumerate(self.buffer):
            buffer.insert(central, obs[:, i], states[:, i], critic_states[:, i],
                          actions[:, i], log_probs[:, i], values[:, i],
                          team_rewards, masks[:, i])

    def _append_episode_metrics(self, infos):
        if self._episode_energy_j is None:
            raise RuntimeError("No episode energy was captured from environment info.")
        self._cumulative_energy_j += self._episode_energy_j
        row = {"episode": self._completed_episodes,
               "environment_steps": self._completed_episodes * self.episode_length * self.n_rollout_threads,
               "elapsed_seconds": time.perf_counter() - self._started_at,
               "critic_value_loss": self._critic_info["value_loss"],
               "critic_return_rmse": self._critic_info["return_rmse"],
               "critic_explained_variance": self._critic_info["explained_variance"],
               "episode_energy_j": self._episode_energy_j,
               "episode_energy_mwh": self._episode_energy_j / 3.6e9,
               "cumulative_energy_j": self._cumulative_energy_j,
               "cumulative_energy_mwh": self._cumulative_energy_j / 3.6e9,
               # Shared team reward: count one turbine, not both copies.
               "episode_return": float(self.buffer[0].rewards.sum(axis=0).mean()),
               "episode_end_reward": float(self.buffer[0].rewards[-1].mean())}
        for key in ("scenario_id", "episode_seed", "wind_mode"):
            row[key] = self._episode_conditions.get(key, "")
        for key in ("ambient_wind_speed_m_s", "ambient_wind_direction_deg"):
            row["end_" + key] = self._episode_conditions.get(key, "")
        for i, info in enumerate(infos):
            row[f"actor{i}_policy_loss"] = float(info["policy_loss"])
            row[f"actor{i}_ppo_clip_fraction"] = float(info["ppo_clip_fraction"])
            row[f"actor{i}_individual_return"] = float(self.individual_rewards[:, :, i].sum(axis=0).mean())
            row[f"actor{i}_individual_value_loss"] = self._individual_value_info[i]["loss"]
            row[f"actor{i}_individual_value_rmse"] = self._individual_value_info[i]["rmse"]
        # Close after each row so progress is immediately readable, including
        # while a subsequent evaluation/video is running or if a run stops early.
        with self.metrics_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=self._csv_columns).writerow(row)
        self._episode_energy_j = None

    def _critic_samples(self, buffer):
        unused_advantages = np.zeros_like(buffer.rewards)
        trainer = self.critic_trainer
        if trainer._use_recurrent_policy:
            return buffer.recurrent_generator(unused_advantages, trainer.num_mini_batch,
                                              trainer.data_chunk_length)
        if trainer._use_naive_recurrent:
            return buffer.naive_recurrent_generator(unused_advantages, trainer.num_mini_batch)
        return buffer.feed_forward_generator(unused_advantages, trainer.num_mini_batch)

    def _append_rollout_steps(self, team_values, team_returns, team_advantage,
                              individual_returns, individual_values, individual_advantages,
                              mixed_advantages, ppo_advantages):
        # Observations/actions in buffers are at interval start; power and
        # applied rates from info belong to the corresponding interval end.
        fields = ["episode", "tick", "environment_index", "time_start_s", "time_end_s",
                  "scenario_id", "episode_seed", "wind_mode", "ambient_wind_speed_m_s",
                  "ambient_wind_direction_deg",
                  "farm_power_w", "team_reward", "team_value", "team_return_to_go",
                  "team_gae_advantage"]
        for i in range(self.num_agents):
            fields.extend(f"actor{i}_{name}" for name in (
                "wind_speed_m_s", "wind_direction_deg", "yaw_deg", "sampled_rate_deg_s",
                "applied_rate_deg_s", "power_w", "reward", "return_to_go",
                "aligned_reference_value", "individual_value", "individual_gae_advantage",
                "mixed_advantage", "ppo_advantage"))
        exists = self.rollout_path.exists()
        with self.rollout_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not exists:
                writer.writeheader()
            for step in range(self.episode_length):
                for env_index, env in enumerate(self.envs.envs):
                    dt = getattr(env, "dt", np.nan)
                    row = {"episode": self._completed_episodes + 1, "tick": step,
                           "environment_index": env_index, "time_start_s": step * dt,
                           "time_end_s": (step + 1) * dt,
                           "farm_power_w": self._rollout_farm_power[step, env_index],
                           "team_reward": self.buffer[0].rewards[step, env_index, 0],
                           "team_value": team_values[step, env_index, 0],
                           "team_return_to_go": team_returns[step, env_index, 0],
                           "team_gae_advantage": team_advantage[step, env_index, 0]}
                    row.update(self._rollout_conditions[step][env_index])
                    for i, buffer in enumerate(self.buffer):
                        speed, direction, yaw = buffer.obs[step, env_index]
                        mixed = mixed_advantages[i]
                        values = (speed, direction, yaw, buffer.actions[step, env_index, 0],
                                  *self._rollout_physical[step, env_index, i],
                                  self.individual_rewards[step, env_index, i, 0],
                                  individual_returns[i][step, env_index, 0],
                                  self.individual_reference[step, env_index, i, 0],
                                  individual_values[i][step, env_index, 0],
                                  individual_advantages[i][step, env_index, 0],
                                  mixed[step, env_index, 0], ppo_advantages[i][step, env_index, 0])
                        for name, value in zip(("wind_speed_m_s", "wind_direction_deg", "yaw_deg",
                                                "sampled_rate_deg_s", "applied_rate_deg_s", "power_w",
                                                "reward", "return_to_go", "aligned_reference_value",
                                                "individual_value", "individual_gae_advantage",
                                                "mixed_advantage", "ppo_advantage"), values):
                            row[f"actor{i}_{name}"] = value
                    writer.writerow(row)

    def _train_individual_critics(self, reference, targets, values, realized_returns):
        """Fit the average residual in frozen GAE targets for each state."""
        self._individual_value_info = []
        states = reference.share_obs[:-1].reshape(-1, *reference.share_obs.shape[2:])
        count = len(states)
        if count < self.critic_trainer.num_mini_batch:
            raise ValueError("Individual critics need at least one sample per minibatch.")
        for i, (critic, optimizer) in enumerate(zip(self.individual_critics, self.individual_optimizers)):
            desired = ((targets[i] - self.individual_reference[:-1, :, i]) /
                       self.individual_horizons[:-1]).reshape(-1, 1)
            desired = torch.as_tensor(desired, dtype=torch.float32, device=self.device)
            losses = []
            for _ in range(self.critic_trainer.ppo_epoch):
                for indices in np.array_split(torch.randperm(count).numpy(), self.critic_trainer.num_mini_batch):
                    predicted = self._individual_critic_residual(critic, states[indices])
                    loss = nn.functional.smooth_l1_loss(predicted, desired[indices])
                    optimizer.zero_grad()
                    loss.backward()
                    if self.critic_trainer._use_max_grad_norm:
                        nn.utils.clip_grad_norm_(critic.parameters(), self.critic_trainer.max_grad_norm)
                    optimizer.step()
                    losses.append(loss.item())
            error = values[i][:-1] - realized_returns[i]
            self._individual_value_info.append({"loss": float(np.mean(losses)),
                                                "rmse": float(np.sqrt(np.mean(error ** 2)))})

    def train(self):
        # Both actors use rollout values. Update the critic only after both
        # actor updates finish, keeping their advantages fixed during PPO.
        reference = self.buffer[0]
        for buffer in self.buffer[1:]:
            for field in ("share_obs", "rewards", "masks", "value_preds", "returns"):
                if not np.allclose(getattr(reference, field), getattr(buffer, field)):
                    raise ValueError(f"Shared farm critic requires identical {field} across agents.")
        # Snapshot the rollout-time value error before updating any critics.
        diagnostics = _critic_return_diagnostics(reference, self.critic_trainer.value_normalizer)
        # Each individual's aligned reference plus its learned residual forms
        # its value estimate. Freeze both individual and team advantages while
        # updating actors; fit their respective critics afterward.
        def standardize(x):
            return (x - np.mean(x)) / (np.std(x) + 1e-5)

        infos = []
        all_individual_returns, all_individual_values = [], []
        all_individual_advantages, all_individual_targets = [], []
        all_mixed_advantages, all_ppo_advantages = [], []
        team_values = team_advantage_snapshot = None
        for i, (trainer, buffer) in enumerate(zip(self.trainer, self.buffer)):
            trainer.prep_training()
            individual_returns = _discounted_returns(
                self.individual_rewards[:, :, i], buffer.masks, buffer.gamma)
            with torch.no_grad():
                shape = buffer.share_obs.shape
                central = buffer.share_obs.reshape(-1, *shape[2:])
                residual = self._individual_critic_residual(self.individual_critics[i], central)
                residual = residual.detach().cpu().numpy().reshape(shape[0], shape[1], 1)
            individual_values = _individual_values(self.individual_reference[:, :, i], residual,
                                                   self.individual_horizons, buffer.masks)
            individual_advantage, individual_targets = _individual_gae(
                self.individual_rewards[:, :, i], individual_values, buffer.masks,
                buffer.gamma, buffer.gae_lambda, buffer._use_gae)
            if trainer.value_normalizer is not None:
                baseline = trainer.value_normalizer.denormalize(buffer.value_preds[:-1])
            else:
                baseline = buffer.value_preds[:-1]
            team_advantage = buffer.returns[:-1] - baseline
            if i == 0:
                team_values = baseline.copy()
                team_advantage_snapshot = team_advantage.copy()
            mixed = ((1 - self.team_reward_weight) * standardize(individual_advantage) +
                     self.team_reward_weight * standardize(team_advantage))
            # Reuse light_mappo's sequence/minibatch handling and actor update.
            # Its train() subtracts baseline and normalizes once more; restore
            # the actual farm returns before fitting the shared critic.
            farm_returns = buffer.returns[:-1].copy()
            buffer.returns[:-1] = baseline + mixed
            # Match the subsequent RMAPPO normalization, including float32
            # rounding during buffer assignment and baseline subtraction.
            final_advantage = standardize(buffer.returns[:-1] - baseline)
            all_individual_returns.append(individual_returns)
            all_individual_values.append(individual_values)
            all_individual_advantages.append(individual_advantage)
            all_individual_targets.append(individual_targets)
            all_mixed_advantages.append(mixed)
            all_ppo_advantages.append(final_advantage)
            trainer._clip_fractions = []
            try:
                info = trainer.train(buffer)
                info["ppo_clip_fraction"] = (float(np.mean(trainer._clip_fractions))
                                             if trainer._clip_fractions else float("nan"))
                infos.append(info)
            finally:
                buffer.returns[:-1] = farm_returns

        team_returns = _discounted_returns(reference.rewards, reference.masks, reference.gamma)
        self._append_rollout_steps(team_values, team_returns, team_advantage_snapshot,
                                   all_individual_returns, all_individual_values, all_individual_advantages,
                                   all_mixed_advantages, all_ppo_advantages)

        self._train_individual_critics(reference, all_individual_targets,
                                       all_individual_values, all_individual_returns)

        value_loss_total = grad_norm_total = 0.0
        trainer = self.critic_trainer
        for _ in range(trainer.ppo_epoch):
            # One copy of the farm trajectory, not a duplicate per turbine.
            for sample in self._critic_samples(reference):
                # GAE stores raw returns; regress in average per-discounted-step
                # units so the last-step value cannot inherit an episode-scale
                # magnitude merely from the network's output convention.
                values, _ = self.critic.per_step_values(sample[0], sample[3], sample[8])
                old_values, returns, active_masks = (
                    torch.as_tensor(sample[i], **trainer.tpdv) for i in (5, 6, 9))
                horizon = self.critic.horizon(sample[0], **trainer.tpdv)
                if torch.any(horizon <= 0):
                    raise ValueError("Team critic minibatch contains a terminal state with no remaining steps.")
                loss = trainer.cal_value_loss(values, old_values / horizon, returns / horizon, active_masks)
                self.critic_optimizer.zero_grad()
                (trainer.value_loss_coef * loss).backward()
                grad_norm = (nn.utils.clip_grad_norm_(self.critic.parameters(), trainer.max_grad_norm)
                             if trainer._use_max_grad_norm else get_gard_norm(self.critic.parameters()))
                self.critic_optimizer.step()
                value_loss_total += loss.item()
                grad_norm_total += float(grad_norm)
        updates = trainer.ppo_epoch * trainer.num_mini_batch
        self._critic_info = {"value_loss": value_loss_total / updates,
                             "critic_grad_norm": grad_norm_total / updates,
                             **diagnostics}
        for info, buffer in zip(infos, self.buffer):
            info.pop("value_loss")
            info.pop("critic_grad_norm")
            buffer.after_update()
        self.individual_reference[0] = self.individual_reference[-1].copy()
        self._completed_episodes += 1
        self._append_episode_metrics(infos)
        total_episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        if self._completed_episodes % self.numeric_eval_interval == 0 or self._completed_episodes == total_episodes:
            self._evaluate_episode(self._completed_episodes)
        return infos

    def log_train(self, train_infos, total_num_steps):
        super().log_train(train_infos, total_num_steps)
        for key, value in self._critic_info.items():
            self.writter.add_scalar(f"critic/{key}", value, total_num_steps)

    def _checkpoint_metadata(self):
        env = self.envs.envs[0]
        layout = getattr(env, "layout_d", None)
        return {
            "format_version": 6, "num_agents": self.num_agents,
            "action_head": "tanh_squashed_gaussian",
            "critic_observation_shape": list(self.envs.share_observation_space[0].shape),
            "remaining_time_index": self.remaining_time_index,
            "layout_d": None if layout is None else np.asarray(layout).tolist(),
            "total_rated_power_w": getattr(env, "total_rated_power_w", None),
            "individual_value": "aligned_reference_plus_discounted_remaining_horizon_residual",
            "team_value": "discounted_remaining_horizon_times_per_step_value",
        }

    def save(self):
        directory = Path(self.save_dir)
        (directory / "checkpoint_metadata.json").write_text(json.dumps(self._checkpoint_metadata(), indent=2))
        for i, policy in enumerate(self.policy):
            torch.save(policy.actor.state_dict(), directory / f"actor_agent{i}.pt")
            torch.save(self.individual_critics[i].state_dict(), directory / f"individual_critic_agent{i}.pt")
        torch.save(self.critic.state_dict(), directory / "critic.pt")
        normalizer = self.critic_trainer.value_normalizer
        if normalizer is not None and not self.critic_trainer._use_popart:
            torch.save(normalizer.state_dict(), directory / "value_normalizer.pt")

    def restore(self):
        directory = Path(self.model_dir)
        metadata_path = directory / "checkpoint_metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Checkpoint predates the bounded action head and expanded critic state; start a new run.")
        metadata = json.loads(metadata_path.read_text())
        expected = self._checkpoint_metadata()
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("Incompatible checkpoint; format 6 requires matching agent count, layout, ratings and critic schema.")
        if not (directory / "critic.pt").is_file():
            raise ValueError("Expected a shared-critic checkpoint (critic.pt); start a new run for old checkpoints.")
        for i, policy in enumerate(self.policy):
            policy.actor.load_state_dict(torch.load(directory / f"actor_agent{i}.pt",
                                                     map_location=self.device, weights_only=True))
            self.individual_critics[i].load_state_dict(torch.load(
                directory / f"individual_critic_agent{i}.pt", map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(directory / "critic.pt",
                                               map_location=self.device, weights_only=True))
        normalizer = self.critic_trainer.value_normalizer
        if normalizer is not None and not self.critic_trainer._use_popart:
            normalizer.load_state_dict(torch.load(directory / "value_normalizer.pt",
                                                  map_location=self.device, weights_only=True))
