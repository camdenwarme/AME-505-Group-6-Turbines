"""Separate turbine actors with one centralized critic, using light_mappo."""

import copy
import csv
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from algorithms.algorithm.r_mappo import RMAPPO
from runner.separated.env_runner import EnvRunner as SeparatedRunner
from utils.util import get_gard_norm


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
    """Common rewards/termination imply one value target per farm transition."""

    def __init__(self, config):
        if not config["all_args"].use_centralized_V:
            raise ValueError("SharedCriticRunner requires the centralized value function.")
        # Delay restoration until the original runner's policies are connected
        # to the shared critic and its single optimizer/normalizer.
        model_dir = config["all_args"].model_dir
        args = copy.copy(config["all_args"])
        args.model_dir = None
        super().__init__({**config, "all_args": args})
        self.model_dir = model_dir
        self._episode_callback = config.get("episode_callback")
        self._completed_episodes = 0
        self._episode_energy_j = None
        self._cumulative_energy_j = 0.0
        self._started_at = time.perf_counter()
        self.metrics_path = Path(self.run_dir) / "training_metrics.csv"
        self._csv_columns = ("episode", "environment_steps", "elapsed_seconds",
                             *(f"actor{i}_policy_loss" for i in range(self.num_agents)),
                             "critic_value_loss", "episode_energy_j", "episode_energy_mwh",
                             "cumulative_energy_j", "cumulative_energy_mwh")
        if not self.metrics_path.exists():
            with self.metrics_path.open("w", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=self._csv_columns).writeheader()
        self.critic = self.policy[0].critic
        self.critic_optimizer = self.policy[0].critic_optimizer
        for policy in self.policy:
            policy.critic = self.critic
            policy.critic_optimizer = self.critic_optimizer
        self.critic_trainer = RMAPPO(args, self.policy[0], device=self.device)
        self.trainer = [_ActorTrainer(args, policy, device=self.device) for policy in self.policy]
        for trainer in self.trainer:
            trainer.value_normalizer = self.critic_trainer.value_normalizer
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

    def insert(self, data):
        # DummyVecEnv resets completed environments before returning. The info
        # snapshot still contains the terminal episode energy before that reset.
        # Count farm energy once, not once per turbine sharing the same reward.
        self._episode_energy_j = float(np.mean([
            env_infos[0]["episode_energy_j"] for env_infos in data[3]]))
        super().insert(data)

    def _append_episode_metrics(self, infos):
        if self._episode_energy_j is None:
            raise RuntimeError("No episode energy was captured from environment info.")
        self._cumulative_energy_j += self._episode_energy_j
        row = {"episode": self._completed_episodes,
               "environment_steps": self._completed_episodes * self.episode_length * self.n_rollout_threads,
               "elapsed_seconds": time.perf_counter() - self._started_at,
               "critic_value_loss": self._critic_info["value_loss"],
               "episode_energy_j": self._episode_energy_j,
               "episode_energy_mwh": self._episode_energy_j / 3.6e9,
               "cumulative_energy_j": self._cumulative_energy_j,
               "cumulative_energy_mwh": self._cumulative_energy_j / 3.6e9}
        for i, info in enumerate(infos):
            row[f"actor{i}_policy_loss"] = float(info["policy_loss"])
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

    def train(self):
        # Both actors use rollout values and the same unchanged normalization
        # statistics. Update the critic only after both actor updates finish.
        reference = self.buffer[0]
        for buffer in self.buffer[1:]:
            for field in ("share_obs", "rewards", "masks", "value_preds", "returns"):
                if not np.allclose(getattr(reference, field), getattr(buffer, field)):
                    raise ValueError(f"Shared farm critic requires identical {field} across agents.")
        infos = []
        for trainer, buffer in zip(self.trainer, self.buffer):
            trainer.prep_training()
            infos.append(trainer.train(buffer))

        value_loss_total = grad_norm_total = 0.0
        trainer = self.critic_trainer
        for _ in range(trainer.ppo_epoch):
            # One copy of the farm trajectory, not a duplicate per turbine.
            for sample in self._critic_samples(reference):
                values, _ = self.critic(sample[0], sample[3], sample[8])
                old_values, returns, active_masks = (
                    torch.as_tensor(sample[i], **trainer.tpdv) for i in (5, 6, 9))
                loss = trainer.cal_value_loss(values, old_values, returns, active_masks)
                self.critic_optimizer.zero_grad()
                (trainer.value_loss_coef * loss).backward()
                grad_norm = (nn.utils.clip_grad_norm_(self.critic.parameters(), trainer.max_grad_norm)
                             if trainer._use_max_grad_norm else get_gard_norm(self.critic.parameters()))
                self.critic_optimizer.step()
                value_loss_total += loss.item()
                grad_norm_total += float(grad_norm)
        updates = trainer.ppo_epoch * trainer.num_mini_batch
        self._critic_info = {"value_loss": value_loss_total / updates,
                             "critic_grad_norm": grad_norm_total / updates}
        for info, buffer in zip(infos, self.buffer):
            info.pop("value_loss")
            info.pop("critic_grad_norm")
            buffer.after_update()
        self._completed_episodes += 1
        self._append_episode_metrics(infos)
        total_episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        if self._completed_episodes % 10 == 0 or self._completed_episodes == total_episodes:
            self._evaluate_episode(self._completed_episodes)
        return infos

    def log_train(self, train_infos, total_num_steps):
        super().log_train(train_infos, total_num_steps)
        for key, value in self._critic_info.items():
            self.writter.add_scalar(f"critic/{key}", value, total_num_steps)

    def save(self):
        directory = Path(self.save_dir)
        for i, policy in enumerate(self.policy):
            torch.save(policy.actor.state_dict(), directory / f"actor_agent{i}.pt")
        torch.save(self.critic.state_dict(), directory / "critic.pt")
        normalizer = self.critic_trainer.value_normalizer
        if normalizer is not None and not self.critic_trainer._use_popart:
            torch.save(normalizer.state_dict(), directory / "value_normalizer.pt")

    def restore(self):
        directory = Path(self.model_dir)
        if not (directory / "critic.pt").is_file():
            raise ValueError("Expected a shared-critic checkpoint (critic.pt); start a new run for old checkpoints.")
        for i, policy in enumerate(self.policy):
            policy.actor.load_state_dict(torch.load(directory / f"actor_agent{i}.pt",
                                                     map_location=self.device, weights_only=True))
        self.critic.load_state_dict(torch.load(directory / "critic.pt",
                                               map_location=self.device, weights_only=True))
        normalizer = self.critic_trainer.value_normalizer
        if normalizer is not None and not self.critic_trainer._use_popart:
            normalizer.load_state_dict(torch.load(directory / "value_normalizer.pt",
                                                  map_location=self.device, weights_only=True))
