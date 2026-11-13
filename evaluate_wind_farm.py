"""Numerically evaluate a policy and a zero-yaw-rate baseline without training."""

import numpy as np
import torch


@torch.no_grad()
def evaluate_episode(env, policies=None):
    """Return physical energy and reward for one deterministic episode.

    With policies=None the turbines retain their initially aligned headings.
    Evaluation runs in its own environment and never writes to training buffers.
    """
    modes = [] if policies is None else [p.actor.training for p in policies]
    try:
        if policies is not None:
            for policy in policies:
                policy.actor.eval()
            states = [np.zeros((1, p.actor._recurrent_N, p.actor.hidden_size), np.float32)
                      for p in policies]
        obs = env.reset()
        team_return = 0.0
        individual_returns = np.zeros(2, dtype=float)
        mask = np.ones((1, 1), np.float32)
        for _ in range(env.episode_length):
            if policies is None:
                actions = np.zeros((2, 1), dtype=np.float32)
            else:
                actions = []
                for i, policy in enumerate(policies):
                    action, states[i] = policy.act(obs[i:i + 1], states[i], mask,
                                                   deterministic=True)
                    actions.append(action.detach().cpu().numpy()[0])
                actions = np.stack(actions)
            obs, rewards, dones, infos = env.step(actions)
            team_return += float(infos[0]["team_reward"])
            individual_returns += rewards[:, 0]
            if np.all(dones):
                break
        return {"farm_energy_mwh": float(infos[0]["episode_energy_j"]) / 3.6e9,
                "turbine0_energy_mwh": float(infos[0]["turbine_episode_energy_j"]) / 3.6e9,
                "turbine1_energy_mwh": float(infos[1]["turbine_episode_energy_j"]) / 3.6e9,
                "team_return": team_return,
                "turbine0_return": float(individual_returns[0]),
                "turbine1_return": float(individual_returns[1])}
    finally:
        if policies is not None:
            for policy, mode in zip(policies, modes):
                policy.actor.train(mode)
