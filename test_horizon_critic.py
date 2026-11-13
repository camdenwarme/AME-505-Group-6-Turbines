"""Check discounted horizons, critic scaling and optimizer compatibility."""

import unittest
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from horizon_critic import HorizonScaledCritic, discounted_horizon


class _Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.v_out = nn.Linear(7, 1)

    def forward(self, obs, states, masks):
        obs = torch.as_tensor(obs, dtype=self.v_out.weight.dtype, device=self.v_out.weight.device)
        states = torch.as_tensor(states, dtype=obs.dtype, device=obs.device)
        masks = torch.as_tensor(masks, dtype=obs.dtype, device=obs.device)
        return self.v_out(obs), states * masks.unsqueeze(-1)


class HorizonCriticTests(unittest.TestCase):
    def test_horizon_matches_discounted_sum_at_edge_cases(self):
        steps = np.array([0, 1, 2, 20, 200])
        for gamma in (0.0, 0.5, 0.99, 0.999999, 1.0):
            with self.subTest(gamma=gamma):
                expected = np.array([sum(gamma ** j for j in range(int(n))) for n in steps])
                np.testing.assert_allclose(discounted_horizon(steps, gamma), expected, rtol=1e-9, atol=1e-9)
                actual = discounted_horizon(torch.tensor(steps, dtype=torch.float64), gamma)
                np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-9, atol=1e-9)
        for gamma in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                discounted_horizon(1, gamma)

    def test_values_and_recurrent_states_scale_without_changing_optimizer_parameters(self):
        base = _Critic()
        optimizer = torch.optim.Adam(base.parameters(), lr=0.01)
        before = {id(p) for group in optimizer.param_groups for p in group["params"]}
        critic = HorizonScaledCritic(base, episode_length=200, gamma=1.0)
        self.assertEqual(before, {id(p) for p in critic.parameters()})
        self.assertTrue(torch.equal(base.v_out.weight, torch.zeros_like(base.v_out.weight)))
        self.assertTrue(torch.equal(base.v_out.bias, torch.zeros_like(base.v_out.bias)))

        with torch.no_grad():
            base.v_out.bias.fill_(0.3)
        obs = np.zeros((4, 7), np.float32)
        obs[:, 6] = [1, 0.5, 1 / 200, 0]
        states = np.ones((4, 1, 3), np.float32)
        masks = np.array([[1], [1], [0], [1]], np.float32)
        raw, states_out = critic(obs, states, masks)
        normalized, _ = critic.per_step_values(obs, states, masks)
        torch.testing.assert_close(raw.squeeze(), torch.tensor([60.0, 30.0, 0.3, 0.0]))
        torch.testing.assert_close(normalized.squeeze(), torch.full((4,), 0.3))
        torch.testing.assert_close(states_out, torch.tensor(states) * torch.tensor(masks).unsqueeze(-1))
        # The existing optimizer updates exactly the wrapped critic's parameters.
        optimizer.zero_grad()
        normalized.square().mean().backward()
        optimizer.step()
        self.assertNotEqual(base.v_out.bias.item(), 0.3)

    def test_discounted_scaling_and_horizon_validation(self):
        critic = HorizonScaledCritic(_Critic(), episode_length=4, gamma=0.5)
        with torch.no_grad():
            critic.critic.v_out.bias.fill_(2)
        obs = torch.zeros((3, 7))
        obs[:, 6] = torch.tensor([1, 0.25, 0])
        values, _ = critic(obs, torch.zeros(3, 1, 1), torch.ones(3, 1))
        torch.testing.assert_close(values.squeeze(), torch.tensor([3.75, 2, 0]))
        with self.assertRaises(ValueError):
            critic.horizon(torch.zeros(2, 6))
        with self.assertRaises(ValueError):
            HorizonScaledCritic(_Critic(), episode_length=0, gamma=1)

    def test_real_recurrent_mappo_critic_keeps_recurrent_state(self):
        from gym import spaces
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "light_mappo"))
        from config import get_config
        from algorithms.algorithm.r_actor_critic import R_Critic

        args = get_config().parse_args([])
        args.use_recurrent_policy = True
        args.use_feature_normalization = False
        base = R_Critic(args, spaces.Box(-np.inf, np.inf, (19,), np.float32))
        critic = HorizonScaledCritic(base, episode_length=200, gamma=1)
        with torch.no_grad():
            base.v_out.bias.fill_(0.1)
            obs = np.zeros((2, 19), np.float32)
            obs[:, 6] = [1, 0.5]
            states = np.ones((2, args.recurrent_N, args.hidden_size), np.float32)
            masks = np.array([[1], [0]], np.float32)
            raw, actual_states = critic(obs, states, masks)
            per_step, expected_states = critic.per_step_values(obs, states, masks)
            torch.testing.assert_close(raw, per_step * torch.tensor([[200.0], [100.0]]))
            torch.testing.assert_close(actual_states, expected_states)


if __name__ == "__main__":
    unittest.main()
