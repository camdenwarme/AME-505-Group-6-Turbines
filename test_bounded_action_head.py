"""Check the action transformation and PPO likelihood independently of OFF."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from bounded_action_head import TanhBoundedActionHead, install_bounded_action_head


class _Gaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mean = nn.Linear(1, 1)
        self.logstd = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.fc_mean.weight)
        nn.init.zeros_(self.fc_mean.bias)

    def forward(self, x):
        return torch.distributions.Normal(self.fc_mean(x), self.logstd.exp())


class BoundedActionHeadTests(unittest.TestCase):
    def test_sampled_action_and_update_density_match_reference(self):
        torch.manual_seed(13)
        head = TanhBoundedActionHead(_Gaussian(), [-6], [6])
        x = torch.zeros((1000, 1))
        actions, old_log_probs = head(x)
        new_log_probs, entropy = head.evaluate_actions(x, actions)
        self.assertTrue(torch.all(actions.abs() < 6))
        torch.testing.assert_close(old_log_probs, new_log_probs)
        self.assertTrue(torch.isfinite(entropy))
        # Compare the physical-action density to PyTorch's explicit transforms.
        base = torch.distributions.Normal(torch.zeros_like(actions), torch.ones_like(actions) / 6)
        reference = torch.distributions.TransformedDistribution(base, [
            torch.distributions.TanhTransform(),
            torch.distributions.AffineTransform(0, 6),
        ])
        torch.testing.assert_close(new_log_probs, reference.log_prob(actions), rtol=1e-5, atol=1e-5)
        # The initial noise is about 1 deg/s, rather than 6 deg/s.
        self.assertGreater(actions.std().item(), 0.8)
        self.assertLess(actions.std().item(), 1.1)

    def test_deterministic_action_optimizer_and_extreme_actions(self):
        gaussian = _Gaussian()
        old_head = SimpleNamespace(continuous_action=True, action_out=gaussian)
        policy = SimpleNamespace(actor=SimpleNamespace(act=old_head))
        optimizer = torch.optim.Adam(gaussian.parameters(), lr=0.001)
        original = {id(p) for group in optimizer.param_groups for p in group["params"]}
        install_bounded_action_head(policy, SimpleNamespace(low=[-6], high=[6]))
        head = policy.actor.act
        self.assertEqual(original, {id(p) for p in head.parameters()})
        action, _ = head(torch.zeros(1, 1), deterministic=True)
        torch.testing.assert_close(action, torch.zeros_like(action))
        logp, entropy = head.evaluate_actions(torch.zeros(2, 1), torch.tensor([[-6.0], [6.0]]))
        self.assertTrue(torch.isfinite(logp).all())
        (-logp.mean() - 0.01 * entropy).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
