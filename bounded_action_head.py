"""Bounded, differentiable turbine actions for light_mappo's Gaussian actor.

This adapter keeps the upstream actor's Gaussian parameters and optimizer, but
maps Gaussian samples smoothly into the environment's finite actuator bounds.
The density used by PPO includes the change-of-variables Jacobian.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TanhBoundedActionHead(nn.Module):
    """Apply a scaled tanh to Gaussian actions with consistent PPO densities.

    The latent Gaussian is expressed in action units: a zero-mean standard
    deviation of one therefore starts near one degree/second, rather than six
    degrees/second as it would with an unscaled tanh input.
    """

    def __init__(self, gaussian_head, low, high):
        super().__init__()
        self.action_out = gaussian_head
        parameter = next(gaussian_head.parameters())
        low = torch.as_tensor(low, dtype=parameter.dtype, device=parameter.device)
        high = torch.as_tensor(high, dtype=parameter.dtype, device=parameter.device)
        if (not torch.isfinite(low).all() or not torch.isfinite(high).all()
                or not torch.all(high > low)):
            raise ValueError("Tanh action bounds must be finite and strictly ordered.")
        # Bounds are part of the environment, not learned checkpoint state.
        self.register_buffer("center", (high + low) / 2, persistent=False)
        self.register_buffer("scale", (high - low) / 2, persistent=False)

    @staticmethod
    def _log_jacobian(scaled_latent):
        # log(sech(x)^2), written to remain finite for large |x|.
        return 2 * (math.log(2.0) - scaled_latent - F.softplus(-2 * scaled_latent))

    def _transform(self, latent):
        return self.center + self.scale * torch.tanh(latent / self.scale)

    def _log_prob(self, distribution, action):
        unit = (action - self.center) / self.scale
        # Float arithmetic can round tanh to exactly +/-1. Compute both rollout
        # and update likelihoods from the same clamped, stored action.
        epsilon = 1e-6
        unit = unit.clamp(-1 + epsilon, 1 - epsilon)
        scaled_latent = torch.atanh(unit)
        latent = scaled_latent * self.scale
        return (distribution.log_prob(latent) - self._log_jacobian(scaled_latent)).sum(-1, keepdim=True)

    def forward(self, x, available_actions=None, deterministic=False):
        del available_actions
        distribution = self.action_out(x)
        latent = distribution.mean if deterministic else distribution.sample()
        action = self._transform(latent)
        return action, self._log_prob(distribution, action)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        del available_actions
        distribution = self.action_out(x)
        log_probs = self._log_prob(distribution, action)
        # The transformed distribution has no closed-form entropy. A fresh
        # reparameterized Monte Carlo sample estimates H(Z) + E[log |da/dz|].
        latent = distribution.rsample()
        entropy = (distribution.entropy() + self._log_jacobian(latent / self.scale)).sum(-1, keepdim=True)
        if active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum().clamp_min(1)
        else:
            entropy = entropy.mean()
        return log_probs, entropy


def install_bounded_action_head(policy, action_space):
    """Replace ACTLayer while retaining all original trainable parameters."""
    old_head = policy.actor.act
    if not getattr(old_head, "continuous_action", False):
        raise ValueError("Tanh bounded actions require a continuous Box action space.")
    policy.actor.act = TanhBoundedActionHead(old_head.action_out, action_space.low, action_space.high)
