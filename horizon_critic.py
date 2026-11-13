"""Remaining-horizon scaling for the shared farm critic.

The wrapped critic predicts an average discounted reward per remaining step.
The public forward method returns value in the original team-reward units, so
rollout collection and GAE need no separate scaling operation.
"""

import math

import numpy as np
import torch
from torch import nn


def discounted_horizon(remaining_steps, gamma):
    """Return sum(gamma**j for j in range(n)) for integer steps n >= 0.

    Inputs may be numpy arrays, torch tensors, or scalar values. The expm1
    formulation is stable for discount factors close to one. Episode times
    are discrete, so near-integer floating-point step counts are rounded.
    """
    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    if torch.is_tensor(remaining_steps):
        steps = torch.round(remaining_steps).clamp_min(0)
        if gamma == 1:
            return steps
        if gamma == 0:
            return (steps > 0).to(dtype=steps.dtype)
        return -torch.expm1(steps * math.log(gamma)) / (1 - gamma)
    steps = np.maximum(np.rint(np.asarray(remaining_steps, dtype=float)), 0)
    if gamma == 1:
        return steps
    if gamma == 0:
        return np.asarray(steps > 0, dtype=float)
    return -np.expm1(steps * math.log(gamma)) / (1 - gamma)


class HorizonScaledCritic(nn.Module):
    """Wrap an R_Critic while preserving its parameters and recurrent state.

    Central-observation feature `remaining_feature` is the fraction of the
    fixed episode horizon left. The wrapped scalar head is initialized to zero;
    this initialization is overwritten normally when restoring a checkpoint.
    """

    def __init__(self, critic, episode_length, gamma, remaining_feature=6):
        super().__init__()
        if int(episode_length) != episode_length or episode_length < 1:
            raise ValueError("episode_length must be a positive integer")
        # Validate gamma before touching the original critic parameters.
        discounted_horizon(0, gamma)
        if not isinstance(critic.v_out, nn.Linear) or critic.v_out.out_features != 1:
            raise ValueError("HorizonScaledCritic requires a scalar linear value head")
        self.critic = critic
        self.episode_length = int(episode_length)
        self.gamma = float(gamma)
        self.remaining_feature = int(remaining_feature)
        if self.remaining_feature < 0:
            raise ValueError("remaining_feature must be nonnegative")
        nn.init.zeros_(critic.v_out.weight)
        nn.init.zeros_(critic.v_out.bias)

    def horizon(self, cent_obs, *, dtype=None, device=None):
        """Get each observation's discounted remaining-step count."""
        if dtype is None or device is None:
            parameter = next(self.critic.parameters())
            dtype = parameter.dtype if dtype is None else dtype
            device = parameter.device if device is None else device
        obs = torch.as_tensor(cent_obs, dtype=dtype, device=device)
        if obs.ndim < 2 or obs.shape[-1] <= self.remaining_feature:
            raise ValueError("Central observations do not contain the remaining-time feature")
        fraction = obs[..., self.remaining_feature:self.remaining_feature + 1].clamp(0, 1)
        return discounted_horizon(fraction * self.episode_length, self.gamma)

    def per_step_values(self, cent_obs, rnn_states, masks):
        """Predict reward per discounted remaining step for critic fitting."""
        return self.critic(cent_obs, rnn_states, masks)

    def forward(self, cent_obs, rnn_states, masks):
        per_step, next_states = self.per_step_values(cent_obs, rnn_states, masks)
        horizon = self.horizon(cent_obs, dtype=per_step.dtype, device=per_step.device)
        return horizon * per_step, next_states
