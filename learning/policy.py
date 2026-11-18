"""Partitioned actor-critic for Phase 1 (RMA style), on stock SB3 1.5.0.

    e (privileged)        -> env_net (μ)  -> z (latent_size, Tanh)
    [x_t, a_{t-1}], z     -> policy_net   -> latent_pi -> action_net (SB3; PPO wraps it in Tanh)
    [x_t, a_{t-1}], z     -> value_net    -> latent_vf -> value (SB3)

The observation is [x_t, a_{t-1} (state_size) | e (env_size) | anything after, e.g. the Phase 2 history,
which is ignored]. Actor and critic share μ, so value gradients shape z too. In Phase 2 the actor
(policy_net + action_net) is frozen and z is replaced by a history encoder's estimate.
"""

from typing import Any, Dict, List, Tuple, Type

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


def mlp(input_size: int, sizes: List[int], activation_fn: Type[nn.Module]) -> nn.Sequential:
    layers = []
    for size in sizes:
        layers += [nn.Linear(input_size, size), activation_fn()]
        input_size = size
    return nn.Sequential(*layers)


class PartitionedExtractor(nn.Module):
    """SB3 1.5 MlpExtractor interface: forward -> (latent_pi, latent_vf), forward_actor, forward_critic,
    latent_dim_pi, latent_dim_vf."""

    def __init__(self, state_size: int, env_size: int, latent_size: int, env_net_arch: List[int],
                 pi_arch: List[int], vf_arch: List[int], activation_fn: Type[nn.Module]):
        super(PartitionedExtractor, self).__init__()
        self.state_size = state_size
        self.env_size = env_size
        # μ: e -> z. Tanh output, so z (and the student's ẑ) live in [-1, 1].
        self.env_net = nn.Sequential(mlp(env_size, env_net_arch, activation_fn),
                                     nn.Linear(env_net_arch[-1] if env_net_arch else env_size, latent_size), nn.Tanh())
        self.policy_net = mlp(state_size + latent_size, pi_arch, activation_fn)
        self.value_net = mlp(state_size + latent_size, vf_arch, activation_fn)
        self.latent_dim_pi = pi_arch[-1] if pi_arch else state_size + latent_size
        self.latent_dim_vf = vf_arch[-1] if vf_arch else state_size + latent_size

    def split(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """features -> ([x_t, a_{t-1}], e)."""
        return features[:, :self.state_size], features[:, self.state_size:self.state_size + self.env_size]

    def latent(self, features: th.Tensor) -> th.Tensor:
        """z = μ(e)."""
        return self.env_net(self.split(features)[1])

    def actor(self, state: th.Tensor, z: th.Tensor) -> th.Tensor:
        return self.policy_net(th.cat((state, z), dim=1))

    def critic(self, state: th.Tensor, z: th.Tensor) -> th.Tensor:
        return self.value_net(th.cat((state, z), dim=1))

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        state, e = self.split(features)
        z = self.env_net(e)
        return self.actor(state, z), self.critic(state, z)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        state, e = self.split(features)
        return self.actor(state, self.env_net(e))

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        state, e = self.split(features)
        return self.critic(state, self.env_net(e))


class PartitionedPolicy(ActorCriticPolicy):
    """ActorCriticPolicy with a PartitionedExtractor. Saved with policy.save() and reloaded with
    PartitionedPolicy(**saved["data"]) + load_state_dict, like MlpPolicy."""

    def __init__(self, *args, state_size: int = 16, env_size: int = 10, latent_size: int = 8,
                 env_net_arch: List[int] = (64, 64), pi_arch: List[int] = (256, 256),
                 vf_arch: List[int] = (512, 512), **kwargs):
        # set before super().__init__, which builds the networks through _build_mlp_extractor
        self.state_size = int(state_size)
        self.env_size = int(env_size)
        self.latent_size = int(latent_size)
        self.env_net_arch = [int(s) for s in env_net_arch]
        self.pi_arch = [int(s) for s in pi_arch]
        self.vf_arch = [int(s) for s in vf_arch]
        super(PartitionedPolicy, self).__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PartitionedExtractor(self.state_size, self.env_size, self.latent_size,
                                                  self.env_net_arch, self.pi_arch, self.vf_arch, self.activation_fn)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super(PartitionedPolicy, self)._get_constructor_parameters()
        data.update(state_size=self.state_size, env_size=self.env_size, latent_size=self.latent_size,
                    env_net_arch=self.env_net_arch, pi_arch=self.pi_arch, vf_arch=self.vf_arch)
        return data
