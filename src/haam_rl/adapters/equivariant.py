"""A network that knows the buffers are interchangeable.

Swapping two buffers' contents in the plant swaps the corresponding actions and nothing else, so
the best policy is *permutation-equivariant*. A plain MLP over the flattened state has to learn
that symmetry from data (5! = 120 orderings of the same situation). This network has it built in:

    context  c   = MLP(incoming, last painted, look-ahead, phase, occupancy)
    buffer   e_b = MLP(contents of buffer b, c)            same weights for the 5 buffers
    pooled   g   = mean_b e_b
    logit(i, o)  = MLP(e_i, e_o, g, [i == o])              same weights for the 25 pairs
    value        = MLP(g, c)

It plugs into MaskablePPO as `BufferPairPolicy`: the custom part replaces SB3's MLP extractor and
produces the 25 logits directly (the action head is an identity).
Requires the one-hot observation with phase and occupancy (`ObservationConfig(encoding=onehot)`).
"""
from __future__ import annotations

import torch
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from haam_rl.domain.model import BUFFER_CAPACITY, BUFFER_COUNT, COLOR_COUNT

_CODE = COLOR_COUNT + 1                 # one-hot width of one vehicle position
_EXTRAS = 4                             # phase (3) + occupancy (1) at the end of the observation
_SLOTS_START = 2 * _CODE                # after incoming and last painted
_SLOTS_END = _SLOTS_START + BUFFER_COUNT * BUFFER_CAPACITY * _CODE


def _mlp(sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for k in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[k], sizes[k + 1]))
        layers.append(activation())
    return nn.Sequential(*layers)


class BufferPairNetwork(nn.Module):
    """Actor-critic body. `forward_actor` returns the 25 action logits (index = in * 5 + out)."""

    def __init__(self, observation_size: int, embed_dim: int = 128, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        positions, rest = divmod(observation_size - _EXTRAS, _CODE)
        if rest or positions < 2 + BUFFER_COUNT * BUFFER_CAPACITY:
            raise ValueError("BufferPairNetwork needs the one-hot observation with phase and occupancy")
        context_size = observation_size - (_SLOTS_END - _SLOTS_START)
        d = embed_dim
        self.context = _mlp([context_size, d], activation)
        self.buffer = _mlp([BUFFER_CAPACITY * _CODE + d, d, d], activation)
        self.pair = nn.Sequential(_mlp([3 * d + 1, d], activation), nn.Linear(d, 1))
        self.critic = _mlp([2 * d, d], activation)
        self.latent_dim_pi = BUFFER_COUNT * BUFFER_COUNT
        self.latent_dim_vf = d
        self.register_buffer("diagonal", torch.eye(BUFFER_COUNT).reshape(1, BUFFER_COUNT, BUFFER_COUNT, 1))

    def _embed(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]
        slots = features[:, _SLOTS_START:_SLOTS_END].reshape(batch, BUFFER_COUNT, BUFFER_CAPACITY * _CODE)
        context = self.context(torch.cat([features[:, :_SLOTS_START], features[:, _SLOTS_END:]], dim=1))
        buffers = self.buffer(torch.cat([slots, context.unsqueeze(1).expand(-1, BUFFER_COUNT, -1)], dim=2))
        return buffers, context

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        buffers, _ = self._embed(features)
        batch, n, d = buffers.shape
        pooled = buffers.mean(dim=1)
        pairs = torch.cat([
            buffers.unsqueeze(2).expand(batch, n, n, d),           # input buffer i
            buffers.unsqueeze(1).expand(batch, n, n, d),           # output buffer o
            pooled.reshape(batch, 1, 1, d).expand(batch, n, n, d),
            self.diagonal.expand(batch, n, n, 1),
        ], dim=3)
        return self.pair(pairs).reshape(batch, n * n)            # row-major: index = i * 5 + o

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        buffers, context = self._embed(features)
        return self.critic(torch.cat([buffers.mean(dim=1), context], dim=1))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)


class BufferPairPolicy(MaskableActorCriticPolicy):
    """MaskablePPO policy whose body is a BufferPairNetwork (use `policy_kwargs={"embed_dim": ...}`)."""

    def __init__(self, *args, embed_dim: int = 128, **kwargs) -> None:
        self.embed_dim = embed_dim
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BufferPairNetwork(self.features_dim, self.embed_dim, self.activation_fn)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        self.action_net = nn.Identity()  # the network already outputs one logit per action
        output = self.mlp_extractor.pair[-1]  # small initial logits: a near-uniform starting policy
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
