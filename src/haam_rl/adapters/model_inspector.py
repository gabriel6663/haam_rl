"""Read-only introspection of a trained policy for the web inspector: weights, activations, saliency.

SB3's MlpPolicy has two separate MLPs (actor and critic) on top of a flatten layer; this module
walks both and replays the forward pass layer by layer so every intermediate value can be shown.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
from torch import nn

from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.domain.model import ACTION_COUNT, PlantState, decode_action

_ACTIVATIONS = {nn.Tanh: "tanh", nn.ReLU: "relu", nn.ELU: "elu", nn.LeakyReLU: "leaky_relu", nn.SiLU: "silu"}
_FUNCTIONS = {"tanh": torch.tanh, "relu": torch.relu, "elu": nn.functional.elu,
              "leaky_relu": nn.functional.leaky_relu, "silu": nn.functional.silu}


def action_labels() -> list[str]:
    return [f"in B{i + 1} / out B{o + 1}" for i, o in map(decode_action, range(ACTION_COUNT))]


def _branch_layers(sequential: nn.Sequential, head: nn.Linear, head_kind: str) -> list[tuple[nn.Linear, str]]:
    """[(linear layer, activation applied after it)], ending with the output head."""
    modules = list(sequential)
    layers = []
    for index, module in enumerate(modules):
        if isinstance(module, nn.Linear):
            following = modules[index + 1] if index + 1 < len(modules) else None
            layers.append((module, _ACTIVATIONS.get(type(following), "identity")))
    return [*layers, (head, head_kind)]


def _stats(values: np.ndarray) -> dict[str, float]:
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()),
            "max": float(values.max()), "l2": float(np.linalg.norm(values))}


class PolicyInspector:
    """Works with any trained policy; the layer-by-layer view is available for MLP policies only.

    kind "mlp": SB3 MlpPolicy, two plain MLPs (actor and critic) that the web draws as a network.
    kind "structured": any other body (e.g. BufferPairPolicy); the web shows the decision,
    the saliency and every weight matrix, but no layer-by-layer drawing.
    """

    def __init__(self, policy: SB3Policy) -> None:
        self.policy = policy
        sb3 = policy.model.policy
        self.device = sb3.device
        self.kind = "mlp" if hasattr(sb3.mlp_extractor, "policy_net") else "structured"
        self.branches = {} if self.kind != "mlp" else {
            "policy": _branch_layers(sb3.mlp_extractor.policy_net, sb3.action_net, "logits"),
            "value": _branch_layers(sb3.mlp_extractor.value_net, sb3.value_net, "value"),
        }

    def architecture(self) -> dict:
        encoder, config = self.policy.encoder, self.policy.config
        branches = {
            name: [
                {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "activation": activation,
                    "weight": layer.weight.detach().cpu().numpy().round(5).tolist(),
                    "bias": layer.bias.detach().cpu().numpy().round(5).tolist(),
                    "weight_stats": _stats(layer.weight.detach().cpu().numpy()),
                }
                for layer, activation in layers
            ]
            for name, layers in self.branches.items()
        }
        matrices = [
            {"name": name, "shape": list(weight.shape), "weight": weight.detach().cpu().numpy().round(5).tolist(),
             "weight_stats": _stats(weight.detach().cpu().numpy())}
            for name, weight in self.policy.model.policy.named_parameters() if weight.ndim == 2
        ]
        return {
            "name": self.policy.name,
            "kind": self.kind,
            "config": config.to_dict(),
            "input_size": encoder.size,
            "input_groups": [asdict(g) for g in encoder.groups],
            "input_labels": encoder.raw_labels(),
            "action_labels": action_labels(),
            "parameter_count": int(sum(p.numel() for p in self.policy.model.policy.parameters())),
            "branches": branches,
            "matrices": [] if self.kind == "mlp" else matrices,
        }

    def _forward(self, layers: list[tuple[nn.Linear, str]], x: torch.Tensor) -> tuple[list[dict], torch.Tensor]:
        trace = []
        for layer, activation in layers:
            pre = layer(x)
            post = _FUNCTIONS[activation](pre) if activation in _FUNCTIONS else pre
            trace.append({"pre": pre, "post": post})
            x = post
        return trace, x

    def _logits_and_value(self, x: torch.Tensor) -> tuple[list[dict], list[dict], torch.Tensor, torch.Tensor]:
        if self.kind == "mlp":
            policy_trace, logits = self._forward(self.branches["policy"], x)
            with torch.no_grad():
                value_trace, value = self._forward(self.branches["value"], x)
            return policy_trace, value_trace, logits, value
        sb3 = self.policy.model.policy
        features = sb3.extract_features(x, sb3.pi_features_extractor)
        logits = sb3.action_net(sb3.mlp_extractor.forward_actor(features))
        with torch.no_grad():
            value = sb3.value_net(sb3.mlp_extractor.forward_critic(sb3.extract_features(x, sb3.vf_features_extractor)))
        return [], [], logits, value

    def inspect(self, state: PlantState) -> dict:
        mask = self.policy.mask(state)
        observation = self.policy.encoder.encode(state)
        x = torch.as_tensor(observation, device=self.device).reshape(1, -1).requires_grad_(True)
        policy_trace, value_trace, logits, value = self._logits_and_value(x)

        mask_t = torch.as_tensor(mask, device=self.device).reshape(1, -1)
        probabilities = torch.softmax(logits.masked_fill(~mask_t, float("-inf")), dim=-1)
        chosen = int(probabilities.argmax().item())

        # Saliency: gradient of the chosen action's logit w.r.t. each input, and gradient x input.
        logits[0, chosen].backward()
        gradient = x.grad.detach().cpu().numpy().ravel()
        attribution = gradient * observation

        def dump(trace: list[dict]) -> list[dict]:
            return [{k: v.detach().cpu().numpy().ravel().round(5).tolist() for k, v in t.items()} for t in trace]

        groups = self.policy.encoder.groups
        return {
            "model_action": chosen,
            "input": observation.round(5).tolist(),
            "positions": self.policy.encoder.positions(state).tolist(),
            "policy_layers": dump(policy_trace),
            "value_layers": dump(value_trace),
            "logits": logits.detach().cpu().numpy().ravel().round(5).tolist(),
            "unmasked_probabilities": torch.softmax(logits, dim=-1).detach().cpu().numpy().ravel().round(6).tolist(),
            "model_probabilities": probabilities.detach().cpu().numpy().ravel().round(6).tolist(),
            "model_mask": mask.tolist(),
            "value": float(value.item()),
            "saliency": gradient.round(6).tolist(),
            "attribution": attribution.round(6).tolist(),
            "group_attribution": [float(attribution[g.start:g.stop].sum()) for g in groups],
            "group_saliency": [float(np.abs(gradient[g.start:g.stop]).max()) for g in groups],
        }
