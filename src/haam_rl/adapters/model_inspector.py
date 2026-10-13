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


class ValueInspector:
    """Introspection of one value network (a judge of the value-guided planner).

    The network reads each buffer with the same encoder (43 inputs: 5 slots one-hot, the last painted
    color one-hot and the occupancy), pools the 5 buffer embeddings (mean and max) and maps them to
    V(s), the expected color changes among the next vehicles. The web draws the encoder for any
    buffer and the head; activations and saliency come from a replayed forward pass.
    """

    kind = "value"

    def __init__(self, name: str, network: nn.Module) -> None:
        from haam_rl.adapters.value_search import _CODE

        self.name, self.network, self.code = name, network.eval(), _CODE
        self.encoder = _branch_layers(network.buffer[:-2], network.buffer[-2], "relu")
        self.head = _branch_layers(network.head[:-1], network.head[-1], "identity")

    def _input_groups(self) -> list[dict]:
        code, groups = self.code, []
        for slot in range(5):
            groups.append({"label": f"slot {slot + 1}", "kind": "color", "start": slot * code, "stop": (slot + 1) * code})
        groups.append({"label": "último", "kind": "color", "start": 5 * code, "stop": 6 * code})
        groups.append({"label": "ocupación", "kind": "scalar", "start": 6 * code, "stop": 6 * code + 1})
        return groups

    def _input_labels(self) -> list[str]:
        colors = ["∅"] + [f"c{c}" for c in range(1, self.code)]
        labels = [f"slot{s + 1}:{c}" for s in range(5) for c in colors]
        return labels + [f"último:{c}" for c in colors] + ["ocupación"]

    def architecture(self) -> dict:
        def describe(layers: list[tuple[nn.Linear, str]]) -> list[dict]:
            return [{"in_features": layer.in_features, "out_features": layer.out_features, "activation": activation,
                     "weight": layer.weight.detach().numpy().round(5).tolist(),
                     "bias": layer.bias.detach().numpy().round(5).tolist(),
                     "weight_stats": _stats(layer.weight.detach().numpy())} for layer, activation in layers]

        hidden = self.head[0][0].in_features // 2
        pooled = [{"label": "media", "kind": "pool", "start": 0, "stop": hidden},
                  {"label": "máximo", "kind": "pool", "start": hidden, "stop": 2 * hidden}]
        names = ["Codificador de buffer", "Codificador de buffer", "Cabeza", "Cabeza"]
        sources = [self._input_labels(), None, [f"media n{i}" for i in range(hidden)] + [f"máx n{i}" for i in range(hidden)], None]
        layers = [layer for layer, _ in self.encoder] + [layer for layer, _ in self.head]
        matrices = [{"name": f"{names[k]} · capa {k % 2 + 1}", "shape": list(layer.weight.shape),
                     "weight": layer.weight.detach().numpy().round(5).tolist(),
                     "weight_stats": _stats(layer.weight.detach().numpy()), "src_labels": sources[k]}
                    for k, layer in enumerate(layers)]
        return {
            "name": self.name, "kind": self.kind, "config": None,
            "parameter_count": int(sum(p.numel() for p in self.network.parameters())),
            "branches": {"buffer": describe(self.encoder), "head": describe(self.head)},
            "branch_groups": {"buffer": self._input_groups(), "head": pooled},
            "attr_groups": [{"label": "entrante (no lo ve)", "kind": "color"}, {"label": "último", "kind": "color"}] +
                           [{"label": f"B{b + 1} slot {s + 1}", "kind": "color"} for b in range(5) for s in range(5)],
            "action_labels": action_labels(),
            "matrices": matrices,
        }

    def value(self, state: PlantState) -> float:
        from haam_rl.adapters.value_search import encode_leaves

        with torch.no_grad():
            return float(self.network(torch.as_tensor(encode_leaves([state]))).item())

    def inspect(self, state: PlantState) -> dict:
        from haam_rl.adapters.value_search import encode_leaves

        x = torch.as_tensor(encode_leaves([state])).requires_grad_(True)  # (1, 5 buffers, 43)
        trace, h = [], x
        for layer, activation in self.encoder:
            pre = layer(h)
            post = _FUNCTIONS[activation](pre) if activation in _FUNCTIONS else pre
            trace.append({"pre": pre, "post": post})
            h = post
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=1)
        head_trace, out = [], pooled
        for layer, activation in self.head:
            pre = layer(out)
            post = _FUNCTIONS[activation](pre) if activation in _FUNCTIONS else pre
            head_trace.append({"pre": pre, "post": post})
            out = post
        out.sum().backward()
        gradient = x.grad.detach().numpy()[0]           # (5, 43)
        attribution = gradient * x.detach().numpy()[0]
        code = self.code
        last = slice(5 * code, 6 * code)
        group_attribution = [0.0, float(attribution[:, last].sum())]
        group_saliency = [0.0, float(np.abs(gradient[:, last]).max())]
        for b in range(5):
            for s in range(5):
                part = slice(s * code, (s + 1) * code)
                group_attribution.append(float(attribution[b, part].sum()))
                group_saliency.append(float(np.abs(gradient[b, part]).max()))

        def rows(t: torch.Tensor) -> list:
            return t.detach().numpy()[0].round(5).tolist()

        return {
            "kind": self.kind,
            "value": float(out.item()),
            "positions": state.as_vector().tolist(),
            "buffer_inputs": rows(x),
            "buffer_layers": [[{k: v.detach().numpy()[0, b].round(5).tolist() for k, v in t.items()} for t in trace]
                              for b in range(5)],
            "pooled": rows(pooled),
            "head_layers": [{k: v.detach().numpy().ravel().round(5).tolist() for k, v in t.items()} for t in head_trace],
            "group_attribution": group_attribution,
            "group_saliency": group_saliency,
            "occupancy_attribution": float(attribution[:, 6 * code].sum()),
        }
