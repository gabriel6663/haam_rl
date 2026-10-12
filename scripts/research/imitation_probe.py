"""Imitation probe: which network can best reproduce the planner's decisions on unseen scenarios?

    python scripts/research/imitation_probe.py --episodes 600 --workers 4

1. The HAAM-masked planner (look-ahead 10, width 32) plays `--episodes` training scenarios; every
   decision is recorded (one-hot observation with look-ahead 10, mask, planner action).
2. Each architecture is trained with the same optimiser and epochs on 80 % of the scenarios.
3. Reported: accuracy on the remaining 20 % of scenarios, counting only decisions with more than
   one allowed action (forced decisions are trivially right and would hide the differences).

This isolates representation and capacity from the difficulties of reinforcement learning.
Results: artifacts/research/imitation_probe.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from haam_rl.adapters.distillation import _teacher_episodes
from haam_rl.adapters.equivariant import BufferPairNetwork
from haam_rl.adapters.observation import ObservationEncoder
from haam_rl.application.use_cases import LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.heuristics import action_mask

ARCHITECTURES = [  # (kind, hidden sizes or embedding, activation, epochs, learning rate)
    ("mlp", [64, 64], "tanh", 40, 1e-3),
    ("mlp", [256, 256], "tanh", 40, 1e-3),
    ("mlp", [64, 64, 64], "tanh", 40, 1e-3),
    ("mlp", [256, 256, 256], "tanh", 40, 1e-3),
    ("mlp", [512, 512, 512], "tanh", 40, 5e-4),
    ("mlp", [256, 256, 256], "relu", 40, 1e-3),
    ("bufferpair", [64], "relu", 40, 1e-3),
    ("bufferpair", [128], "relu", 40, 1e-3),
]


def build_dataset(path: Path, episodes: int, workers: int) -> None:
    config = ExperimentConfig.load(Path("configs/onehot_lookahead.json"))
    teacher = LookaheadPlannerPolicy(25, 10, 32, mask_mode=config.mask_mode)
    seeds = list(range(5_000_000, 5_000_000 + episodes))
    chunks = workers * 4
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_teacher_episodes, [config] * chunks, [teacher] * chunks, [seeds[k::chunks] for k in range(chunks)]))
    encoder = ObservationEncoder(config.observation, 25)
    states = [s for p in parts for s in p.states]
    np.savez_compressed(path, obs=np.stack([encoder.encode(s) for s in states]).astype(np.float32),
                        masks=np.stack([action_mask(s, 25, config.mask_mode) for s in states]),
                        actions=np.concatenate([p.actions for p in parts]), episode=np.concatenate([p.episode for p in parts]))


def build_network(kind: str, sizes: list[int], activation: str, inputs: int) -> nn.Module:
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    if kind == "mlp":
        layers, previous = [], inputs
        for size in sizes:
            layers += [nn.Linear(previous, size), act()]
            previous = size
        return nn.Sequential(*layers, nn.Linear(previous, 25))
    network = BufferPairNetwork(inputs, sizes[0], act)

    class Actor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = network

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network.forward_actor(x)

    return Actor()


def probe(data_path: str, spec: tuple) -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(0)
    kind, sizes, activation, epochs, lr = spec
    data = np.load(data_path)
    obs, masks, actions = (torch.as_tensor(data[k]) for k in ("obs", "masks", "actions"))
    episodes = np.unique(data["episode"])
    held_out_episodes = np.random.default_rng(0).choice(episodes, len(episodes) // 5, replace=False)
    test = torch.as_tensor(np.isin(data["episode"], held_out_episodes))
    train_rows = torch.nonzero(~test).squeeze(1)
    model = build_network(kind, sizes, activation, obs.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    started, curve = time.perf_counter(), []

    def accuracy(selection: torch.Tensor) -> float:
        with torch.no_grad():
            logits = model(obs[selection]).masked_fill(~masks[selection], float("-inf"))
        hits = logits.argmax(1) == actions[selection]
        choice = masks[selection].sum(1) > 1
        return round(float(hits[choice].float().mean()), 4)

    for epoch in range(epochs):
        model.train()
        order = train_rows[torch.randperm(len(train_rows))]
        for start in range(0, len(order), 512):
            rows = order[start:start + 512]
            logits = model(obs[rows]).masked_fill(~masks[rows], float("-inf"))
            loss = nn.functional.cross_entropy(logits, actions[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            curve.append((epoch + 1, accuracy(~test), accuracy(test)))
    return {"kind": kind, "arch": sizes, "act": activation, "params": sum(p.numel() for p in model.parameters()),
            "minutes": round((time.perf_counter() - started) / 60, 1), "curve": curve}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data", type=Path, default=Path("artifacts/research/teacher_haam.npz"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/imitation_probe.jsonl"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        build_dataset(args.data, args.episodes, args.workers)
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "w", encoding="utf-8") as handle:
        for result in pool.map(probe, [str(args.data)] * len(ARCHITECTURES), ARCHITECTURES):
            print(json.dumps(result), flush=True)
            handle.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
