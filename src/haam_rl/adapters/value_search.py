"""Search guided by a learned value function ("planner + network", in the spirit of AlphaZero).

The beam planner ranks partial plans by color changes so far plus an estimate of what the state
they leave behind will cost later. By default that estimate is 0.2 x fragmentation, a hand-made
proxy. Here a network learns the estimate from experience instead:

    V(state) ~ color changes among the next `window` painted vehicles, if the planner keeps acting

The input is only what a leaf of the search knows: the buffers and the last painted color (the
arrivals after the known horizon are unknown). The network is invariant to the order of the
buffers. Training data are the planner's own trajectories, so V evaluates the policy that will
actually act; the process can be repeated with the improved planner (policy iteration).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from haam_rl.adapters.runner import evaluation_twin
from haam_rl.application.ports import Decision, Policy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.dynamics import is_terminal
from haam_rl.domain.heuristics import MaskMode, action_mask
from haam_rl.domain.model import (
    ACTION_COUNT,
    BUFFER_CAPACITY,
    BUFFER_COUNT,
    COLOR_COUNT,
    TOTAL_SLOTS,
    PlantState,
)
from haam_rl.domain.planning import best_plan

_CODE = COLOR_COUNT + 1


def encode_leaves(states: Sequence[PlantState]) -> np.ndarray:
    """(n, 5, 5*7 + 7 + 1): per buffer its slots one-hot, the last painted color and the occupancy."""
    n = len(states)
    colors = np.zeros((n, BUFFER_COUNT, BUFFER_CAPACITY), dtype=np.int64)
    last = np.empty(n, dtype=np.int64)
    occupancy = np.empty(n, dtype=np.float32)
    for k, state in enumerate(states):
        for b, buffer in enumerate(state.buffers):
            colors[k, b, :len(buffer)] = buffer
        last[k] = state.last_painted_color
        occupancy[k] = state.occupancy / TOTAL_SLOTS
    eye = np.eye(_CODE, dtype=np.float32)
    slots = eye[colors].reshape(n, BUFFER_COUNT, BUFFER_CAPACITY * _CODE)
    shared = np.concatenate([eye[last], occupancy[:, None]], axis=1)  # same for every buffer
    return np.concatenate([slots, np.repeat(shared[:, None, :], BUFFER_COUNT, axis=1)], axis=2)


class ValueNetwork(nn.Module):
    """Permutation-invariant regression: one shared MLP per buffer, mean + max pooling, MLP head."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        size = BUFFER_CAPACITY * _CODE + _CODE + 1
        self.buffer = nn.Sequential(nn.Linear(size, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, leaves: torch.Tensor) -> torch.Tensor:
        embedded = self.buffer(leaves)
        pooled = torch.cat([embedded.mean(dim=1), embedded.max(dim=1).values], dim=1)
        return self.head(pooled).squeeze(-1)


class LeafValue:
    """Batched, cached V(state) for the planner (`LeafEstimate`).

    Given several networks it averages them: an ensemble of judges has less variance than one.
    """

    def __init__(self, network: ValueNetwork | Sequence[ValueNetwork], scale: float = 1.0,
                 cache_size: int = 200_000) -> None:
        self.networks = [n.eval() for n in (network if isinstance(network, Sequence) else [network])]
        self.scale = scale
        self.cache: OrderedDict[tuple, float] = OrderedDict()
        self.cache_size = cache_size

    def __call__(self, states: Sequence[PlantState]) -> list[float]:
        """Estimated future color changes; exactly 0 for a finished episode (nothing left to paint).

        Values are rounded to 1e-5: float32 results can differ in the last digit with the batch
        size, and rounding keeps decisions independent of how the work was batched or split.
        """
        keys = [(s.buffers, s.last_painted_color) for s in states]
        missing = [k for k, key in enumerate(keys) if key not in self.cache and not is_terminal(states[k])]
        if missing:
            leaves = torch.as_tensor(encode_leaves([states[k] for k in missing]))
            with torch.no_grad():
                values = torch.stack([network(leaves) for network in self.networks]).mean(dim=0).numpy()
            for k, value in zip(missing, values):
                self.cache[keys[k]] = round(float(value), 5)
        result = []
        for state, key in zip(states, keys):
            if is_terminal(state):
                result.append(0.0)
            else:
                self.cache.move_to_end(key)  # least-recently-used eviction
                result.append(self.scale * self.cache[key])
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def __getstate__(self) -> dict:  # do not ship the cache to worker processes
        return {**self.__dict__, "cache": OrderedDict()}


@dataclass(frozen=True)
class ValueSettings:
    window: int = 30          # V predicts the color changes among the next `window` painted vehicles
    scenarios: int = 1200     # training scenarios played by the data policy
    epochs: int = 30
    hidden: int = 128
    learning_rate: float = 1e-3
    batch_size: int = 1024
    workers: int = 4


def future_changes(marks: list[int], window: int) -> list[float | None]:
    """Target of the state before each decision: changes among the next `window` paints, starting
    with that decision's own paint. None when the episode ends first.

    `marks` holds one entry per decision: -1 if nothing was painted, else 1 (color changed) or 0.
    """
    paint_steps = [k for k, mark in enumerate(marks) if mark >= 0]
    cumulative = np.concatenate([[0], np.cumsum([marks[k] for k in paint_steps])])
    targets: list[float | None] = []
    first = 0  # index of the first paint at or after decision k
    for k in range(len(marks)):
        while first < len(paint_steps) and paint_steps[first] < k:
            first += 1
        end = first + window
        targets.append(float(cumulative[end] - cumulative[first]) if end <= len(paint_steps) else None)
    return targets


def _play(config: ExperimentConfig, policy: Policy, seeds: list[int], window: int) -> tuple[list[PlantState], list[float]]:
    states, targets = [], []
    for seed in seeds:
        twin = evaluation_twin(config)  # accepts any physically valid action, as the planner needs
        state, done, visited, marks = twin.reset(seed), False, [], []
        while not done:
            visited.append(state)
            state, _, done, info = twin.step(policy.decide(state).action)
            marks.append(int(info.color_changed) if info.painted_color else -1)
        for s, target in zip(visited, future_changes(marks, window)):
            if target is not None:
                states.append(s)
                targets.append(target)
    return states, targets


def collect(config: ExperimentConfig, policy: Policy, seeds: list[int], window: int, workers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunks = [seeds[k::workers * 4] for k in range(workers * 4)]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_play, [config] * len(chunks), [policy] * len(chunks), chunks, [window] * len(chunks)))
    states = [s for p in parts for s in p[0]]
    targets = np.asarray([t for p in parts for t in p[1]], dtype=np.float32)
    episode = np.concatenate([np.full(len(p[1]), k) for k, p in enumerate(parts)])  # chunk id, for a held-out split
    return encode_leaves(states), targets, episode


def train_value(leaves: np.ndarray, targets: np.ndarray, groups: np.ndarray, settings: ValueSettings,
                seed: int = 0) -> tuple[ValueNetwork, dict[str, float]]:
    """Fit V with a held-out split by chunk of scenarios; report R^2 on the held-out part."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    held_out = np.isin(groups, rng.choice(np.unique(groups), max(1, len(np.unique(groups)) // 5), replace=False))
    x, y = torch.as_tensor(leaves), torch.as_tensor(targets)
    train_rows = torch.as_tensor(np.flatnonzero(~held_out))
    network = ValueNetwork(settings.hidden)
    optimizer = torch.optim.Adam(network.parameters(), lr=settings.learning_rate)
    for _ in range(settings.epochs):
        network.train()
        train_rows = train_rows[torch.randperm(len(train_rows))]
        for start in range(0, len(train_rows), settings.batch_size):
            rows = train_rows[start:start + settings.batch_size]
            loss = nn.functional.mse_loss(network(x[rows]), y[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    network.eval()
    with torch.no_grad():
        test = torch.as_tensor(np.flatnonzero(held_out))
        predicted = network(x[test])
        residual = float(((predicted - y[test]) ** 2).mean())
        variance = float(y[test].var())
    return network, {"held_out_mse": residual, "held_out_r2": 1 - residual / variance, "samples": len(y)}


def save_value(network: ValueNetwork, settings: ValueSettings, metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps({"settings": asdict(settings), "metrics": metrics}, indent=2),
                                         encoding="utf-8")


def load_value(path: Path) -> ValueNetwork:
    meta = json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    network = ValueNetwork(meta["settings"]["hidden"])
    network.load_state_dict(torch.load(path, map_location="cpu"))
    return network.eval()


class ValueGuidedPlannerPolicy:
    """The look-ahead planner with a learned leaf estimate: score = changes + scale x V(leaf)."""

    def __init__(self, value_path: Path | Sequence[Path], fill_target: int, horizon: int | None = 10,
                 width: int = 32, scale: float = 1.0, mask_mode: MaskMode = MaskMode.PHYSICAL) -> None:
        paths = [value_path] if isinstance(value_path, (str, Path)) else list(value_path)
        self.value_paths = tuple(Path(p) for p in paths)
        self.fill_target, self.horizon, self.width = fill_target, horizon, width
        self.scale, self.mask_mode = scale, MaskMode(mask_mode)
        self.estimate = LeafValue([load_value(p) for p in self.value_paths], scale)
        judges = "" if len(self.value_paths) == 1 else f",judges={len(self.value_paths)}"
        self.name = f"value_planner(h={horizon},w={width},scale={scale}{judges})"

    def __getstate__(self) -> dict:  # worker processes reload the network from its file
        return {k: v for k, v in self.__dict__.items() if k != "estimate"}

    def __setstate__(self, state: dict) -> None:
        torch.set_num_threads(1)
        self.__dict__.update(state)
        self.estimate = LeafValue([load_value(p) for p in self.value_paths], self.scale)

    def proposals(self, state: PlantState) -> dict[int, float]:
        score, action = best_plan(state, self.fill_target, self.horizon, self.width,
                                  mask_mode=self.mask_mode, estimate=self.estimate)
        return {action: score}

    def decide(self, state: PlantState) -> Decision:
        (action,) = self.proposals(state)
        probabilities = np.zeros(ACTION_COUNT)
        probabilities[action] = 1.0
        return Decision(action, probabilities, action_mask(state, self.fill_target, self.mask_mode))
