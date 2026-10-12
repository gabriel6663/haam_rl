"""Experiment configuration, stored as JSON next to every trained model.

A model zip is only meaningful together with the plant it was trained on, how its observation was
encoded and which mask it was trained under; `ExperimentConfig` records all three.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from haam_rl.domain.heuristics import MaskMode
from haam_rl.domain.model import TOTAL_SLOTS
from haam_rl.domain.reward import RewardConfig
from haam_rl.domain.scenarios import PAPER_COLOR_PROBABILITIES


class Encoding(str, Enum):
    RAW = "raw"        # the paper: color ids as numbers (0..6)
    ONEHOT = "onehot"  # one 7-way indicator per slot


@dataclass(frozen=True)
class PlantConfig:
    vehicles: int = 100
    fill_target: int = TOTAL_SLOTS  # vehicles held before painting starts; 25 = full buffers, as in FlexSim
    color_probabilities: tuple[float, ...] = PAPER_COLOR_PROBABILITIES

    def __post_init__(self) -> None:
        if not 1 <= self.fill_target <= TOTAL_SLOTS:
            raise ValueError(f"fill_target must be in 1..{TOTAL_SLOTS}")
        if self.vehicles < 1:
            raise ValueError("vehicles must be positive")


@dataclass(frozen=True)
class ObservationConfig:
    encoding: Encoding = Encoding.ONEHOT
    lookahead: int = 0          # vehicles queued after the incoming one that the agent can see
    include_phase: bool = True  # only used by the one-hot encoding


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 500_000
    n_envs: int = 8
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"  # "constant", or "linear": decays to 0 at the end of training
    n_steps: int = 256          # per environment, so a rollout holds n_envs * n_steps transitions
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    policy: str = "mlp"         # "mlp" (SB3 MlpPolicy) or "buffer_pair" (see adapters/equivariant.py)
    net_arch: tuple[int, ...] = (64, 64)  # mlp: hidden layers of the actor and of the critic
    embed_dim: int = 128        # buffer_pair: width of the buffer and pair embeddings
    activation: str = "tanh"
    eval_freq: int = 20_000     # in environment steps (summed over all environments)
    device: str = "cpu"         # a small MLP is faster on CPU than on GPU
    init_from: str | None = None  # model zip whose policy weights initialise training (fine-tuning)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "improved"
    plant: PlantConfig = field(default_factory=PlantConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    mask_mode: MaskMode = MaskMode.HAAM
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        return _from_dict(cls, data)

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, **sections: dict[str, Any]) -> ExperimentConfig:
        """`config.with_overrides(training={"total_timesteps": 1000})` returns a modified copy."""
        data = self.to_dict()
        for section, values in sections.items():
            if isinstance(values, dict):
                data[section].update(values)
            else:
                data[section] = values
        return ExperimentConfig.from_dict(data)


def parse_overrides(assignments: list[str]) -> dict[str, Any]:
    """`["training.policy=buffer_pair", "name=x"]` -> keyword arguments for `with_overrides`.

    Values are read as JSON when possible (numbers, lists, null), otherwise kept as strings.
    """
    sections: dict[str, Any] = {}
    for assignment in assignments:
        key, equals, raw = assignment.partition("=")
        if not equals:
            raise ValueError(f"Expected section.field=value, got {assignment!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        section, _dot, field_name = key.partition(".")
        if field_name:
            sections.setdefault(section, {})[field_name] = value
        else:
            sections[section] = value
    return sections


def legacy_config() -> ExperimentConfig:
    """Models saved before configs existed: raw 27-value input, buffers kept at 5 vehicles."""
    return ExperimentConfig(
        name="legacy",
        plant=PlantConfig(fill_target=5),
        observation=ObservationConfig(encoding=Encoding.RAW, lookahead=0, include_phase=False),
    )


def config_path_for(model_path: Path) -> Path:
    return Path(model_path).with_suffix(".json")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value, kind = data[f.name], f.type if not isinstance(f.type, str) else _resolve(f.type)
        if is_dataclass(kind):
            value = _from_dict(kind, value)
        elif isinstance(kind, type) and issubclass(kind, Enum):
            value = kind(value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _resolve(type_name: str) -> Any:
    known = {
        "PlantConfig": PlantConfig, "ObservationConfig": ObservationConfig, "RewardConfig": RewardConfig,
        "TrainingConfig": TrainingConfig, "MaskMode": MaskMode, "Encoding": Encoding,
    }
    return known.get(type_name, object)
