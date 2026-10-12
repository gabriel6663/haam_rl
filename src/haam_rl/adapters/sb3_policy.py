"""A trained MaskablePPO model exposed as an application `Policy`."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from haam_rl.adapters.observation import ObservationEncoder
from haam_rl.application.ports import Decision
from haam_rl.config import ExperimentConfig, config_path_for, legacy_config
from haam_rl.domain.heuristics import action_mask
from haam_rl.domain.model import PlantState

# torch.distributions checks on every construction that probabilities are a valid simplex. SB3
# builds one distribution per environment step, and those checks took ~15 % of training time.
# The inputs come from a softmax, so the checks can never fail here.
torch.distributions.Distribution.set_default_validate_args(False)


class SB3Policy:
    """Encodes the state the way the model was trained, applies the model's own mask, acts greedily.

    Greedy masked action = argmax of the logits over the allowed actions, exactly what
    `MaskablePPO.predict(..., deterministic=True)` returns, without building a distribution object.
    """

    def __init__(self, model: MaskablePPO, config: ExperimentConfig, name: str = "model",
                 path: Path | None = None) -> None:
        self.model = model
        self.path = path  # set when loaded from disk; lets the policy be sent to worker processes
        self.config = config
        self.name = name
        self.fill_target = config.plant.fill_target
        self.encoder = ObservationEncoder(config.observation, self.fill_target)
        expected = model.observation_space.shape
        if expected != (self.encoder.size,):
            raise ValueError(f"{name}: model expects input {expected}, config encodes {self.encoder.size} features")

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> SB3Policy:
        path = Path(path)
        sidecar = config_path_for(path)
        config = ExperimentConfig.load(sidecar) if sidecar.exists() else legacy_config()
        return cls(MaskablePPO.load(path, device=device), config, name=path.stem, path=path)

    def __getstate__(self) -> dict:
        """Pickle by file path: worker processes reload the model instead of copying torch objects."""
        if self.path is None:
            raise TypeError("Only policies loaded from a file can be sent to other processes")
        return {"path": str(self.path), "name": self.name}

    def __setstate__(self, state: dict) -> None:
        torch.set_num_threads(1)
        loaded = SB3Policy.load(Path(state["path"]))
        self.__dict__.update(loaded.__dict__)
        self.name = state["name"]

    def mask(self, state: PlantState) -> np.ndarray:
        return action_mask(state, self.fill_target, self.config.mask_mode)

    def logits(self, observations: np.ndarray) -> torch.Tensor:
        """Actor logits for a batch of encoded observations, shape (batch, 25)."""
        policy = self.model.policy
        with torch.no_grad():
            obs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
            features = policy.extract_features(obs, policy.pi_features_extractor)
            return policy.action_net(policy.mlp_extractor.forward_actor(features))

    def batch_probabilities(self, states: Sequence[PlantState]) -> tuple[np.ndarray, np.ndarray]:
        """Masked action probabilities and masks for many states with a single forward pass."""
        masks = np.stack([self.mask(s) for s in states])
        logits = self.logits(np.stack([self.encoder.encode(s) for s in states]))
        allowed = torch.as_tensor(masks, device=logits.device)
        probabilities = torch.softmax(logits.masked_fill(~allowed, float("-inf")), dim=-1)
        return probabilities.cpu().numpy(), masks

    def decide_batch(self, states: Sequence[PlantState]) -> list[Decision]:
        probabilities, masks = self.batch_probabilities(states)
        return [Decision(int(np.argmax(p)), p, m) for p, m in zip(probabilities, masks)]

    def decide(self, state: PlantState) -> Decision:
        return self.decide_batch([state])[0]
