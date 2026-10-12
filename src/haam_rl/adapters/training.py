"""MaskablePPO training on the Python twin, with model selection by validation color changes."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch import nn

from haam_rl.adapters.equivariant import BufferPairPolicy
from haam_rl.adapters.gym_env import PaintShopEnv
from haam_rl.adapters.runner import run_scenarios
from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.config import ExperimentConfig, config_path_for
from haam_rl.domain.scenarios import VALIDATION_SEEDS

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


def policy_spec(config: ExperimentConfig) -> tuple[object, dict]:
    """SB3 policy class and keyword arguments for the network the config asks for."""
    t = config.training
    if t.policy == "mlp":
        return "MlpPolicy", {"net_arch": list(t.net_arch), "activation_fn": ACTIVATIONS[t.activation]}
    if t.policy == "buffer_pair":
        return BufferPairPolicy, {"embed_dim": t.embed_dim, "activation_fn": ACTIVATIONS[t.activation]}
    raise ValueError(f"Unknown policy {t.policy!r}")


class ColorChangeEvalCallback(BaseCallback):
    """Every `eval_freq` steps, run the greedy policy on the validation scenarios.

    The checkpoint with the fewest mean color changes is kept as `best_model.zip`. Mean episode
    reward (what SB3's own callback tracks) is a proxy; color changes are the actual metric.
    """

    def __init__(self, config: ExperimentConfig, run_dir: Path, eval_freq: int) -> None:
        super().__init__()
        self.config = config
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.best = float("inf")
        self.history: list[tuple[int, float]] = []
        self._last_eval = -eval_freq

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            self.run_evaluation()
        return True

    def _on_training_end(self) -> None:
        self.run_evaluation()

    def run_evaluation(self) -> None:
        policy = SB3Policy(self.model, self.config)
        results = run_scenarios(self.config, policy, VALIDATION_SEEDS)
        mean = float(np.mean([r.color_changes for r in results]))
        self.history.append((self.num_timesteps, mean))
        self.logger.record("eval/color_changes", mean)
        if mean < self.best:
            self.best = mean
            save_model(self.model, self.config, self.run_dir / "best_model.zip")
        if self.verbose:
            print(f"[{self.num_timesteps:>8}] validation color changes {mean:.2f} (best {self.best:.2f})", flush=True)


def save_model(model: MaskablePPO, config: ExperimentConfig, path: Path) -> None:
    model.save(path)
    config.save(config_path_for(path))


def _learning_rate(t) -> float | object:
    if t.lr_schedule == "constant":
        return t.learning_rate
    if t.lr_schedule == "linear":
        return lambda progress_remaining: t.learning_rate * progress_remaining  # SB3: 1 -> 0
    raise ValueError(f"Unknown lr_schedule {t.lr_schedule!r}")


def train(config: ExperimentConfig, seed: int, run_dir: Path, vec_env: str = "dummy",
          tensorboard: bool = False, verbose: int = 1) -> ColorChangeEvalCallback:
    run_dir.mkdir(parents=True, exist_ok=True)
    t = config.training
    env = make_vec_env(
        PaintShopEnv, n_envs=t.n_envs, seed=seed, env_kwargs={"config": config},
        vec_env_cls=SubprocVecEnv if vec_env == "subproc" and t.n_envs > 1 else DummyVecEnv,
    )
    policy_class, policy_kwargs = policy_spec(config)
    model = MaskablePPO(
        policy_class, env, seed=seed, device=t.device, verbose=0,
        learning_rate=_learning_rate(t), n_steps=t.n_steps, batch_size=t.batch_size, n_epochs=t.n_epochs,
        gamma=t.gamma, gae_lambda=t.gae_lambda, clip_range=t.clip_range, ent_coef=t.ent_coef,
        vf_coef=t.vf_coef, max_grad_norm=t.max_grad_norm, target_kl=t.target_kl,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_dir / "tensorboard") if tensorboard else None,
    )
    if t.init_from:
        # Start from pretrained weights (e.g. a policy distilled from the planner); same architecture.
        pretrained = MaskablePPO.load(t.init_from, device=t.device)
        model.policy.load_state_dict(pretrained.policy.state_dict())
        torch.manual_seed(seed)  # loading built a second model and consumed random numbers: re-seed for reproducibility
    callback = ColorChangeEvalCallback(config, run_dir, t.eval_freq)
    callback.verbose = verbose
    try:
        model.learn(total_timesteps=t.total_timesteps, callback=callback)
    finally:
        env.close()
    save_model(model, config, run_dir / "final_model.zip")
    with open(run_dir / "learning_curve.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timesteps", "validation_color_changes"])
        writer.writerows(callback.history)
    return callback
