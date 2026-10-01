from __future__ import annotations

import argparse
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor

from haam_rl.adapters.gym_env import PaintShopEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vehicles", type=int, default=100)
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging (requires `pip install tensorboard`)")
    args = parser.parse_args()

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    train_env = Monitor(PaintShopEnv(args.vehicles))
    eval_env = Monitor(PaintShopEnv(args.vehicles))
    device = "cpu"
    model = MaskablePPO(
        "MlpPolicy", train_env, device=device, seed=args.seed, verbose=1,
        learning_rate=1e-4, n_steps=2048, batch_size=256, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=str(artifacts / "tensorboard") if args.tensorboard else None,
    )
    callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(artifacts / "best"),
        eval_freq=10_000,
        n_eval_episodes=30,
        deterministic=True,
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(artifacts / "maskable_ppo_haam")
    print(f"Saved model using {device}: {artifacts / 'maskable_ppo_haam.zip'}")


if __name__ == "__main__":
    main()
