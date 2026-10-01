from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from haam_rl.adapters.gym_env import PaintShopEnv


def run_episode(env: PaintShopEnv, chooser, seed: int) -> int:
    env.reset(seed=seed)
    while True:
        mask = env.action_masks()
        action = chooser(env, mask)
        _, _, terminated, _, info = env.step(action)
        if terminated:
            return int(info["color_changes"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("artifacts/maskable_ppo_haam.zip"))
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--seed", type=int, default=10_000)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(
            f"Model not found: {args.model}. Train first with `python scripts/train.py --timesteps 300000 --seed 42`."
        )
    model = MaskablePPO.load(args.model, device="auto")
    seeds = range(args.seed, args.seed + args.scenarios)
    rl = [run_episode(PaintShopEnv(), lambda env, mask: int(model.predict(env.twin.state.as_vector(), action_masks=mask, deterministic=True)[0]), seed) for seed in seeds]
    # Deterministic HAAM heuristic: highest-priority valid action (top buffer tie-break).
    baseline = [run_episode(PaintShopEnv(), lambda _env, mask: int(np.flatnonzero(mask)[0]), seed) for seed in range(args.seed, args.seed + args.scenarios)]
    print(f"Cambios de color IA: {np.mean(rl):.2f} +/- {np.std(rl):.2f}")
    print(f"Cambios de color Heuristica: {np.mean(baseline):.2f} +/- {np.std(baseline):.2f}")
    print(f"Mejora: {(np.mean(baseline) - np.mean(rl)) / np.mean(baseline) * 100:.2f}%")


if __name__ == "__main__":
    main()
