"""Learn the planner's leaf value from played games, then plan with it.

    python scripts/train_value.py --out artifacts/value/v1.pt --workers 8
    python scripts/train_value.py --data-policy planner heuristic --scenarios 3000 --out artifacts/value/v4.pt

--data-policy: one or more of "planner" (look-ahead planner with the fragmentation estimate),
"heuristic", or the path of a value network (games played by the value-guided planner). With
several policies the scenarios are split evenly between them: the value network judges many
hypothetical, often poor, states during search, so diverse games teach it better than good ones.
Training scenarios use seeds far from validation and test seeds.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from haam_rl.adapters.value_search import (
    ValueGuidedPlannerPolicy,
    ValueSettings,
    collect,
    save_value,
    train_value,
)
from haam_rl.application.ports import Policy
from haam_rl.application.use_cases import HeuristicPolicy, LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig

VALUE_SEED_OFFSET = 7_000_000


def data_policy(name: str, fill_target: int, horizon: int, width: int, scale: float) -> Policy:
    if name == "planner":
        return LookaheadPlannerPolicy(fill_target, horizon, width)
    if name == "heuristic":
        return HeuristicPolicy(fill_target)
    return ValueGuidedPlannerPolicy(Path(name), fill_target, horizon, width, scale=scale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-policy", nargs="+", default=["planner"])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--data-scale", type=float, default=0.5, help="value scale of a value-guided data policy")
    parser.add_argument("--seed", type=int, default=0)
    for f in fields(ValueSettings):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(1)

    settings = ValueSettings(**{f.name: getattr(args, f.name) for f in fields(ValueSettings)})
    config = ExperimentConfig()
    first = VALUE_SEED_OFFSET + args.seed * 100_000
    share = settings.scenarios // len(args.data_policy)
    parts, names = [], []
    started = time.perf_counter()
    for k, name in enumerate(args.data_policy):
        policy = data_policy(name, config.plant.fill_target, args.horizon, args.width, args.data_scale)
        seeds = list(range(first + k * share, first + (k + 1) * share))
        leaves, targets, groups = collect(config, policy, seeds, settings.window, settings.workers)
        parts.append((leaves, targets, groups + k * 1_000))  # keep held-out groups distinct per policy
        names.append(policy.name)
        print(f"collected {len(targets)} states from {len(seeds)} games of {policy.name} "
              f"(mean target {targets.mean():.2f})", flush=True)
    leaves, targets, groups = (np.concatenate([p[i] for p in parts]) for i in range(3))
    print(f"data collected in {(time.perf_counter() - started) / 60:.1f} min", flush=True)
    network, metrics = train_value(leaves, targets, groups, settings, args.seed)
    metrics["data_policy"] = " + ".join(names)
    save_value(network, settings, metrics, args.out)
    print(f"held-out R2 {metrics['held_out_r2']:.3f}, MSE {metrics['held_out_mse']:.3f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
