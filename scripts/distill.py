"""Distill the look-ahead planner into a neural policy (DAgger), optionally per seed.

    python scripts/distill.py --config configs/onehot_lookahead.json --seeds 0 1 2 --workers 4

The student uses the config's observation, mask and network; the teacher is the planner with the
same mask and look-ahead. Runs go to artifacts/runs/<config name>_dagger/seed_<n>/ and can be
evaluated like any trained model, or fine-tuned with PPO through `training.init_from`.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import torch

from haam_rl.adapters.distillation import DistillationSettings, distill
from haam_rl.config import ExperimentConfig, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE",
                        help="override any config field, e.g. training.policy=buffer_pair training.embed_dim=128")
    parser.add_argument("--name", help="experiment name of the student (default: <config name>_dagger)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    for f in fields(DistillationSettings):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    args = parser.parse_args()

    torch.set_num_threads(1)
    config = ExperimentConfig.load(args.config)
    config = config.with_overrides(**parse_overrides(args.set)) if args.set else config
    config = config.with_overrides(name=args.name or f"{config.name}_dagger")
    settings = DistillationSettings(**{f.name: getattr(args, f.name) for f in fields(DistillationSettings)})
    for seed in args.seeds:
        run_dir = args.out / config.name / f"seed_{seed}"
        print(f"=== {config.name} seed {seed} -> {run_dir}", flush=True)
        started = time.perf_counter()
        best = distill(config, settings, seed, run_dir)
        print(f"=== done in {(time.perf_counter() - started) / 60:.1f} min, best validation color changes {best:.2f}", flush=True)


if __name__ == "__main__":
    main()
