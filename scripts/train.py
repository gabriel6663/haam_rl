"""Train one MaskablePPO model per seed for an experiment config.

    python scripts/train.py --config configs/onehot_lookahead.json --seeds 0 1 2 3 4

Each run writes artifacts/runs/<config name>/seed_<seed>/ with best_model.zip (lowest validation
color changes), final_model.zip, their .json configs and learning_curve.csv.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from haam_rl.adapters.training import train
from haam_rl.config import ExperimentConfig, parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="experiment JSON (see configs/)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--timesteps", type=int, help="override training.total_timesteps")
    parser.add_argument("--n-envs", type=int, help="override training.n_envs")
    parser.add_argument("--device", help="override training.device (cpu, cuda, auto)")
    parser.add_argument("--set", nargs="*", default=[], metavar="SECTION.FIELD=VALUE",
                        help="override any config field, e.g. training.policy=buffer_pair name=my_run")
    parser.add_argument("--out", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy",
                        help="dummy: one process (low memory); subproc: one process per environment")
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch CPU threads; 1 is fastest for a small MLP and avoids oversubscription "
                             "when several trainings run in parallel")
    parser.add_argument("--tensorboard", action="store_true", help="requires `pip install tensorboard`")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    config = ExperimentConfig.load(args.config)
    overrides = {k: v for k, v in {"total_timesteps": args.timesteps, "n_envs": args.n_envs, "device": args.device}.items() if v is not None}
    if overrides:
        config = config.with_overrides(training=overrides)
    if args.set:
        config = config.with_overrides(**parse_overrides(args.set))

    for seed in args.seeds:
        run_dir = args.out / config.name / f"seed_{seed}"
        print(f"=== {config.name} seed {seed} -> {run_dir}", flush=True)
        started = time.perf_counter()
        callback = train(config, seed, run_dir, vec_env=args.vec_env, tensorboard=args.tensorboard)
        minutes = (time.perf_counter() - started) / 60
        print(f"=== done in {minutes:.1f} min, best validation color changes {callback.best:.2f}", flush=True)


if __name__ == "__main__":
    main()
