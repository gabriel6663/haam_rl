"""Train several experiment configs with several seeds, as many (config, seed) runs at a time as asked.

    python scripts/run_experiments.py --configs configs/*.json --seeds 0 1 2 --timesteps 3000000 --parallel 12

Each run uses one CPU core (PyTorch limited to 1 thread). Logs go to
artifacts/logs/<config name>_seed<n>.log; models to artifacts/runs/<config name>/seed_<n>/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from haam_rl.config import ExperimentConfig


def run(config: Path, seed: int, timesteps: int | None, logs: Path) -> tuple[str, int]:
    name = ExperimentConfig.load(config).name
    log = logs / f"{name}_seed{seed}.log"
    command = [sys.executable, "-u", "scripts/train.py", "--config", str(config), "--seeds", str(seed)]
    if timesteps:
        command += ["--timesteps", str(timesteps)]
    with open(log, "w", encoding="utf-8") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    return f"{name} seed {seed}", code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=Path, nargs="*", default=sorted(Path("configs").glob("*.json")))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--timesteps", type=int, help="override training.total_timesteps of every config")
    parser.add_argument("--parallel", type=int, default=4, help="runs trained at the same time (one core each)")
    args = parser.parse_args()

    logs = Path("artifacts/logs")
    logs.mkdir(parents=True, exist_ok=True)
    jobs = [(config, seed) for config in args.configs for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = [pool.submit(run, config, seed, args.timesteps, logs) for config, seed in jobs]
        for future in futures:
            name, code = future.result()
            print(f"{name}: {'ok' if code == 0 else f'failed (exit {code})'}", flush=True)


if __name__ == "__main__":
    main()
