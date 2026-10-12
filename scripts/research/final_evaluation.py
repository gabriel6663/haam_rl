"""Regenerate every number of the publication from the trained artifacts.

    python scripts/research/final_evaluation.py --value artifacts/value/v1.pt \
        --committee-network artifacts/runs/bufferpair64_physical_dagger_ppo/seed_2/best_model.zip

Runs, in order (outputs in artifacts/eval/):
  final_paper.json / final_extended.json   every policy family on the 30 / 500 test scenarios
  ensemble_paper.json / ensemble_extended.json   the paper's voting ensemble (one-hot PPO, 5 seeds)
  fill_sweep.json                          heuristic and planner vs the buffer fill target
  latency.json                             per-decision latency (run on an idle machine)
Then: python scripts/build_publication.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

RUNS = [  # experiment directories evaluated as policies (all trained for the paper plant, fill 25)
    "paper", "onehot", "onehot_lookahead", "onehot_lookahead_physical", "arch_64x64", "arch_256x256x256",
    "bufferpair64_ppo", "bufferpair64_lookahead_dagger", "bufferpair64_dagger_ppo",
    "bufferpair64_physical_dagger", "bufferpair64_physical_dagger_ppo", "bufferpair64_physical_valuedagger",
    "bufferpair128_physical_judgesdagger",
]


def run(arguments: list[str]) -> None:
    print("$ python " + " ".join(arguments), flush=True)
    subprocess.run([sys.executable, *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--value", required=True, help="value network file(s) of the value-guided planner, comma-separated")
    parser.add_argument("--committee-network", type=Path, required=True, help="network used for rollouts in the committee")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--skip", nargs="*", default=[], choices=["final", "ensemble", "fill", "latency"])
    args = parser.parse_args()

    runs = [f"artifacts/runs/{name}" for name in RUNS if Path(f"artifacts/runs/{name}").is_dir()]
    workers = ["--workers", str(args.workers)]
    if "final" not in args.skip:
        for seed_set in ("paper", "extended"):
            run(["scripts/evaluate.py", "--seed-set", seed_set, "--models", *runs,
                 "--planner", "10", "--planner-mask", "physical", "haam", "--oracle",
                 "--value", args.value, "--value-oracle", "--committee", str(args.committee_network),
                 *workers, "--out", f"artifacts/eval/final_{seed_set}.json"])
    if "ensemble" not in args.skip:
        for seed_set in ("paper", "extended"):
            run(["scripts/evaluate.py", "--seed-set", seed_set, "--models", "artifacts/runs/onehot", "--ensemble",
                 *workers, "--out", f"artifacts/eval/ensemble_{seed_set}.json"])
    if "fill" not in args.skip:
        run(["scripts/fill_sweep.py", "--targets", "25", "24", "23", "22", "20", "18", "15", "10", *workers,
             "--out", "artifacts/eval/fill_sweep.json"])
    if "latency" not in args.skip:
        run(["scripts/benchmark.py", "--models", str(args.committee_network), "--value", args.value,
             "--committee", str(args.committee_network),
             "--out", "artifacts/eval/latency.json"])


if __name__ == "__main__":
    main()
