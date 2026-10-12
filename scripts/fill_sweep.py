"""Color changes of the heuristic, the look-ahead planner and the oracle vs the buffer fill target.

    python scripts/fill_sweep.py --targets 25 24 22 20 18 15 10 --workers 8

The paper's plant fills the 25 slots before painting. With full buffers the incoming vehicle can
only go into the slot that was just freed, so input decisions disappear; this sweep measures how
much keeping a few slots free is worth for each policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from haam_rl.adapters.runner import run_scenarios
from haam_rl.application.use_cases import HeuristicPolicy, LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig, PlantConfig
from haam_rl.domain.scenarios import EXTENDED_TEST_SEEDS, TEST_SEEDS

SEED_SETS = {"paper": TEST_SEEDS, "extended": EXTENDED_TEST_SEEDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=int, nargs="+", default=[25, 24, 23, 22, 20, 18, 15, 10])
    parser.add_argument("--planner-horizon", type=int, default=10)
    parser.add_argument("--planner-width", type=int, default=32, help="0 skips the planner")
    parser.add_argument("--oracle-width", type=int, default=0, help="oracle beam width; 0 skips it (slow)")
    parser.add_argument("--seed-set", choices=SEED_SETS, default="paper")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/fill_sweep.json"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    seeds = list(SEED_SETS[args.seed_set])
    rows = []
    print(f"{'fill':>4} {'heuristic':>10} {'planner':>10} {'oracle':>10}", flush=True)
    for target in args.targets:
        config = ExperimentConfig(plant=PlantConfig(fill_target=target))
        policies = {"heuristic": HeuristicPolicy(target)}
        if args.planner_width:
            policies["planner"] = LookaheadPlannerPolicy(target, args.planner_horizon, args.planner_width)
        if args.oracle_width:
            policies["oracle"] = LookaheadPlannerPolicy(target, None, args.oracle_width)
        row = {"fill_target": target}
        for name, policy in policies.items():
            row[name] = [r.color_changes for r in run_scenarios(config, policy, seeds, workers=args.workers)]
        rows.append(row)
        cells = [f"{np.mean(row[name]):>10.2f}" if name in row else f"{'-':>10}" for name in ("heuristic", "planner", "oracle")]
        print(f"{target:>4} " + " ".join(cells), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"seed_set": args.seed_set, "seeds": seeds, "rows": rows}, indent=2), encoding="utf-8")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
