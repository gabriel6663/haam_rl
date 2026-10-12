"""Beam width and fragmentation weight of the look-ahead planner, tuned on the validation seeds.

    python scripts/research/planner_tuning.py --workers 8

Measures how much a wider search helps (is the search saturated?) and how sensitive the planner
is to the weight of its hand-made leaf estimate. Results: artifacts/research/planner_tuning.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from haam_rl.adapters.runner import run_scenarios
from haam_rl.application.use_cases import LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.scenarios import VALIDATION_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--widths", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/planner_tuning.json"))
    args = parser.parse_args()
    rows = []
    for width in args.widths:
        for weight in args.weights:
            started = time.perf_counter()
            policy = LookaheadPlannerPolicy(25, 10, width, fragmentation_weight=weight)
            values = [r.color_changes for r in run_scenarios(ExperimentConfig(), policy, VALIDATION_SEEDS, args.workers)]
            rows.append({"width": width, "weight": weight, "mean": float(np.mean(values)), "per_scenario": values,
                         "seconds": round(time.perf_counter() - started, 1)})
            print(f"width {width:>4} weight {weight}: {np.mean(values):.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
