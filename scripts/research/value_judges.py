"""How good is each value network as the planner's judge, and does averaging several help?

    python scripts/research/value_judges.py --values artifacts/value/v1.pt artifacts/value/v2.pt ... --workers 12

On the 100 validation scenarios: the value-guided planner (look-ahead 10, width 32, scale 0.5)
with every value network alone, then with the average of the first k networks for k = 2..N.
Results: artifacts/research/value_judges.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from haam_rl.adapters.runner import run_scenarios
from haam_rl.adapters.value_search import ValueGuidedPlannerPolicy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.scenarios import VALIDATION_SEEDS


def evaluate_judge(paths: list[Path], scale: float, workers: int) -> dict:
    started = time.perf_counter()
    policy = ValueGuidedPlannerPolicy(paths, 25, 10, 32, scale=scale)
    values = [r.color_changes for r in run_scenarios(ExperimentConfig(), policy, VALIDATION_SEEDS, workers)]
    metrics = [json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))["metrics"] for p in paths]
    return {"values": [p.stem for p in paths], "scale": scale, "mean": float(np.mean(values)), "per_scenario": values,
            "data_policies": [m.get("data_policy") for m in metrics], "held_out_r2": [m.get("held_out_r2") for m in metrics],
            "seconds": round(time.perf_counter() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--values", type=Path, nargs="+", required=True, help="in the order used for the averages")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--scale-sweep", type=float, nargs="*", default=[0.5, 0.75, 1.0, 1.25, 1.5],
                        help="scales tried with the average of all networks")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/value_judges.json"))
    args = parser.parse_args()
    rows = {"single": [], "averaged": [], "scales": []}
    for path in args.values:
        rows["single"].append(evaluate_judge([path], args.scale, args.workers))
        print(f"{path.stem:>4} alone: {rows['single'][-1]['mean']:.2f}", flush=True)
    for k in range(2, len(args.values) + 1):
        rows["averaged"].append(evaluate_judge(args.values[:k], args.scale, args.workers))
        print(f"average of {k}: {rows['averaged'][-1]['mean']:.2f}", flush=True)
    for scale in args.scale_sweep:
        rows["scales"].append(evaluate_judge(args.values, scale, args.workers))
        print(f"average of all, scale {scale}: {rows['scales'][-1]['mean']:.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
