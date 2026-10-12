"""Several decision makers together: votes, rollouts and committees, with two different referees.

    python scripts/research/committee_study.py --network artifacts/runs/bufferpair64_physical_dagger_ppo/seed_2/best_model.zip \
        --values artifacts/value/v1.pt artifacts/value/v2.pt --workers 12

On the 100 validation scenarios, every candidate scored over the known horizon by
  (a) the hand-made referee: changes + 0.2 x fragmentation, and
  (b) the learned referee: changes + scale x averaged value networks (default 0.75),
for: the network alone, rollout(network), rollout(heuristic), the beam planner, and the committee
of planner + rollout(network) + rollout(heuristic). Results: artifacts/research/committee_study.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from haam_rl.adapters.runner import run_scenarios
from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.adapters.value_search import LeafValue, load_value
from haam_rl.application.hybrid import CommitteePolicy, PlannerProposer, RolloutPolicy
from haam_rl.application.use_cases import HeuristicPolicy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.planning import fragmentation_estimate
from haam_rl.domain.scenarios import VALIDATION_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--values", type=Path, nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("artifacts/research/committee_study.json"))
    args = parser.parse_args()

    network = SB3Policy.load(args.network)
    heuristic = HeuristicPolicy(25)
    referees = {"fragmentation": fragmentation_estimate(0.2),
                "value": LeafValue([load_value(p) for p in args.values], scale=args.scale)}
    candidates = [("network alone", network)]
    for referee_name, referee in referees.items():
        planner = PlannerProposer(25, 10, 32, estimate=referee)
        rollout_network = RolloutPolicy(network, 25, 10, estimate=referee)
        rollout_heuristic = RolloutPolicy(heuristic, 25, 10, estimate=referee)
        candidates += [
            (f"rollout(network) [{referee_name}]", rollout_network),
            (f"rollout(heuristic) [{referee_name}]", rollout_heuristic),
            (f"planner [{referee_name}]", planner),
            (f"committee [{referee_name}]", CommitteePolicy([planner, rollout_network, rollout_heuristic], 25)),
        ]
    results = {"network": str(args.network), "values": [p.stem for p in args.values], "rows": {}}
    for name, policy in candidates:
        started = time.perf_counter()
        values = [r.color_changes for r in run_scenarios(ExperimentConfig(), policy, VALIDATION_SEEDS, args.workers)]
        results["rows"][name] = {"mean": float(np.mean(values)), "per_scenario": values,
                                 "seconds": round(time.perf_counter() - started, 1)}
        print(f"{name:<40} {np.mean(values):.2f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
