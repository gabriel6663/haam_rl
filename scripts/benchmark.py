"""Per-decision latency of each policy, the number FlexSim waits for at every call.

    python scripts/benchmark.py --models artifacts/runs/bufferpair64_lookahead_dagger/seed_0/best_model.zip

Decisions are timed one at a time in a single process (no batching), on the states visited by the
heuristic in the first test scenarios, after a warm-up. Run it on an otherwise idle machine.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from haam_rl.adapters.runner import evaluation_twin
from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.adapters.value_search import ValueGuidedPlannerPolicy
from haam_rl.application.hybrid import CommitteePolicy, PlannerProposer, RolloutPolicy
from haam_rl.application.use_cases import HeuristicPolicy, LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.heuristics import MaskMode
from haam_rl.domain.scenarios import TEST_SEEDS


def visited_states(scenarios: int) -> list:
    twin, heuristic, states = evaluation_twin(ExperimentConfig()), HeuristicPolicy(25), []
    for seed in list(TEST_SEEDS)[:scenarios]:
        state, done = twin.reset(seed), False
        while not done:
            states.append(state)
            state, _, done, _ = twin.step(heuristic.decide(state).action)
    return states


def time_policy(policy, states: list, warmup: int = 20) -> dict[str, float]:
    for state in states[:warmup]:
        policy.decide(state)
    latencies = []
    for state in states:
        started = time.perf_counter()
        policy.decide(state)
        latencies.append((time.perf_counter() - started) * 1000)
    data = np.asarray(latencies)
    return {"decisions": len(data), "mean_ms": round(float(data.mean()), 3), "p50_ms": round(float(np.median(data)), 3),
            "p95_ms": round(float(np.percentile(data, 95)), 3), "max_ms": round(float(data.max()), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--value", nargs="*", default=[],
                        help="value network file(s) for the value-guided planner; comma-separated files form one averaged judge")
    parser.add_argument("--committee", type=Path, help="network zip: also time the committee (needs --value)")
    parser.add_argument("--scenarios", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    torch.set_num_threads(1)

    states = visited_states(args.scenarios)
    policies = [("heuristic", HeuristicPolicy(25))]
    for horizon in (10, 20):
        for mask in (MaskMode.PHYSICAL, MaskMode.HAAM):
            planner = LookaheadPlannerPolicy(25, horizon, 32, mask_mode=mask)
            policies.append((planner.name, planner))
    policies += [(path.as_posix(), SB3Policy.load(path)) for path in args.models]
    for spec in args.value:
        paths = [Path(p) for p in spec.split(",")]
        guided = ValueGuidedPlannerPolicy(paths, 25, 10, 32, scale=0.75)
        policies.append((f"{guided.name}[{'+'.join(p.stem for p in paths)}]", guided))
        if args.committee:
            network = SB3Policy.load(args.committee)
            committee = CommitteePolicy([PlannerProposer(25, 10, 32, estimate=guided.estimate),
                                         RolloutPolicy(network, 25, 10, estimate=guided.estimate),
                                         RolloutPolicy(HeuristicPolicy(25), 25, 10, estimate=guided.estimate)], 25)
            policies.append((f"committee[{'+'.join(p.stem for p in paths)}]", committee))

    results = {}
    for name, policy in policies:
        results[name] = time_policy(policy, states)
        r = results[name]
        print(f"{name:<70} mean {r['mean_ms']:>7.3f} ms  p50 {r['p50_ms']:>7.3f}  p95 {r['p95_ms']:>7.3f}  max {r['max_ms']:>7.3f}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
