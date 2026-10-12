"""Compare policies on fixed scenarios, every one paired against the heuristic.

    python scripts/evaluate.py --models artifacts/runs/onehot --ensemble --planner 10 --oracle
    python scripts/evaluate.py --seed-set extended --planner 5 10 20 --planner-mask physical haam

--models      model zips or run directories (every best_model.zip below them is used). Models of
              the same experiment also get a "mean of N seeds" row: per scenario, the average of
              its seeds, i.e. the expected result of one training run.
--ensemble    adds the paper's hard/soft voting ensemble of all given models.
--planner H.. adds the look-ahead planner for each horizon H (and each --planner-mask).
--oracle      adds the planner with perfect information (a reference, not deployable).
--value V..   adds the value-guided planner for each value network file (look-ahead of --value-horizon);
              --value-oracle and --committee NETWORK add its perfect-information version and the
              committee of that planner with rollouts of a network and of the heuristic.

Seed sets: paper = 30 test scenarios (10000-10029), extended = 500 (30000-30499),
validation = the 100 used for model selection (20000-20099).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from haam_rl.adapters.runner import run_scenarios
from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.adapters.value_search import LeafValue, ValueGuidedPlannerPolicy, load_value
from haam_rl.application.evaluation import paired_comparison
from haam_rl.application.hybrid import CommitteePolicy, PlannerProposer, RolloutPolicy
from haam_rl.application.use_cases import EnsemblePolicy, HeuristicPolicy, LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig, PlantConfig
from haam_rl.domain.heuristics import MaskMode
from haam_rl.domain.scenarios import EXTENDED_TEST_SEEDS, TEST_SEEDS, VALIDATION_SEEDS

SEED_SETS = {"paper": TEST_SEEDS, "extended": EXTENDED_TEST_SEEDS, "validation": VALIDATION_SEEDS}


def find_models(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        found += sorted(path.rglob("best_model.zip")) if path.is_dir() else [path]
    if paths and not found:
        raise SystemExit(f"No models found in {paths}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", type=Path, nargs="*", default=[])
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--vote-threshold", type=float, help="hard-voting threshold; default M/2 - 1 (paper Eq. 2)")
    parser.add_argument("--planner", type=int, nargs="*", default=[], metavar="HORIZON")
    parser.add_argument("--planner-width", type=int, default=32)
    parser.add_argument("--planner-mask", nargs="+", default=["physical"], choices=[m.value for m in MaskMode])
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--oracle-width", type=int, default=64)
    parser.add_argument("--value", nargs="*", default=[], metavar="VALUE_PT[,VALUE_PT...]",
                        help="value network file(s); comma-separated files are averaged into one judge")
    parser.add_argument("--value-scale", type=float, default=0.75)
    parser.add_argument("--value-horizon", type=int, default=10)
    parser.add_argument("--value-width", type=int, default=32)
    parser.add_argument("--value-oracle", action="store_true", help="also the value-guided planner with perfect information")
    parser.add_argument("--committee", type=Path, nargs="*", default=[], metavar="NETWORK_ZIP",
                        help="committee: value planner + rollout(network) + rollout(heuristic), refereed by the value")
    parser.add_argument("--fill-target", type=int, help="plant fill target when no model is given (default 25)")
    parser.add_argument("--seed-set", choices=SEED_SETS, default="paper")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, help="write per-scenario results and statistics to this JSON file")
    args = parser.parse_args()

    seeds = list(SEED_SETS[args.seed_set])
    model_paths = find_models(args.models)
    models = [SB3Policy.load(path) for path in model_paths]
    if models:
        plants = {m.config.plant for m in models}
        if len(plants) != 1:
            raise SystemExit(f"Models were trained on different plants: {plants}")
        config = models[0].config
    else:
        plant = PlantConfig() if args.fill_target is None else PlantConfig(fill_target=args.fill_target)
        config = ExperimentConfig(plant=plant)
    fill = config.plant.fill_target

    policies: list[tuple[str, str, object]] = [("heuristic", "heuristic", HeuristicPolicy(fill))]
    policies += [(path.as_posix(), "model", model) for path, model in zip(model_paths, models)]
    if args.ensemble and len(models) > 1:
        ensemble = EnsemblePolicy(models, hard_threshold=args.vote_threshold)
        policies.append((ensemble.name, "ensemble", ensemble))
    for mask in args.planner_mask:
        for horizon in args.planner:
            planner = LookaheadPlannerPolicy(fill, horizon, args.planner_width, mask_mode=MaskMode(mask))
            policies.append((planner.name, "planner", planner))
        if args.oracle:
            oracle = LookaheadPlannerPolicy(fill, None, args.oracle_width, mask_mode=MaskMode(mask))
            policies.append((oracle.name, "oracle", oracle))

    for value_spec in args.value:
        value_paths = [Path(p) for p in value_spec.split(",")]
        label = "+".join(p.stem for p in value_paths)
        guided = ValueGuidedPlannerPolicy(value_paths, fill, args.value_horizon, args.value_width, scale=args.value_scale)
        policies.append((f"{guided.name}[{label}]", "value_planner", guided))
        if args.value_oracle:
            oracle = ValueGuidedPlannerPolicy(value_paths, fill, None, args.oracle_width, scale=args.value_scale)
            policies.append((f"value_oracle(w={args.oracle_width})[{label}]", "oracle", oracle))
        for network_path in args.committee:
            referee = LeafValue([load_value(p) for p in value_paths], args.value_scale)
            network = SB3Policy.load(network_path)
            committee = CommitteePolicy([
                PlannerProposer(fill, args.value_horizon, args.value_width, estimate=referee),
                RolloutPolicy(network, fill, args.value_horizon, estimate=referee),
                RolloutPolicy(HeuristicPolicy(fill), fill, args.value_horizon, estimate=referee),
            ], fill)
            name = f"committee(value planner + rollout(network) + rollout(heuristic))[{label}]"
            policies.append((name, "committee", committee))

    per_scenario: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}
    for name, kind, policy in policies:
        started = time.perf_counter()
        results = run_scenarios(config, policy, seeds, workers=args.workers)
        per_scenario[name] = [r.color_changes for r in results]
        meta[name] = {"kind": kind, "wall_seconds": round(time.perf_counter() - started, 2),
                      "decisions": sum(len(r.actions) for r in results)}
        print(f"  evaluated {name} in {meta[name]['wall_seconds']}s", flush=True)

    # One row per experiment: per scenario, the mean over that experiment's seeds.
    by_experiment: dict[str, list[str]] = defaultdict(list)
    for path, model in zip(model_paths, models):
        by_experiment[model.config.name].append(path.as_posix())
    for experiment, names in by_experiment.items():
        if len(names) > 1:
            row = f"{experiment} (media de {len(names)} semillas)"
            per_scenario[row] = np.mean([per_scenario[n] for n in names], axis=0).round(4).tolist()
            seed_means = [float(np.mean(per_scenario[n])) for n in names]
            meta[row] = {"kind": "seed_mean", "members": names, "seed_means": seed_means}

    baseline = per_scenario["heuristic"]
    print(f"\nplant: {config.plant.vehicles} vehicles, fill_target={fill}, "
          f"seed set '{args.seed_set}' ({len(seeds)} scenarios)\n")
    header = f"{'policy':<58} {'mean':>6} {'std':>5} {'reduction':>9} {'95% CI reduction':>17} {'p':>8} {'W/T/L':>10}"
    print(header + "\n" + "-" * len(header))
    summary = {}
    for name, values in per_scenario.items():
        stats = paired_comparison(values, baseline)
        summary[name] = {**meta[name], "mean": float(np.mean(values)), "std": float(np.std(values)),
                         "per_scenario": values, "vs_heuristic": stats.__dict__}
        if name == "heuristic":
            print(f"{name:<58} {np.mean(values):>6.2f} {np.std(values):>5.2f}")
            continue
        ci = f"[{stats.reduction_ci95[0]:+.1f},{stats.reduction_ci95[1]:+.1f}]%"
        print(f"{name[-58:]:<58} {np.mean(values):>6.2f} {np.std(values):>5.2f} {stats.reduction_pct:>8.1f}% "
              f"{ci:>17} {stats.p_value:>8.5f} {stats.wins:>4}/{stats.ties}/{stats.losses}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seed_set": args.seed_set, "seeds": seeds, "plant": config.to_dict()["plant"], "results": summary}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
