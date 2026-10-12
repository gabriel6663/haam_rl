"""Run a policy over many scenarios on the Python twin, as fast as that kind of policy allows.

Neural policies (anything with `decide_batch`) run every scenario in lockstep, batching inference.
Search policies (heuristic, planner, oracle) are CPU-bound per decision, so their scenarios are
split across processes. Either way every scenario is simulated independently and deterministically,
so the results do not depend on the path taken or on the number of workers.
"""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

from haam_rl.adapters.python_twin import PythonPaintShopTwin
from haam_rl.application.evaluation import EpisodeResult, evaluate
from haam_rl.application.ports import Policy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.heuristics import MaskMode


def evaluation_twin(config: ExperimentConfig) -> PythonPaintShopTwin:
    """Twin that accepts any physically valid action: each policy enforces its own mask."""
    return PythonPaintShopTwin(config.with_overrides(mask_mode=MaskMode.PHYSICAL.value))


def _run_chunk(config: ExperimentConfig, policy: Policy, seeds: list[int]) -> list[EpisodeResult]:
    return evaluate(lambda: evaluation_twin(config), policy, seeds)


def run_scenarios(config: ExperimentConfig, policy: Policy, seeds: Iterable[int], workers: int = 1) -> list[EpisodeResult]:
    """Results in the order of `seeds`."""
    seeds = list(seeds)
    # Batched policies are fastest in lockstep in this process; everything else runs per scenario.
    if workers <= 1 or hasattr(policy, "decide_batch") or len(seeds) < 2:
        return _run_chunk(config, policy, seeds)
    chunk_count = min(len(seeds), workers * 4)  # several chunks per worker evens out slow scenarios
    chunks = [seeds[k::chunk_count] for k in range(chunk_count)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_run_chunk, [config] * chunk_count, [policy] * chunk_count, chunks))
    by_seed = {result.seed: result for part in parts for result in part}
    return [by_seed[seed] for seed in seeds]
