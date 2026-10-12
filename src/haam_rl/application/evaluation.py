"""Running policies over scenarios and comparing them with paired statistics."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from haam_rl.application.ports import PlantPort, Policy
from haam_rl.application.use_cases import decide_all


@dataclass
class EpisodeResult:
    seed: int
    color_changes: int
    painted: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)


def run_episode(plant: PlantPort, policy: Policy, seed: int) -> EpisodeResult:
    return evaluate(lambda: plant, policy, [seed])[0]


def evaluate(make_plant: Callable[[], PlantPort], policy: Policy, seeds: Iterable[int]) -> list[EpisodeResult]:
    """Run every scenario side by side ("lockstep").

    Episodes are independent, so the results are the same as running them one after another, but
    a policy with `decide_batch` (a neural network) answers all running scenarios in one forward
    pass per step instead of one pass per scenario.
    """
    seeds = list(seeds)
    plants = [make_plant() for _ in seeds]
    states = [plant.reset(seed) for plant, seed in zip(plants, seeds)]
    results = [EpisodeResult(seed, 0) for seed in seeds]
    running = list(range(len(seeds)))
    while running:
        decisions = decide_all(policy, [states[k] for k in running])
        still_running = []
        for k, decision in zip(running, decisions):
            states[k], _, done, info = plants[k].step(decision.action)
            results[k].actions.append(decision.action)
            if info.painted_color:
                results[k].painted.append(info.painted_color)
            results[k].color_changes = info.color_changes
            if not done:
                still_running.append(k)
        running = still_running
    return results


@dataclass(frozen=True)
class PairedComparison:
    """Candidate vs baseline on the same scenarios (fewer color changes is better).

    Uncertainty comes from the scenarios: both bootstraps resample scenarios (pairs kept together),
    and the p-value is a two-sided sign-flip permutation test of the mean paired difference.
    """

    candidate_mean: float
    baseline_mean: float
    mean_difference: float                 # candidate - baseline (negative = candidate better)
    standard_error: float                  # of the mean difference
    ci95: tuple[float, float]              # bootstrap interval of the mean difference
    reduction_pct: float                   # (baseline - candidate) / baseline * 100
    reduction_ci95: tuple[float, float]    # bootstrap interval of reduction_pct
    p_value: float
    wins: int
    ties: int
    losses: int


def paired_comparison(candidate: Sequence[float], baseline: Sequence[float], resamples: int = 100_000,
                      seed: int = 0) -> PairedComparison:
    a, b = np.asarray(candidate, dtype=float), np.asarray(baseline, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("Paired comparison needs two non-empty lists of equal length")
    n, diff = a.size, a - b
    rng = np.random.default_rng(seed)

    boot_diff, boot_reduction, extreme = [], [], 0
    observed = abs(diff.mean())
    for size in _chunks(resamples, max(1, 2_000_000 // n)):  # bounded memory for large n
        rows = rng.integers(0, n, size=(size, n))
        a_mean, b_mean = a[rows].mean(axis=1), b[rows].mean(axis=1)
        boot_diff.append(a_mean - b_mean)
        boot_reduction.append(np.divide(b_mean - a_mean, b_mean, out=np.zeros(size), where=b_mean != 0) * 100)
        signs = rng.choice((-1.0, 1.0), size=(size, n))
        extreme += int(np.sum(np.abs((signs * diff).mean(axis=1)) >= observed - 1e-12))
    boot_diff, boot_reduction = np.concatenate(boot_diff), np.concatenate(boot_reduction)

    return PairedComparison(
        candidate_mean=float(a.mean()),
        baseline_mean=float(b.mean()),
        mean_difference=float(diff.mean()),
        standard_error=float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        ci95=(float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))),
        reduction_pct=float((b.mean() - a.mean()) / b.mean() * 100) if b.mean() else 0.0,
        reduction_ci95=(float(np.percentile(boot_reduction, 2.5)), float(np.percentile(boot_reduction, 97.5))),
        p_value=(extreme + 1) / (resamples + 1),
        wins=int(np.sum(diff < 0)),
        ties=int(np.sum(diff == 0)),
        losses=int(np.sum(diff > 0)),
    )


def _chunks(total: int, size: int) -> list[int]:
    return [min(size, total - start) for start in range(0, total, size)]
