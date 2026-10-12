"""Turn evaluation JSON files into result tables, so that reported numbers are never typed by hand.

    python scripts/report.py artifacts/eval/final_paper.json artifacts/eval/final_extended.json \
        --out artifacts/eval/report.md

For every policy: mean color changes with the 95 % interval of the mean (bootstrap over scenarios),
reduction vs the heuristic with its 95 % interval, p-value of the paired permutation test and
wins / ties / losses per scenario. Rows can be renamed and ordered with --labels (a JSON file
mapping evaluation names to display names; policies not listed are left out).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from haam_rl.application.evaluation import paired_comparison


def mean_interval(values: list[float], resamples: int = 20_000, seed: int = 0) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rows = np.random.default_rng(seed).integers(0, data.size, size=(resamples, data.size))
    means = data[rows].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def format_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def table(evaluation: dict, labels: dict[str, str] | None) -> str:
    results = evaluation["results"]
    names = list(labels) if labels else list(results)
    lines = [
        "| Política | Cambios (media) | IC 95 % media | Reducción vs heurística | IC 95 % reducción | p | G / E / P |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in names:
        if name not in results:
            continue
        row, stats = results[name], results[name]["vs_heuristic"]
        low, high = mean_interval(row["per_scenario"])
        label = labels[name] if labels else name
        if name == "heuristic":
            lines.append(f"| {label} | {row['mean']:.2f} | {low:.2f}–{high:.2f} | — | — | — | — |")
            continue
        r_low, r_high = stats["reduction_ci95"]
        lines.append(
            f"| {label} | {row['mean']:.2f} | {low:.2f}–{high:.2f} | {stats['reduction_pct']:.1f} % | "
            f"{r_low:.1f} a {r_high:.1f} % | {format_p(stats['p_value'])} | "
            f"{stats['wins']} / {stats['ties']} / {stats['losses']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evaluations", type=Path, nargs="+")
    parser.add_argument("--labels", type=Path, help="JSON {evaluation name: display name}, in display order")
    parser.add_argument("--compare", nargs=2, action="append", default=[], metavar=("CANDIDATE", "BASELINE"),
                        help="add a paired comparison between two policies of the same evaluation file")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Spanish labels on Windows consoles

    labels = json.loads(args.labels.read_text(encoding="utf-8")) if args.labels else None
    sections = []
    for path in args.evaluations:
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        plant = evaluation["plant"]
        sections.append(
            f"## {path.stem}\n\n{len(evaluation['seeds'])} escenarios (conjunto '{evaluation['seed_set']}'), "
            f"{plant['vehicles']} vehículos, fill_target = {plant['fill_target']}.\n\n{table(evaluation, labels)}"
        )
        results = evaluation["results"]
        for candidate, baseline in args.compare:
            if candidate not in results or baseline not in results:
                continue
            stats = paired_comparison(results[candidate]["per_scenario"], results[baseline]["per_scenario"])
            names = (labels or {}).get(candidate, candidate), (labels or {}).get(baseline, baseline)
            sections.append(
                f"{names[0]} frente a {names[1]}: diferencia media {stats.mean_difference:+.2f} cambios "
                f"(IC 95 % {stats.ci95[0]:+.2f} a {stats.ci95[1]:+.2f}), reducción {stats.reduction_pct:.1f} % "
                f"(IC 95 % {stats.reduction_ci95[0]:.1f} a {stats.reduction_ci95[1]:.1f} %), "
                f"p = {format_p(stats.p_value)}, G / E / P = {stats.wins} / {stats.ties} / {stats.losses}."
            )
    text ="\n\n".join(sections) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
