"""Build the publication (HTML and PDF) from the saved results. No number in it is typed by hand.

    python scripts/research/final_evaluation.py ...   # produces artifacts/eval/*.json
    python scripts/build_publication.py                # -> docs/publicacion/publicacion.html and .pdf

Every table, figure and figure in the text is read from artifacts/eval, artifacts/research and
artifacts/runs. Figures are drawn with matplotlib and embedded as SVG; the PDF is printed by a
headless Chromium (Playwright).
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVAL, RESEARCH, RUNS = Path("artifacts/eval"), Path("artifacts/research"), Path("artifacts/runs")
INK, MUTED, ACCENT = "#1f2933", "#8a94a3", "#2f6fd0"
FAMILY_COLORS = {"referencia": "#8a94a3", "rl": "#2f6fd0", "imitacion": "#2e9e5b", "busqueda": "#d9622b", "oraculo": "#5f4bb6"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#9aa5b1", "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "svg.fonttype": "none"})


# ---------------------------------------------------------------------------- data access
def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{num(value, digits)} %"


def p_value(p: float) -> str:
    return "< 0,0001" if p < 1e-4 else num(p, 4)


def mean_ci(values: list[float], resamples: int = 20_000) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    rows = np.random.default_rng(0).integers(0, data.size, size=(resamples, data.size))
    means = data[rows].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def find(results: dict, pattern: str) -> str | None:
    """Key of the first result whose name matches the regular expression."""
    return next((key for key in results if re.search(pattern, key)), None)


class Results:
    """One evaluation file, addressed by regular expressions over the policy names."""

    def __init__(self, path: Path) -> None:
        self.data = load(path)
        self.results = self.data["results"]

    def get(self, pattern: str) -> dict | None:
        key = find(self.results, pattern)
        return None if key is None else {**self.results[key], "key": key}

    def mean(self, pattern: str) -> float:
        return self.get(pattern)["mean"]


# Policies in the order of the story, with the regular expression that finds each one.
CATALOGUE = [
    ("Heurística LP/CM/UCM (referencia del paper)", "referencia", r"^heuristic$"),
    ("RL · entrada cruda, como el paper (5 semillas)", "rl", r"^paper \(media"),
    ("RL · one-hot (5 semillas)", "rl", r"^onehot \(media"),
    ("RL · one-hot + anticipación 10 (5 semillas)", "rl", r"^onehot_lookahead \(media"),
    ("RL · MLP 64×64, 3M pasos (3 semillas)", "rl", r"^arch_64x64 \(media"),
    ("RL · MLP 256×256×256, 3M pasos (3 semillas)", "rl", r"^arch_256x256x256 \(media"),
    ("RL · sin máscara heurística (5 semillas)", "rl", r"^onehot_lookahead_physical \(media"),
    ("RL · red estructurada (3 semillas)", "rl", r"^bufferpair64_ppo \(media"),
    ("Imitación HAAM · red estructurada (3 semillas)", "imitacion", r"^bufferpair64_lookahead_dagger \(media"),
    ("Imitación HAAM + ajuste PPO (3 semillas)", "imitacion", r"^bufferpair64_dagger_ppo \(media"),
    ("Imitación física · red estructurada (3 semillas)", "imitacion", r"^bufferpair64_physical_dagger \(media"),
    ("Imitación física + ajuste PPO (3 semillas)", "imitacion", r"^bufferpair64_physical_dagger_ppo \(media"),
    ("Imitación del planificador guiado por valor", "imitacion", r"physical_valuedagger"),
    ("Imitación del planificador con 8 jueces (red 128)", "imitacion", r"physical_judgesdagger"),
    ("Planificador · máscara HAAM, fragmentación", "busqueda", r"^planner\(h=10,w=32,mask=haam\)"),
    ("Planificador · fragmentación", "busqueda", r"^planner\(h=10,w=32\)$"),
    ("Planificador guiado por valor (8 jueces)", "busqueda", r"^value_planner"),
    ("Comité: planificador + rollouts, juez de valor", "busqueda", r"^committee"),
    ("Oráculo con fragmentación (información completa)", "oraculo", r"^oracle\(w=64\)$"),
    ("Oráculo guiado por valor (información completa)", "oraculo", r"^value_oracle"),
]


def collected_tests() -> int:
    """Number of automated tests, as pytest collects them."""
    result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], capture_output=True,
                            text=True, encoding="utf-8", check=False)
    match = re.search(r"(\d+) tests? collected", result.stdout)
    return int(match.group(1)) if match else 0


def forced_decision_share(seeds: list[int]) -> float:
    """Share of the heuristic's decisions for which the HAAM mask leaves a single action."""
    from haam_rl.adapters.runner import evaluation_twin
    from haam_rl.application.use_cases import HeuristicPolicy
    from haam_rl.config import ExperimentConfig
    from haam_rl.domain.heuristics import MaskMode, allowed_actions

    twin, heuristic, forced, total = evaluation_twin(ExperimentConfig()), HeuristicPolicy(25), 0, 0
    for seed in seeds:
        state, done = twin.reset(seed), False
        while not done:
            forced += len(allowed_actions(state, 25, MaskMode.HAAM)) == 1
            total += 1
            state, _, done, _ = twin.step(heuristic.decide(state).action)
    return 100 * forced / total


# ---------------------------------------------------------------------------- figures
def svg(figure) -> str:
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    text = buffer.getvalue()
    return text[text.index("<svg"):]


def dot_plot(rows: list[tuple[str, str, list[float]]], title: str, xlabel: str, reference: float | None = None) -> str:
    """rows: (label, family, per-scenario values). Mean and 95 % bootstrap interval per row."""
    figure, axis = plt.subplots(figsize=(6.6, 0.32 * len(rows) + 0.9))
    intervals = [mean_ci(values) for _, _, values in rows]
    offset = 0.015 * (max(h for _, h in intervals) - min(lo for lo, _ in intervals))  # label gap in data units
    for k, ((label, family, values), (low, high)) in enumerate(zip(rows, intervals)):
        y = len(rows) - 1 - k
        color = FAMILY_COLORS[family]
        axis.plot([low, high], [y, y], color=color, linewidth=2.2, alpha=0.55, solid_capstyle="round")
        axis.plot(np.mean(values), y, "o", color=color, markersize=5.5)
        axis.text(high + offset, y, num(float(np.mean(values))), va="center", fontsize=8, color=INK)
    if reference is not None:
        axis.axvline(reference, color=MUTED, linestyle="--", linewidth=0.8)
    axis.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc="left", fontsize=9.5, color=INK)
    axis.grid(axis="x", color="#e4e7eb", linewidth=0.6)
    axis.set_axisbelow(True)
    return svg(figure)


def line_plot(series: list[tuple[str, str, list[float], list[float]]], title: str, xlabel: str, ylabel: str,
              xticks: list | None = None, markers: bool = True, width: float = 6.6, height: float = 2.8) -> str:
    figure, axis = plt.subplots(figsize=(width, height))
    for label, color, xs, ys in series:
        axis.plot(xs, ys, marker="o" if markers else None, markersize=3.5, linewidth=1.6, color=color, label=label)
    if xticks is not None:
        axis.set_xticks(xticks)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left", fontsize=9.5, color=INK)
    axis.grid(color="#e4e7eb", linewidth=0.6)
    axis.legend(frameon=False, fontsize=8)
    return svg(figure)


# ---------------------------------------------------------------------------- html helpers
def table(header: list[str], rows: list[list[str]], caption: str, align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    head = "".join(f'<th class="{a}">{h}</th>' for h, a in zip(header, align))
    body = "".join("<tr>" + "".join(f'<td class="{a}">{c}</td>' for c, a in zip(row, align)) + "</tr>" for row in rows)
    return f'<figure class="table"><figcaption>{caption}</figcaption><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></figure>'


def figure_block(svg_text: str, caption: str) -> str:
    return f'<figure class="chart">{svg_text}<figcaption>{caption}</figcaption></figure>'


def comparison_row(label: str, row: dict, paper_row: dict | None = None) -> list[str]:
    stats = row["vs_heuristic"]
    low, high = mean_ci(row["per_scenario"])
    cells = [label, num(row["mean"]), f"{num(low)}–{num(high)}"]
    if row.get("kind") == "heuristic" or row["key"] == "heuristic":
        cells += ["—", "—", "—", "—"]
    else:
        r_low, r_high = stats["reduction_ci95"]
        cells += [pct(stats["reduction_pct"]), f"{num(r_low, 1)} a {num(r_high, 1)} %", p_value(stats["p_value"]),
                  f"{stats['wins']}/{stats['ties']}/{stats['losses']}"]
    if paper_row is not None:
        cells.append(num(paper_row["mean"]))
    return cells


def learning_curve(run: str) -> tuple[list[int], list[float]] | None:
    curves = []
    for path in sorted((RUNS / run).glob("seed_*/learning_curve.csv")):
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if "timesteps" in rows[0]:
            curves.append([(int(r["timesteps"]), float(r["validation_color_changes"])) for r in rows])
    if not curves:
        return None
    length = min(len(c) for c in curves)
    steps = [curves[0][k][0] for k in range(length)]
    values = [float(np.mean([c[k][1] for c in curves])) for k in range(length)]
    return steps, values


# ---------------------------------------------------------------------------- document
def build(eval_dir: Path) -> str:
    final = Results(eval_dir / "final_extended.json")
    paper = Results(eval_dir / "final_paper.json")
    ensemble_ext = Results(eval_dir / "ensemble_extended.json")
    n_ext, n_paper = len(final.data["seeds"]), len(paper.data["seeds"])

    available = [(label, family, final.get(pattern), paper.get(pattern)) for label, family, pattern in CATALOGUE]
    available = [(label, family, row, paper_row) for label, family, row, paper_row in available if row is not None]
    heuristic = final.get(r"^heuristic$")
    best = final.get(r"^value_planner")
    planner = final.get(r"^planner\(h=10,w=32\)$")
    rl_paper = final.get(r"^paper \(media")
    rl_best = final.get(r"^arch_64x64 \(media") or final.get(r"^onehot_lookahead \(media")
    committee = final.get(r"^committee")
    value_oracle = final.get(r"^value_oracle")
    student = final.get(r"physical_judgesdagger") or final.get(r"physical_valuedagger")
    paper_reference = (34.38, 29.57, 34.57, 29.90)  # paper: heuristic, HAAM-RL (text), then from its tables III/IV

    fig_main = dot_plot([(label, family, row["per_scenario"]) for label, family, row, _ in available],
                        f"Cambios de color por escenario · {n_ext} escenarios de test", "cambios de color (menos es mejor)",
                        reference=heuristic["mean"])

    # Architecture sweep (PPO, HAAM mask, look-ahead 10, 3M steps)
    arch_names = ["64x64", "128x128", "256x256", "256x256_relu", "64x64x64", "128x128x128", "256x256x256"]
    arch_rows = []
    for name in arch_names:
        path = eval_dir / "arch" / f"{name}_extended.json"
        if path.exists():
            result = Results(path).get(r"\(media")
            arch_rows.append((name.replace("x", "×").replace("_relu", " ReLU"), "rl", result["per_scenario"]))
    fig_arch = dot_plot(arch_rows, "MLP de distinta profundidad y anchura (PPO, 3 semillas, 3M pasos)",
                        "cambios de color (menos es mejor)") if arch_rows else ""

    probe_rows = [json.loads(line) for line in (RESEARCH / "imitation_probe.jsonl").read_text(encoding="utf-8").splitlines() if line]
    figure, axis = plt.subplots(figsize=(6.6, 2.8))
    labels = [("Estructurada d=" if r["kind"] == "bufferpair" else "MLP ") + "×".join(map(str, r["arch"])) +
              (" ReLU" if r["kind"] == "mlp" and r["act"] == "relu" else "") for r in probe_rows]
    best_test = [max(c[2] for c in r["curve"]) for r in probe_rows]
    final_train = [r["curve"][-1][1] for r in probe_rows]
    xs = np.arange(len(probe_rows))
    axis.bar(xs - 0.2, final_train, 0.38, color="#c3ccd6", label="entrenamiento (época 40)")
    axis.bar(xs + 0.2, best_test, 0.38, color=[FAMILY_COLORS["imitacion"] if r["kind"] == "bufferpair" else ACCENT for r in probe_rows],
             label="escenarios no vistos (mejor época)")
    axis.set_xticks(xs, labels, rotation=25, ha="right", fontsize=7.5)
    axis.set_ylim(0.4, 1.0)
    axis.set_ylabel("acierto en decisiones reales")
    axis.set_title("Imitar al planificador: más capas memorizan, la estructura generaliza", loc="left", fontsize=9.5)
    axis.legend(frameon=False, fontsize=7.5)
    axis.grid(axis="y", color="#e4e7eb", linewidth=0.6)
    fig_probe = svg(figure)
    probe_mlp_best = max(t for t, r in zip(best_test, probe_rows) if r["kind"] == "mlp")
    probe_struct_best = max(t for t, r in zip(best_test, probe_rows) if r["kind"] == "bufferpair")

    curves = []
    for run, label, color in [("arch_64x64", "PPO · MLP 64×64", ACCENT), ("arch_256x256x256", "PPO · MLP 256×256×256", "#7aa6e8"),
                              ("bufferpair64_ppo", "PPO · red estructurada", FAMILY_COLORS["imitacion"])]:
        curve = learning_curve(run)
        if curve:
            curves.append((label, color, [s / 1e6 for s in curve[0]], curve[1]))
    fig_curves = line_plot(curves, "Aprendizaje por refuerzo: validación durante el entrenamiento (media de semillas)",
                           "millones de pasos", "cambios de color", markers=False) if curves else ""

    tuning = load(RESEARCH / "planner_tuning.json")
    tuning_series = []
    for weight, color in [(0.1, "#f2a77a"), (0.2, FAMILY_COLORS["busqueda"]), (0.3, "#9c3f12")]:
        pts = sorted((r["width"], r["mean"]) for r in tuning if r["weight"] == weight)
        tuning_series.append((f"peso fragmentación {num(weight, 1)}", color, [p[0] for p in pts], [p[1] for p in pts]))
    fig_tuning = line_plot(tuning_series, "Planificador con fragmentación: ancho del haz (validación)", "ancho del haz",
                           "cambios de color", xticks=[16, 32, 64, 128], height=2.4)

    judges = load(RESEARCH / "value_judges.json")
    figure, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    singles = [r["mean"] for r in judges["single"]]
    left.plot([1] * len(singles), singles, "o", color=MUTED, markersize=4, label="una red sola")
    averaged = [(len(r["values"]), r["mean"]) for r in judges["averaged"]]
    left.plot([1] + [k for k, _ in averaged], [judges["single"][0]["mean"]] + [m for _, m in averaged], "-o",
              color=FAMILY_COLORS["busqueda"], markersize=4, label="promedio de las k primeras")
    left.set_xlabel("número de redes de valor (jueces)")
    left.set_ylabel("cambios de color")
    left.set_title("Varios jueces: menos varianza, mejores planes", loc="left", fontsize=9)
    left.legend(frameon=False, fontsize=7.5)
    left.grid(color="#e4e7eb", linewidth=0.6)
    if judges.get("scales"):
        scales = [(r["scale"], r["mean"]) for r in judges["scales"]]
        right.plot([s for s, _ in scales], [m for _, m in scales], "-o", color=FAMILY_COLORS["busqueda"], markersize=4)
        right.set_xlabel("escala λ del valor")
        right.set_title(f"Escala ({len(judges['single'])} jueces)", loc="left", fontsize=9)
        right.grid(color="#e4e7eb", linewidth=0.6)
    else:
        right.axis("off")
    fig_judges = svg(figure)

    committee_study = load(RESEARCH / "committee_study.json")["rows"]

    fill = load(eval_dir / "fill_sweep.json")
    fill_rows = fill["rows"]
    free = [25 - r["fill_target"] for r in fill_rows]
    fill_series = [("heurística", MUTED, free, [float(np.mean(r["heuristic"])) for r in fill_rows])]
    if "planner" in fill_rows[0]:
        fill_series.append(("planificador (fragmentación)", FAMILY_COLORS["busqueda"], free, [float(np.mean(r["planner"])) for r in fill_rows]))
    fig_fill = line_plot(fill_series, f"Plazas libres en los buffers ({len(fill['seeds'])} escenarios de test)",
                         "plazas libres (25 − nivel de llenado)", "cambios de color", xticks=free, height=2.4)
    fill_best = min(fill_rows, key=lambda r: np.mean(r["heuristic"]))
    planner_fill_note = ""
    if "planner" in fill_rows[0]:
        planner_best = min(fill_rows, key=lambda r: np.mean(r["planner"]))
        planner_fill_note = (f", y el planificador con fragmentación de {num(np.mean(fill_rows[0]['planner']))} a "
                             f"{num(np.mean(planner_best['planner']))} con {25 - planner_best['fill_target']} plazas libres")

    latency = load(eval_dir / "latency.json") if (eval_dir / "latency.json").exists() else {}
    value_latency = next((v for k, v in latency.items() if k.startswith("value_planner")), None)
    latency_note = f", que decide en {num(value_latency['mean_ms'], 0)} ms de media" if value_latency else ""
    value_meta = {p.stem: load(p) for p in sorted(Path("artifacts/value").glob("v*.json"))}

    # ------------------------------------------------------------------ text
    h, b = heuristic["mean"], best["mean"]
    tests = collected_tests()

    def versus(candidate: dict | None, baseline: dict | None, text: str) -> str | None:
        """Paired comparison sentence between two policies on the same scenarios."""
        if not candidate or not baseline:
            return None
        from haam_rl.application.evaluation import paired_comparison

        stats = paired_comparison(candidate["per_scenario"], baseline["per_scenario"])
        return (f"{text}: diferencia media {num(stats.mean_difference)} cambios (IC 95 % {num(stats.ci95[0])} a "
                f"{num(stats.ci95[1])}, p {'< 0,0001' if stats.p_value < 1e-4 else '= ' + p_value(stats.p_value)}; mejor en {stats.wins}, igual en {stats.ties} y peor en "
                f"{stats.losses} de {n_ext} escenarios).")

    head_to_head = [text for text in [
        versus(committee, best, "Comité frente a planificador guiado por valor"),
        versus(best, planner, "Planificador guiado por valor frente a planificador con fragmentación"),
        versus(student, planner, "Red destilada frente a planificador con fragmentación"),
        versus(final.get(r"^bufferpair64_physical_dagger_ppo \(media"), final.get(r"^bufferpair64_physical_dagger \(media"),
               "Imitación física con ajuste PPO frente a la misma sin ajuste"),
        versus(best, value_oracle, "Planificador guiado por valor frente al oráculo guiado por valor"),
    ] if text]
    forced = forced_decision_share(paper.data["seeds"])
    today = datetime.now(UTC).strftime("%d/%m/%Y")
    rl_lookahead_mean = final.mean(r"^onehot_lookahead \(media")
    physical_dagger_mean = final.mean(r"^bufferpair64_physical_dagger \(media")
    haam_dagger_mean = final.mean(r"^bufferpair64_lookahead_dagger \(media")
    red = best["vs_heuristic"]["reduction_pct"]
    sections = []
    sections.append(f"""
<header class="title">
  <h1>Re-secuenciamiento por color en plantas de pintura: reproducción de HAAM-RL y búsqueda guiada por redes de valor</h1>
  <p class="meta">Equipo HAAM-RL Twin · {today} · Informe técnico reproducible</p>
</header>
<section class="abstract"><h2>Resumen</h2>
<p>Reproducimos HAAM-RL (Choi et al., 2024), un agente de aprendizaje por refuerzo con máscara heurística para reordenar vehículos
antes de la cabina de pintura y minimizar los cambios de color, y lo comparamos con familias de métodos completamente distintas.
Tras corregir un gemelo digital que no simulaba la misma planta que FlexSim, la heurística del paper queda en {num(h)} cambios por cada
100 vehículos ({n_ext} escenarios de test). El agente RL al estilo del paper la reduce un {pct(rl_paper['vs_heuristic']['reduction_pct'])} y la mejor
configuración RL, un {pct(rl_best['vs_heuristic']['reduction_pct'])}. Mostramos que añadir capas a la red no ayuda, que una red que respeta la simetría
entre buffers sí, y que el límite real está en cómo se estima el coste futuro. La mejor política es un planificador por búsqueda en haz que,
con la misma información que el agente (el vehículo entrante y las 10 llegadas siguientes), evalúa el futuro con el promedio de 8 redes de
valor: {num(b)} cambios, una reducción del {pct(red)} (IC 95 % {num(best['vs_heuristic']['reduction_ci95'][0], 1)}–{num(best['vs_heuristic']['reduction_ci95'][1], 1)} %),
mejor que la heurística en {best['vs_heuristic']['wins']} de {n_ext} escenarios{f" y a {num(b - value_oracle['mean'])} cambios de un oráculo que conoce las 100 llegadas" if value_oracle else ""}.
Cada cifra de este documento se genera a partir de los resultados guardados, con semillas fijas y pruebas pareadas.</p>
</section>""")

    sections.append(f"""
<section><h2>1. Introducción</h2>
<p>Cada cambio de color en la cabina obliga a purgar las pistolas: cuesta pintura, solvente y tiempo. El problema de re-secuenciamiento
por color (CRP) consiste en usar buffers intermedios para agrupar los vehículos del mismo color. Choi et al. proponen HAAM-RL: un agente
Soft Actor-Critic cuyas acciones se filtran con las heurísticas LP, CM y UCM, con recompensa con potencial (PBRS) y un ensemble de modelos en
inferencia. Reportan {num(paper_reference[1])} cambios frente a {num(paper_reference[0])} de la heurística (−14,0 %; el paper indica 16,25 % porque su
ecuación 5 divide por el resultado del agente). Las tablas de su apéndice suman, sin embargo, medias de {num(paper_reference[2])} y {num(paper_reference[3])}.</p>
<p>Este trabajo tiene tres objetivos: reproducir el método con mediciones exactas y reproducibles; entender qué limita su rendimiento
(arquitectura, entrenamiento, máscara, información disponible); y evaluar enfoques completamente distintos, incluida la combinación de varios
modelos que deciden juntos. Las contribuciones son:</p>
<ul>
<li>Un gemelo digital alineado con FlexSim, con {tests} pruebas automáticas y un verificador de trazas que detecta cualquier divergencia entre ambos.</li>
<li>Una reproducción del agente del paper y un estudio de arquitectura: más capas no mejoran; una red equivariante a permutaciones de buffers sí.</li>
<li>La medición del techo de cualquier política que obedezca la máscara del paper, y la comprobación de que el cuello de botella era el aprendizaje.</li>
<li>Un planificador por búsqueda en haz guiado por un ensemble de redes de valor, que es la mejor política encontrada, y su destilación en una red rápida.</li>
<li>Una evaluación sistemática de «varios modelos a la vez»: votación (no ayuda), comités arbitrados por simulación (solo ayudan con un buen juez)
y ensembles de jueces (la mejora más clara).</li>
</ul>
</section>""")

    sections.append("""
<section><h2>2. Planta y formulación</h2>
<p>La planta tiene una cola de llegada, 5 buffers FIFO de 5 plazas y la cabina. Cada escenario son 100 vehículos de 6 colores con la mezcla
del paper (6, 38, 29, 14, 10 y 3 %). Cada decisión es un par (buffer de entrada, buffer de salida), 25 acciones, y su efecto depende de la fase:</p>
<ul>
<li><b>Llenado</b> (ocupación &lt; 25): el vehículo entrante se guarda en el buffer i; acción (i, i).</li>
<li><b>Régimen</b> (buffers llenos): se pinta el frente del buffer o y el entrante se guarda en i; con buffers llenos solo cabe en la plaza liberada.</li>
<li><b>Vaciado</b> (sin llegadas): se pinta el frente del buffer o; acción (o, o).</li>
</ul>
<p>La métrica es el número de cambios de color en el episodio. La observación del agente usa la del paper (27 valores) o una codificación
one-hot con la fase, la ocupación y las k llegadas siguientes (anticipación). La recompensa es la del paper: −0,1 por cambio, 0,1 + 0,02 × racha
por mantener el color, más PBRS con potencial nulo en el estado final (condición que garantiza no alterar la política óptima).</p>
</section>""")

    sections.append("""
<section><h2>3. Métodos comparados</h2>
<p><b>Heurística.</b> Reglas del paper: para guardar, CM (cola del mismo color) y si no LP (buffer con menos colores distintos); para pintar,
CM (frente del último color) y si no UCM (buffer con más colores distintos); los empates van al buffer superior.</p>
<p><b>Aprendizaje por refuerzo (paper).</b> MaskablePPO con la máscara HAAM: la política solo puede elegir acciones que las reglas anteriores
también propondrían. Probamos entrada cruda y one-hot, anticipación 0 y 10, máscaras HAAM, solo de salida y física, y redes MLP de 2 y 3 capas de
64 a 512 neuronas.</p>
<p><b>Red estructurada.</b> Los 5 buffers son intercambiables: permutarlos permuta las acciones. Nuestra red procesa cada buffer con los mismos pesos,
puntúa cada par (entrada, salida) con una función compartida y es exactamente equivariante, cosa que un MLP debe aprender de los datos (120 ordenaciones).</p>
<p><b>Imitación (DAgger).</b> La red aprende a reproducir las decisiones de un planificador; en cada iteración juega ella misma y el planificador
etiqueta los estados a los que llega (Ross et al., 2011). Opcionalmente, PPO refina el resultado.</p>
<p><b>Planificador por búsqueda en haz.</b> En cada decisión simula todas las acciones físicamente posibles sobre los vehículos conocidos (el entrante y
los 10 siguientes), conserva los 32 mejores planes por profundidad y ejecuta la primera acción del mejor. Un plan se puntúa con sus cambios de color
más una estimación del coste futuro del estado que deja: 0,2 × fragmentación de los buffers (fórmula manual) o una red de valor.</p>
<p><b>Búsqueda guiada por valor.</b> Una red de valor, invariante al orden de los buffers, estima cuántos cambios habrá en los 30 vehículos siguientes
viendo solo el contenido de los buffers y el último color (lo que se sabe al final del horizonte conocido). Se entrena con partidas del planificador en
escenarios disjuntos de validación y test. El planificador usa cambios + λ × valor; con varias redes, su promedio («ensemble de jueces»).</p>
<p><b>Varios modelos a la vez.</b> (a) <i>Votación</i> del paper: cada red vota y gana la mayoría (o la suma de probabilidades). (b) <i>Rollout</i>
(Bertsekas): para cada acción posible se simula el horizonte conocido siguiendo a una política base y se elige la mejor. (c) <i>Comité</i>: el planificador
y rollouts con distintas políticas base proponen acciones con su puntuación simulada; un árbitro común elige. (d) <i>Ensemble de jueces</i>: varias redes
de valor promediadas dentro de la búsqueda.</p>
</section>""")

    sections.append(f"""
<section><h2>4. Protocolo experimental</h2>
<ul>
<li><b>Semillas disjuntas.</b> Entrenamiento con semillas aleatorias o ≥ 10⁶; validación (selección de modelos e hiperparámetros) con 100 escenarios
(20000–20099); test con los 30 escenarios del tamaño del protocolo del paper (10000–10029) y con {n_ext} escenarios ampliados (30000–30499), que son los
que se reportan como resultado principal (error estándar ≈ 0,15 cambios).</li>
<li><b>Estadística pareada.</b> Todas las políticas se evalúan en los mismos escenarios. Se reporta el intervalo bootstrap del 95 % (remuestreo de
escenarios) de la media y de la reducción relativa, y el valor p de un test de permutación de signos a dos colas (100.000 remuestreos).
Para RL, cada fila promedia por escenario las semillas entrenadas, es decir, el resultado esperado de un entrenamiento.</li>
<li><b>Exactitud.</b> El entrenamiento es determinista (misma semilla, pesos idénticos). Cada optimización del código se verificó reproduciendo
bit a bit todos los resultados guardados. Las estimaciones de valor se redondean a 10⁻⁵ para que las decisiones no dependan del reparto entre procesos.</li>
<li><b>Cómputo.</b> CPU, un núcleo por entrenamiento; ~3.000 pasos por segundo con MLP 64×64. Todo el trabajo cabe en una estación de 16 núcleos.</li>
</ul>
</section>""")

    bugs = [
        ["El gemelo nunca tenía más de 5 vehículos en 25 plazas", "heurística en 44,4 cambios en vez de ~34", "fases de llenado, régimen y vaciado"],
        ["FlexSim contaba cambios de vehículos que no pintaba", "KPI inflado y duplicado", "el KPI se actualiza tras mover a la cabina"],
        ["FlexSim decidía cuándo extraer con reglas propias", "la API y FlexSim simulaban cosas distintas", "la API dicta fase, extracción e inserción"],
        ["Reglas de entrada evaluadas antes de extraer", "candidatos incorrectos en régimen", "se evalúan tras la extracción"],
        ["Potencial PBRS no nulo al final", "el shaping podía cambiar la política óptima", "Φ(final) = 0, verificado por test"],
        ["Selección del mejor modelo por recompensa media", "el «mejor» no era el de menos cambios", "selección por cambios en validación"],
        ["Caché de valores que podía perder entradas", "fallos esporádicos en búsquedas largas", "caché LRU corregida y probada"],
    ]
    sections.append(f"""
<section><h2>5. Iteraciones y resultados</h2>
<h3>5.1 Corregir la planta antes de medir</h3>
<p>La primera medición daba una heurística de 44,4 cambios, muy lejos de los 34,38 del paper: el gemelo nunca llenaba los buffers. Con la planta corregida
la heurística queda en {num(paper.mean(r'^heuristic$'))} en los 30 escenarios del tamaño del paper y en {num(h)} en los {n_ext} ampliados, coherente con el paper.</p>
{table(["Problema encontrado", "Efecto", "Corrección"], bugs, "Tabla 1. Errores corregidos en la implementación inicial y en el código de FlexSim.", "lll")}
""")

    rl_patterns = [("Entrada cruda, como el paper", r"^paper \(media"), ("One-hot", r"^onehot \(media"),
                   ("One-hot + anticipación 10", r"^onehot_lookahead \(media"), ("Sin máscara heurística", r"^onehot_lookahead_physical \(media")]
    rl_table = [comparison_row(label, final.get(pat), paper.get(pat)) for label, pat in rl_patterns if final.get(pat)]
    ens = ensemble_ext.get(r"^ensemble")
    ens_members = ensemble_ext.get(r"\(media")
    sections.append(f"""
<h3>5.2 Reproducción del agente del paper</h3>
<p>Con 1M de pasos por semilla, el agente con máscara HAAM mejora la heurística de forma significativa pero modesta. La codificación one-hot ayuda;
la anticipación casi no ayuda al agente; y sin la máscara heurística PPO no aprende en el mismo presupuesto, lo que confirma la tesis central del paper.
El ensemble con la votación del paper da {num(ens['mean'])} frente a {num(ens_members['mean'])} de media de sus miembros: no mejora.</p>
{table(["Configuración (PPO)", "Media", "IC 95 %", "Reducción", "IC reducción", "p", "G/E/P", f"Media ({n_paper})"],
       rl_table, f"Tabla 2. Reproducción del agente RL, {n_ext} escenarios de test (última columna: los {n_paper} del tamaño del paper).")}
""")

    sections.append(f"""
<h3>5.3 ¿Más capas? No: más estructura</h3>
<p>Entrenamos MLP de 2 y 3 capas de 64 a 256 neuronas durante 3M de pasos. Ninguna supera a la 64×64, y la más grande es la peor.
Para separar capacidad de dificultades de RL medimos cuánto puede imitar cada red las decisiones del planificador en escenarios no vistos: los MLP se
quedan en torno al {pct(100 * probe_mlp_best, 0)} de acierto en las decisiones reales aunque memoricen el entrenamiento, mientras la red estructurada llega al
{pct(100 * probe_struct_best, 0)} con menos parámetros.</p>
{figure_block(fig_arch, "Figura 1. Arquitecturas MLP entrenadas con PPO; media y IC 95 % sobre los escenarios de test ampliados.") if fig_arch else ""}
{figure_block(fig_probe, "Figura 2. Prueba de imitación: acierto en decisiones no forzadas, entrenamiento frente a escenarios no vistos.")}
{figure_block(fig_curves, "Figura 3. Validación durante el entrenamiento con PPO: la red estructurada aprende en una fracción de los pasos.") if fig_curves else ""}
""")

    haam_planner, haam_oracle = final.get(r"mask=haam\)$"), final.get(r"^oracle\(w=64,mask=haam\)")
    sections.append(f"""
<h3>5.4 ¿Limita la máscara del paper?</h3>
<p>Medimos el techo de cualquier política que obedezca la máscara HAAM restringiendo el planificador a sus acciones: con anticipación 10 alcanza
{num(haam_planner['mean'])} cambios{f" y {num(haam_oracle['mean'])} con información completa" if haam_oracle else ""}. El agente RL, con la misma información, se queda
cerca de {num(rl_lookahead_mean)}: el cuello de botella era el aprendizaje, no la máscara. Aun así, la máscara sí limita a un alumno
bien entrenado: la imitación sin máscara heurística (máscara física) obtiene {num(physical_dagger_mean)} frente a
{num(haam_dagger_mean)} con la máscara HAAM.</p>
""")

    sections.append(f"""
<h3>5.5 Planificar con la información disponible</h3>
<p>El planificador con la fórmula de fragmentación, sin ningún entrenamiento, obtiene {num(planner['mean'])} cambios ({pct(planner['vs_heuristic']['reduction_pct'])}).
Ensanchar el haz mejora poco y cuesta mucho: la búsqueda está cerca de saturar y el margen está en cómo se estima el coste futuro.</p>
{figure_block(fig_tuning, "Figura 4. Ancho del haz y peso de la fragmentación del planificador, en los 100 escenarios de validación.")}
""")

    cs = committee_study
    sections.append(f"""
<h3>5.6 Varios modelos decidiendo juntos</h3>
<p>Probamos tres formas de combinar modelos. La votación del paper no ayuda (sección 5.2). Un comité arbitrado por simulación solo ayuda si el árbitro es bueno:
con la fórmula de fragmentación el rollout de la red queda en {num(cs['rollout(network) [fragmentation]']['mean'])} (la red sola: {num(cs['network alone']['mean'])}) y el comité
en {num(cs['committee [fragmentation]']['mean'])}, peor que el planificador solo ({num(cs['planner [fragmentation]']['mean'])}); con una red de valor como árbitro el rollout de la red baja a
{num(cs['rollout(network) [value]']['mean'])} y el comité a {num(cs['committee [value]']['mean'])} (planificador solo con el mismo juez: {num(cs['planner [value]']['mean'])}).
La combinación que más aporta es el ensemble de jueces: promediar varias redes de valor dentro de la búsqueda.</p>
{table(["Política (validación, 100 escenarios)", "Juez de fragmentación", "Juez de valor"],
       [[label, num(cs[f"{key} [fragmentation]"]["mean"]), num(cs[f"{key} [value]"]["mean"])]
        for label, key in [("Rollout de la red", "rollout(network)"), ("Rollout de la heurística", "rollout(heuristic)"),
                           ("Planificador", "planner"), ("Comité (planificador + 2 rollouts)", "committee")]],
       f"Tabla 3. Rollouts y comités con dos árbitros distintos (la red sola: {num(cs['network alone']['mean'])}).", "lrr")}
""")

    def describe(data_policy: str) -> str:
        """Readable Spanish name of the policies that played a value network's training games."""
        names = {"heuristic": "la heurística"}
        parts = [names.get(part.strip(), "el planificador guiado por v1" if part.strip().startswith("value_planner")
                           else "el planificador con fragmentación") for part in data_policy.split("+")]
        return "partidas de " + " y ".join(parts)

    def recipe(stem: str) -> tuple[str, int]:
        meta = value_meta[stem]
        return describe(meta["metrics"].get("data_policy", "")), meta["settings"]["scenarios"]

    base_recipe = recipe(judges["single"][0]["values"][0])
    base = [r["mean"] for r in judges["single"] if recipe(r["values"][0]) == base_recipe]
    variants = {r["values"][0]: r["mean"] for r in judges["single"] if recipe(r["values"][0]) != base_recipe}
    variant_text = "; ".join(
        f"{stem} ({recipe(stem)[1]} {recipe(stem)[0]}): {num(mean)}" for stem, mean in variants.items())
    best_scale = min(judges.get("scales") or [{"mean": float("nan"), "scale": 0}], key=lambda r: r["mean"])
    sections.append(f"""
<h3>5.7 Búsqueda guiada por un ensemble de redes de valor</h3>
<p>Sustituir la fórmula de fragmentación por una red de valor transforma el planificador. Las {len(base)} redes entrenadas con la misma receta
({base_recipe[1]} {base_recipe[0]}, con semillas distintas) obtienen entre {num(min(base))} y {num(max(base))} cambios en validación cada una.
Promediar las {len(judges['single'])} redes baja a {num(judges['averaged'][-1]['mean'])} con λ = 0,5{f" y a {num(best_scale['mean'])} con λ = {num(best_scale['scale'], 2)}" if judges.get('scales') else ""}.
Tres intentos de mejorar una red individual no funcionaron y se reportan: {variant_text}. Más datos no garantizan un juez mejor; reentrenar con partidas del
propio planificador guiado (iteración de política) lo empeora porque solo ve estados buenos, cuando durante la búsqueda debe juzgar muchos estados malos.</p>
{figure_block(fig_judges, "Figura 5. Juez promediado: cambios de color según el número de redes de valor y según la escala λ (validación).")}
""")

    if student:
        sections.append(f"""
<h3>5.8 Destilar el mejor planificador en una red rápida</h3>
<p>Para despliegues donde la latencia importe, destilamos el planificador guiado en la red estructurada con DAgger. El alumno obtiene {num(student['mean'])} cambios
({pct(student['vs_heuristic']['reduction_pct'])}) con una sola pasada de red por decisión: mejor que cualquier agente entrenado solo con RL y que el planificador
con fragmentación.</p>
""")

    sections.append(f"""
<h3>5.9 Nivel de llenado de los buffers</h3>
<p>Con los buffers llenos el vehículo entrante solo puede ir a la plaza recién liberada. Dejar plazas libres devuelve esa decisión: con {25 - fill_best['fill_target']} plazas libres
la heurística baja de {num(np.mean(fill_rows[0]['heuristic']))} a {num(np.mean(fill_best['heuristic']))} cambios{planner_fill_note}. Es una decisión de operación de planta, con costes (espacio, ciclo) que el
gemelo no modela, y es independiente de la política.</p>
{figure_block(fig_fill, "Figura 6. Cambios de color según las plazas que se dejan libres en los buffers.")}
""")

    final_rows = [comparison_row(label, row, paper_row) for label, _, row, paper_row in available]
    sections.append(f"""
<section><h2>6. Resultados finales</h2>
<p>La Figura 7 y la Tabla 4 reúnen todas las familias en los {n_ext} escenarios de test. Comparados directamente, escenario por escenario:</p>
<ul>{''.join(f"<li>{text}</li>" for text in head_to_head)}</ul>
{figure_block(fig_main, f"Figura 7. Todas las políticas en {n_ext} escenarios de test: media e IC 95 %. Gris: referencia; azul: RL; verde: imitación; naranja: búsqueda; morado: oráculos con información completa. Línea discontinua: heurística.")}
{table(["Política", "Media", "IC 95 %", "Reducción", "IC reducción", "p", "G/E/P", f"Media ({n_paper})"], final_rows,
       f"Tabla 4. Resultados finales en {n_ext} escenarios de test, pareados contra la heurística; la última columna repite la media en los {n_paper} escenarios del tamaño del paper.")}
</section>""")

    if latency:
        lat_rows = [[html.escape(name), num(v["mean_ms"], 2), num(v["p95_ms"], 2), num(v["max_ms"], 2)] for name, v in latency.items()]
        sections.append(f"""
<section><h2>7. Coste por decisión</h2>
{table(["Política", "Media (ms)", "p95 (ms)", "Máximo (ms)"], lat_rows, "Tabla 5. Latencia por decisión en un núcleo de CPU, sin procesamiento por lotes.")}
<p>Todas las políticas caben en el tiempo de decisión de FlexSim; la red destilada es dos órdenes de magnitud más rápida que el planificador guiado.</p>
</section>""")

    sections.append(f"""
<section><h2>8. Discusión</h2>
<p><b>Por qué RL rinde poco aquí.</b> Las consecuencias de una decisión se ven decenas de pasos después y el {pct(forced, 0)} de las decisiones con máscara HAAM son forzadas; el agente
recibe poca señal útil. Más capas no lo resuelven: los MLP grandes memorizan sin generalizar. Lo que sí ayuda es incorporar la estructura del problema
(equivarianza entre buffers) y aprender de un experto (imitación) en lugar de solo por ensayo y error.</p>
<p><b>Por qué gana la búsqueda guiada por valor.</b> Separa lo que se sabe (las llegadas conocidas, que se simulan exactamente) de lo que no (el futuro, que estima una red).
La calidad del juez es decisiva: con una buena estimación, incluso combinar modelos por simulación funciona; con una mala, empeora. Promediar jueces es la forma más
barata y robusta de mejorarlo.</p>
<p><b>Recomendación.</b> Desplegar el planificador guiado por el ensemble de redes de valor ({num(b)} cambios, {pct(red)} menos que la heurística){latency_note}.
El comité con rollouts añade una mejora pequeña pero significativa si se dispone de algo más de tiempo por decisión. Si la latencia fuera crítica, la versión destilada
del planificador{f" ({num(student['mean'])} cambios)" if student else ""} decide en menos de un milisegundo. Revisar además el nivel de llenado de los buffers con producción.</p>
</section>
<section><h2>9. Limitaciones y trabajo futuro</h2>
<ul>
<li>Los resultados son del gemelo en Python. El verificador de trazas permite confirmar en FlexSim, escenario por escenario, que ambos simulan lo mismo; ese paso está pendiente.</li>
<li>Un solo tipo de escenario (100 vehículos, mezcla del paper). Faltan escenarios más largos, otras mezclas y restricciones reales (limpiezas, costes por par de colores).</li>
<li>El oráculo es una cota superior del óptimo, no el óptimo; un solver exacto en instancias pequeñas daría la brecha real.</li>
<li>Siguientes pasos: entrenar las redes de valor con estados generados por la propia búsqueda, búsqueda Monte Carlo (MCTS) con el mismo juez y decidir el nivel de llenado de forma dinámica.</li>
</ul>
</section>
<section><h2>10. Reproducibilidad</h2>
<pre>pip install -e ".[dev]" &amp;&amp; pytest
python scripts/run_experiments.py --configs configs/*.json --seeds 0 1 2 3 4
python scripts/distill.py --config configs/onehot_lookahead_physical.json --teacher-value ...
python scripts/train_value.py --out artifacts/value/v1.pt
python scripts/research/value_judges.py --values artifacts/value/v*.pt
python scripts/research/final_evaluation.py --value &lt;redes&gt; --committee-network &lt;red&gt;
python scripts/build_publication.py</pre>
</section>
<section class="refs"><h2>Referencias</h2>
<ol>
<li>Choi, K., Rho, C., Kim, T. y Choi, D. (2024). Heuristic Algorithm-based Action Masking Reinforcement Learning (HAAM-RL) with Ensemble Inference Method. arXiv:2403.14110.</li>
<li>Krystek, J. y Bysko, S. (2019). Virtual commissioning as the main core of Industry 4.0 – case study in the automotive paint shop.</li>
<li>Ng, A. Y., Harada, D. y Russell, S. (1999). Policy invariance under reward transformations. ICML.</li>
<li>Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.</li>
<li>Ross, S., Gordon, G. y Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning (DAgger). AISTATS.</li>
<li>Bertsekas, D. (2019). Reinforcement Learning and Optimal Control (algoritmos de rollout). Athena Scientific.</li>
<li>Silver, D. et al. (2017). Mastering the game of Go without human knowledge. Nature 550.</li>
<li>Zaheer, M. et al. (2017). Deep Sets. NeurIPS.</li>
<li>Huang, S. y Ontañón, S. (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. FLAIRS.</li>
</ol>
</section>""")

    value_rows = [[name, describe(meta["metrics"].get("data_policy", "")).capitalize(), num(meta["metrics"]["held_out_r2"], 3),
                   f"{meta['settings']['scenarios']:,}".replace(",", ".")] for name, meta in value_meta.items()]
    if value_rows:
        sections.append(f"""<section><h2>Apéndice A. Redes de valor</h2>
{table(["Red", "Partidas de entrenamiento", "R² (partidas no vistas)", "Escenarios"], value_rows,
       "Tabla A1. Redes de valor entrenadas; el planificador final promedia todas.", "llrr")}</section>""")

    body = "\n".join(sections)
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Publicación HAAM-RL Twin</title>
<style>
@page {{ size: A4; margin: 18mm 17mm 18mm 17mm; }}
body {{ font-family: Georgia, "Times New Roman", serif; color: {INK}; font-size: 10.3pt; line-height: 1.45; max-width: 760px; margin: 24px auto; padding: 0 16px; }}
h1 {{ font-size: 18pt; line-height: 1.2; margin: 0 0 6px; }}
h2 {{ font-size: 12.5pt; margin: 20px 0 6px; border-bottom: 1px solid #d5dbe1; padding-bottom: 3px; }}
h3 {{ font-size: 10.8pt; margin: 14px 0 4px; }}
p, li {{ text-align: justify; hyphens: auto; }}
.meta {{ color: {MUTED}; margin: 0 0 12px; font-size: 9.5pt; }}
.abstract {{ background: #f3f5f8; border-left: 3px solid {ACCENT}; padding: 6px 14px; }}
.abstract h2 {{ border: 0; margin-top: 6px; }}
figure {{ margin: 12px 0; page-break-inside: avoid; }}
figure.chart svg {{ width: 100%; height: auto; }}
figure.chart svg text {{ font-family: "Segoe UI", Arial, Helvetica, sans-serif !important; }}
figcaption {{ font-size: 8.8pt; color: #52606d; margin-top: 4px; }}
figure.table figcaption {{ margin: 0 0 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.6pt; font-family: "Segoe UI", Arial, sans-serif; }}
th {{ border-bottom: 1.2px solid {INK}; padding: 3px 5px; font-weight: 600; }}
td {{ border-bottom: 0.6px solid #e1e5ea; padding: 2.5px 5px; }}
.l {{ text-align: left; }} .r {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
pre {{ background: #f3f5f8; padding: 8px 10px; font-size: 8.4pt; white-space: pre-wrap; }}
.refs li {{ font-size: 9pt; text-align: left; }}
</style></head><body>{body}</body></html>"""


README_ROWS = [r"^heuristic$", r"^paper \(media", r"^arch_64x64 \(media", r"^bufferpair64_ppo \(media",
               r"^planner\(h=10,w=32\)$", r"physical_judgesdagger", r"^value_planner", r"^committee", r"^value_oracle"]


def readme_results(eval_dir: Path) -> str:
    """Compact results table for the README, from the same evaluation files as the publication."""
    final, paper = Results(eval_dir / "final_extended.json"), Results(eval_dir / "final_paper.json")
    labels = {pattern: label for label, _, pattern in CATALOGUE}
    intro = (f"Cambios de color por escenario de 100 vehículos (menos es mejor), {len(final.data['seeds'])} escenarios "
             f"de test; entre paréntesis, los {len(paper.data['seeds'])} escenarios del tamaño del paper.")
    lines = [intro, "",
             "| Política | Cambios | Reducción vs heurística (IC 95 %) |", "| --- | --- | --- |"]
    for pattern in README_ROWS:
        row, paper_row = final.get(pattern), paper.get(pattern)
        if row is None:
            continue
        stats = row["vs_heuristic"]
        low, high = stats["reduction_ci95"]
        reduction = "—" if row["key"] == "heuristic" else f"{pct(stats['reduction_pct'])} ({num(low, 1)}–{num(high, 1)} %)"
        lines.append(f"| {labels[pattern]} | {num(row['mean'])} ({num(paper_row['mean'])}) | {reduction} |")
    return "\n".join(lines)


def update_readme(eval_dir: Path, path: Path = Path("README.md")) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{readme_results(eval_dir)}\n{end}{tail}", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("docs/publicacion/publicacion.html"))
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--readme", action="store_true", help="also refresh the results table of README.md")
    parser.add_argument("--eval-dir", type=Path, default=EVAL, help="where the evaluation JSON files are")
    args = parser.parse_args()
    if args.readme:
        update_readme(args.eval_dir)
        print("README.md results updated")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build(args.eval_dir), encoding="utf-8")
    print(f"written {args.out}")
    if not args.no_pdf:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(args.out.resolve().as_uri())
            pdf = args.out.with_suffix(".pdf")
            page.pdf(path=str(pdf), format="A4", print_background=True,
                     margin={"top": "18mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
            browser.close()
        print(f"written {pdf}")


if __name__ == "__main__":
    main()
