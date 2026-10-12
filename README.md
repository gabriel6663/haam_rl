# HAAM-RL Twin

Reproducción, análisis y superación de **HAAM-RL** (Choi et al., 2024, [arXiv:2403.14110](https://arxiv.org/abs/2403.14110)) para el re-secuenciamiento por color en una planta de pintura automotriz: decidir en qué buffer FIFO se guarda cada vehículo y de cuál se pinta, para minimizar los cambios de color.

La publicación completa, con todas las pruebas e iteraciones, está en [docs/publicacion/publicacion.pdf](docs/publicacion/publicacion.pdf) (fuente: `docs/publicacion/publicacion.html`). La genera `scripts/build_publication.py` directamente desde los resultados guardados, así que ninguna cifra está escrita a mano.

## Resultados principales

<!-- RESULTS:START -->
Cambios de color por escenario de 100 vehículos (menos es mejor), 500 escenarios de test; entre paréntesis, los 30 escenarios del tamaño del paper.

| Política | Cambios | Reducción vs heurística (IC 95 %) |
| --- | --- | --- |
| Heurística LP/CM/UCM (referencia del paper) | 31,73 (32,33) | — |
| RL · entrada cruda, como el paper (5 semillas) | 31,11 (31,15) | 2,0 % (1,2–2,7 %) |
| RL · MLP 64×64, 3M pasos (3 semillas) | 29,32 (29,94) | 7,6 % (6,8–8,4 %) |
| RL · red estructurada (3 semillas) | 28,45 (29,01) | 10,3 % (9,5–11,1 %) |
| Planificador · fragmentación | 22,99 (23,27) | 27,6 % (26,8–28,4 %) |
| Imitación del planificador con 8 jueces (red 128) | 21,79 (21,83) | 31,3 % (30,5–32,1 %) |
| Planificador guiado por valor (8 jueces) | 16,83 (16,63) | 46,9 % (46,4–47,5 %) |
| Comité: planificador + rollouts, juez de valor | 16,69 (16,77) | 47,4 % (46,8–48,0 %) |
| Oráculo guiado por valor (información completa) | 14,25 (14,47) | 55,1 % (54,6–55,6 %) |
<!-- RESULTS:END -->

## Qué incluye

| Componente | Para qué sirve |
| --- | --- |
| Gemelo digital en Python | Mismas reglas que el modelo de FlexSim; 13.000 pasos por segundo |
| Heurística LP/CM/UCM | La referencia del paper |
| Aprendizaje por refuerzo (MaskablePPO) | El método del paper, con máscara heurística, PBRS y redes MLP o estructuradas |
| Red estructurada (`buffer_pair`) | Trata los 5 buffers como intercambiables (equivariante); aprende mucho más rápido que un MLP |
| Imitación (DAgger) | Destila un planificador en una red rápida |
| Planificador por búsqueda en haz | Simula las llegadas conocidas; sin entrenamiento |
| Planificador guiado por valor | El planificador con redes de valor que estiman el coste futuro; **la mejor política del proyecto** |
| Rollout, comité y ensembles | «Varios modelos a la vez»: votación del paper, comités arbitrados por simulación y ensembles de jueces |
| API HTTP | La llama FlexSim en cada decisión; también sirve el inspector web |
| Inspector web | Red, pesos, activaciones, sensibilidad, gemelo paso a paso y comparación de políticas |
| Verificador de trazas | Comprueba, decisión por decisión, que FlexSim simula lo mismo que el gemelo |

## Instalación

Requiere Python 3.11–3.13. En cualquier sistema operativo:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

No hace falta GPU: las redes son pequeñas y el cuello de botella es la simulación, así que todo corre en CPU (un núcleo por entrenamiento). Para generar la publicación: `pip install -e ".[docs]"` y `playwright install chromium`.

## Uso rápido

```bash
# Servir la mejor política para FlexSim (y el inspector web en http://127.0.0.1:8000/ui/)
HAAM_PLANNER=10 HAAM_VALUE=artifacts/value/v1.pt,artifacts/value/v5.pt,... \
  python -m uvicorn haam_rl.adapters.http_api:app --host 127.0.0.1 --port 8000

# Entrenar un agente RL (una carpeta por semilla en artifacts/runs/<experimento>/seed_<n>/)
python scripts/train.py --config configs/onehot_lookahead.json --seeds 0 1 2

# Destilar el planificador guiado por valor en una red rápida
python scripts/distill.py --config configs/onehot_lookahead_physical.json --teacher-value artifacts/value/v1.pt \
  --set training.policy=buffer_pair training.embed_dim=128 training.activation=relu

# Entrenar una red de valor (juez) con partidas del planificador
python scripts/train_value.py --out artifacts/value/v1.pt --workers 8

# Comparar políticas en 500 escenarios de test, pareadas contra la heurística
python scripts/evaluate.py --seed-set extended --models artifacts/runs/onehot --planner 10 \
  --value artifacts/value/v1.pt,artifacts/value/v5.pt
```

En Windows con PowerShell, las variables de entorno se definen con `$env:HAAM_PLANNER = "10"`.

## Política recomendada para FlexSim

El planificador guiado por el promedio de varias redes de valor (`HAAM_PLANNER=10` + `HAAM_VALUE` con todas las redes de `artifacts/value/`). Usa la misma información que el agente del paper: el vehículo entrante y los 10 siguientes de la cola. Decide en decenas de milisegundos. Si la latencia fuera crítica, la red destilada de ese planificador decide en menos de un milisegundo con algo menos de calidad.

Las variables de entorno de la API:

| Variable | Efecto |
| --- | --- |
| `HAAM_PLANNER=10` | Arranca con el planificador; el valor es la anticipación |
| `HAAM_VALUE=a.pt,b.pt,...` | Con `HAAM_PLANNER`: redes de valor que guían la búsqueda (varias se promedian) |
| `HAAM_MODELS=a.zip,b.zip` | Arranca con un modelo (o un ensemble si son varios) |
| `HAAM_TRACE=traza.jsonl` | Registra cada decisión para verificarla con `scripts/check_trace.py` |

Sin variables, la API arranca con la heurística. Desde la web, el botón «Modelos…» permite cambiar de política sin reiniciar.

## Integración con FlexSim

[flexsim/decision_trigger.fs](flexsim/decision_trigger.fs) es el trigger de decisión. En cada llamada:

1. Envía a `POST /step` el estado de 27 valores y las próximas llegadas de la cola.
2. Ejecuta lo que la API responde (`phase`, `input_buffer`, `output_buffer`): primero extrae hacia pintura, después guarda el vehículo entrante.
3. Actualiza el KPI de cambios de color solo cuando un vehículo entra realmente a la máquina.

Para validar FlexSim contra el gemelo, levanta la API con `HAAM_TRACE=flexsim.jsonl`, corre escenarios en FlexSim y ejecuta `python scripts/check_trace.py flexsim.jsonl`. El script reproduce cada decisión con las reglas del gemelo, informa de cualquier diferencia de estado y recalcula los cambios de color de cada episodio.

Cada decisión es una pareja (buffer de entrada, buffer de salida), con `action_id = entrada * 5 + salida`:

| Fase | Cuándo | Qué hace la acción |
| --- | --- | --- |
| llenado | hay vehículo esperando y menos de `fill_target` en buffers | guarda en el buffer *i*; acción (*i*, *i*) |
| régimen | hay vehículo esperando y buffers en `fill_target` | pinta el frente del buffer *o*, luego guarda en *i* |
| vaciado | no hay vehículo esperando | pinta el frente del buffer *o*; acción (*o*, *o*) |

## Experimentos y reproducibilidad

Cada experimento de RL es un JSON en `configs/` (`configs/arch/` para el barrido de arquitectura, `configs/bufferpair/` para la red estructurada, `configs/extra/` para el nivel de llenado). Cualquier campo se puede cambiar desde la línea de comandos con `--set seccion.campo=valor`. Cada modelo guardado lleva su JSON al lado, con planta, observación, máscara, recompensa y entrenamiento, así que siempre se puede recargar sabiendo cómo se entrenó.

Semillas separadas, para que la selección de modelos nunca vea el test:

| Uso | Semillas |
| --- | --- |
| Entrenamiento | aleatorias o ≥ 1.000.000 |
| Validación (selección de modelos e hiperparámetros) | 20000–20099 |
| Test del tamaño del paper | 10000–10029 |
| Test ampliado (resultado principal) | 30000–30499 |

Todos los números de la publicación se regeneran con:

```bash
python scripts/research/final_evaluation.py --value <redes de valor separadas por comas> --committee-network <red>
python scripts/build_publication.py
```

| Script | Qué hace |
| --- | --- |
| `scripts/train.py`, `scripts/run_experiments.py` | Entrenar con PPO, una o muchas configuraciones y semillas en paralelo |
| `scripts/distill.py` | DAgger: destilar un planificador en una red |
| `scripts/train_value.py` | Entrenar una red de valor con partidas de una o varias políticas |
| `scripts/evaluate.py` | Evaluar cualquier combinación de políticas con estadística pareada |
| `scripts/report.py` | Tablas en Markdown desde los JSON de evaluación |
| `scripts/fill_sweep.py` | Barrido del nivel de llenado de los buffers |
| `scripts/benchmark.py` | Latencia por decisión de cada política |
| `scripts/check_trace.py` | Verificar una traza de FlexSim contra el gemelo |
| `scripts/research/*.py` | Estudios de la publicación (imitación, ajuste del planificador, jueces, comités, evaluación final) |
| `scripts/build_publication.py` | Generar la publicación (HTML y PDF) |

El entrenamiento es determinista (misma semilla, mismos pesos) y las evaluaciones no dependen del número de procesos.

## Estructura del código (arquitectura hexagonal)

```
src/haam_rl/
  domain/        reglas puras: sin ML ni HTTP
    model.py         estado, acciones, fases
    dynamics.py      qué acciones son ejecutables y qué hacen
    heuristics.py    reglas LP / CM / UCM, máscaras y política heurística
    reward.py        recompensa del paper + PBRS
    scenarios.py     llegadas y semillas de validación y test
    planning.py      búsqueda en haz sobre las llegadas conocidas, con estimación de hojas intercambiable
  application/   casos de uso que solo dependen del dominio
    ports.py         interfaces (planta, política, decisión)
    use_cases.py     heurística, planificador y ensemble con la votación del paper
    hybrid.py        rollout y comité: varios modelos arbitrados por simulación
    evaluation.py    episodios en paralelo lógico y comparación pareada (bootstrap y permutación)
  adapters/      todo lo que toca librerías externas
    python_twin.py   gemelo digital
    gym_env.py       entorno Gymnasium para SB3
    observation.py   codificación de la entrada (cruda / one-hot / anticipación)
    equivariant.py   red estructurada por buffers (BufferPairPolicy)
    sb3_policy.py    modelo entrenado como política, con inferencia por lotes
    training.py      PPO y selección del mejor modelo por cambios de color
    distillation.py  DAgger
    value_search.py  redes de valor y planificador guiado por valor
    runner.py        evaluación rápida: por lotes o en procesos
    model_inspector.py  pesos, activaciones y sensibilidad para la web
    http_api.py      API para FlexSim y el inspector
  config.py      configuración de experimentos (JSON)
web/index.html   inspector (sin dependencias externas)
flexsim/         trigger de FlexSim
docs/publicacion publicación generada
tests/           dominio, aplicación, políticas híbridas y API
```

## Inspector web

Se abre en `http://127.0.0.1:8000/ui/` con la API levantada. Permite:

- **Elegir la política.** Heurística, uno o varios modelos (ensemble), el planificador o el planificador guiado por una o varias redes de valor.
- **Ver la red.** Para MLP: actor y crítico capa a capa, con aristas por peso o por peso × activación. Para la red estructurada: todas sus matrices de pesos.
- **Editar un estado** a mano (o pegar el vector de FlexSim) y ver la decisión, las probabilidades, la acción de la heurística y los votos del ensemble.
- **Medir la sensibilidad** de cada posición del estado.
- **Jugar episodios** del gemelo paso a paso, con la política, con la heurística o a mano.
- **Comparar** la política con la heurística (y con el oráculo) en N escenarios.
