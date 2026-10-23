# HAAM-RL Twin

Reproducción, análisis y superación de **HAAM-RL** (Choi et al., 2024, [arXiv:2403.14110](https://arxiv.org/abs/2403.14110)) para el re-secuenciamiento por color en una planta de pintura automotriz: decidir en qué buffer FIFO se guarda cada vehículo y de cuál se pinta, para minimizar los cambios de color.

La publicación completa, con todas las pruebas e iteraciones, está en [docs/publicacion/publicacion.pdf](docs/publicacion/publicacion.pdf).

## Resultados principales

<!-- RESULTS:START -->
Cambios de color por escenario de 100 vehículos (menos es mejor), 500 escenarios de test; entre paréntesis, los 30 escenarios del tamaño del paper. Los planificadores, el comité y la red destilada conocen los 10 vehículos siguientes de la cola; si la planta solo conoce el próximo, ver [Qué información necesita cada política](#qué-información-necesita-cada-política).

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

## Qué información necesita cada política

En cada decisión, FlexSim envía a la API el estado de la planta (el vehículo que espera, el último color pintado y los 25 puestos de los buffers) y, opcionalmente, la **cola**: los colores de los vehículos que vienen detrás, en orden. El agente del paper solo usa el estado; conocer la cola («anticipación») es la mejora que el paper propone como trabajo futuro.

- **Sin cola (anticipación 0).** Solo se conoce el próximo vehículo. Sirve en cualquier planta.
- **Con cola (anticipación 10).** La planta conoce la secuencia de los 10 vehículos siguientes, y FlexSim los tiene en orden en su cola de espera. Solo vale si esa secuencia es fiable.

La misma política en los dos casos (500 escenarios de test; entre paréntesis, la reducción frente a la heurística; tiempo medio por decisión en un núcleo, sin lotes):

<!-- INFORMATION:START -->
| Política | Sin cola: cambios | Sin cola: tiempo por decisión | Cola de 10: cambios | Cola de 10: tiempo por decisión |
| --- | --- | --- | --- | --- |
| Heurística LP/CM/UCM | 31,73 | 0,02 ms (p95 0,03) | 31,73 | 0,02 ms (p95 0,03) |
| RL · one-hot, MLP 64×64 (5 semillas) | 29,62 (6,7 %) | 0,19 ms (p95 0,23) | 30,09 (5,2 %) | 0,19 ms (p95 0,23) |
| Planificador con fragmentación | 29,97 (5,6 %) | 0,91 ms (p95 7,5) | 22,99 (27,6 %) | 6,28 ms (p95 14) |
| Planificador guiado por 8 redes de valor | 24,34 (23,3 %) | 4,45 ms (p95 20) | 16,83 (46,9 %) | 42,86 ms (p95 116) |
| Comité: planificador + rollouts, juez de valor | — | — | 16,69 (47,4 %) | 54,47 ms (p95 132) |
<!-- INFORMATION:END -->

Ventajas de cada modo:

| | Sin cola | Con cola de 10 |
| --- | --- | --- |
| Qué necesita de la planta | Solo el próximo vehículo | La secuencia de los 10 siguientes, fiable |
| Mejora frente a la heurística | Unos 23 % menos cambios | Unos 47 % menos cambios (algo más con el comité) |
| Tiempo por decisión | Unos 4 ms (≈ 10 veces más rápido) | Unos 43 ms (55 ms con el comité) |
| Riesgo | Ninguno adicional | Si la cola real no coincide con la anunciada, planifica sobre llegadas equivocadas |
| Configuración | `HAAM_PLANNER=0` + `HAAM_VALUE`; trigger con `LOOKAHEAD = 0` | `HAAM_PLANNER=10` + `HAAM_VALUE` (+ `HAAM_COMMITTEE`); trigger con `LOOKAHEAD = 10` |

Sin cola, el planificador guiado por valor sigue siendo la mejor opción: es mejor que la heurística en casi todos los escenarios y que cualquier agente de RL, incluso los que veían la cola. Las redes de valor no ven la cola en ningún caso (solo juzgan el contenido de los buffers), así que sirven en los dos modos. El comité no se evaluó sin cola, porque su red se entrenó viendo la cola.

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

**Requisitos.** Python 3.11, 3.12 o 3.13 (`python --version`). No hace falta GPU: las redes son pequeñas (unos 55.000 parámetros cada red de valor) y el cuello de botella es la simulación, así que todo corre en CPU. Las dependencias principales son PyTorch, Stable-Baselines3 + sb3-contrib (PPO con máscara de acciones), Gymnasium, NumPy y FastAPI; las instala `pip`.

**1. Crear un entorno virtual** en la carpeta del proyecto, para no mezclar estas librerías con las del sistema:

```powershell
# Windows (PowerShell)
cd C:\ruta\a\haam_rl_twin
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
cd /ruta/a/haam_rl_twin
python -m venv .venv
source .venv/bin/activate
```

Si PowerShell no deja ejecutar `Activate.ps1`, permítelo una vez para tu usuario con `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. El entorno se activa en cada terminal nueva antes de trabajar; el prompt muestra `(.venv)`.

**2. Instalar el proyecto** (en modo editable, con las herramientas de pruebas):

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Con una GPU NVIDIA, `pip` puede descargar la versión de PyTorch con CUDA, que es mucho más pesada y no aporta nada aquí. Para instalar solo la de CPU, antes del paso anterior: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

**3. Comprobar la instalación:**

```bash
pytest
```

Todas las pruebas deben pasar (tardan unos 15 segundos).

**4. Archivos entrenados.** La carpeta `artifacts/` (redes de valor, modelos y resultados) no está en git. Si te la pasaron aparte, cópiala en la raíz del proyecto: con `artifacts/value/v1.pt` … `v8.pt` ya se puede usar la mejor política sin entrenar nada. Si no la tienes, la sección siguiente explica cómo entrenarlo todo.

**5. Probar que funciona:** levanta la API (ver [Política recomendada para FlexSim](#política-recomendada-para-flexsim)) y abre el inspector en `http://127.0.0.1:8000/ui/`.

## Entrenamiento

Todos los comandos se ejecutan desde la raíz del proyecto, con el entorno activado. Cada script admite `--help`. Lo que se entrena queda en `artifacts/`; volver a entrenar con el mismo `--out` **sobrescribe** el archivo, así que para experimentar conviene usar otro nombre o carpeta (por ejemplo `artifacts/value_nuevas/`; la API solo carga archivos que estén dentro de `artifacts/`).

### Las 8 redes de valor (los jueces del planificador)

**Qué son.** Una red de valor mira el contenido de los 5 buffers, el último color pintado y la ocupación, y estima **cuántos cambios de color habrá en los próximos 30 vehículos pintados**. No ve el vehículo entrante ni la cola. El planificador la usa para juzgar el futuro de cada plan: elige la acción que minimiza `cambios del plan + 0,75 × V(estado final)`. Una red sola es ruidosa; el promedio de 8 redes («juez conjunto») decide mucho mejor (en validación, 17,10 cambios frente a 19,4–20,0 de cada red sola).

**Arquitectura.** Igual en las 8 (`ValueNetwork` en `src/haam_rl/adapters/value_search.py`): un mismo codificador lee cada buffer (43 entradas → 128 → 128, ReLU), los 5 códigos se combinan con su media y su máximo (así el resultado no depende del orden de los buffers), y una cabeza (256 → 128 → 1) da V(s).

**Cómo se entrena una red** (`scripts/train_value.py`), en dos fases:

1. **Recolección de datos.** Una «política de datos» juega `--scenarios` escenarios completos del gemelo. De cada estado se guarda la entrada de la red y el objetivo: los cambios de color que hubo realmente en los 30 vehículos pintados siguientes. Los escenarios usan semillas a partir de 7.000.000 + 100.000 × `--seed`, separadas de las de validación y test. Corre en paralelo con `--workers` procesos.
2. **Ajuste.** Regresión con error cuadrático medio: Adam, tasa de aprendizaje 0,001, lotes de 1024, 30 épocas. El 20 % de los escenarios se reserva para medir el R² en partidas no vistas; ese R² se guarda junto a la red y se muestra en la web.

Cada red produce dos archivos: `vN.pt` (los pesos) y `vN.json` (configuración, R² y política de datos).

**Receta exacta de las 8 redes del proyecto.** Cinco usan la receta base con semillas distintas; las otras tres son variantes que individualmente fueron peores, pero aportan diversidad al promedio:

```bash
# Receta base: 1200 partidas del planificador con fragmentación (anticipación 10, ancho 32)
python scripts/train_value.py --data-policy planner --scenarios 1200 --seed 0 --workers 6 --out artifacts/value/v1.pt
python scripts/train_value.py --data-policy planner --scenarios 1200 --seed 4 --workers 6 --out artifacts/value/v5.pt
python scripts/train_value.py --data-policy planner --scenarios 1200 --seed 5 --workers 6 --out artifacts/value/v6.pt
python scripts/train_value.py --data-policy planner --scenarios 1200 --seed 6 --workers 6 --out artifacts/value/v7.pt
python scripts/train_value.py --data-policy planner --scenarios 1200 --seed 7 --workers 6 --out artifacts/value/v8.pt

# v2: partidas del planificador guiado por v1 (iteración de política); requiere v1
python scripts/train_value.py --data-policy artifacts/value/v1.pt --scenarios 1200 --seed 1 --workers 6 --out artifacts/value/v2.pt

# v3: más datos (3000 partidas)
python scripts/train_value.py --data-policy planner --scenarios 3000 --seed 2 --workers 6 --out artifacts/value/v3.pt

# v4: datos mezclados, mitad planificador y mitad heurística
python scripts/train_value.py --data-policy planner heuristic --scenarios 3000 --seed 3 --workers 6 --out artifacts/value/v4.pt
```

| Red | Partidas de entrenamiento | R² en partidas no vistas |
| --- | --- | --- |
| v1, v5, v6, v7, v8 | 1200 del planificador con fragmentación (semillas 0, 4, 5, 6, 7) | 0,65–0,70 |
| v2 | 1200 del planificador guiado por v1 | 0,56 |
| v3 | 3000 del planificador con fragmentación | 0,68 |
| v4 | 1500 del planificador + 1500 de la heurística | 0,51 |

Cada red tarda del orden de 10–20 minutos con 6 procesos; casi todo el tiempo es la recolección de partidas. Las redes son independientes entre sí (salvo v2, que necesita v1), así que se pueden entrenar varias a la vez si hay núcleos libres. Ajusta `--workers` a los núcleos de tu equipo.

**Comprobar el juez** en los 100 escenarios de validación, antes de mirar el test:

```bash
python scripts/evaluate.py --seed-set validation --value-horizon 0 \
  --value artifacts/value/v1.pt,artifacts/value/v2.pt,artifacts/value/v3.pt,artifacts/value/v4.pt,artifacts/value/v5.pt,artifacts/value/v6.pt,artifacts/value/v7.pt,artifacts/value/v8.pt
```

Con las 8 redes del proyecto da 24,82 cambios frente a 32,18 de la heurística (−22,9 %); si unas redes nuevas quedan claramente por encima, algo salió mal. `--value-horizon 0` evalúa el planificador sin cola (solo el próximo vehículo); `10` lo evalúa con la cola. La escala 0,75 es la elegida en validación (`--value-scale`).

**Usar redes nuevas.** Basta con listarlas en `HAAM_VALUE` (separadas por comas) al levantar la API, o elegirlas en la web en «Modelos…» → «red de valor». Se puede usar cualquier número de redes: se promedian.

### Agentes de aprendizaje por refuerzo (el método del paper)

`scripts/train.py` entrena con MaskablePPO una configuración de `configs/` y deja una carpeta por semilla en `artifacts/runs/<experimento>/seed_<n>/`. Durante el entrenamiento evalúa periódicamente en validación y guarda como `best_model.zip` el punto con menos cambios de color (y `final_model.zip` al terminar). Cada modelo lleva al lado un `.json` con toda su configuración.

```bash
# El agente como en el paper (entrada cruda, máscara heurística), 5 semillas
python scripts/train.py --config configs/paper.json --seeds 0 1 2 3 4 --timesteps 1000000

# La red estructurada por buffers, que aprende mejor que un MLP
python scripts/train.py --config configs/bufferpair/ppo_scratch.json --seeds 0 1 2

# Cualquier campo se cambia sin editar el JSON
python scripts/train.py --config configs/onehot.json --set training.net_arch=[256,256] training.learning_rate=0.0001

# Muchas configuraciones y semillas en paralelo (una por núcleo)
python scripts/run_experiments.py --seeds 0 1 2 3 4 --timesteps 1000000 --parallel 4   # todas las de configs/
```

Con 1M de pasos, un entrenamiento tarda unos 8–10 minutos por semilla en un núcleo. En Windows conviene dejar los valores por defecto (`--vec-env dummy --threads 1`), porque muchos procesos con PyTorch agotan la memoria.

### Red destilada (imitación del planificador guiado)

Para cuando haga falta decidir en menos de un milisegundo, `scripts/distill.py` enseña a una red a imitar al planificador guiado por las 8 redes de valor (DAgger: la red juega, el planificador corrige sus decisiones y la red reaprende). La red del proyecto (`bufferpair128_physical_judgesdagger`) se entrenó así, en unos 90 minutos:

```bash
python scripts/distill.py --config configs/onehot_lookahead_physical.json --seeds 0 --workers 13 --iterations 16 \
  --teacher-value artifacts/value/v1.pt,artifacts/value/v2.pt,artifacts/value/v3.pt,artifacts/value/v4.pt,artifacts/value/v5.pt,artifacts/value/v6.pt,artifacts/value/v7.pt,artifacts/value/v8.pt \
  --teacher-value-scale 0.75 --name bufferpair128_physical_judgesdagger \
  --set training.policy=buffer_pair training.embed_dim=128 training.activation=relu
```

Esta red ve la cola de 10 vehículos (la usa también el comité).

En Windows con PowerShell, las barras invertidas `\` del final de línea de estos ejemplos se reemplazan por el acento grave `` ` ``, o se escribe todo en una sola línea; las variables de entorno se definen con `$env:NOMBRE = "valor"`.

## Política recomendada para FlexSim

El planificador guiado por el promedio de las 8 redes de valor de `artifacts/value/`. La anticipación depende de lo que sepa la planta (ver [Qué información necesita cada política](#qué-información-necesita-cada-política)):

```powershell
# Solo se conoce el próximo vehículo (por defecto en flexsim/decision_trigger.fs: LOOKAHEAD = 0)
$env:HAAM_PLANNER = "0"
$env:HAAM_VALUE   = "artifacts/value/v1.pt,artifacts/value/v2.pt,artifacts/value/v3.pt,artifacts/value/v4.pt,artifacts/value/v5.pt,artifacts/value/v6.pt,artifacts/value/v7.pt,artifacts/value/v8.pt"
python -m uvicorn haam_rl.adapters.http_api:app --host 127.0.0.1 --port 8000

# La planta conoce los 10 siguientes (poner LOOKAHEAD = 10 en el trigger)
$env:HAAM_PLANNER   = "10"
$env:HAAM_COMMITTEE = "artifacts/runs/bufferpair128_physical_judgesdagger/seed_0/best_model.zip"   # opcional
```

`GET /health` confirma el modo (`value_planner` o `committee`) y la anticipación (`lookahead`). Si la latencia fuera crítica con cola, la red destilada de ese planificador decide en menos de un milisegundo con algo menos de calidad.

Las variables de entorno de la API:

| Variable | Efecto |
| --- | --- |
| `HAAM_PLANNER=0` o `10` | Arranca con el planificador; el valor es la anticipación (cuántos vehículos de la cola mira) |
| `HAAM_VALUE=a.pt,b.pt,...` | Con `HAAM_PLANNER`: redes de valor que guían la búsqueda (varias se promedian) |
| `HAAM_MODELS=a.zip,b.zip` | Arranca con un modelo (o un ensemble si son varios) |
| `HAAM_TRACE=traza.jsonl` | Registra cada decisión para verificarla con `scripts/check_trace.py` |

Sin variables, la API arranca con la heurística. Desde la web, el botón «Modelos…» permite cambiar de política sin reiniciar.

## Integración con FlexSim

[flexsim/decision_trigger.fs](flexsim/decision_trigger.fs) es el trigger de decisión. En cada llamada:

1. Envía a `POST /step` el estado de 27 valores y, si `LOOKAHEAD` > 0, las próximas llegadas de la cola.
2. Ejecuta lo que la API responde (`phase`, `input_buffer`, `output_buffer`): primero extrae hacia pintura, después guarda el vehículo entrante.
3. Actualiza el KPI de cambios de color solo cuando un vehículo entra realmente a la máquina.

Al inicio del trigger se configuran `TOTAL_VEHICLES` (vehículos por episodio, 100) y `LOOKAHEAD` (0 por defecto). Si los vehículos llegan con el tiempo y la cola queda vacía antes de que hayan llegado todos, el trigger espera al siguiente disparo en vez de vaciar los buffers; el modelo se detiene cuando se pintaron los `TOTAL_VEHICLES`.

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

La evaluación final (todas las políticas en los 30 y los 500 escenarios de test, latencias y barrido de llenado) se repite con:

```bash
python scripts/research/final_evaluation.py --value <redes de valor separadas por comas> --committee-network <red>
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
docs/publicacion publicación (PDF)
tests/           dominio, aplicación, políticas híbridas y API
```

## Inspector web

Se abre en `http://127.0.0.1:8000/ui/` con la API levantada. Permite:

- **Elegir la política.** Heurística, uno o varios modelos (ensemble), el planificador o el planificador guiado por una o varias redes de valor.
- **Ver la red.** Para MLP: actor y crítico capa a capa, con aristas por peso o por peso × activación. Para la red estructurada: todas sus matrices de pesos. Con el planificador guiado por valor (o el comité), cada una de las redes de valor («jueces»), elegidas en el selector «Red»: el codificador aplicado a cada buffer (B1–B5) y la cabeza que estima V(s), con sus pesos, activaciones y sensibilidad.
- **Ver por qué decide el planificador guiado.** En «Decisión», el botón «juez» muestra para cada acción el puntaje que minimiza el planificador a un paso (cambio de color + λ·V del estado siguiente, promediando los jueces), y la lista de V(s) de cada juez. Sin cola (anticipación 0) es exactamente su criterio en llenado y régimen.
- **Editar un estado** a mano (o pegar el vector de FlexSim) y ver la decisión, las probabilidades, la acción de la heurística y los votos del ensemble.
- **Medir la sensibilidad** de cada posición del estado.
- **Jugar episodios** del gemelo paso a paso, con la política, con la heurística o a mano.
- **Comparar** la política con la heurística (y con el oráculo) en N escenarios.
