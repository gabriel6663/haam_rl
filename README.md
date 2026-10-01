# HAAM-RL Twin

Implementacion nueva, separada del prototipo anterior, para el problema de re-secuenciacion de pintura. Usa un gemelo digital Python rapido para entrenar y deja FlexSim como adaptador de validacion, no como cuello de botella de cada iteracion.

## Que replica del paper

- Estado de 27 valores: color entrante, ultimo color pintado y 5 buffers de 5 posiciones.
- Espacio discreto de 25 acciones: `buffer_entrada x buffer_salida`.
- Buffers FIFO, capacidad 5, vehiculos de seis colores y distribucion del paper.
- Restricciones fisicas: no extraer de vacio, no insertar en lleno (salvo que esa misma accion primero extraiga del buffer), y no aceptar acciones cuando no queda vehiculo entrante.
- Mascara HAAM: LP/CM para entrada y CM/UCM para salida. La mascara se aplica a la distribucion de la politica con `MaskablePPO`; no es solo una penalizacion posterior.
- Recompensa de color y Potential-Based Reward Shaping (PBRS): el potencial reduce fragmentacion y prioriza autos disponibles del mismo color que el ultimo pintado.

No implementa ensemble aun: primero se debe demostrar que un modelo individual supera al baseline con las mismas semillas. Cuando eso ocurra, se pueden guardar varios modelos y sumar sus probabilidades enmascaradas.

## Arquitectura hexagonal

- `domain/`: reglas, estado, heuristicas y recompensa; no conoce Gym, HTTP ni FlexSim.
- `application/`: puertos y caso de uso de una decision.
- `adapters/`: gemelo digital, entorno Gymnasium y API HTTP. FlexSim puede llamar a la API sin contaminar el dominio.

## Instalacion en Windows + RTX 3060 Ti

Instala un driver NVIDIA actual. Luego, desde PowerShell en esta carpeta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Elegir el comando CUDA actual del selector oficial de PyTorch:
# https://pytorch.org/get-started/locally/
pip install -e ".[dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La red es pequena; la 3060 Ti es suficiente. El gemelo Python es intencionalmente rapido: la GPU no compensa un simulador externo lento por HTTP en cada paso.

## Ejecutar

```powershell
python scripts/train.py --timesteps 300000 --seed 42
python scripts/evaluate.py --model artifacts/maskable_ppo_haam.zip --scenarios 30
python -m uvicorn haam_rl.adapters.http_api:app --host 127.0.0.1 --port 8000
pytest
```

TensorBoard es opcional: el entrenamiento funciona sin instalarlo. Si quieren sus graficos, ejecuten `pip install tensorboard` y agreguen `--tensorboard` al comando de entrenamiento.

La evaluacion durante entrenamiento usa `MaskableEvalCallback`; no lo reemplacen por el `EvalCallback` de Stable-Baselines3 porque este ultimo no aplica la mascara y puede enviar acciones invalidas al entorno.

El modelo se guarda en `artifacts/maskable_ppo_haam.zip`. `evaluate.py` compara 30 escenarios fijos contra la heuristica determinista. Antes de afirmar mejora, agreguen validacion equivalente contra FlexSim y reporten: cambios de color, tasa de acciones invalidas (debe ser 0), throughput y distribucion de resultados.
