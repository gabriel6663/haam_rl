from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sb3_contrib import MaskablePPO

from haam_rl.application.use_cases import allowed_actions
from haam_rl.domain.model import PlantState

app = FastAPI(title="HAAM-RL Twin Bridge", version="0.1.0")
MODEL_PATH = Path(os.getenv("HAAM_MODEL_PATH", "artifacts/best/best_model.zip"))
model: MaskablePPO | None = MaskablePPO.load(MODEL_PATH, device="auto") if MODEL_PATH.exists() else None


class SystemState(BaseModel):
    state_vector: list[int] = Field(min_length=27, max_length=27)


class ActionResponse(BaseModel):
    action_id: int
    valid_action_count: int


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": model is not None, "model_path": str(MODEL_PATH)}


@app.post("/step", response_model=ActionResponse)
def take_action(payload: SystemState) -> ActionResponse:
    state = PlantState.from_vector(payload.state_vector)
    print(state)
    mask = allowed_actions(state)
    if not mask.any():
        raise HTTPException(status_code=409, detail="No valid HAAM action for this state")
    if model is None:
        action = int(np.flatnonzero(mask)[0])  # safe deterministic fallback, never an invalid action
    else:
        action, _ = model.predict(state.as_vector(), action_masks=mask, deterministic=True)
        action = int(action)
    return ActionResponse(action_id=action, valid_action_count=int(mask.sum()))
