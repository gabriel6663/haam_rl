"""HTTP bridge used by FlexSim (`POST /step`) and by the web inspector (everything else).

Run:  python -m uvicorn haam_rl.adapters.http_api:app --host 127.0.0.1 --port 8000
UI:   http://127.0.0.1:8000/ui/

Environment variables:
    HAAM_PLANNER        start with the look-ahead planner; the value is its horizon (e.g. 10).
    HAAM_VALUE          with HAAM_PLANNER: value network file(s), comma-separated (averaged), that guide
                        the planner's search. This is the recommended policy.
    HAAM_COMMITTEE      with HAAM_VALUE: a network zip; the value-guided planner then decides together
                        with rollouts of that network and of the heuristic (slightly better, slower).
    HAAM_MODELS         comma-separated model zips to load at start (several = ensemble).
    Without either variable the API answers with the heuristic until a policy is loaded.
    HAAM_ARTIFACTS_DIR  where /models looks for zips (default: artifacts)
    HAAM_TRACE          append every /step request and answer to this JSONL file; replay it with
                        scripts/check_trace.py to verify that FlexSim follows the twin's rules
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from haam_rl.adapters.model_inspector import PolicyInspector
from haam_rl.adapters.python_twin import PythonPaintShopTwin
from haam_rl.adapters.runner import evaluation_twin
from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.adapters.value_search import ValueGuidedPlannerPolicy
from haam_rl.application.evaluation import run_episode
from haam_rl.application.hybrid import CommitteePolicy, PlannerProposer, RolloutPolicy
from haam_rl.application.ports import Policy
from haam_rl.application.use_cases import EnsemblePolicy, HeuristicPolicy, LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig, config_path_for
from haam_rl.domain.dynamics import phase_of, physical_action_mask
from haam_rl.domain.heuristics import MaskMode, action_mask, heuristic_action
from haam_rl.domain.model import ACTION_COUNT, STATE_VECTOR_SIZE, Phase, PlantState, decode_action

app = FastAPI(title="HAAM-RL Twin Bridge", version="0.3.0")
# Local research tool: the UI may be opened from file:// or another port.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ARTIFACTS_DIR = Path(os.getenv("HAAM_ARTIFACTS_DIR", "artifacts")).resolve()
TRACE_PATH = Path(os.environ["HAAM_TRACE"]) if os.getenv("HAAM_TRACE") else None
_trace_lock = threading.Lock()
WEB_DIR = Path(os.getenv("HAAM_WEB_DIR", Path(__file__).resolve().parents[3] / "web"))


# ---------------------------------------------------------------- loaded policy


class Registry:
    """The policy currently answering requests: heuristic, planner, one model or an ensemble."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.members: list[SB3Policy] = []
        self.inspectors: list[PolicyInspector] = []
        self.paths: list[Path] = []
        self.config = ExperimentConfig()
        self.policy: Policy = HeuristicPolicy(self.config.plant.fill_target)

    def load(self, paths: list[Path], hard_threshold: float | None = None,
             planner: PlannerSpec | None = None) -> None:
        members = [SB3Policy.load(p) for p in paths]
        policy: Policy
        if planner is not None:
            config = ExperimentConfig().with_overrides(plant={"fill_target": planner.fill_target})
            if planner.value:
                values = [_artifact(v.strip(), ".pt") for v in planner.value.split(",") if v.strip()]
                policy = ValueGuidedPlannerPolicy(values, planner.fill_target, planner.horizon, planner.width,
                                                  planner.value_scale)
                if planner.committee_network:
                    referee = policy.estimate
                    network = SB3Policy.load(_artifact(planner.committee_network, ".zip"))
                    policy = CommitteePolicy([
                        PlannerProposer(planner.fill_target, planner.horizon, planner.width, estimate=referee),
                        RolloutPolicy(network, planner.fill_target, planner.horizon, estimate=referee),
                        RolloutPolicy(HeuristicPolicy(planner.fill_target), planner.fill_target, planner.horizon,
                                      estimate=referee),
                    ], planner.fill_target)
            else:
                policy = LookaheadPlannerPolicy(planner.fill_target, planner.horizon, planner.width)
        elif not members:
            config = ExperimentConfig()
            policy = HeuristicPolicy(config.plant.fill_target)
        else:
            config = members[0].config
            policy = members[0] if len(members) == 1 else EnsemblePolicy(members, hard_threshold=hard_threshold)
        with self._lock:
            self.members, self.paths, self.config, self.policy = members, [Path(p).resolve() for p in paths], config, policy
            self.inspectors = [PolicyInspector(m) for m in members]

    @property
    def mode(self) -> str:
        if isinstance(self.policy, CommitteePolicy):
            return "committee"
        if isinstance(self.policy, ValueGuidedPlannerPolicy):
            return "value_planner"
        if isinstance(self.policy, LookaheadPlannerPolicy):
            return "planner"
        return "heuristic" if not self.members else "model" if len(self.members) == 1 else "ensemble"

    @property
    def mask_mode(self) -> MaskMode:
        """Mask the current policy decides under (the heuristic proposes only HAAM actions)."""
        if self.members:
            return self.config.mask_mode
        return getattr(self.policy, "mask_mode", MaskMode.HAAM)

    @property
    def lookahead(self) -> int:
        if isinstance(self.policy, (LookaheadPlannerPolicy, ValueGuidedPlannerPolicy)):
            return self.policy.horizon
        if isinstance(self.policy, CommitteePolicy):
            return self.policy.members[0].horizon
        return max((m.config.observation.lookahead for m in self.members), default=0)


class PlannerSpec(BaseModel):
    horizon: int = Field(default=10, ge=0, le=50)
    width: int = Field(default=32, ge=1, le=256)
    fill_target: int = Field(default=25, ge=1, le=25)
    value: str | None = Field(default=None, description="value network(s) (.pt, relative to artifacts/, "
                                                        "comma-separated = averaged judges) guiding the search")
    value_scale: float = Field(default=0.75, gt=0)
    committee_network: str | None = Field(default=None, description="network zip for a committee with the value planner")


def _artifact(relative: str, suffix: str) -> Path:
    """A file inside artifacts/; anything outside it, or of another type, is refused."""
    path = Path(relative) if Path(relative).is_absolute() else (ARTIFACTS_DIR / relative)
    path = path.resolve()
    if not path.is_relative_to(ARTIFACTS_DIR) or path.suffix != suffix or not path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found inside artifacts/: {relative}")
    return path


registry = Registry()
if os.getenv("HAAM_PLANNER"):
    value = ",".join(str(Path(v).resolve()) for v in os.environ["HAAM_VALUE"].split(",")) if os.getenv("HAAM_VALUE") else None
    committee = str(Path(os.environ["HAAM_COMMITTEE"]).resolve()) if os.getenv("HAAM_COMMITTEE") else None
    registry.load([], planner=PlannerSpec(horizon=int(os.environ["HAAM_PLANNER"]), value=value, committee_network=committee))
elif os.getenv("HAAM_MODELS"):
    registry.load([Path(p.strip()) for p in os.environ["HAAM_MODELS"].split(",") if p.strip()])


# ---------------------------------------------------------------- request / response bodies


class StateRequest(BaseModel):
    state_vector: list[int] = Field(min_length=STATE_VECTOR_SIZE, max_length=STATE_VECTOR_SIZE,
                                    description="incoming, last painted, then 5 buffers x 5 slots (front first)")
    upcoming: list[int] = Field(default_factory=list, max_length=100,
                                description="vehicles queued after the incoming one, in arrival order")


class StepResponse(BaseModel):
    action_id: int
    input_buffer: int   # 1-based, as named in FlexSim
    output_buffer: int  # 1-based
    phase: Phase
    insert: bool        # store the incoming vehicle in input_buffer
    extract: bool       # move the front of output_buffer to the painting machine
    valid_action_count: int
    decided_by: str


class InspectRequest(StateRequest):
    member: int = 0


class LoadRequest(BaseModel):
    paths: list[str] = Field(default_factory=list, description="relative to artifacts/; empty = heuristic")
    vote_threshold: float | None = None
    planner: PlannerSpec | None = Field(default=None, description="use the look-ahead planner instead of models")


class ResetRequest(BaseModel):
    seed: int | None = None


class SimStepRequest(BaseModel):
    mode: Literal["model", "heuristic", "action"] = "model"
    action_id: int | None = Field(default=None, ge=0, lt=ACTION_COUNT)


class CompareRequest(BaseModel):
    first_seed: int = 10_000
    scenarios: int = Field(default=30, ge=1, le=200)
    oracle: bool = False
    oracle_width: int = Field(default=16, ge=1, le=256)


# ---------------------------------------------------------------- helpers


def _parse_state(request: StateRequest) -> PlantState:
    try:
        return PlantState.from_vector(request.state_vector, request.upcoming)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _fill_target() -> int:
    return registry.config.plant.fill_target


def _decision_payload(state: PlantState) -> dict:
    """Everything the UI shows about one decision, for any loaded policy."""
    fill = _fill_target()
    phase = phase_of(state, fill)
    if phase is Phase.DONE:
        raise HTTPException(status_code=409, detail="Episode is over: nothing waiting and every buffer empty")
    decision = registry.policy.decide(state)
    member_actions = decision.details.get("member_actions", []) if registry.mode == "ensemble" else []
    i, o = decode_action(decision.action)
    return {
        "phase": phase.value,
        "action_id": decision.action,
        "decoded_action": {"input_buffer": i, "output_buffer": o},
        "probabilities": np.round(decision.probabilities, 6).tolist(),
        "policy_mask": decision.mask.tolist(),
        "physical_mask": physical_action_mask(state, fill).tolist(),
        "haam_mask": action_mask(state, fill, MaskMode.HAAM).tolist(),
        "heuristic_action_id": heuristic_action(state, fill),
        "details": decision.details,
        "members": [{"name": m.name, "action_id": a} for m, a in zip(registry.members, member_actions)],
    }


# ---------------------------------------------------------------- FlexSim bridge


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "mode": registry.mode,
        "models": [p.as_posix() for p in registry.paths],
        "fill_target": _fill_target(),
        "lookahead": registry.lookahead,
        "mask_mode": registry.mask_mode.value,
        "vehicles": registry.config.plant.vehicles,
    }


@app.post("/step", response_model=StepResponse)
def step(request: StateRequest) -> StepResponse:
    """One decision. FlexSim must do exactly what `insert` / `extract` say, in that order: extract first."""
    state = _parse_state(request)
    phase = phase_of(state, _fill_target())
    if phase is Phase.DONE:
        raise HTTPException(status_code=409, detail="Episode is over: nothing waiting and every buffer empty")
    decision = registry.policy.decide(state)
    i, o = decode_action(decision.action)
    response = StepResponse(
        action_id=decision.action, input_buffer=i + 1, output_buffer=o + 1, phase=phase,
        insert=phase.inserts, extract=phase.extracts,
        valid_action_count=int(decision.mask.sum()), decided_by=registry.mode,
    )
    if TRACE_PATH is not None:
        record = {"time": datetime.now(UTC).isoformat(), "fill_target": _fill_target(),
                  "state_vector": request.state_vector, "upcoming": request.upcoming,
                  "response": response.model_dump(mode="json")}
        with _trace_lock, open(TRACE_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    return response


# ---------------------------------------------------------------- models and introspection


@app.get("/models")
def list_models() -> dict:
    found = sorted(ARTIFACTS_DIR.rglob("*.zip")) if ARTIFACTS_DIR.exists() else []
    models = []
    for path in found:
        sidecar = config_path_for(path)
        config = ExperimentConfig.load(sidecar) if sidecar.exists() else None
        models.append({
            "path": path.relative_to(ARTIFACTS_DIR).as_posix(),
            "experiment": config.name if config else "legacy",
            "encoding": config.observation.encoding.value if config else "raw",
            "lookahead": config.observation.lookahead if config else 0,
            "mask_mode": config.mask_mode.value if config else "haam",
            "fill_target": config.plant.fill_target if config else 5,
        })
    loaded = [p.relative_to(ARTIFACTS_DIR).as_posix() for p in registry.paths if p.is_relative_to(ARTIFACTS_DIR)]
    values = []
    for path in sorted(ARTIFACTS_DIR.rglob("*.pt")) if ARTIFACTS_DIR.exists() else []:
        meta = path.with_suffix(".json")
        metrics = json.loads(meta.read_text(encoding="utf-8")).get("metrics", {}) if meta.exists() else {}
        values.append({"path": path.relative_to(ARTIFACTS_DIR).as_posix(), "held_out_r2": metrics.get("held_out_r2")})
    return {"loaded": loaded, "models": models, "values": values}


@app.post("/models/load")
def load_models(request: LoadRequest) -> dict:
    paths = [_artifact(relative, ".zip") for relative in request.paths]
    try:
        registry.load(paths, request.vote_threshold, request.planner)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return health()


@app.get("/model/architecture")
def architecture(member: int = Query(default=0, ge=0)) -> dict:
    if member >= len(registry.inspectors):
        raise HTTPException(status_code=404, detail="No trained model loaded (the API is using the heuristic)")
    return registry.inspectors[member].architecture()


@app.post("/inspect")
def inspect(request: InspectRequest) -> dict:
    """The decision of the loaded policy, plus the internals of one member network when there is one."""
    state = _parse_state(request)
    payload = _decision_payload(state)
    payload["network"] = registry.inspectors[request.member].inspect(state) if request.member < len(registry.inspectors) else None
    return payload


# ---------------------------------------------------------------- interactive twin


@dataclass
class Session:
    twin: PythonPaintShopTwin
    painted: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False
    # Requests run in a thread pool: one step at a time per session, or two steps could both
    # decide on the same state and the second would apply an action that is no longer valid.
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, Session] = {}
_MAX_SESSIONS = 32


def _snapshot(session_id: str, session: Session) -> dict:
    state = session.twin.state
    return {
        "session_id": session_id,
        "state_vector": state.as_vector().tolist(),
        "remaining_arrivals": list(state.remaining_arrivals),
        "phase": phase_of(state, session.twin.config.plant.fill_target).value,
        "painted": session.painted,
        "actions": session.actions,
        "color_changes": session.twin.color_changes,
        "total_reward": round(session.total_reward, 4),
        "physical_mask": session.twin.action_mask().tolist(),
        "done": session.done,
    }


def _session(session_id: str) -> Session:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown session; call /sim/reset first")
    return _sessions[session_id]


@app.post("/sim/reset")
def sim_reset(request: ResetRequest) -> dict:
    if len(_sessions) >= _MAX_SESSIONS:
        _sessions.pop(next(iter(_sessions)))
    twin = evaluation_twin(registry.config)  # accepts any physically valid action
    twin.reset(request.seed)
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = Session(twin)
    return _snapshot(session_id, _sessions[session_id])


@app.post("/sim/{session_id}/step")
def sim_step(session_id: str, request: SimStepRequest) -> dict:
    session = _session(session_id)
    with session.lock:
        if session.done:
            raise HTTPException(status_code=409, detail="Episode finished; reset the simulation")
        state = session.twin.state
        if request.mode == "model":
            action = registry.policy.decide(state).action
        elif request.mode == "heuristic":
            action = heuristic_action(state, session.twin.config.plant.fill_target)
        else:
            if request.action_id is None or not session.twin.action_mask()[request.action_id]:
                raise HTTPException(status_code=422, detail=f"Action {request.action_id} is not physically executable now")
            action = request.action_id
        _, reward, done, info = session.twin.step(action)
        if info.painted_color:
            session.painted.append(info.painted_color)
        session.actions.append(action)
        session.total_reward += reward
        session.done = done
        return _snapshot(session_id, session)


@app.post("/sim/compare")
def sim_compare(request: CompareRequest) -> dict:
    """Loaded policy vs heuristic (and optionally the perfect-information planner) on the same seeds."""
    plant = registry.config.plant
    twin = evaluation_twin(registry.config)
    heuristic = HeuristicPolicy(plant.fill_target)
    oracle = LookaheadPlannerPolicy(plant.fill_target, horizon=None, width=request.oracle_width)
    rows = []
    for seed in range(request.first_seed, request.first_seed + request.scenarios):
        model = run_episode(twin, registry.policy, seed)
        baseline = run_episode(twin, heuristic, seed)
        row = {"seed": seed, "model": model.color_changes, "heuristic": baseline.color_changes,
               "model_painted": model.painted, "heuristic_painted": baseline.painted}
        if request.oracle:
            row["oracle"] = run_episode(twin, oracle, seed).color_changes
        rows.append(row)
    mean = {key: float(np.mean([r[key] for r in rows])) for key in ("model", "heuristic", "oracle") if key in rows[0]}
    return {
        "policy": registry.mode,
        "rows": rows,
        "means": mean,
        "reduction_pct": (mean["heuristic"] - mean["model"]) / mean["heuristic"] * 100 if mean["heuristic"] else 0.0,
    }


@app.middleware("http")
async def revalidate_web_files(request, call_next):
    """Browsers must re-check the inspector page on every load, so an update is never hidden by a cache."""
    response = await call_next(request)
    if request.url.path.startswith("/ui"):
        response.headers["Cache-Control"] = "no-cache"
    return response


if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
