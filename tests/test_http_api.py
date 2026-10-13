from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from haam_rl.adapters import http_api
from haam_rl.adapters.training import train
from haam_rl.config import ExperimentConfig

client = TestClient(http_api.app)
STATE = [3, 2, 2, 2, 3, 0, 0, 3, 3, 0, 0, 0, 1, 4, 0, 0, 0, 2, 0, 0, 0, 0, 5, 2, 2, 0, 0]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory) -> Path:
    """A barely trained model: enough to exercise loading, inspection and the ensemble."""
    run_dir = tmp_path_factory.mktemp("run")
    config = ExperimentConfig().with_overrides(
        name="tiny", observation={"lookahead": 3},
        training={"total_timesteps": 512, "n_envs": 2, "n_steps": 128, "batch_size": 64, "eval_freq": 10**9})
    train(config, seed=0, run_dir=run_dir, verbose=0)
    return run_dir / "final_model.zip"


@pytest.fixture(autouse=True)
def heuristic_by_default():
    http_api.registry.load([])
    yield
    http_api.registry.load([])


def test_step_tells_flexsim_what_to_do():
    body = client.post("/step", json={"state_vector": STATE}).json()
    assert body["phase"] == "fill" and body["insert"] and not body["extract"]
    assert body["input_buffer"] == body["output_buffer"]  # FILL actions live on the diagonal
    full = [2, 1] + [1] * 25
    body = client.post("/step", json={"state_vector": full}).json()
    assert body["phase"] == "steady" and body["insert"] and body["extract"]


def test_step_rejects_finished_and_malformed_states():
    assert client.post("/step", json={"state_vector": [0] * 27}).status_code == 409
    assert client.post("/step", json={"state_vector": [9] + [0] * 26}).status_code == 422


def test_simulation_session_runs_to_completion():
    snapshot = client.post("/sim/reset", json={"seed": 7}).json()
    while not snapshot["done"]:
        snapshot = client.post(f"/sim/{snapshot['session_id']}/step", json={"mode": "heuristic"}).json()
    assert len(snapshot["painted"]) == 100


def test_manual_action_must_be_physically_executable():
    snapshot = client.post("/sim/reset", json={"seed": 7}).json()
    blocked = snapshot["physical_mask"].index(False)
    response = client.post(f"/sim/{snapshot['session_id']}/step", json={"mode": "action", "action_id": blocked})
    assert response.status_code == 422


def test_inspect_with_one_model_and_with_an_ensemble(tiny_model: Path, monkeypatch):
    monkeypatch.setattr(http_api, "ARTIFACTS_DIR", tiny_model.parent)
    assert client.post("/models/load", json={"paths": [tiny_model.name]}).json()["mode"] == "model"
    body = client.post("/inspect", json={"state_vector": STATE, "upcoming": [1, 2, 3]}).json()
    network = body["network"]
    assert len(network["policy_layers"]) == 3
    assert all(p == 0 for p, ok in zip(network["model_probabilities"], network["model_mask"]) if not ok)
    groups = client.get("/model/architecture").json()["input_groups"]
    assert len(groups) == 27 + 3 + 2  # positions, look-ahead, phase, occupancy

    assert client.post("/models/load", json={"paths": [tiny_model.name] * 3}).json()["mode"] == "ensemble"
    body = client.post("/inspect", json={"state_vector": STATE}).json()
    assert body["details"]["vote_mode"] in {"hard", "soft"} and len(body["members"]) == 3


def test_models_outside_artifacts_are_refused():
    assert client.post("/models/load", json={"paths": ["../pyproject.toml"]}).status_code == 404


def test_compare_with_oracle():
    body = client.post("/sim/compare", json={"scenarios": 2, "oracle": True, "oracle_width": 4}).json()
    assert body["policy"] == "heuristic" and body["means"]["model"] == body["means"]["heuristic"]
    assert body["means"]["oracle"] <= body["means"]["heuristic"]


def test_planner_policy_can_be_selected():
    health = client.post("/models/load", json={"planner": {"horizon": 3, "width": 4}}).json()
    assert health["mode"] == "planner" and health["lookahead"] == 3
    body = client.post("/step", json={"state_vector": STATE, "upcoming": [2, 3, 3]}).json()
    assert body["decided_by"] == "planner" and body["phase"] == "fill"


def test_models_listing_works_with_a_relative_startup_path(tiny_model: Path, monkeypatch):
    monkeypatch.setattr(http_api, "ARTIFACTS_DIR", tiny_model.parent.resolve())
    monkeypatch.chdir(tiny_model.parent)
    http_api.registry.load([Path(tiny_model.name)])  # as HAAM_MODELS would pass it
    assert client.get("/models").json()["loaded"] == [tiny_model.name]


def _emulate_flexsim(trace_path: Path, monkeypatch, corrupt_kpi: bool = False) -> int:
    """Drive one episode through POST /step the way the FlexSim trigger does; return its color changes."""
    from haam_rl.adapters.python_twin import PythonPaintShopTwin

    monkeypatch.setattr(http_api, "TRACE_PATH", trace_path)
    twin = PythonPaintShopTwin(ExperimentConfig().with_overrides(mask_mode="physical", plant={"vehicles": 40}))
    state, done = twin.reset(3), False
    while not done:
        vector = state.as_vector().tolist()
        if corrupt_kpi and twin.color_changes > 2:
            vector[1] = 0  # the old FlexSim bug: last painted color out of sync with the machine
        body = client.post("/step", json={"state_vector": vector, "upcoming": list(state.remaining_arrivals[:10])}).json()
        state, _, done, _ = twin.step(body["action_id"])
    return twin.color_changes


def test_trace_of_a_faithful_simulator_replays_exactly(tmp_path: Path, monkeypatch):
    import subprocess
    import sys

    trace = tmp_path / "trace.jsonl"
    changes = _emulate_flexsim(trace, monkeypatch)
    result = subprocess.run([sys.executable, "scripts/check_trace.py", str(trace)], capture_output=True, text=True, check=False,
                            encoding="utf-8")
    assert result.returncode == 0, result.stdout
    assert f"40 painted, {changes} color changes" in result.stdout and "0 mismatches" in result.stdout


def test_trace_checker_catches_a_desynchronised_simulator(tmp_path: Path, monkeypatch):
    import subprocess
    import sys

    trace = tmp_path / "trace.jsonl"
    _emulate_flexsim(trace, monkeypatch, corrupt_kpi=True)
    result = subprocess.run([sys.executable, "scripts/check_trace.py", str(trace)], capture_output=True, text=True, check=False,
                            encoding="utf-8")
    assert result.returncode == 1 and "twin expected" in result.stdout


def test_value_guided_planner_can_be_selected(tmp_path: Path, monkeypatch):
    import torch

    from haam_rl.adapters.value_search import ValueNetwork, ValueSettings, save_value

    monkeypatch.setattr(http_api, "ARTIFACTS_DIR", tmp_path.resolve())
    save_value(ValueNetwork(8), ValueSettings(hidden=8), {"held_out_r2": 0.5}, tmp_path / "value" / "v.pt")
    torch.set_num_threads(1)
    assert client.get("/models").json()["values"] == [{"path": "value/v.pt", "held_out_r2": 0.5}]
    health = client.post("/models/load", json={"planner": {"horizon": 3, "width": 4, "value": "value/v.pt"}}).json()
    assert health["mode"] == "value_planner"
    assert client.post("/step", json={"state_vector": STATE, "upcoming": [2, 3, 3]}).json()["decided_by"] == "value_planner"
    assert client.post("/models/load", json={"planner": {"value": "../x.pt"}}).status_code == 404


def test_health_reports_the_mask_of_the_active_policy():
    assert client.get("/health").json()["mask_mode"] == "haam"  # heuristic
    assert client.post("/models/load", json={"planner": {"horizon": 2, "width": 2}}).json()["mask_mode"] == "physical"


def test_inspector_page_is_never_served_stale():
    response = client.get("/ui/")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-cache"


def test_concurrent_steps_on_one_session_are_serialised():
    from concurrent.futures import ThreadPoolExecutor

    client.post("/models/load", json={"planner": {"horizon": 3, "width": 4}})
    session = client.post("/sim/reset", json={"seed": 4}).json()["session_id"]
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda _: client.post(f"/sim/{session}/step", json={"mode": "model"}).status_code, range(24)))
    assert set(codes) == {200}
    snapshot = client.post(f"/sim/{session}/step", json={"mode": "heuristic"}).json()
    assert len(snapshot["actions"]) == 25


def test_committee_with_value_referee_can_be_selected(tiny_model: Path, tmp_path: Path, monkeypatch):
    import shutil

    from haam_rl.adapters.value_search import ValueNetwork, ValueSettings, save_value

    monkeypatch.setattr(http_api, "ARTIFACTS_DIR", tmp_path.resolve())
    shutil.copy(tiny_model, tmp_path / "net.zip")
    shutil.copy(tiny_model.with_suffix(".json"), tmp_path / "net.json")
    save_value(ValueNetwork(8), ValueSettings(hidden=8), {}, tmp_path / "v.pt")
    planner = {"horizon": 3, "width": 4, "value": "v.pt", "committee_network": "net.zip"}
    assert client.post("/models/load", json={"planner": planner}).json()["mode"] == "committee"
    assert client.post("/step", json={"state_vector": STATE, "upcoming": [2, 3, 3]}).json()["decided_by"] == "committee"


def test_value_planner_exposes_its_judges():
    values = "value/v1.pt,value/v2.pt"
    if not all((http_api.ARTIFACTS_DIR / v).exists() for v in values.split(",")):
        pytest.skip("value networks not trained in this checkout")
    health = client.post("/models/load", json={"planner": {"horizon": 0, "value": values}}).json()
    assert health["mode"] == "value_planner" and health["networks"] == ["value/v1.pt", "value/v2.pt"]
    architecture = client.get("/model/architecture?member=1").json()
    assert architecture["kind"] == "value" and architecture["name"] == "value/v2.pt"
    body = client.post("/inspect", json={"state_vector": STATE, "member": 1}).json()
    judges = body["judges"]
    assert body["network"]["value"] == pytest.approx(judges["values"][1], abs=1e-3)
    scores = [s for s in judges["action_scores"] if s is not None]
    # With look-ahead 0 in FILL the planner ranks exactly by the one-step judge score.
    assert judges["action_scores"][body["action_id"]] == min(scores)
