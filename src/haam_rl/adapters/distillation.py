"""Distill a planner (look-ahead or value-guided) into a neural policy with DAgger (Ross et al., 2011).

Iteration 0 records the planner's own trajectories. Every later iteration lets the *student* play,
asks the planner what it would have done in each state the student reached, adds those labels to
the dataset and retrains. Learning from its own mistakes is what makes the student robust; pure
behaviour cloning only sees the states the planner visits.

The student is an ordinary MaskablePPO policy with the experiment's observation, mask and network,
so it can be evaluated, inspected in the web UI, served by the API or fine-tuned with PPO
(`training.init_from`). Its critic is fitted to the discounted returns of the recorded episodes,
which gives PPO fine-tuning a sensible starting value function.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from haam_rl.adapters.gym_env import PaintShopEnv
from haam_rl.adapters.python_twin import PythonPaintShopTwin
from haam_rl.adapters.runner import run_scenarios
from haam_rl.adapters.sb3_policy import SB3Policy
from haam_rl.adapters.training import policy_spec, save_model
from haam_rl.adapters.value_search import ValueGuidedPlannerPolicy
from haam_rl.application.ports import Policy
from haam_rl.application.use_cases import LookaheadPlannerPolicy
from haam_rl.config import ExperimentConfig
from haam_rl.domain.model import PlantState
from haam_rl.domain.scenarios import VALIDATION_SEEDS

TRAINING_SEED_OFFSET = 1_000_000  # distillation scenarios never overlap validation or test seeds


@dataclass(frozen=True)
class DistillationSettings:
    teacher_horizon: int = 10
    teacher_width: int = 32
    teacher_value: str = ""         # value network file(s), comma-separated: the teacher is the value-guided planner
    teacher_value_scale: float = 0.75
    iterations: int = 8
    scenarios_per_iteration: int = 160
    epochs_per_iteration: int = 4
    batch_size: int = 512
    learning_rate: float = 1e-3
    value_loss_weight: float = 0.5
    workers: int = 4


@dataclass
class Dataset:
    observations: list[np.ndarray]
    masks: list[np.ndarray]
    actions: list[np.ndarray]
    returns: list[np.ndarray]

    def add(self, observations, masks, actions, returns) -> None:
        self.observations.append(observations)
        self.masks.append(masks)
        self.actions.append(actions)
        self.returns.append(returns)

    def arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(np.concatenate(part) for part in (self.observations, self.masks, self.actions, self.returns))

    def __len__(self) -> int:
        return sum(len(a) for a in self.actions)


def _label(teacher: Policy, states: list[PlantState]) -> list[int]:
    return [teacher.decide(state).action for state in states]


def label_states(pool: ProcessPoolExecutor, teacher: Policy, states: list[PlantState],
                 chunks: int) -> np.ndarray:
    """Teacher action for every state, computed in parallel (the planner is CPU-bound)."""
    parts = [states[k::chunks] for k in range(chunks)]
    actions = np.empty(len(states), dtype=np.int64)
    for k, chunk_actions in enumerate(pool.map(_label, [teacher] * chunks, parts)):
        actions[k::chunks] = chunk_actions
    return actions


@dataclass
class Rollout:
    states: list[PlantState]
    actions: np.ndarray
    rewards: np.ndarray
    episode: np.ndarray  # index of the episode each step belongs to; steps of an episode are in order


def _teacher_episodes(config: ExperimentConfig, teacher: Policy, seeds: list[int]) -> Rollout:
    states, actions, rewards, episode = [], [], [], []
    for seed in seeds:
        twin = PythonPaintShopTwin(config)
        state, done = twin.reset(seed), False
        while not done:
            action = teacher.decide(state).action
            states.append(state)
            state, reward, done, _ = twin.step(action)
            actions.append(action)
            rewards.append(reward)
            episode.append(seed)
    return Rollout(states, np.asarray(actions), np.asarray(rewards, dtype=np.float32), np.asarray(episode))


def teacher_rollout(pool: ProcessPoolExecutor, config: ExperimentConfig, teacher: Policy,
                    seeds: list[int], chunks: int) -> Rollout:
    parts = list(pool.map(_teacher_episodes, [config] * chunks, [teacher] * chunks,
                          [seeds[k::chunks] for k in range(chunks)]))
    return Rollout([s for p in parts for s in p.states], np.concatenate([p.actions for p in parts]),
                   np.concatenate([p.rewards for p in parts]), np.concatenate([p.episode for p in parts]))


def student_rollout(config: ExperimentConfig, student: SB3Policy, seeds: list[int]) -> Rollout:
    """Play every scenario in lockstep with batched greedy decisions of the student."""
    twins = [PythonPaintShopTwin(config) for _ in seeds]
    states = [twin.reset(seed) for twin, seed in zip(twins, seeds)]
    visited, actions, rewards, episode = [], [], [], []
    running = list(range(len(seeds)))
    while running:
        current = [states[k] for k in running]
        still = []
        for k, state, decision in zip(running, current, student.decide_batch(current)):
            visited.append(state)
            states[k], reward, done, _ = twins[k].step(decision.action)
            actions.append(decision.action)
            rewards.append(reward)
            episode.append(seeds[k])
            if not done:
                still.append(k)
        running = still
    return Rollout(visited, np.asarray(actions), np.asarray(rewards, dtype=np.float32), np.asarray(episode))


def discounted_returns(rewards: np.ndarray, episode: np.ndarray, gamma: float) -> np.ndarray:
    """Return-to-go of each step within its own episode (steps of an episode appear in order)."""
    returns = np.zeros_like(rewards)
    following: dict[int, float] = {}
    for index in range(len(rewards) - 1, -1, -1):
        k = int(episode[index])
        following[k] = rewards[index] + gamma * following.get(k, 0.0)
        returns[index] = following[k]
    return returns


def make_teacher(config: ExperimentConfig, settings: DistillationSettings) -> Policy:
    """The planner the student imitates, with the student's own mask and look-ahead."""
    if settings.teacher_value:
        paths = [Path(p.strip()) for p in settings.teacher_value.split(",") if p.strip()]
        return ValueGuidedPlannerPolicy(paths, config.plant.fill_target, settings.teacher_horizon,
                                        settings.teacher_width, settings.teacher_value_scale, config.mask_mode)
    return LookaheadPlannerPolicy(config.plant.fill_target, settings.teacher_horizon, settings.teacher_width,
                                  mask_mode=config.mask_mode)


def new_student(config: ExperimentConfig, seed: int) -> MaskablePPO:
    policy_class, policy_kwargs = policy_spec(config)
    return MaskablePPO(policy_class, PaintShopEnv(config), seed=seed, device=config.training.device,
                       n_steps=config.training.n_steps, policy_kwargs=policy_kwargs)


def fit(model: MaskablePPO, data: Dataset, settings: DistillationSettings, rng: np.random.Generator) -> dict[str, float]:
    """Supervised epochs: cross-entropy towards the teacher's action, MSE towards the returns."""
    policy = model.policy
    observations, masks, actions, returns = (torch.as_tensor(a) for a in data.arrays())
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    policy.set_training_mode(True)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "accuracy": 0.0}
    batches = 0
    for _ in range(settings.epochs_per_iteration):
        order = torch.as_tensor(rng.permutation(len(actions)))
        for start in range(0, len(order), settings.batch_size):
            rows = order[start:start + settings.batch_size]
            obs = observations[rows]
            pi_features = policy.extract_features(obs, policy.pi_features_extractor)
            vf_features = policy.extract_features(obs, policy.vf_features_extractor)
            logits = policy.action_net(policy.mlp_extractor.forward_actor(pi_features))
            logits = logits.masked_fill(~masks[rows], float("-inf"))
            values = policy.value_net(policy.mlp_extractor.forward_critic(vf_features)).squeeze(-1)
            policy_loss = torch.nn.functional.cross_entropy(logits, actions[rows])
            value_loss = torch.nn.functional.mse_loss(values, returns[rows])
            loss = policy_loss + settings.value_loss_weight * value_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["accuracy"] += (logits.argmax(dim=-1) == actions[rows]).float().mean().item()
            batches += 1
    policy.set_training_mode(False)
    return {k: v / max(batches, 1) for k, v in totals.items()}


def distill(config: ExperimentConfig, settings: DistillationSettings, seed: int, run_dir: Path,
            verbose: bool = True) -> float:
    """Run DAgger; keep the student with the fewest validation color changes. Returns that value."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "distillation.json").write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    teacher = make_teacher(config, settings)
    model = new_student(config, seed)
    student = SB3Policy(model, config)  # wraps the live model: always reflects the latest weights
    data = Dataset([], [], [], [])
    chunks = settings.workers * 4
    best, history = float("inf"), []
    with ProcessPoolExecutor(max_workers=settings.workers) as pool:
        for iteration in range(settings.iterations + 1):
            first = TRAINING_SEED_OFFSET + seed * 100_000 + iteration * settings.scenarios_per_iteration
            seeds = list(range(first, first + settings.scenarios_per_iteration))
            if iteration == 0:
                played = teacher_rollout(pool, config, teacher, seeds, chunks)
                labels = played.actions
            else:
                played = student_rollout(config, student, seeds)
                labels = label_states(pool, teacher, played.states, chunks)
            data.add(
                np.stack([student.encoder.encode(s) for s in played.states]).astype(np.float32),
                np.stack([student.mask(s) for s in played.states]),
                labels.astype(np.int64),
                discounted_returns(played.rewards, played.episode, config.training.gamma),
            )
            agreement = float(np.mean(played.actions == labels))
            losses = fit(model, data, settings, rng)
            results = run_scenarios(config, student, VALIDATION_SEEDS)
            validation = float(np.mean([r.color_changes for r in results]))
            if validation < best:
                best = validation
                save_model(model, config, run_dir / "best_model.zip")
            history.append({"iteration": iteration, "samples": len(data), "student_agreement": round(agreement, 4),
                            **{k: round(v, 4) for k, v in losses.items()}, "validation_color_changes": validation})
            if verbose:
                print(f"[iter {iteration}] samples {len(data):>7}  student=teacher on {agreement:.1%} of its states  "
                      f"train acc {losses['accuracy']:.3f}  validation {validation:.2f} (best {best:.2f})", flush=True)
    save_model(model, config, run_dir / "final_model.zip")
    with open(run_dir / "learning_curve.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return best
