"""Replay a FlexSim decision trace (written by the API with HAAM_TRACE) against the twin's rules.

    python scripts/check_trace.py flexsim_trace.jsonl

For every decision the twin predicts the next state from the reported state and the action taken,
and compares it with the state FlexSim reports at the next decision. Any difference means FlexSim
and the twin are not simulating the same plant (wrong move, KPI out of sync, a vehicle lost...).
Also recomputes the color changes of each episode from the painted vehicles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from haam_rl.domain.dynamics import apply_action, is_terminal
from haam_rl.domain.model import NO_VEHICLE, PlantState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes, errors, changes, painted = 0, 0, 0, 0
    predicted: PlantState | None = None
    for index, record in enumerate(records):
        state = PlantState.from_vector(record["state_vector"], record.get("upcoming", []))
        starts_episode = state.last_painted_color == NO_VEHICLE and state.occupancy == 0
        if starts_episode:
            if episodes:
                print(f"episode {episodes}: {painted} painted, {changes} color changes")
            episodes, changes, painted = episodes + 1, 0, 0
        elif predicted is not None:
            expected = predicted.as_vector().tolist()
            if predicted.incoming_color == NO_VEHICLE and state.incoming_color != NO_VEHICLE and not record.get("upcoming"):
                expected[0] = state.incoming_color  # the next arrival was not in the look-ahead sent before
            if expected != record["state_vector"]:
                errors += 1
                if errors <= args.max_errors:
                    print(f"decision {index}: FlexSim reports {record['state_vector']}\n"
                          f"{'':>{len(str(index)) + 11}}twin expected {expected}")
        action = record["response"]["action_id"]
        try:
            transition = apply_action(state, action, record["fill_target"])
        except ValueError as error:
            errors += 1
            print(f"decision {index}: action {action} is not executable: {error}")
            predicted = None
            continue
        changes += transition.color_changed
        painted += transition.painted_color != NO_VEHICLE
        predicted = None if is_terminal(transition.state) else transition.state
    if episodes:
        print(f"episode {episodes}: {painted} painted, {changes} color changes")
    print(f"\n{len(records)} decisions, {episodes} episodes, {errors} mismatches")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
