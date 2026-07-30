from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import EntitySpec


@dataclass(frozen=True)
class FrozenAssignment:
    """Initial-window identity assignment, frozen for the whole video."""

    entity_to_track: dict[str, str | None]
    residual_track_ids: tuple[str, ...]
    unmatched_entity_ids: tuple[str, ...]
    total_cost: float


def freeze_initial_assignment(
    entities: Sequence[EntitySpec],
    track_ids: Sequence[str],
    cost_matrix: np.ndarray,
    *,
    maximum_match_cost: float,
    unmatched_entity_cost: float = 1.0,
    unmatched_track_cost: float = 1.0,
    maximum_entities: int = 16,
) -> FrozenAssignment:
    """Solve rectangular identity matching with explicit unmatched dustbins.

    Scene adapters build the cost matrix from the immutable condition frame and
    an initial observation window. Future motion must not contribute to it.
    Non-finite or over-threshold entries are treated as forbidden matches.
    """

    entity_list = list(entities)
    all_track_list = [str(track_id) for track_id in track_ids]
    costs = np.asarray(cost_matrix, dtype=np.float64)
    expected_shape = (len(entity_list), len(all_track_list))
    if costs.shape != expected_shape:
        raise ValueError(
            f"cost_matrix must have shape {expected_shape}, got {costs.shape}"
        )
    if len({item.entity_id for item in entity_list}) != len(entity_list):
        raise ValueError("entity IDs must be unique")
    if len(set(all_track_list)) != len(all_track_list):
        raise ValueError("track IDs must be unique")
    if not isinstance(maximum_entities, int) or maximum_entities < 0:
        raise ValueError("maximum_entities must be a non-negative integer")
    if len(entity_list) > maximum_entities:
        raise ValueError(
            f"at most {maximum_entities} expected entities are supported"
        )
    penalties = (
        maximum_match_cost,
        unmatched_entity_cost,
        unmatched_track_cost,
    )
    if any(not math.isfinite(float(value)) or value < 0 for value in penalties):
        raise ValueError("matching thresholds and costs must be finite and >= 0")
    finite_costs = costs[np.isfinite(costs)]
    if finite_costs.size and float(finite_costs.min()) < 0.0:
        raise ValueError("finite matching costs must be non-negative")

    # Exact dynamic programming over the expected-entity mask. Complexity is
    # O(num_tracks * num_entities * 2**num_entities), so an arbitrarily noisy
    # prediction can contribute all residual tracks without making runtime
    # exponential in the number of hallucinated candidates.
    empty_assignment: tuple[int | None, ...] = (None,) * len(entity_list)
    states: dict[int, tuple[float, tuple[int | None, ...]]] = {
        0: (0.0, empty_assignment)
    }
    for track_index in range(len(all_track_list)):
        next_states: dict[int, tuple[float, tuple[int | None, ...]]] = {}
        for used_mask, (state_cost, assignment) in states.items():
            _keep_best(
                next_states,
                used_mask,
                (
                    state_cost + unmatched_track_cost,
                    assignment,
                ),
            )
            for entity_index in range(len(entity_list)):
                entity_bit = 1 << entity_index
                if used_mask & entity_bit:
                    continue
                match_cost = float(costs[entity_index, track_index])
                if (
                    not math.isfinite(match_cost)
                    or match_cost > maximum_match_cost
                ):
                    continue
                candidate_assignment = list(assignment)
                candidate_assignment[entity_index] = track_index
                _keep_best(
                    next_states,
                    used_mask | entity_bit,
                    (
                        state_cost + match_cost,
                        tuple(candidate_assignment),
                    ),
                )
        states = next_states

    candidates = [
        (
            state_cost
            + (
                len(entity_list) - used_mask.bit_count()
            )
            * unmatched_entity_cost,
            assignment,
        )
        for used_mask, (state_cost, assignment) in states.items()
    ]
    total_cost, assignment = min(candidates, key=_assignment_sort_key)
    used_track_ids = {
        all_track_list[index]
        for index in assignment
        if index is not None
    }
    entity_to_track = {
        entity.entity_id: (
            all_track_list[track_index]
            if track_index is not None
            else None
        )
        for entity, track_index in zip(entity_list, assignment)
    }
    return FrozenAssignment(
        entity_to_track=entity_to_track,
        residual_track_ids=tuple(
            track_id
            for track_id in all_track_list
            if track_id not in used_track_ids
        ),
        unmatched_entity_ids=tuple(
            entity.entity_id
            for entity, track_index in zip(entity_list, assignment)
            if track_index is None
        ),
        total_cost=float(total_cost),
    )


def _assignment_sort_key(
    candidate: tuple[float, tuple[int | None, ...]],
) -> tuple[float, tuple[int, ...]]:
    cost, assignment = candidate
    # Deterministic tie-breaking: real tracks in input order, dustbin last.
    normalized = tuple(
        index if index is not None else 1_000_000 for index in assignment
    )
    return round(float(cost), 12), normalized


def _keep_best(
    states: dict[int, tuple[float, tuple[int | None, ...]]],
    mask: int,
    candidate: tuple[float, tuple[int | None, ...]],
) -> None:
    current = states.get(mask)
    if current is None or (
        _assignment_sort_key(candidate) < _assignment_sort_key(current)
    ):
        states[mask] = candidate
