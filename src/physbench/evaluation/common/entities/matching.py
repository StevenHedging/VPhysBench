from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
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
    maximum_tracks: int = 16,
) -> FrozenAssignment:
    """Solve rectangular identity matching with explicit unmatched dustbins.

    Scene adapters build the cost matrix from the immutable condition frame and
    an initial observation window. Future motion must not contribute to it.
    Non-finite or over-threshold entries are treated as forbidden matches.
    """

    entity_list = list(entities)
    track_list = [str(track_id) for track_id in track_ids]
    costs = np.asarray(cost_matrix, dtype=np.float64)
    expected_shape = (len(entity_list), len(track_list))
    if costs.shape != expected_shape:
        raise ValueError(
            f"cost_matrix must have shape {expected_shape}, got {costs.shape}"
        )
    if len({item.entity_id for item in entity_list}) != len(entity_list):
        raise ValueError("entity IDs must be unique")
    if len(set(track_list)) != len(track_list):
        raise ValueError("track IDs must be unique")
    if len(track_list) > maximum_tracks:
        raise ValueError(
            f"at most {maximum_tracks} initial tracks are supported"
        )
    penalties = (
        maximum_match_cost,
        unmatched_entity_cost,
        unmatched_track_cost,
    )
    if any(not math.isfinite(float(value)) or value < 0 for value in penalties):
        raise ValueError("matching thresholds and costs must be finite and >= 0")

    @lru_cache(maxsize=None)
    def solve(
        entity_index: int,
        used_mask: int,
    ) -> tuple[float, tuple[int | None, ...]]:
        if entity_index == len(entity_list):
            residual_count = len(track_list) - used_mask.bit_count()
            return residual_count * unmatched_track_cost, ()

        tail_cost, tail_assignment = solve(entity_index + 1, used_mask)
        best = (
            unmatched_entity_cost + tail_cost,
            (None,) + tail_assignment,
        )
        for track_index in range(len(track_list)):
            if used_mask & (1 << track_index):
                continue
            match_cost = float(costs[entity_index, track_index])
            if not math.isfinite(match_cost) or match_cost > maximum_match_cost:
                continue
            candidate_tail_cost, candidate_tail = solve(
                entity_index + 1,
                used_mask | (1 << track_index),
            )
            candidate = (
                match_cost + candidate_tail_cost,
                (track_index,) + candidate_tail,
            )
            if _assignment_sort_key(candidate) < _assignment_sort_key(best):
                best = candidate
        return best

    total_cost, assignment = solve(0, 0)
    used = {index for index in assignment if index is not None}
    entity_to_track = {
        entity.entity_id: (
            track_list[track_index] if track_index is not None else None
        )
        for entity, track_index in zip(entity_list, assignment)
    }
    return FrozenAssignment(
        entity_to_track=entity_to_track,
        residual_track_ids=tuple(
            track_id
            for index, track_id in enumerate(track_list)
            if index not in used
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
