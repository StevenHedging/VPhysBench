"""Object identity, lifecycle, and set-scoring primitives for scene evaluators."""

from .contracts import (
    EntityMatch,
    EntitySpec,
    LifecyclePolicy,
    ObjectTrack,
    ReferenceCapability,
    VisibilityState,
    exposure_by_id,
)
from .matching import FrozenAssignment, freeze_initial_assignment
from .scoring import (
    EntityIntegrityScore,
    GatedCaseComposition,
    GOSPADecomposition,
    GeometricComposition,
    PositionComparison,
    compare_positions,
    compose_gated_case_score,
    compose_weighted_geometric,
    distance_similarity,
    score_entity_integrity,
)

__all__ = [
    "EntityIntegrityScore",
    "EntityMatch",
    "EntitySpec",
    "FrozenAssignment",
    "GatedCaseComposition",
    "GOSPADecomposition",
    "GeometricComposition",
    "LifecyclePolicy",
    "ObjectTrack",
    "PositionComparison",
    "ReferenceCapability",
    "VisibilityState",
    "compare_positions",
    "compose_gated_case_score",
    "compose_weighted_geometric",
    "distance_similarity",
    "exposure_by_id",
    "freeze_initial_assignment",
    "score_entity_integrity",
]
