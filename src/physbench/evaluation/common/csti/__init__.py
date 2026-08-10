"""Cumulative Spatio-Temporal IoU evaluation primitives."""

from .adapters import (
    build_csti_input_from_aligned_masks,
    build_csti_input_from_frame_matches,
    normalize_observer_mask,
)
from .contracts import CSTIConfig, CSTIContractError, CSTIEntityTube, CSTIInput
from .metric import (
    evaluate_csti,
    not_applicable_csti_metric,
    score_postcondition_tube,
    score_postcondition_tube_reference,
    zero_csti_metric,
)

__all__ = [
    "CSTIConfig",
    "CSTIContractError",
    "CSTIEntityTube",
    "CSTIInput",
    "build_csti_input_from_aligned_masks",
    "build_csti_input_from_frame_matches",
    "evaluate_csti",
    "not_applicable_csti_metric",
    "normalize_observer_mask",
    "score_postcondition_tube",
    "score_postcondition_tube_reference",
    "zero_csti_metric",
]
