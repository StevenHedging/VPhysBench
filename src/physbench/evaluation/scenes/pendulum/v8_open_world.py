"""Annotation-constrained condition geometry for ``pendulum_state_v8``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from ...common.errors import ReferenceAnalysisError
from .open_world import PendulumStructureSpec, _structure_masks
from .v7_open_world import ConditionStructureDecision
from .v8_identity import PendulumSubjectAnchor


PENDULUM_V8_OBSERVER_VERSION = "frozen_subject_anchor_fused_v1"


@dataclass(frozen=True)
class AnnotatedConditionDecision:
    structure: PendulumStructureSpec
    anchor: PendulumSubjectAnchor
    candidates: tuple[Mapping[str, object], ...]
    winning_margin: float
    provenance: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_version": PENDULUM_V8_OBSERVER_VERSION,
            "structure": self.structure.to_dict(),
            "anchor": dict(self.anchor.provenance),
            "candidates": [dict(candidate) for candidate in self.candidates],
            "winning_margin": self.winning_margin,
            "provenance": dict(self.provenance),
        }


def select_annotated_condition_structure_v8(
    decision: ConditionStructureDecision,
    *,
    anchor: PendulumSubjectAnchor,
    expected_radius_length_ratio: float,
    config: Mapping[str, Any],
) -> AnnotatedConditionDecision:
    """Select the only v7 proposal independently supported by the bob mask.

    The v7 score remains useful visual evidence, but cannot establish semantic
    identity.  Absolute anchor, geometry, and source-agreement gates run before
    ranking, and a spatially distinct near-tie invalidates the reference.
    """

    anchor_mask = np.asarray(anchor.mask) > 0
    if anchor_mask.ndim != 2 or not np.any(anchor_mask):
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_subject_missing_v8",
            "frozen pendulum subject anchor is empty or malformed",
        )
    dilation_ratio = _finite_nonnegative(
        config.get("v8_anchor_dilation_radius_ratio", 0.75),
        name="v8_anchor_dilation_radius_ratio",
    )
    minimum_containment = _unit_interval(
        config.get("v8_minimum_circle_anchor_containment", 0.45),
        name="v8_minimum_circle_anchor_containment",
    )
    maximum_distance = _finite_positive(
        config.get("v8_maximum_anchor_center_distance_radii", 1.75),
        name="v8_maximum_anchor_center_distance_radii",
    )
    minimum_sources = _positive_integer(
        config.get("v8_minimum_identity_source_agreement", 3),
        name="v8_minimum_identity_source_agreement",
    )
    expected_ratio = _finite_positive(
        expected_radius_length_ratio,
        name="expected_radius_length_ratio",
    )
    minimum_geometry_ratio_score = _unit_interval(
        config.get("v8_minimum_anchor_geometry_ratio_score", 0.45),
        name="v8_minimum_anchor_geometry_ratio_score",
    )
    ambiguity_margin = _unit_interval(
        config.get("v8_condition_identity_ambiguity_margin", 0.05),
        name="v8_condition_identity_ambiguity_margin",
    )

    dilation_px = int(round(anchor.equivalent_radius_px * dilation_ratio))
    if dilation_px > 0:
        diameter = 2 * dilation_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (diameter, diameter),
        )
        supported_mask = cv2.dilate(
            anchor_mask.astype(np.uint8),
            kernel,
        ) > 0
    else:
        supported_mask = anchor_mask

    audited: list[dict[str, object]] = []
    for index, hypothesis in enumerate(decision.hypotheses):
        audited.append(
            _audit_candidate(
                hypothesis,
                index=index,
                supported_mask=supported_mask,
                anchor=anchor,
                minimum_containment=minimum_containment,
                maximum_distance=maximum_distance,
                minimum_sources=minimum_sources,
                expected_radius_length_ratio=expected_ratio,
                minimum_geometry_ratio_score=(
                    minimum_geometry_ratio_score
                ),
            )
        )

    containment_supported = [
        candidate
        for candidate in audited
        if bool(candidate["gates"]["anchor_containment"])
    ]
    if not containment_supported:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_subject_missing_v8",
            "no v7 condition proposal overlaps the frozen bob annotation",
        )
    spatially_supported = [
        candidate
        for candidate in containment_supported
        if bool(candidate["gates"]["anchor_center_distance"])
    ]
    if not spatially_supported:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_subject_misaligned_v8",
            "annotation-overlapping proposals are too far from the bob center",
        )
    eligible = [
        candidate
        for candidate in spatially_supported
        if bool(candidate["gates"]["source_agreement"])
    ]
    if not eligible:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_subject_weak_v8",
            "annotation-supported proposals lack independent v7 sources",
        )
    geometry_supported = [
        candidate
        for candidate in eligible
        if bool(candidate["gates"]["anchor_geometry_ratio"])
    ]
    if not geometry_supported:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_geometry_incompatible_v8",
            "annotation-supported proposals imply a pivot distance that is "
            "incompatible with the frozen bob radius and Case geometry",
        )

    geometry_supported.sort(
        key=lambda candidate: (
            float(candidate["selection_score"]),
            float(candidate["anchor_geometry_ratio_score"]),
            float(candidate["identity_score"]),
            int(candidate["source_agreement_count"]),
            float(candidate["v7_score"]),
            -int(candidate["hypothesis_index"]),
        ),
        reverse=True,
    )
    winner = geometry_supported[0]
    distinct = [
        candidate
        for candidate in geometry_supported[1:]
        if _spatially_distinct(winner, candidate)
    ]
    runner_score = max(
        (float(candidate["selection_score"]) for candidate in distinct),
        default=0.0,
    )
    winning_margin = float(winner["selection_score"]) - runner_score
    if distinct and winning_margin < ambiguity_margin:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_subject_ambiguous_v8",
            "spatially distinct annotation-supported bob proposals are tied "
            f"within {winning_margin:.4f}",
        )

    # The hidden subject annotation, rather than an internal highlight or a
    # partially visible Hough circle, owns bob identity and scale.  Eligible
    # visual proposals contribute only the independently observed pivot.
    center = np.asarray(anchor.centroid_xy, dtype=np.float64)
    pivot = np.asarray(winner["pivot_xy"], dtype=np.float64)
    radius = float(anchor.equivalent_radius_px)
    _, subject_mask = _structure_masks(
        anchor_mask.shape,
        pivot_xy=pivot,
        bob_xy=center,
        radius_px=radius,
    )
    bob_mask = np.where(anchor_mask, 255, 0).astype(np.uint8)
    subject_mask = cv2.bitwise_or(subject_mask, bob_mask)
    structure = PendulumStructureSpec(
        pivot_xy=pivot,
        bob_xy=center,
        bob_radius_px=radius,
        bob_mask=bob_mask,
        subject_mask=subject_mask,
        confidence=float(np.clip(winner["selection_score"], 0.0, 1.0)),
        source="condition_v8_frozen_subject_anchor_fused_v1",
    )
    return AnnotatedConditionDecision(
        structure=structure,
        anchor=anchor,
        candidates=tuple(audited),
        winning_margin=winning_margin,
        provenance={
            "observer_version": PENDULUM_V8_OBSERVER_VERSION,
            "selection_policy": "absolute_gates_then_unique_identity_rank_v1",
            "selected_hypothesis_index": int(winner["hypothesis_index"]),
            "anchor_dilation_px": dilation_px,
            "thresholds": {
                "minimum_circle_anchor_containment": minimum_containment,
                "maximum_anchor_center_distance_radii": maximum_distance,
                "minimum_identity_source_agreement": minimum_sources,
                "minimum_anchor_geometry_ratio_score": (
                    minimum_geometry_ratio_score
                ),
                "expected_radius_length_ratio": expected_ratio,
                "condition_identity_ambiguity_margin": ambiguity_margin,
            },
        },
    )


def _audit_candidate(
    hypothesis: Mapping[str, object],
    *,
    index: int,
    supported_mask: np.ndarray,
    anchor: PendulumSubjectAnchor,
    minimum_containment: float,
    maximum_distance: float,
    minimum_sources: int,
    expected_radius_length_ratio: float,
    minimum_geometry_ratio_score: float,
) -> dict[str, object]:
    try:
        center = _point(hypothesis.get("center_xy"), name="center_xy")
        pivot = _point(hypothesis.get("pivot_xy"), name="pivot_xy")
        radius = _finite_positive(
            hypothesis.get("radius_px"),
            name="radius_px",
        )
        source_count = _nonnegative_integer(
            hypothesis.get("source_agreement_count"),
            name="source_agreement_count",
        )
        edge_score = _unit_interval(
            hypothesis.get("edge_closure_score"),
            name="edge_closure_score",
        )
        body_score = _unit_interval(
            hypothesis.get("body_contrast_score"),
            name="body_contrast_score",
        )
        v7_score = _unit_interval(
            hypothesis.get("score"),
            name="score",
        )
    except (TypeError, ValueError) as exc:
        return {
            "hypothesis_index": index,
            "hypothesis": dict(hypothesis),
            "eligible": False,
            "malformed": True,
            "malformed_reason": f"{type(exc).__name__}: {exc}",
            "gates": {
                "anchor_containment": False,
                "anchor_center_distance": False,
                "source_agreement": False,
                "anchor_geometry_ratio": False,
            },
        }

    circle = np.zeros(supported_mask.shape, dtype=np.uint8)
    cv2.circle(
        circle,
        tuple(int(round(value)) for value in center),
        max(1, int(round(radius))),
        1,
        -1,
    )
    circle_binary = circle > 0
    circle_area = int(np.count_nonzero(circle_binary))
    intersection = int(np.count_nonzero(circle_binary & supported_mask))
    containment = float(intersection / max(circle_area, 1))
    distance_radii = float(
        np.linalg.norm(center - anchor.centroid_xy) / max(radius, 1.0)
    )
    identity_score = float(
        0.55 * containment
        + 0.20 * math.exp(-distance_radii)
        + 0.15 * edge_score
        + 0.10 * body_score
    )
    anchor_pivot_length = float(
        np.linalg.norm(anchor.centroid_xy - pivot)
    )
    observed_anchor_ratio = float(
        anchor.equivalent_radius_px / max(anchor_pivot_length, 1.0)
    )
    geometry_ratio_score = float(
        math.exp(
            -abs(
                math.log(
                    observed_anchor_ratio / expected_radius_length_ratio
                )
            )
        )
    )
    # Geometry dominates the choice among proposals already supported by the
    # frozen bob mask.  This prevents a bright spot inside the bob from
    # supplying a short local pseudo-pivot merely because its circle has
    # perfect containment.
    selection_score = float(
        0.90 * geometry_ratio_score + 0.10 * identity_score
    )
    gates = {
        "anchor_containment": containment >= minimum_containment,
        "anchor_center_distance": distance_radii <= maximum_distance,
        "source_agreement": source_count >= minimum_sources,
        "anchor_geometry_ratio": (
            geometry_ratio_score >= minimum_geometry_ratio_score
        ),
    }
    return {
        "hypothesis_index": index,
        "hypothesis": dict(hypothesis),
        "center_xy": center.tolist(),
        "pivot_xy": pivot.tolist(),
        "radius_px": radius,
        "v7_score": v7_score,
        "source_agreement_count": source_count,
        "edge_closure_score": edge_score,
        "body_contrast_score": body_score,
        "anchor_circle_containment": containment,
        "anchor_center_distance_radii": distance_radii,
        "anchor_pivot_length_px": anchor_pivot_length,
        "observed_anchor_radius_length_ratio": observed_anchor_ratio,
        "anchor_geometry_ratio_score": geometry_ratio_score,
        "identity_score": identity_score,
        "selection_score": selection_score,
        "gates": gates,
        "eligible": all(gates.values()),
        "malformed": False,
    }


def _spatially_distinct(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> bool:
    distance = float(
        np.linalg.norm(
            np.asarray(first["center_xy"], dtype=np.float64)
            - np.asarray(second["center_xy"], dtype=np.float64)
        )
    )
    return distance > 1.5 * max(
        float(first["radius_px"]),
        float(second["radius_px"]),
        1.0,
    )


def _point(value: object, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite 2D point")
    return point


def _unit_interval(value: object, *, name: str) -> float:
    output = _finite_nonnegative(value, name=name)
    if output > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return output


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    output = float(value)
    if not math.isfinite(output) or output < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return output


def _finite_positive(value: object, *, name: str) -> float:
    output = _finite_nonnegative(value, name=name)
    if output <= 0.0:
        raise ValueError(f"{name} must be positive")
    return output


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a nonnegative integer")
    return int(value)


def _positive_integer(value: object, *, name: str) -> int:
    output = _nonnegative_integer(value, name=name)
    if output == 0:
        raise ValueError(f"{name} must be positive")
    return output


__all__ = [
    "AnnotatedConditionDecision",
    "PENDULUM_V8_OBSERVER_VERSION",
    "select_annotated_condition_structure_v8",
]
