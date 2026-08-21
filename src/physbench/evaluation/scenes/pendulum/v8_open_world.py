"""Annotation-constrained condition geometry for ``pendulum_state_v8``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from ...common.errors import ReferenceAnalysisError
from .open_world import PendulumStructureSpec, _line_segments, _structure_masks
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
    condition_frame: np.ndarray | None = None,
    expected_initial_angle_deg: float | None = None,
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

    hypotheses = list(decision.hypotheses)
    anchor_line_evidence = None
    if condition_frame is not None and expected_initial_angle_deg is not None:
        anchor_line_evidence = infer_anchor_guided_pivot_v8(
            condition_frame,
            anchor=anchor,
            expected_initial_angle_deg=expected_initial_angle_deg,
            config=config,
        )
        if anchor_line_evidence is not None:
            hypotheses.append(
                {
                    "center_xy": anchor.centroid_xy.tolist(),
                    "pivot_xy": anchor_line_evidence["pivot_xy"],
                    "radius_px": float(anchor.equivalent_radius_px),
                    "length_px": float(
                        np.linalg.norm(
                            anchor.centroid_xy
                            - np.asarray(
                                anchor_line_evidence["pivot_xy"],
                                dtype=np.float64,
                            )
                        )
                    ),
                    "score": float(anchor_line_evidence["score"]),
                    "source_agreement_count": max(minimum_sources, 3),
                    "edge_closure_score": float(
                        anchor_line_evidence["score"]
                    ),
                    "body_contrast_score": 1.0,
                    "source": "v8_anchor_guided_fragmented_string",
                    "supporting_segments": int(
                        anchor_line_evidence["supporting_segments"]
                    ),
                }
            )

    audited: list[dict[str, object]] = []
    for index, hypothesis in enumerate(hypotheses):
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
            "anchor_guided_string_evidence": anchor_line_evidence,
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


def infer_anchor_guided_pivot_v8(
    condition_frame: np.ndarray,
    *,
    anchor: PendulumSubjectAnchor,
    expected_initial_angle_deg: float,
    config: Mapping[str, Any],
) -> dict[str, object] | None:
    """Fuse collinear string fragments terminating at the annotated bob.

    Pale strings can be split by Canny into two or more segments.  Each
    accepted segment must independently point through the frozen bob and agree
    with the annotated initial-angle magnitude.  The upper endpoint of the
    longest supported span supplies the pivot; the bob remains annotation
    owned.
    """

    frame = np.asarray(condition_frame)
    if (
        frame.ndim != 3
        or frame.shape[:2] != anchor.mask.shape
        or frame.shape[2] != 3
        or frame.dtype != np.uint8
    ):
        raise ValueError("condition frame and pendulum anchor must align")
    angle = _finite_nonnegative(
        expected_initial_angle_deg,
        name="expected_initial_angle_deg",
    )
    if angle > 90.0:
        raise ValueError("expected_initial_angle_deg must not exceed 90")
    tolerance = _finite_positive(
        config.get("v8_anchor_string_angle_tolerance_deg", 26.0),
        name="v8_anchor_string_angle_tolerance_deg",
    )
    line_distance_ratio = _finite_positive(
        config.get("v8_anchor_string_maximum_line_distance_radii", 1.25),
        name="v8_anchor_string_maximum_line_distance_radii",
    )
    center = np.asarray(anchor.centroid_xy, dtype=np.float64)
    radius = max(float(anchor.equivalent_radius_px), 1.0)
    supported: list[dict[str, object]] = []
    for first, second in _line_segments(frame, config=config):
        upper, lower = (
            (first, second) if first[1] <= second[1] else (second, first)
        )
        vector = lower - upper
        length = float(np.linalg.norm(vector))
        if length < max(0.65 * radius, 6.0) or vector[1] <= 0.0:
            continue
        segment_angle = math.degrees(
            math.atan2(abs(float(vector[0])), float(vector[1]))
        )
        angle_error = abs(segment_angle - angle)
        if angle_error > tolerance:
            continue
        projection = float(
            np.dot(center - upper, vector) / max(length * length, 1.0)
        )
        if projection < 0.65:
            continue
        closest = upper + projection * vector
        line_distance = float(np.linalg.norm(center - closest))
        if line_distance > line_distance_ratio * radius:
            continue
        direction_sign = int(np.sign(vector[0]))
        angle_score = math.exp(-angle_error / max(tolerance, 1.0))
        distance_score = math.exp(-line_distance / radius)
        supported.append(
            {
                "upper": upper,
                "lower": lower,
                "direction_sign": direction_sign,
                "angle_error_deg": angle_error,
                "line_distance_px": line_distance,
                "score": 0.55 * angle_score + 0.45 * distance_score,
            }
        )
    if not supported:
        return None

    groups: dict[int, list[dict[str, object]]] = {}
    for segment in supported:
        groups.setdefault(int(segment["direction_sign"]), []).append(segment)
    ranked: list[tuple[float, np.ndarray, list[dict[str, object]]]] = []
    for group in groups.values():
        pivot = min(group, key=lambda value: float(value["upper"][1]))[
            "upper"
        ]
        span = float(np.linalg.norm(center - pivot))
        if span < 2.0 * radius:
            continue
        mean_score = float(np.mean([value["score"] for value in group]))
        coverage = min(sum(
            float(np.linalg.norm(value["lower"] - value["upper"]))
            for value in group
        ) / span, 1.0)
        ranked.append((0.75 * mean_score + 0.25 * coverage, pivot, group))
    if not ranked:
        return None
    score, pivot, group = max(
        ranked,
        key=lambda value: (value[0], len(value[2]), -float(value[1][1])),
    )
    return {
        "pivot_xy": np.asarray(pivot, dtype=np.float64).tolist(),
        "score": float(np.clip(score, 0.0, 1.0)),
        "supporting_segments": len(group),
        "policy": "anchor_collinear_fragment_fusion_v1",
    }


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
