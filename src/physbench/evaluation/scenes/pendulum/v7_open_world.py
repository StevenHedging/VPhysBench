"""Versioned pendulum observation helpers for ``pendulum_state_v7``.

The v6 evaluator is intentionally frozen.  This module implements the v7
condition-only anchor, proposal fusion, persistence-aware residual promotion,
and symmetric topology observation without changing any v6 code path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from ...common.entities.observer import (
    EvidenceTier,
    ObjectDetection,
    OpenWorldObservation,
    OpenWorldTrack,
    track_open_world_detections,
)
from ...common.entities.timeline import CommonTimeGrid
from ...common.errors import ReferenceAnalysisError
from ...common.masks.quality import mask_centroid
from .open_world import (
    PendulumStructureSpec,
    _circle_candidates,
    _circle_string_evidence,
    _condition_change_mask,
    _directed_bob_detections,
    _histogram_intersection,
    _line_segments,
    _mask_histogram,
    _mask_overlap_fraction,
    _structure_masks,
    _weak_directed_string_circle,
    audit_pendulum_topology,
)
from .scoring import PendulumTrace, TraceQualityError, _estimate_period


PENDULUM_V7_OBSERVER_VERSION = "condition_causal_fused_v4"
PENDULUM_V7_TOPOLOGY_VERSION = "symmetric_multisource_v2"


@dataclass(frozen=True)
class ConditionStructureDecision:
    """Condition-only pendulum anchor plus auditable hypotheses."""

    structure: PendulumStructureSpec
    hypotheses: tuple[Mapping[str, object], ...]
    confidence_margin: float
    source_agreement: float
    rejection_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_version": PENDULUM_V7_OBSERVER_VERSION,
            "structure": self.structure.to_dict(),
            "confidence_margin": self.confidence_margin,
            "source_agreement": self.source_agreement,
            "rejection_counts": {
                str(key): int(value)
                for key, value in self.rejection_counts.items()
            },
            "hypotheses": [dict(value) for value in self.hypotheses],
        }


def _circle_edge_support(
    edges: np.ndarray,
    *,
    center_xy: Sequence[float],
    radius_px: float,
) -> float:
    center = tuple(int(round(value)) for value in center_xy)
    radius = max(2, int(round(radius_px)))
    annulus = np.zeros(edges.shape, dtype=np.uint8)
    cv2.circle(annulus, center, radius + 2, 255, -1)
    cv2.circle(annulus, center, max(1, radius - 2), 0, -1)
    selected = annulus > 0
    if not np.any(selected):
        return 0.0
    # A closed one-pixel circle occupies only part of this four-pixel annulus.
    density = float(np.mean(edges[selected] > 0))
    return float(np.clip(density / 0.22, 0.0, 1.0))


def _circle_body_contrast(
    frame: np.ndarray,
    *,
    center_xy: Sequence[float],
    radius_px: float,
) -> float:
    center = tuple(int(round(value)) for value in center_xy)
    radius = max(2, int(round(radius_px)))
    inner = np.zeros(frame.shape[:2], dtype=np.uint8)
    outer = np.zeros_like(inner)
    cv2.circle(inner, center, max(1, int(round(0.72 * radius))), 255, -1)
    cv2.circle(outer, center, max(radius + 2, int(round(1.65 * radius))), 255, -1)
    cv2.circle(outer, center, max(radius + 1, int(round(1.15 * radius))), 0, -1)
    if not np.any(inner) or not np.any(outer):
        return 0.0
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float64)
    delta = float(
        np.linalg.norm(
            np.mean(lab[inner > 0], axis=0)
            - np.mean(lab[outer > 0], axis=0)
        )
    )
    return float(np.clip(delta / 45.0, 0.0, 1.0))


def _extend_partial_string_pivot(
    *,
    center_xy: Sequence[float],
    raw_pivot_xy: Sequence[float],
    radius_px: float,
    expected_radius_length_ratio: float,
    frame_shape: tuple[int, ...],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, object]]:
    """Extend a partial Hough string to its condition-physical pivot.

    ``HoughLinesP`` often returns only the terminal portion of a long string.
    Treating that segment's upper endpoint as the pivot makes an otherwise
    correct bob appear to have the wrong pendulum length.  The Case supplies
    a radius/length ratio, so v7 may extend the *observed line direction* to
    the declared physical scale.  No future frame or reference trajectory is
    used.
    """

    center = np.asarray(center_xy, dtype=np.float64)
    raw_pivot = np.asarray(raw_pivot_xy, dtype=np.float64)
    delta = center - raw_pivot
    raw_length = float(np.linalg.norm(delta))
    expected_length = float(radius_px / expected_radius_length_ratio)
    diagnostics: dict[str, object] = {
        "raw_pivot_xy": raw_pivot.tolist(),
        "raw_length_px": raw_length,
        "physical_expected_length_px": expected_length,
        "physical_pivot_extended": False,
        "physical_pivot_extension_scale": 1.0,
    }
    if raw_length <= 1e-9 or expected_length <= 1e-9:
        return raw_pivot, diagnostics
    minimum_fraction = float(
        config.get(
            "v7_condition_pivot_extension_minimum_length_fraction",
            0.78,
        )
    )
    maximum_scale = float(
        config.get("v7_condition_pivot_extension_maximum_scale", 4.0)
    )
    scale = expected_length / raw_length
    if raw_length >= minimum_fraction * expected_length or scale > maximum_scale:
        return raw_pivot, diagnostics
    candidate = center - delta / raw_length * expected_length
    height, width = frame_shape[:2]
    margin = float(
        config.get("v7_condition_pivot_extension_canvas_margin_px", 4.0)
    )
    if not (
        margin <= candidate[0] <= width - 1.0 - margin
        and margin <= candidate[1] <= height - 1.0 - margin
        and candidate[1] < center[1] - float(radius_px)
    ):
        return raw_pivot, diagnostics
    diagnostics.update(
        {
            "physical_pivot_extended": True,
            "physical_pivot_extension_scale": scale,
        }
    )
    return candidate, diagnostics


def _support_continues_below_candidate(
    *,
    center_xy: Sequence[float],
    pivot_xy: Sequence[float],
    radius_px: float,
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    minimum_alignment: float,
    maximum_axis_distance_radius_ratio: float,
    minimum_extension_radius_ratio: float,
) -> tuple[bool, dict[str, float | None]]:
    """Reject a circle placed on a through-going support or stand edge.

    A pendulum bob is the terminal body of its pivot-to-bob branch.  A Hough
    circle on the vertical stand can satisfy circle, appearance, physical
    radius/length, and upward-line tests simultaneously.  It is nevertheless
    not terminal: a long, nearly collinear segment continues below it.  This
    test uses only current-condition pixels and the candidate's own branch;
    no reference or prediction coordinates participate.
    """

    center = np.asarray(center_xy, dtype=np.float64)
    pivot = np.asarray(pivot_xy, dtype=np.float64)
    radius = float(radius_px)
    alignment_threshold = float(minimum_alignment)
    axis_distance_ratio = float(maximum_axis_distance_radius_ratio)
    extension_ratio = float(minimum_extension_radius_ratio)
    if (
        center.shape != (2,)
        or pivot.shape != (2,)
        or not np.isfinite(center).all()
        or not np.isfinite(pivot).all()
        or not math.isfinite(radius)
        or radius <= 0.0
        or not 0.0 <= alignment_threshold <= 1.0
        or not math.isfinite(axis_distance_ratio)
        or axis_distance_ratio < 0.0
        or not math.isfinite(extension_ratio)
        or extension_ratio <= 0.0
    ):
        raise ValueError("support-continuation inputs are invalid")
    branch = center - pivot
    branch_length = float(np.linalg.norm(branch))
    if branch_length <= 1e-9:
        return False, {
            "support_continuation_alignment": None,
            "support_continuation_axis_distance_radius_ratio": None,
            "support_continuation_extension_radius_ratio": None,
        }
    branch_unit = branch / branch_length
    best_alignment: float | None = None
    best_axis_distance_ratio: float | None = None
    best_extension_ratio: float | None = None
    rejected = False
    for first, second in segments:
        start = np.asarray(first, dtype=np.float64)
        end = np.asarray(second, dtype=np.float64)
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= 1e-9:
            continue
        alignment = abs(float(np.dot(direction / length, branch_unit)))
        projection_denominator = float(np.dot(direction, direction))
        fraction = float(
            np.clip(
                np.dot(center - start, direction)
                / max(projection_denominator, 1e-12),
                0.0,
                1.0,
            )
        )
        closest = start + fraction * direction
        axis_distance_ratio_value = float(
            np.linalg.norm(center - closest) / radius
        )
        projections = (
            float(np.dot(start - center, branch_unit)),
            float(np.dot(end - center, branch_unit)),
        )
        forward_extension_ratio = max(projections) / radius
        crosses_candidate = min(projections) <= radius
        if (
            alignment >= alignment_threshold
            and axis_distance_ratio_value <= axis_distance_ratio
            and crosses_candidate
            and forward_extension_ratio >= extension_ratio
        ):
            rejected = True
            if (
                best_extension_ratio is None
                or forward_extension_ratio > best_extension_ratio
            ):
                best_alignment = alignment
                best_axis_distance_ratio = axis_distance_ratio_value
                best_extension_ratio = forward_extension_ratio
    return rejected, {
        "support_continuation_alignment": best_alignment,
        "support_continuation_axis_distance_radius_ratio": (
            best_axis_distance_ratio
        ),
        "support_continuation_extension_radius_ratio": best_extension_ratio,
    }


def detect_condition_structure_v7(
    frame: np.ndarray,
    *,
    config: Mapping[str, Any],
    expected_radius_length_ratio: float,
    expected_initial_angle_deg: float | None = None,
) -> ConditionStructureDecision:
    """Select a bob from condition pixels using independent evidence.

    Hough is only a proposal source.  A proposal must also agree with an
    upward string/pivot, the declared physical radius-to-length ratio, and
    raw-image body evidence.  No Case identifier, fixed image coordinate, or
    future frame contributes to this decision.
    """

    expected_ratio = float(expected_radius_length_ratio)
    if not math.isfinite(expected_ratio) or expected_ratio <= 0.0:
        raise ValueError("expected_radius_length_ratio must be positive")
    expected_angle = (
        None
        if expected_initial_angle_deg is None
        else abs(float(expected_initial_angle_deg))
    )
    if expected_angle is not None and (
        not math.isfinite(expected_angle) or expected_angle > 90.0
    ):
        raise ValueError("expected_initial_angle_deg must be in [-90, 90]")
    circles = _circle_candidates(
        frame,
        config=config,
        maximum_candidates=int(
            config.get("maximum_condition_circle_candidates", 200)
        ),
    )
    segments = _line_segments(frame, config=config)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(
        gray,
        float(config.get("line_canny_low", 35.0)),
        float(config.get("line_canny_high", 120.0)),
    )
    height, width = frame.shape[:2]
    diagonal = math.hypot(width, height)
    hypotheses: list[dict[str, object]] = []
    rejection_counts = {
        "minimum_visible_angle": 0,
        "physics_ratio": 0,
        "support_continues_below_candidate": 0,
    }
    for x, y, radius in circles:
        if (
            x - radius <= 1.0
            or x + radius >= width - 1.0
            or y - radius <= 1.0
            or y + radius >= height - 1.0
        ):
            continue
        line_score, raw_pivot = _circle_string_evidence(
            (x, y, radius),
            segments,
            frame_shape=frame.shape,
        )
        if raw_pivot is None:
            continue
        center = np.asarray([x, y], dtype=np.float64)
        pivot, pivot_diagnostics = _extend_partial_string_pivot(
            center_xy=center,
            raw_pivot_xy=raw_pivot,
            radius_px=radius,
            expected_radius_length_ratio=expected_ratio,
            frame_shape=frame.shape,
            config=config,
        )
        length = float(np.linalg.norm(center - pivot))
        if (
            length
            < float(config.get("minimum_length_radius_ratio", 2.5))
            * radius
            or length
            > float(config.get("maximum_length_canvas_ratio", 0.8))
            * diagonal
            or float(pivot[1]) >= y - radius
        ):
            continue
        observed_ratio = radius / max(length, 1.0)
        ratio_score = float(
            math.exp(-abs(math.log(observed_ratio / expected_ratio)))
        )
        if ratio_score < float(
            config.get(
                "v7_condition_minimum_physics_ratio_score",
                0.50,
            )
        ):
            rejection_counts["physics_ratio"] += 1
            continue
        edge_score = _circle_edge_support(
            edges,
            center_xy=center,
            radius_px=radius,
        )
        body_score = _circle_body_contrast(
            frame,
            center_xy=center,
            radius_px=radius,
        )
        observed_angle = abs(
            math.degrees(
                math.atan2(
                    float(center[0] - pivot[0]),
                    float(center[1] - pivot[1]),
                )
            )
        )
        if expected_angle is not None:
            minimum_visible_angle = min(
                expected_angle,
                float(
                    config.get(
                        "v7_condition_minimum_visible_angle_deg",
                        7.0,
                    )
                ),
            )
            if observed_angle + 1e-12 < minimum_visible_angle:
                rejection_counts["minimum_visible_angle"] += 1
                continue
        support_continuation, support_diagnostics = (
            _support_continues_below_candidate(
                center_xy=center,
                pivot_xy=pivot,
                radius_px=radius,
                segments=segments,
                minimum_alignment=float(
                    config.get(
                        "v7_condition_support_continuation_minimum_alignment",
                        0.90,
                    )
                ),
                maximum_axis_distance_radius_ratio=float(
                    config.get(
                        "v7_condition_support_continuation_"
                        "maximum_axis_distance_radius_ratio",
                        1.20,
                    )
                ),
                minimum_extension_radius_ratio=float(
                    config.get(
                        "v7_condition_support_continuation_"
                        "minimum_extension_radius_ratio",
                        2.50,
                    )
                ),
            )
        )
        if support_continuation:
            rejection_counts["support_continues_below_candidate"] += 1
            continue
        angle_score = (
            None
            if expected_angle is None
            else float(
                math.exp(
                    -abs(observed_angle - expected_angle)
                    / float(config.get("v7_initial_angle_scale_deg", 7.5))
                )
            )
        )
        full_string_edge_score = _raw_string_edge_score(
            frame,
            pivot_xy=pivot,
            bob_xy=center,
            half_width_px=int(
                config.get("v7_condition_full_string_half_width_px", 2)
            ),
        )
        # Hough supplies the proposal; these terms are independent
        # validations.  Line geometry and the physical radius/length ratio
        # dominate identity selection.  The annotated initial angle is a
        # deliberately weak ranking cue because an OOD condition image can
        # preserve the entity while perturbing its apparent pose.
        base_score = (
            float(
                0.42 * line_score
                + 0.30 * ratio_score
                + 0.16 * edge_score
                + 0.12 * body_score
            )
            if angle_score is None
            else float(
                0.35 * line_score
                + 0.30 * ratio_score
                + 0.05 * angle_score
                + 0.20 * edge_score
                + 0.10 * body_score
            )
        )
        extension_scale = float(
            pivot_diagnostics["physical_pivot_extension_scale"]
        )
        extension_penalty = (
            float(
                config.get(
                    "v7_condition_pivot_extension_log_penalty", 0.12
                )
            )
            * max(0.0, math.log(max(extension_scale, 1.0)))
            if bool(pivot_diagnostics["physical_pivot_extended"])
            else 0.0
        )
        full_string_weight = float(
            config.get("v7_condition_full_string_edge_weight", 0.15)
        )
        score = float(
            np.clip(
                (1.0 - full_string_weight) * base_score
                + full_string_weight * full_string_edge_score
                - extension_penalty,
                0.0,
                1.0,
            )
        )
        agreement_flags = [
            line_score >= float(config.get("v7_minimum_line_agreement", 0.25)),
            ratio_score >= float(config.get("v7_minimum_ratio_agreement", 0.55)),
            edge_score >= float(config.get("v7_minimum_edge_agreement", 0.18)),
            body_score >= float(config.get("v7_minimum_body_agreement", 0.10)),
        ]
        if angle_score is not None:
            agreement_flags.append(
                angle_score
                >= float(config.get("v7_minimum_angle_agreement", 0.35))
            )
        hypotheses.append(
            {
                "center_xy": [float(x), float(y)],
                "pivot_xy": np.asarray(pivot, dtype=np.float64).tolist(),
                "radius_px": float(radius),
                "length_px": length,
                "radius_length_ratio": observed_ratio,
                "observed_initial_angle_deg": observed_angle,
                "line_score": float(line_score),
                "physics_ratio_score": ratio_score,
                "physics_angle_score": angle_score,
                "edge_closure_score": edge_score,
                "body_contrast_score": body_score,
                "full_string_edge_score": full_string_edge_score,
                "pre_extension_score": base_score,
                "pivot_extension_penalty": extension_penalty,
                "source_agreement_count": int(sum(agreement_flags)),
                **pivot_diagnostics,
                **support_diagnostics,
                "score": score,
            }
        )
    if not hypotheses:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_structure_missing_v7",
            "condition-only multi-source observer found no circle connected "
            "to an upward pendulum string",
        )
    hypotheses.sort(
        key=lambda value: (
            float(value["score"]),
            int(value["source_agreement_count"]),
            float(value["length_px"]),
        ),
        reverse=True,
    )
    best = hypotheses[0]
    minimum_agreement = int(
        config.get(
            "v7_minimum_source_agreement",
            3 if expected_angle is not None else 2,
        )
    )
    minimum_score = float(config.get("v7_minimum_condition_score", 0.30))
    if (
        int(best["source_agreement_count"]) < minimum_agreement
        or float(best["score"]) < minimum_score
    ):
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_structure_weak_v7",
            "best condition-only bob hypothesis lacks independent source "
            f"agreement (score={float(best['score']):.3f}, "
            f"sources={int(best['source_agreement_count'])})",
        )
    distinct = [
        value
        for value in hypotheses[1:]
        if np.linalg.norm(
            np.asarray(value["center_xy"], dtype=np.float64)
            - np.asarray(best["center_xy"], dtype=np.float64)
        )
        > 1.5
        * max(float(value["radius_px"]), float(best["radius_px"]), 1.0)
    ]
    runner_up = float(distinct[0]["score"]) if distinct else 0.0
    margin = float(best["score"]) - runner_up
    ambiguity_margin = float(config.get("v7_condition_ambiguity_margin", 0.005))
    if distinct and margin < ambiguity_margin:
        raise ReferenceAnalysisError(
            "reference_condition_pendulum_structure_ambiguous_v7",
            "spatially distinct condition-only bob hypotheses are tied "
            f"within {margin:.4f}",
        )
    center = np.asarray(best["center_xy"], dtype=np.float64)
    pivot = np.asarray(best["pivot_xy"], dtype=np.float64)
    radius = float(best["radius_px"])
    bob_mask, subject_mask = _structure_masks(
        (height, width),
        pivot_xy=pivot,
        bob_xy=center,
        radius_px=radius,
    )
    structure = PendulumStructureSpec(
        pivot_xy=pivot,
        bob_xy=center,
        bob_radius_px=radius,
        bob_mask=bob_mask,
        subject_mask=subject_mask,
        confidence=float(np.clip(best["score"], 0.0, 1.0)),
        source="condition_v7_hough_string_physics_edge_body",
    )
    return ConditionStructureDecision(
        structure=structure,
        hypotheses=tuple(hypotheses[:12]),
        confidence_margin=margin,
        source_agreement=float(
            int(best["source_agreement_count"])
            / (5.0 if expected_angle is not None else 4.0)
        ),
        rejection_counts=rejection_counts,
    )


def _mask_relation(
    first: np.ndarray | None,
    second: np.ndarray | None,
) -> tuple[float, float]:
    if first is None or second is None:
        return 0.0, 0.0
    a = np.asarray(first) > 0
    b = np.asarray(second) > 0
    intersection = int(np.count_nonzero(a & b))
    first_area = int(np.count_nonzero(a))
    second_area = int(np.count_nonzero(b))
    union = first_area + second_area - intersection
    return (
        float(intersection / union) if union else 0.0,
        float(intersection / max(min(first_area, second_area), 1)),
    )


def _proposal_pair_should_fuse(
    first: ObjectDetection,
    second: ObjectDetection,
    *,
    center_radius_ratio: float,
    minimum_mask_iou: float,
    minimum_containment: float,
) -> bool:
    iou, containment = _mask_relation(first.mask, second.mask)
    first_radius = math.sqrt(first.area_px2 / math.pi)
    second_radius = math.sqrt(second.area_px2 / math.pi)
    center_distance = float(np.linalg.norm(first.xy - second.xy))
    center_duplicate = center_distance <= center_radius_ratio * max(
        first_radius + second_radius,
        1.0,
    )
    ordinary_duplicate = bool(
        iou >= minimum_mask_iou
        or containment >= minimum_containment
        or (center_duplicate and containment > 0.0)
    )
    if ordinary_duplicate:
        return True
    first_directed = "condition_directed_sam2" in first.sources
    second_directed = "condition_directed_sam2" in second.sources
    terminal_repair = bool(
        (
            first.metadata.get("terminal_residual_repair", False)
            and second_directed
        )
        or (
            second.metadata.get("terminal_residual_repair", False)
            and first_directed
        )
    )
    return terminal_repair


def fuse_pendulum_proposals_v7(
    detections: Sequence[ObjectDetection],
    *,
    center_radius_ratio: float = 0.65,
    minimum_mask_iou: float = 0.18,
    minimum_containment: float = 0.55,
) -> list[ObjectDetection]:
    """Fuse overlapping observation sources before they acquire track IDs."""

    values = list(detections)
    if len({value.frame_index for value in values}) > 1:
        raise ValueError("proposal fusion accepts exactly one frame")
    ordered = sorted(values, key=lambda value: value.detection_id)
    parent = list(range(len(ordered)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(
                first_root, second_root
            )

    # Pairwise connected components make fusion transitive and independent
    # of proposal enumeration order.  A residual may also repair a directed
    # SAM terminal fragment without pretending that both masks overlap.
    for first_index, first in enumerate(ordered):
        for second_index in range(first_index + 1, len(ordered)):
            if _proposal_pair_should_fuse(
                first,
                ordered[second_index],
                center_radius_ratio=center_radius_ratio,
                minimum_mask_iou=minimum_mask_iou,
                minimum_containment=minimum_containment,
            ):
                union(first_index, second_index)
    grouped: dict[int, list[ObjectDetection]] = {}
    for index, value in enumerate(ordered):
        grouped.setdefault(find(index), []).append(value)
    groups = [
        grouped[key]
        for key in sorted(
            grouped,
            key=lambda root: min(
                value.detection_id for value in grouped[root]
            ),
        )
    ]

    output: list[ObjectDetection] = []
    tier_rank = {
        EvidenceTier.AMBIGUOUS: 0,
        EvidenceTier.TENTATIVE: 1,
        EvidenceTier.INDEPENDENT_SALIENT: 2,
        EvidenceTier.PARTICIPANT: 3,
    }
    for group_index, group in enumerate(groups):
        sources = tuple(
            sorted({source for value in group for source in value.sources})
        )
        proposal_sources = tuple(
            sorted(
                {
                    str(source)
                    for value in group
                    for source in value.metadata.get(
                        "proposal_sources", value.sources
                    )
                }
            )
        )
        valid_direct = any(
            "condition_directed_sam2" in value.sources
            and value.metadata.get("identity_anchor_valid") is not False
            for value in group
        )
        invalid_direct = any(
            "condition_directed_sam2" in value.sources
            and value.metadata.get("identity_anchor_valid") is False
            for value in group
        )
        independent_residual = any(
            bool(value.metadata.get("independent_residual_evidence", False))
            for value in group
        )
        entity_class = (
            "pendulum_bob"
            if valid_direct or independent_residual or not invalid_direct
            else "pendulum_bob__replacement"
        )
        best = max(
            group,
            key=lambda value: (
                tier_rank[value.evidence_tier],
                value.confidence,
                value.area_px2,
            ),
        )
        terminal_repairs = [
            value
            for value in group
            if bool(
                value.metadata.get("terminal_residual_repair", False)
            )
        ]
        # Directed SAM supplies identity evidence.  When it contains only a
        # string fragment, a geometry-validated terminal residual supplies
        # localization and the mask.  Otherwise a valid directed mask remains
        # the preferred localization source.
        if terminal_repairs:
            best = max(
                terminal_repairs,
                key=lambda value: (value.confidence, value.area_px2),
            )
        elif independent_residual and not valid_direct:
            best = max(
                (
                    value
                    for value in group
                    if value.metadata.get(
                        "independent_residual_evidence", False
                    )
                ),
                key=lambda value: (value.confidence, value.area_px2),
            )
        tier = max(
            (value.evidence_tier for value in group),
            key=tier_rank.__getitem__,
        )
        metadata: dict[str, object] = {
            "fused_detection_ids": [
                value.detection_id for value in group
            ],
            "proposal_sources": list(proposal_sources),
            "proposal_fusion_count": len(group),
            "identity_anchor_valid": bool(
                valid_direct or independent_residual
            ),
            "directed_source_present": bool(valid_direct or invalid_direct),
            "independent_residual_evidence": independent_residual,
            "terminal_residual_repair": bool(terminal_repairs),
            "identity_evidence_source": (
                "condition_directed_sam2"
                if valid_direct
                else "independent_residual"
            ),
            "localization_evidence_source": (
                "terminal_residual"
                if terminal_repairs
                else (
                    "condition_directed_sam2"
                    if "condition_directed_sam2" in best.sources
                    else "independent_residual"
                )
            ),
        }
        metadata.update(dict(best.metadata))
        metadata["proposal_sources"] = list(proposal_sources)
        metadata["proposal_fusion_count"] = len(group)
        metadata["identity_anchor_valid"] = bool(
            valid_direct or independent_residual
        )
        output.append(
            ObjectDetection(
                frame_index=best.frame_index,
                detection_id=(
                    f"pendulum_v7_fused_{best.frame_index:05d}_"
                    f"{group_index:03d}"
                ),
                xy=best.xy,
                area_px2=best.area_px2,
                entity_class=entity_class,
                mask=best.mask,
                confidence=max(value.confidence for value in group),
                evidence_tier=tier,
                sources=sources,
                metadata=metadata,
            )
        )
    return output


def _pearson_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not a.size:
        return 0.0
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, 0.0, 1.0))


def _condition_preexistence_score(
    frame: np.ndarray,
    condition_frame: np.ndarray,
    *,
    center_xy: Sequence[float],
    radius_px: float,
    search_radius_px: int,
    patch_radius_ratio: float,
    frame_lab: np.ndarray | None = None,
    condition_lab: np.ndarray | None = None,
    frame_gradient: np.ndarray | None = None,
    condition_gradient: np.ndarray | None = None,
) -> float:
    """Measure whether a proposal was already apparatus in the condition.

    The comparison stays at the same image location and searches only a few
    pixels for camera jitter.  Mean-normalized intensity and gradient NCC make
    the score tolerant to modest exposure changes without allowing a newly
    introduced object elsewhere in the condition to explain the proposal.
    """

    if frame.shape != condition_frame.shape:
        raise ValueError("condition preexistence frames must align")
    height, width = frame.shape[:2]
    center = np.asarray(center_xy, dtype=np.float64)
    half = max(
        4,
        int(math.ceil(float(patch_radius_ratio) * float(radius_px))),
    )
    cx, cy = (int(round(float(value))) for value in center)
    x0, x1 = max(0, cx - half), min(width, cx + half + 1)
    y0, y1 = max(0, cy - half), min(height, cy + half + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0
    if frame_lab is None:
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
    if condition_lab is None:
        condition_lab = cv2.cvtColor(
            condition_frame, cv2.COLOR_BGR2LAB
        ).astype(np.float32)
    if frame_lab.shape != frame.shape or condition_lab.shape != frame.shape:
        raise ValueError("precomputed Lab images must match the video frame")
    if frame_gradient is None:
        frame_l = frame_lab[..., 0]
        frame_gradient = cv2.magnitude(
            cv2.Sobel(frame_l, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(frame_l, cv2.CV_32F, 0, 1, ksize=3),
        )
    if condition_gradient is None:
        condition_l = condition_lab[..., 0]
        condition_gradient = cv2.magnitude(
            cv2.Sobel(condition_l, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(condition_l, cv2.CV_32F, 0, 1, ksize=3),
        )
    target_lab = frame_lab[y0:y1, x0:x1]
    target_l = target_lab[..., 0]
    target_gradient = frame_gradient[y0:y1, x0:x1]
    best = 0.0
    search = max(0, int(search_radius_px))
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sx0, sx1 = x0 + dx, x1 + dx
            sy0, sy1 = y0 + dy, y1 + dy
            if sx0 < 0 or sy0 < 0 or sx1 > width or sy1 > height:
                continue
            candidate_lab = condition_lab[sy0:sy1, sx0:sx1]
            candidate_l = candidate_lab[..., 0]
            candidate_gradient = condition_gradient[
                sy0:sy1, sx0:sx1
            ]
            intensity = _pearson_similarity(target_l, candidate_l)
            gradient = _pearson_similarity(
                target_gradient, candidate_gradient
            )
            mean_delta = float(
                np.linalg.norm(
                    np.mean(target_lab, axis=(0, 1))
                    - np.mean(candidate_lab, axis=(0, 1))
                )
            )
            colour = float(math.exp(-mean_delta / 32.0))
            score = 0.55 * intensity + 0.35 * gradient + 0.10 * colour
            best = max(best, float(np.clip(score, 0.0, 1.0)))
    return best


def _terminal_residual_relation(
    *,
    center_xy: np.ndarray,
    radius_px: float,
    area_px2: float,
    directed_center_xy: np.ndarray | None,
    directed_area_px2: float,
    structure: PendulumStructureSpec,
    config: Mapping[str, Any],
) -> dict[str, object]:
    anchor_area = max(
        float(np.count_nonzero(structure.bob_mask)),
        math.pi * structure.bob_radius_px**2,
        1.0,
    )
    output: dict[str, object] = {
        "terminal_residual_repair": False,
        "terminal_directed_area_ratio": (
            float(directed_area_px2 / anchor_area)
            if directed_area_px2 > 0.0
            else None
        ),
        "terminal_residual_area_ratio": float(area_px2 / anchor_area),
    }
    if directed_center_xy is None or directed_area_px2 <= 0.0:
        return output
    pivot = np.asarray(structure.pivot_xy, dtype=np.float64)
    directed_vector = np.asarray(directed_center_xy, dtype=np.float64) - pivot
    residual_vector = np.asarray(center_xy, dtype=np.float64) - pivot
    directed_length = float(np.linalg.norm(directed_vector))
    residual_length = float(np.linalg.norm(residual_vector))
    if directed_length <= 1e-9 or residual_length <= 1e-9:
        return output
    length = max(float(structure.length_px), 1.0)
    directed_length_ratio = directed_length / length
    residual_length_ratio = residual_length / length
    alignment = float(
        np.dot(directed_vector, residual_vector)
        / (directed_length * residual_length)
    )
    residual_unit = residual_vector / residual_length
    perpendicular_radii = float(
        np.linalg.norm(
            directed_vector
            - np.dot(directed_vector, residual_unit) * residual_unit
        )
        / max(float(radius_px), structure.bob_radius_px, 1.0)
    )
    separation_radii = float(
        np.linalg.norm(
            np.asarray(center_xy, dtype=np.float64)
            - np.asarray(directed_center_xy, dtype=np.float64)
        )
        / max(float(radius_px), structure.bob_radius_px, 1.0)
    )
    terminal_improvement = abs(1.0 - directed_length_ratio) - abs(
        1.0 - residual_length_ratio
    )
    directed_area_ratio = directed_area_px2 / anchor_area
    residual_area_ratio = area_px2 / anchor_area
    repair = bool(
        directed_area_ratio
        <= float(
            config.get(
                "v7_terminal_repair_directed_maximum_area_ratio", 0.65
            )
        )
        and float(
            config.get(
                "v7_terminal_repair_residual_minimum_area_ratio", 0.50
            )
        )
        <= residual_area_ratio
        <= float(
            config.get(
                "v7_terminal_repair_residual_maximum_area_ratio", 1.80
            )
        )
        and residual_length_ratio
        >= float(
            config.get(
                "v7_terminal_repair_minimum_length_ratio", 0.78
            )
        )
        and residual_length_ratio
        <= float(
            config.get(
                "v7_terminal_repair_maximum_length_ratio", 1.20
            )
        )
        and residual_length > directed_length
        and terminal_improvement
        >= float(
            config.get(
                "v7_terminal_repair_minimum_length_improvement", 0.025
            )
        )
        and alignment
        >= float(
            config.get("v7_terminal_repair_minimum_alignment", 0.965)
        )
        and perpendicular_radii
        <= float(
            config.get(
                "v7_terminal_repair_maximum_perpendicular_radii", 1.25
            )
        )
        and separation_radii
        <= float(
            config.get(
                "v7_terminal_repair_maximum_separation_radii", 3.50
            )
        )
    )
    output.update(
        {
            "terminal_residual_repair": repair,
            "terminal_directed_length_ratio": directed_length_ratio,
            "terminal_residual_length_ratio": residual_length_ratio,
            "terminal_branch_alignment": alignment,
            "terminal_perpendicular_radii": perpendicular_radii,
            "terminal_separation_radii": separation_radii,
            "terminal_length_improvement": terminal_improvement,
        }
    )
    return output


def _residual_detections(
    frames: Sequence[np.ndarray],
    directed_bob_masks: Sequence[np.ndarray],
    *,
    directed_subject_masks: Sequence[np.ndarray] | None,
    availability: np.ndarray,
    condition_frame: np.ndarray,
    structure: PendulumStructureSpec,
    config: Mapping[str, Any],
) -> tuple[list[list[ObjectDetection]], dict[str, int]]:
    output: list[list[ObjectDetection]] = [[] for _ in frames]
    condition_histogram = _mask_histogram(
        condition_frame, structure.bob_mask
    )
    condition_lab = cv2.cvtColor(
        condition_frame, cv2.COLOR_BGR2LAB
    ).astype(np.float32)
    condition_l = condition_lab[..., 0]
    condition_gradient = cv2.magnitude(
        cv2.Sobel(condition_l, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(condition_l, cv2.CV_32F, 0, 1, ksize=3),
    )
    rejected = {
        "weak_string_circle": 0,
        "apparatus": 0,
        "condition_preexisting_apparatus": 0,
        "single_source": 0,
    }
    for frame_index, frame in enumerate(frames):
        if not bool(availability[frame_index]):
            continue
        change = _condition_change_mask(
            frame,
            condition_frame,
            threshold=float(config.get("condition_change_threshold", 24.0)),
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        frame_l = frame_lab[..., 0]
        frame_gradient = cv2.magnitude(
            cv2.Sobel(frame_l, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(frame_l, cv2.CV_32F, 0, 1, ksize=3),
        )
        frame_edges = cv2.Canny(
            gray,
            float(config.get("line_canny_low", 35.0)),
            float(config.get("line_canny_high", 120.0)),
        )
        segments = _line_segments(frame, config=config)
        directed_center = mask_centroid(directed_bob_masks[frame_index])
        directed_area = float(
            np.count_nonzero(directed_bob_masks[frame_index])
        )
        directed_radius = math.sqrt(max(directed_area, 1.0) / math.pi)
        directed_subject_mask = (
            directed_bob_masks[frame_index]
            if directed_subject_masks is None
            else directed_subject_masks[frame_index]
        )
        for candidate_index, (x, y, radius) in enumerate(
            _circle_candidates(
                frame,
                config=config,
                radius_hint_px=structure.bob_radius_px,
            )
        ):
            center = np.asarray([x, y], dtype=np.float64)
            pivot_distance = float(
                np.linalg.norm(center - structure.pivot_xy)
            )
            if not (
                float(config.get("v7_minimum_residual_length_ratio", 0.60))
                * structure.length_px
                <= pivot_distance
                <= float(config.get("v7_maximum_residual_length_ratio", 1.35))
                * structure.length_px
            ):
                continue
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(
                mask,
                (int(round(x)), int(round(y))),
                max(2, int(round(radius))),
                255,
                -1,
            )
            candidate_area = float(np.count_nonzero(mask))
            line_score, line_pivot = _circle_string_evidence(
                (x, y, radius),
                segments,
                frame_shape=frame.shape,
            )
            changed = _mask_overlap_fraction(mask, change)
            directed_subject_overlap = _mask_overlap_fraction(
                mask, directed_subject_mask
            )
            color = _histogram_intersection(
                condition_histogram,
                _mask_histogram(frame, mask),
            )
            edge_support = _circle_edge_support(
                frame_edges,
                center_xy=center,
                radius_px=radius,
            )
            body_contrast = _circle_body_contrast(
                frame,
                center_xy=center,
                radius_px=radius,
            )
            weak, relation = _weak_directed_string_circle(
                center,
                radius_px=radius,
                pivot_xy=structure.pivot_xy,
                directed_bob_xy=directed_center,
                directed_bob_radius_px=directed_radius,
                change_overlap=changed,
                minimum_body_change_overlap=float(
                    config.get(
                        "minimum_string_candidate_change_overlap", 0.30
                    )
                ),
                bob_clearance_radius_ratio=float(
                    config.get(
                        "direct_duplicate_center_radius_ratio", 2.25
                    )
                ),
            )
            pivot_error = (
                float(
                    np.linalg.norm(line_pivot - structure.pivot_xy)
                    / max(structure.length_px, 1.0)
                )
                if line_pivot is not None
                else math.inf
            )
            anchored = bool(
                line_score
                >= float(config.get("participant_string_score", 0.55))
                and pivot_error
                <= float(
                    config.get(
                        "maximum_residual_pivot_error_ratio", 0.25
                    )
                )
            )
            changed_body = changed >= float(
                config.get("minimum_change_overlap", 0.25)
            )
            appearance = color >= float(
                config.get("v7_minimum_residual_color_similarity", 0.15)
            )
            round_body = bool(
                edge_support
                >= float(config.get("v7_minimum_residual_edge_support", 0.20))
                and body_contrast
                >= float(config.get("v7_minimum_residual_body_contrast", 0.12))
            )
            anchor_center_distance = float(
                np.linalg.norm(center - structure.bob_xy)
            )
            directed_departure_distance = (
                math.inf
                if directed_center is None
                else float(
                    np.linalg.norm(directed_center - structure.bob_xy)
                )
            )
            anchor_copy = bool(
                directed_center is not None
                and directed_departure_distance
                >= float(
                    config.get(
                        "v7_anchor_copy_directed_departure_radius_ratio",
                        2.5,
                    )
                )
                * max(structure.bob_radius_px, 1.0)
                and anchor_center_distance
                <= float(
                    config.get(
                        "v7_anchor_copy_center_radius_ratio",
                        1.25,
                    )
                )
                * max(radius, structure.bob_radius_px, 1.0)
                and appearance
                and round_body
            )
            terminal_relation = _terminal_residual_relation(
                center_xy=center,
                radius_px=radius,
                area_px2=candidate_area,
                directed_center_xy=directed_center,
                directed_area_px2=directed_area,
                structure=structure,
                config=config,
            )
            if weak and not anchor_copy:
                rejected["weak_string_circle"] += 1
                continue
            independent = bool(
                round_body
                and (
                    (changed_body and (anchored or appearance))
                    or anchor_copy
                )
            )
            sam_identity_recovery = bool(
                directed_center is None
                and directed_subject_overlap
                >= float(
                    config.get(
                        "v7_residual_minimum_directed_subject_overlap",
                        0.50,
                    )
                )
                and round_body
                and appearance
                and (
                    anchored
                    or (
                        float(
                            config.get(
                                "v7_terminal_repair_minimum_length_ratio",
                                0.78,
                            )
                        )
                        <= pivot_distance
                        / max(structure.length_px, 1.0)
                        <= float(
                            config.get(
                                "v7_terminal_repair_maximum_length_ratio",
                                1.20,
                            )
                        )
                    )
                )
            )
            geometry_identity_recovery = bool(
                directed_center is None
                and edge_support
                >= float(
                    config.get(
                        "v7_recovery_minimum_edge_support",
                        0.60,
                    )
                )
                and body_contrast
                >= float(
                    config.get(
                        "v7_recovery_minimum_body_contrast",
                        0.55,
                    )
                )
                and color
                >= float(
                    config.get(
                        "v7_recovery_minimum_color_similarity",
                        0.45,
                    )
                )
                and anchored
            )
            independent = bool(
                independent
                or sam_identity_recovery
                or geometry_identity_recovery
            )
            support_half_width = (
                float(config.get("support_exclusion_radius_ratio", 1.2))
                * structure.bob_radius_px
            )
            in_support = bool(
                abs(x - structure.pivot_xy[0]) <= support_half_width
                and y > structure.pivot_xy[1]
            )
            if (
                in_support
                and not anchor_copy
                and not (anchored and changed >= 0.55)
            ):
                rejected["apparatus"] += 1
                continue
            if not independent:
                rejected["single_source"] += 1
                continue
            preexistence_score = _condition_preexistence_score(
                frame,
                condition_frame,
                center_xy=center,
                radius_px=radius,
                search_radius_px=int(
                    config.get(
                        "v7_condition_preexistence_search_radius_px", 3
                    )
                ),
                patch_radius_ratio=float(
                    config.get(
                        "v7_condition_preexistence_patch_radius_ratio",
                        1.50,
                    )
                ),
                frame_lab=frame_lab,
                condition_lab=condition_lab,
                frame_gradient=frame_gradient,
                condition_gradient=condition_gradient,
            )
            condition_preexisting = bool(
                preexistence_score
                >= float(
                    config.get(
                        "v7_condition_preexistence_rejection_threshold",
                        0.84,
                    )
                )
            )
            if (
                condition_preexisting
                and not anchor_copy
                and not bool(
                    terminal_relation["terminal_residual_repair"]
                )
                and not sam_identity_recovery
                and not geometry_identity_recovery
            ):
                rejected["condition_preexisting_apparatus"] += 1
                continue
            proposal_sources = ["hough_circle"]
            if changed_body:
                proposal_sources.append("condition_frame_change")
            if anchored:
                proposal_sources.append("pivot_string_geometry")
            if appearance:
                proposal_sources.append("condition_appearance")
            if anchor_copy:
                proposal_sources.append(
                    "condition_anchor_occupancy_after_directed_departure"
                )
            if sam_identity_recovery:
                proposal_sources.append(
                    "directed_subject_terminal_overlap"
                )
            if geometry_identity_recovery:
                proposal_sources.append(
                    "condition_geometry_causal_recovery"
                )
            output[frame_index].append(
                ObjectDetection(
                    frame_index=frame_index,
                    detection_id=(
                        f"pendulum_v7_residual_{frame_index:05d}_"
                        f"{candidate_index:03d}"
                    ),
                    xy=center,
                    area_px2=candidate_area,
                    entity_class="pendulum_bob",
                    mask=mask,
                    confidence=float(
                        np.clip(
                            0.45
                            + 0.25 * changed
                            + 0.20 * line_score
                            + 0.10 * color,
                            0.0,
                            0.89,
                        )
                    ),
                    # Persistence is evaluated after tracking.  A single
                    # residual frame must never become a free participant.
                    evidence_tier=EvidenceTier.TENTATIVE,
                    sources=("pendulum_v7_residual_candidate",),
                    metadata={
                        "proposal_sources": proposal_sources,
                        "independent_residual_evidence": True,
                        "condition_change_overlap": changed,
                        "anchor_color_similarity": color,
                        "circle_edge_support": edge_support,
                        "body_contrast_score": body_contrast,
                        "round_body_evidence": round_body,
                        "string_score": line_score,
                        "string_pivot_error_ratio": (
                            pivot_error if math.isfinite(pivot_error) else None
                        ),
                        "anchored_string": anchored,
                        "condition_anchor_copy_evidence": anchor_copy,
                        "condition_anchor_center_distance_px": (
                            anchor_center_distance
                        ),
                        "condition_preexistence_score": (
                            preexistence_score
                        ),
                        "condition_novelty_score": (
                            1.0 - preexistence_score
                        ),
                        "condition_preexisting_apparatus": (
                            condition_preexisting
                        ),
                        "directed_subject_overlap": (
                            directed_subject_overlap
                        ),
                        "condition_identity_recovery_candidate": (
                            sam_identity_recovery
                        ),
                        "condition_geometry_recovery_candidate": (
                            geometry_identity_recovery
                        ),
                        "directed_anchor_departure_distance_px": (
                            directed_departure_distance
                            if math.isfinite(directed_departure_distance)
                            else None
                        ),
                        **terminal_relation,
                        **relation,
                    },
                )
            )
    return output, rejected


def _calibrate_residual_tracks(
    tracks: Sequence[OpenWorldTrack],
    *,
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
) -> tuple[tuple[OpenWorldTrack, ...], int]:
    minimum_frames = int(config.get("v7_residual_minimum_frames", 3))
    minimum_duration = float(
        config.get("v7_residual_minimum_duration_s", 0.12)
    )
    promoted = 0
    output: list[OpenWorldTrack] = []
    times = time_grid.times_s
    for track in tracks:
        has_valid_directed = any(
            "condition_directed_sam2" in detection.sources
            and detection.metadata.get("identity_anchor_valid") is not False
            for detection in track.detections
        )
        is_replacement = track.entity_class.endswith("__replacement")
        residual_only = all(
            "pendulum_v7_residual_candidate" in detection.sources
            or bool(
                detection.metadata.get(
                    "independent_residual_evidence", False
                )
            )
            for detection in track.detections
        )
        frame_span = (
            float(
                times[track.detections[-1].frame_index]
                - times[track.detections[0].frame_index]
            )
            if len(track.detections) > 1
            else 0.0
        )
        persistent = bool(
            len(track.detections) >= minimum_frames
            and frame_span + 1e-12 >= minimum_duration
        )
        if has_valid_directed or is_replacement:
            tier = EvidenceTier.PARTICIPANT
        elif residual_only and persistent:
            tier = EvidenceTier.PARTICIPANT
            promoted += 1
        else:
            tier = EvidenceTier.AMBIGUOUS
        output.append(
            OpenWorldTrack(
                track_id=track.track_id,
                detections=track.detections,
                confirmed=track.confirmed,
                evidence_tier=tier,
            )
        )
    return tuple(output), promoted


def _directed_continuity(
    detection: ObjectDetection,
    *,
    accepted: Sequence[ObjectDetection],
    structure: PendulumStructureSpec,
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
) -> tuple[bool, dict[str, object]]:
    """Validate one directed observation against the causal identity history.

    Appearance similarity alone cannot distinguish two visually identical
    bobs.  The condition anchor, pendulum length, and a causal constant-velocity
    prediction therefore act as conservative identity constraints.  Rejected
    detections remain observable as replacement participants; they are never
    silently discarded.
    """

    # Identity thresholds are condition-frozen.  A bloated prediction mask
    # must not relax its own displacement gate.
    radius = max(float(structure.bob_radius_px), 1.0)
    length_ratio = float(
        np.linalg.norm(detection.xy - structure.pivot_xy)
        / max(structure.length_px, 1.0)
    )
    minimum_length_ratio = float(
        config.get("v7_directed_minimum_length_ratio", 0.60)
    )
    maximum_length_ratio = float(
        config.get("v7_directed_maximum_length_ratio", 1.35)
    )
    length_valid = bool(
        minimum_length_ratio <= length_ratio <= maximum_length_ratio
    )
    diagnostics: dict[str, object] = {
        "directed_continuity_length_ratio": length_ratio,
        "directed_continuity_length_valid": length_valid,
        "directed_continuity_history_size": len(accepted),
    }
    if not accepted:
        anchor_distance_radii = float(
            np.linalg.norm(detection.xy - structure.bob_xy) / radius
        )
        anchor_valid = anchor_distance_radii <= float(
            config.get(
                "v7_directed_condition_anchor_maximum_distance_radii",
                2.5,
            )
        )
        diagnostics.update(
            {
                "directed_continuity_mode": "condition_anchor",
                "directed_condition_anchor_distance_radii": (
                    anchor_distance_radii
                ),
                "directed_continuity_position_valid": anchor_valid,
            }
        )
        return bool(length_valid and anchor_valid), diagnostics

    last = accepted[-1]
    times = time_grid.times_s
    frame_steps = int(detection.frame_index - last.frame_index)
    positive_steps = np.diff(times)
    median_step_s = (
        float(np.median(positive_steps))
        if positive_steps.size
        else 0.0
    )
    elapsed_s = float(
        times[detection.frame_index] - times[last.frame_index]
    )
    elapsed_steps = (
        elapsed_s / median_step_s
        if median_step_s > 1e-12
        else float(frame_steps)
    )
    maximum_recovery_steps = float(
        config.get("v7_directed_maximum_recovery_gap_steps", 3.0)
    )
    recovery_gap_valid = bool(
        frame_steps > 0
        and elapsed_steps <= maximum_recovery_steps + 1e-12
    )
    displacement_radii = float(
        np.linalg.norm(detection.xy - last.xy) / radius
    )
    base_jump_limit = float(
        config.get("v7_directed_maximum_jump_radius_ratio", 5.0)
    )
    # Only a velocity-supported short-gap recovery receives a longer
    # displacement horizon.  A one-frame far replacement sees the original
    # frozen limit.
    jump_horizon = (
        max(1.0, elapsed_steps)
        if len(accepted) >= 2 and recovery_gap_valid
        else 1.0
    )
    jump_limit = base_jump_limit * jump_horizon
    jump_valid = displacement_radii <= jump_limit
    velocity_error_radii: float | None = None
    velocity_valid = False
    if len(accepted) >= 2:
        previous = accepted[-2]
        previous_time = float(times[previous.frame_index])
        last_time = float(times[last.frame_index])
        current_time = float(times[detection.frame_index])
        delta = last_time - previous_time
        horizon = current_time - last_time
        if delta > 1e-12 and horizon > 0.0:
            velocity = (last.xy - previous.xy) / delta
            predicted = last.xy + velocity * horizon
            velocity_error_radii = float(
                np.linalg.norm(detection.xy - predicted) / radius
            )
            velocity_valid = velocity_error_radii <= float(
                config.get(
                    "v7_directed_maximum_velocity_error_radius_ratio",
                    4.0,
                )
            )
    position_valid = bool(
        recovery_gap_valid
        and jump_valid
        and (velocity_valid if len(accepted) >= 2 else True)
    )
    diagnostics.update(
        {
            "directed_continuity_mode": (
                "causal_velocity" if len(accepted) >= 2 else "causal_jump"
            ),
            "directed_displacement_radius_ratio": displacement_radii,
            "directed_elapsed_gap_frames": frame_steps,
            "directed_elapsed_gap_s": elapsed_s,
            "directed_elapsed_gap_steps": elapsed_steps,
            "directed_recovery_gap_valid": recovery_gap_valid,
            "directed_jump_limit_radius_ratio": jump_limit,
            "directed_velocity_error_radius_ratio": velocity_error_radii,
            "directed_jump_valid": jump_valid,
            "directed_velocity_valid": velocity_valid,
            "directed_continuity_position_valid": position_valid,
        }
    )
    return bool(length_valid and position_valid), diagnostics


def _as_directed_replacement(
    detection: ObjectDetection,
    *,
    diagnostics: Mapping[str, object],
) -> ObjectDetection:
    metadata = {
        **dict(detection.metadata),
        **dict(diagnostics),
        "identity_anchor_valid": False,
        "identity_continuity_rejected": True,
    }
    return replace(
        detection,
        entity_class="pendulum_bob__replacement",
        sources=tuple(
            sorted(
                {
                    *detection.sources,
                    "identity_continuity_rejected",
                }
            )
        ),
        metadata=metadata,
    )


def discover_pendulum_objects_v7(
    frames: Sequence[np.ndarray],
    *,
    directed_bob_masks: Sequence[np.ndarray],
    directed_subject_masks: Sequence[np.ndarray] | None = None,
    condition_frame: np.ndarray,
    structure: PendulumStructureSpec,
    time_grid: CommonTimeGrid,
    config: Mapping[str, Any],
    available: Sequence[bool] | None = None,
) -> OpenWorldObservation:
    """Observe bobs with per-frame source fusion and persistent residuals."""

    frame_count = len(time_grid.times_s)
    if len(frames) != frame_count or len(directed_bob_masks) != frame_count:
        raise ValueError("pendulum v7 observation timelines must align")
    if (
        directed_subject_masks is not None
        and len(directed_subject_masks) != frame_count
    ):
        raise ValueError(
            "directed subject masks must align with the v7 timeline"
        )
    availability = (
        np.ones(frame_count, dtype=bool)
        if available is None
        else np.asarray(available)
    )
    if availability.shape != (frame_count,) or availability.dtype.kind != "b":
        raise ValueError("available must contain one boolean per frame")
    directed = _directed_bob_detections(
        frames,
        directed_bob_masks,
        availability=availability,
        structure=structure,
        condition_frame=condition_frame,
        config=config,
    )
    residual, rejected = _residual_detections(
        frames,
        directed_bob_masks,
        directed_subject_masks=directed_subject_masks,
        availability=availability,
        condition_frame=condition_frame,
        structure=structure,
        config=config,
    )
    fused = [
        fuse_pendulum_proposals_v7(
            [*directed[index], *residual[index]],
            center_radius_ratio=float(
                config.get("v7_fusion_center_radius_ratio", 0.65)
            ),
            minimum_mask_iou=float(
                config.get("v7_fusion_minimum_mask_iou", 0.18)
            ),
            minimum_containment=float(
                config.get("v7_fusion_minimum_containment", 0.55)
            ),
        )
        for index in range(frame_count)
    ]
    # A validated condition-directed proposal is the observation of the
    # already frozen Case identity, not a free candidate in the residual
    # Hungarian pool.  Keeping that causal channel separate prevents a nearby
    # Hough fit from stealing the identity at an oscillation turning point.
    # Fusion has already happened, so residual evidence can still repair the
    # directed mask before this split.
    condition_detections: list[ObjectDetection] = []
    residual_by_frame: list[list[ObjectDetection]] = [
        [] for _ in range(frame_count)
    ]
    suppressed_near_directed = 0
    directed_continuity_rejections = 0
    residual_identity_recoveries = 0
    recovery_candidates_rejected = 0
    for frame_index, values in enumerate(fused):
        directed_values = [
            value
            for value in values
            if "condition_directed_sam2" in value.sources
            and value.metadata.get("identity_anchor_valid") is not False
            and value.entity_class == "pendulum_bob"
        ]
        directed = (
            max(
                directed_values,
                key=lambda value: (value.confidence, value.area_px2),
            )
            if directed_values
            else None
        )
        directed_continuity_result: (
            tuple[bool, dict[str, object]] | None
        ) = None
        if directed is None:
            recovery_values = [
                value
                for value in values
                if value.entity_class == "pendulum_bob"
                and (
                    bool(
                        value.metadata.get(
                            "condition_identity_recovery_candidate", False
                        )
                    )
                    or bool(
                        value.metadata.get(
                            "condition_geometry_recovery_candidate", False
                        )
                    )
                )
            ]
            if recovery_values:
                valid_recoveries: list[
                    tuple[
                        tuple[float, ...],
                        ObjectDetection,
                        dict[str, object],
                    ]
                ] = []
                maximum_recovery_velocity_error = float(
                    config.get(
                        "v7_recovery_maximum_velocity_error_radius_ratio",
                        2.0,
                    )
                )
                for value in recovery_values:
                    continuity_valid, continuity_diagnostics = (
                        _directed_continuity(
                            value,
                            accepted=condition_detections,
                            structure=structure,
                            time_grid=time_grid,
                            config=config,
                        )
                    )
                    velocity_error = continuity_diagnostics.get(
                        "directed_velocity_error_radius_ratio"
                    )
                    recovery_velocity_valid = bool(
                        len(condition_detections) < 2
                        or (
                            isinstance(velocity_error, (int, float))
                            and not isinstance(velocity_error, bool)
                            and math.isfinite(float(velocity_error))
                            and float(velocity_error)
                            <= maximum_recovery_velocity_error
                        )
                    )
                    if not continuity_valid or not recovery_velocity_valid:
                        recovery_candidates_rejected += 1
                        continue
                    velocity_rank = (
                        0.0
                        if len(condition_detections) < 2
                        else -float(velocity_error)
                    )
                    radius = math.sqrt(
                        max(float(value.area_px2), 1.0) / math.pi
                    )
                    radius_error = abs(
                        radius / max(structure.bob_radius_px, 1.0) - 1.0
                    )
                    valid_recoveries.append(
                        (
                            (
                                velocity_rank,
                                float(
                                    value.metadata.get(
                                        "directed_subject_overlap", 0.0
                                    )
                                ),
                                float(
                                    value.metadata.get(
                                        "body_contrast_score", 0.0
                                    )
                                ),
                                float(
                                    value.metadata.get(
                                        "circle_edge_support", 0.0
                                    )
                                ),
                                float(
                                    value.metadata.get(
                                        "string_score", 0.0
                                    )
                                ),
                                -radius_error,
                                float(value.confidence),
                            ),
                            value,
                            continuity_diagnostics,
                        )
                    )
                if valid_recoveries:
                    _, directed, continuity_diagnostics = max(
                        valid_recoveries,
                        key=lambda item: (
                            item[0],
                            item[1].detection_id,
                        ),
                    )
                    directed_continuity_result = (
                        True,
                        continuity_diagnostics,
                    )
                    directed = replace(
                        directed,
                        evidence_tier=EvidenceTier.PARTICIPANT,
                        sources=tuple(
                            sorted(
                                {
                                    *directed.sources,
                                    "condition_residual_identity_recovery",
                                }
                            )
                        ),
                        metadata={
                            **dict(directed.metadata),
                            "identity_anchor_valid": True,
                            "identity_evidence_source": (
                                "directed_subject_terminal_overlap"
                                if bool(
                                    directed.metadata.get(
                                        "condition_identity_recovery_candidate",
                                        False,
                                    )
                                )
                                else "condition_geometry_causal_continuity"
                            ),
                            "localization_evidence_source": (
                                "condition_residual_circle"
                            ),
                        },
                    )
        selected_condition_detection_id = (
            directed.detection_id if directed is not None else None
        )
        if directed is not None:
            if directed_continuity_result is None:
                continuity_valid, continuity_diagnostics = (
                    _directed_continuity(
                        directed,
                        accepted=condition_detections,
                        structure=structure,
                        time_grid=time_grid,
                        config=config,
                    )
                )
            else:
                continuity_valid, continuity_diagnostics = (
                    directed_continuity_result
                )
            if continuity_valid:
                directed = replace(
                    directed,
                    metadata={
                        **dict(directed.metadata),
                        **continuity_diagnostics,
                        "identity_continuity_rejected": False,
                    },
                )
                condition_detections.append(directed)
                if "condition_residual_identity_recovery" in (
                    directed.sources
                ):
                    residual_identity_recoveries += 1
            else:
                directed = _as_directed_replacement(
                    directed,
                    diagnostics=continuity_diagnostics,
                )
                directed_continuity_rejections += 1
                residual_by_frame[frame_index].append(directed)
        for value in values:
            if (
                selected_condition_detection_id is not None
                and value.detection_id == selected_condition_detection_id
            ):
                continue
            if (
                directed is not None
                and value.entity_class == "pendulum_bob"
                and "condition_directed_sam2" not in value.sources
            ):
                radius = max(
                    math.sqrt(value.area_px2 / math.pi),
                    math.sqrt(directed.area_px2 / math.pi),
                    1.0,
                )
                if float(np.linalg.norm(value.xy - directed.xy)) <= float(
                    config.get(
                        "v7_near_directed_residual_suppression_radius", 2.0
                    )
                ) * radius:
                    suppressed_near_directed += 1
                    continue
            residual_by_frame[frame_index].append(value)
    residual_raw = track_open_world_detections(
        residual_by_frame,
        time_grid=time_grid,
        maximum_gap_s=float(config.get("maximum_tracking_gap_s", 0.25)),
        maximum_assignment_cost=float(
            config.get("maximum_tracking_assignment_cost", 4.0)
        ),
        minimum_scale_px=max(2.0, 0.5 * structure.bob_radius_px),
        maximum_tracks=max(1, int(config.get("maximum_tracks", 24)) - 1),
    )
    residual_tracks, promoted = _calibrate_residual_tracks(
        residual_raw.tracks,
        time_grid=time_grid,
        config=config,
    )
    condition_track = (
        OpenWorldTrack(
            track_id="track_condition_bob",
            detections=tuple(condition_detections),
            confirmed=True,
            evidence_tier=EvidenceTier.PARTICIPANT,
        )
        if condition_detections
        else None
    )
    tracks = (
        ((condition_track,) if condition_track is not None else ())
        + residual_tracks
    )
    formal = [track for track in tracks if track.formal_exposure_weight > 0.0]
    return OpenWorldObservation(
        tracks=tracks,
        overflow_counts=residual_raw.overflow_counts,
        diagnostics={
            **dict(residual_raw.diagnostics),
            "observer": PENDULUM_V7_OBSERVER_VERSION,
            "proposal_fusion_before_tracking": True,
            "source_is_not_identity": True,
            "available_frames": int(np.count_nonzero(availability)),
            "fused_detection_count": int(sum(map(len, fused))),
            "condition_track_observations": len(condition_detections),
            "suppressed_near_directed_residuals": (
                suppressed_near_directed
            ),
            "directed_continuity_rejections": (
                directed_continuity_rejections
            ),
            "residual_identity_recoveries": residual_identity_recoveries,
            "recovery_candidates_rejected": (
                recovery_candidates_rejected
            ),
            "formal_track_count": len(formal),
            "residual_tracks_promoted_by_persistence": promoted,
            "rejected_residual_candidates": rejected,
            "track_saturated": bool(
                np.any(residual_raw.overflow_counts > 0.0)
            ),
        },
    )


def track_masks(
    track: OpenWorldTrack | None,
    *,
    frame_count: int,
    shape: tuple[int, int],
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    masks = [np.zeros(shape, dtype=np.uint8) for _ in range(frame_count)]
    xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
    observed = np.zeros(frame_count, dtype=bool)
    if track is None:
        return masks, xy, observed
    for detection in track.detections:
        index = detection.frame_index
        if detection.mask is not None:
            masks[index] = np.where(
                np.asarray(detection.mask) > 0, 255, 0
            ).astype(np.uint8)
        xy[index] = detection.xy
        observed[index] = True
    return masks, xy, observed


def extract_bob_trace_v7(
    bob_xy: np.ndarray,
    observed: Sequence[bool],
    times_s: Sequence[float],
    *,
    pivot_xy: Sequence[float],
    period_config: Mapping[str, Any],
) -> PendulumTrace:
    """Extract pendulum state from the matched bob, not SAM string extrema."""

    centers = np.asarray(bob_xy, dtype=np.float64)
    valid = np.asarray(observed, dtype=bool)
    times = np.asarray(times_s, dtype=np.float64)
    pivot = np.asarray(pivot_xy, dtype=np.float64)
    if centers.shape != (len(times), 2) or valid.shape != (len(times),):
        raise ValueError("bob trace arrays must align with the time grid")
    valid &= np.isfinite(centers).all(axis=1)
    valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
    if int(np.count_nonzero(valid)) < 3:
        raise TraceQualityError(
            "insufficient_valid_bob_observations_v7",
            "v7 pendulum trace requires at least three matched bob samples",
        )
    indices = np.arange(len(times), dtype=np.float64)
    interpolated = np.array(centers, copy=True)
    for axis in range(2):
        interpolated[~valid, axis] = np.interp(
            indices[~valid],
            indices[valid],
            interpolated[valid, axis],
        )
    displacement = interpolated - pivot
    length = np.linalg.norm(displacement, axis=1)
    if float(np.median(length[valid])) <= 1.0:
        raise TraceQualityError(
            "invalid_pendulum_length_v7",
            "matched bob is too close to the condition-frozen pivot",
        )
    angle = np.arctan2(displacement[:, 0], displacement[:, 1])
    period = _estimate_period(
        angle,
        times,
        minimum_s=float(period_config["minimum_s"]),
        maximum_s=float(period_config["maximum_s"]),
    )
    amplitude = float(
        np.quantile(np.abs(angle - np.median(angle)), 0.95)
    )
    length_cv = float(
        np.std(length[valid]) / max(np.mean(length[valid]), 1e-12)
    )
    return PendulumTrace(
        times_s=times,
        angle_rad=angle,
        bob_xy=interpolated,
        frame_pivot_xy=np.repeat(pivot[None, :], len(times), axis=0),
        pivot_xy=pivot,
        length_px=length,
        valid=valid,
        valid_ratio=valid_ratio,
        pivot_drift_ratio=0.0,
        length_cv=length_cv,
        period_s=period,
        amplitude_rad=amplitude,
    )


def _raw_string_edge_score(
    frame: np.ndarray,
    *,
    pivot_xy: np.ndarray,
    bob_xy: np.ndarray,
    half_width_px: int,
) -> float:
    vector = bob_xy - pivot_xy
    length = float(np.linalg.norm(vector))
    if length <= 8.0:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 28.0, 100.0)
    fractions = np.linspace(0.12, 0.80, 18)
    hits = 0
    radius = max(2, int(half_width_px) + 2)
    for fraction in fractions:
        point = pivot_xy + fraction * vector
        x = int(round(point[0]))
        y = int(round(point[1]))
        y0, y1 = max(0, y - radius), min(edges.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(edges.shape[1], x + radius + 1)
        hits += int(np.any(edges[y0:y1, x0:x1] > 0))
    return float(hits / len(fractions))


def _persistent_signal(
    values: Sequence[float],
    *,
    threshold: float,
    minimum_frames: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    active = array >= float(threshold)
    retained = np.zeros(len(array), dtype=bool)
    start = 0
    while start < len(array):
        if not active[start]:
            start += 1
            continue
        end = start + 1
        while end < len(array) and active[end]:
            end += 1
        if end - start >= minimum_frames:
            retained[start:end] = True
        start = end
    return np.where(retained, array, 0.0)


def observe_pendulum_topology_v7(
    frames: Sequence[np.ndarray],
    assembly_masks: Sequence[np.ndarray],
    *,
    pivot_xy: Sequence[float],
    bob_xy: np.ndarray,
    bob_observed: Sequence[bool],
    bob_masks: Sequence[np.ndarray],
    frame_weights: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Observe topology from mask and raw-image evidence with persistence."""

    base = audit_pendulum_topology(
        assembly_masks,
        pivot_xy=pivot_xy,
        bob_xy=bob_xy,
        bob_observed=bob_observed,
        bob_masks=bob_masks,
        frame_weights=frame_weights,
        string_half_width_px=int(config.get("string_half_width_px", 2)),
        minimum_string_occupancy=float(
            config.get("minimum_string_occupancy", 0.30)
        ),
        branch_penalty_weight=float(
            config.get("branch_string_penalty_weight", 0.65)
        ),
        branch_minimum_length_ratio=float(
            config.get("branch_minimum_length_ratio", 0.12)
        ),
        branch_minimum_elongation=float(
            config.get("branch_minimum_elongation", 3.0)
        ),
        branch_maximum_thickness_radius_ratio=float(
            config.get("branch_maximum_thickness_radius_ratio", 0.75)
        ),
        branch_bob_exclusion_margin_ratio=float(
            config.get("branch_bob_exclusion_margin_ratio", 0.40)
        ),
        branch_support_half_width_radius_ratio=float(
            config.get("branch_support_half_width_radius_ratio", 1.25)
        ),
        branch_support_verticality_threshold=float(
            config.get("branch_support_verticality_threshold", 0.90)
        ),
        branch_pivot_support_verticality_threshold=float(
            config.get(
                "branch_pivot_support_verticality_threshold", 0.65
            )
        ),
    )
    centers = np.asarray(bob_xy, dtype=np.float64)
    observed = np.asarray(bob_observed, dtype=bool)
    pivot = np.asarray(pivot_xy, dtype=np.float64)
    raw_scores: list[float] = []
    mask_scores: list[float] = []
    raw_branch: list[float] = []
    mask_branch: list[float] = []
    for index, row in enumerate(base["per_frame"]):
        if not observed[index] or not np.isfinite(centers[index]).all():
            raw = 0.0
        else:
            raw = _raw_string_edge_score(
                frames[index],
                pivot_xy=pivot,
                bob_xy=centers[index],
                half_width_px=int(config.get("string_half_width_px", 2)),
            )
        raw_scores.append(raw)
        mask_scores.append(float(row["string_intact_score"]))
        mask_branch.append(float(row["branch_string_evidence"]))
        # A raw branch is accepted only when Hough finds a line away from the
        # main string.  The mask channel remains necessary for formal branch
        # evidence, so support hardware from one source cannot trigger it.
        if not observed[index] or not np.isfinite(centers[index]).all():
            raw_branch.append(0.0)
            continue
        main = centers[index] - pivot
        main_length = float(np.linalg.norm(main))
        evidence = 0.0
        if main_length > 8.0:
            main_unit = main / main_length
            for first, second in _line_segments(frames[index], config=config):
                segment = second - first
                span = float(np.linalg.norm(segment))
                if span < float(
                    config.get("branch_minimum_length_ratio", 0.12)
                ) * main_length:
                    continue
                unit = segment / max(span, 1e-9)
                angular = abs(float(np.dot(unit, main_unit)))
                midpoint = 0.5 * (first + second)
                distance_to_pivot = float(np.linalg.norm(midpoint - pivot))
                if (
                    angular < 0.92
                    and distance_to_pivot < 0.9 * main_length
                    and midpoint[1] >= pivot[1]
                ):
                    evidence = max(evidence, 1.0 - angular)
        raw_branch.append(float(np.clip(evidence, 0.0, 1.0)))
    combined_intact = 0.5 * (
        np.asarray(mask_scores) + np.asarray(raw_scores)
    )
    uncertainty = np.abs(
        np.asarray(mask_scores) - np.asarray(raw_scores)
    )
    combined_branch = np.sqrt(
        np.asarray(mask_branch) * np.asarray(raw_branch)
    )
    minimum_frames = int(config.get("v7_topology_persistence_frames", 3))
    persistent_disagreement = _persistent_signal(
        uncertainty,
        threshold=float(
            config.get("v7_topology_source_disagreement_threshold", 0.45)
        ),
        minimum_frames=minimum_frames,
    )
    # Persistent disagreement cannot be treated as evidence that the string is
    # intact.  Use the conservative source only for the persistent conflict
    # window; isolated detector noise retains the averaged observation.
    conservative_intact = np.minimum(
        np.asarray(mask_scores), np.asarray(raw_scores)
    )
    effective_intact = np.where(
        persistent_disagreement > 0.0,
        conservative_intact,
        combined_intact,
    )
    persistent_break = _persistent_signal(
        1.0 - effective_intact,
        threshold=float(config.get("v7_broken_string_threshold", 0.55)),
        minimum_frames=minimum_frames,
    )
    persistent_branch = _persistent_signal(
        combined_branch,
        threshold=float(config.get("v7_branch_threshold", 0.20)),
        minimum_frames=minimum_frames,
    )
    intact = 1.0 - persistent_break
    frame_scores = intact * (
        1.0
        - float(config.get("branch_string_penalty_weight", 0.65))
        * persistent_branch
    )
    weights = np.asarray(frame_weights, dtype=np.float64)
    denominator = max(float(weights.sum()), 1e-12)
    rows = []
    for index, base_row in enumerate(base["per_frame"]):
        rows.append(
            {
                **dict(base_row),
                "raw_edge_string_score": float(raw_scores[index]),
                "mask_string_score": float(mask_scores[index]),
                "topology_source_disagreement": float(
                    uncertainty[index]
                ),
                "persistent_source_disagreement": float(
                    persistent_disagreement[index]
                ),
                "persistent_break_evidence": float(
                    persistent_break[index]
                ),
                "raw_branch_evidence": float(raw_branch[index]),
                "mask_branch_evidence": float(mask_branch[index]),
                "persistent_branch_evidence": float(
                    persistent_branch[index]
                ),
                "topology_frame_score_v7": float(frame_scores[index]),
            }
        )
    return {
        "version": PENDULUM_V7_TOPOLOGY_VERSION,
        "score": float(np.dot(frame_scores, weights) / denominator),
        "string_intact_score": float(np.dot(intact, weights) / denominator),
        "branch_string_exposure_ratio": float(
            np.dot(persistent_branch, weights) / denominator
        ),
        "source_disagreement_ratio": float(
            np.dot(uncertainty, weights) / denominator
        ),
        "persistent_source_disagreement_ratio": float(
            np.dot(persistent_disagreement, weights) / denominator
        ),
        "broken_frame_count": int(np.count_nonzero(persistent_break)),
        "branch_frame_count": int(np.count_nonzero(persistent_branch)),
        "per_frame": rows,
    }


def compare_pendulum_topology_v7(
    reference: Mapping[str, object] | None,
    prediction: Mapping[str, object],
    *,
    frame_weights: Sequence[float],
) -> dict[str, object]:
    """Compare topology symmetrically, or score it absolutely for OOD."""

    prediction_rows = list(prediction["per_frame"])
    weights = np.asarray(frame_weights, dtype=np.float64)
    denominator = max(float(weights.sum()), 1e-12)
    if reference is None:
        values = np.asarray(
            [
                float(row["topology_frame_score_v7"])
                for row in prediction_rows
            ],
            dtype=np.float64,
        )
        score = float(np.dot(values, weights) / denominator)
        mode = "condition_absolute_no_parent_future_pixels"
    else:
        reference_rows = list(reference["per_frame"])
        if len(reference_rows) != len(prediction_rows):
            raise ValueError("topology observations use different timelines")
        values = np.asarray(
            [
                math.exp(
                    -abs(
                        float(prediction_rows[index][
                            "persistent_break_evidence"
                        ])
                        - float(reference_rows[index][
                            "persistent_break_evidence"
                        ])
                    )
                    / 0.25
                )
                * math.exp(
                    -abs(
                        float(prediction_rows[index][
                            "persistent_branch_evidence"
                        ])
                        - float(reference_rows[index][
                            "persistent_branch_evidence"
                        ])
                    )
                    / 0.20
                )
                for index in range(len(prediction_rows))
            ],
            dtype=np.float64,
        )
        score = float(np.dot(values, weights) / denominator)
        mode = "symmetric_reference_prediction_multisource"
    disagreement = float(prediction["source_disagreement_ratio"])
    return {
        "version": PENDULUM_V7_TOPOLOGY_VERSION,
        "score": float(np.clip(score, 0.0, 1.0)),
        "mode": mode,
        "source_disagreement_ratio": disagreement,
        "uncertain": disagreement > 0.45,
        "per_frame_similarity": values.tolist(),
        "reference_observation": (
            None if reference is None else dict(reference)
        ),
        "prediction_observation": dict(prediction),
    }


__all__ = [
    "ConditionStructureDecision",
    "PENDULUM_V7_OBSERVER_VERSION",
    "PENDULUM_V7_TOPOLOGY_VERSION",
    "compare_pendulum_topology_v7",
    "detect_condition_structure_v7",
    "discover_pendulum_objects_v7",
    "extract_bob_trace_v7",
    "fuse_pendulum_proposals_v7",
    "observe_pendulum_topology_v7",
    "track_masks",
]
