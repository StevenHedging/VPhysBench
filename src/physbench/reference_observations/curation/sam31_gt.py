"""Pure identity and candidate-selection contracts for SAM 3.1 GT curation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2:
        raise ValueError("SAM3.1 GT candidate mask must be two-dimensional")
    if value.dtype.kind not in "buif":
        raise ValueError("SAM3.1 GT candidate mask must be numeric or boolean")
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        raise ValueError("SAM3.1 GT candidate mask must be finite")
    result = value > 0
    if not result.any():
        raise ValueError("SAM3.1 GT candidate mask must not be empty")
    return result


@dataclass(frozen=True)
class GtCandidate:
    candidate_id: str
    prompt: str
    backend_object_id: int
    mask: np.ndarray
    confidence: float
    centroid_xy: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    area_pixels: int
    source: str = "full_frame_text"

    @classmethod
    def from_mask(
        cls,
        *,
        candidate_id: str,
        prompt: str,
        backend_object_id: int,
        mask: np.ndarray,
        confidence: float,
        source: str = "full_frame_text",
    ) -> "GtCandidate":
        if not candidate_id or not prompt:
            raise ValueError("SAM3.1 GT candidate identity and prompt are required")
        if not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
            raise ValueError("SAM3.1 GT candidate confidence must lie in [0,1]")
        binary = _binary_mask(mask)
        ys, xs = np.nonzero(binary)
        return cls(
            candidate_id=str(candidate_id),
            prompt=str(prompt),
            backend_object_id=int(backend_object_id),
            mask=binary,
            confidence=float(confidence),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ),
            area_pixels=int(len(xs)),
            source=str(source),
        )

    @property
    def width(self) -> float:
        return float(self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return float(self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def compactness(self) -> float:
        return float(self.area_pixels / max(1.0, self.width * self.height))

    @property
    def touches_border(self) -> bool:
        height, width = self.mask.shape
        left, top, right, bottom = self.bbox_xyxy
        return left == 0 or top == 0 or right == width or bottom == height


@dataclass(frozen=True)
class CollisionPhysicsBinding:
    object_ids: tuple[str, ...]
    radii_m: tuple[float, ...]


@dataclass(frozen=True)
class OrderedGtIdentity:
    object_id: str
    evaluator_object_id: str
    mask_id: str
    candidate: GtCandidate


def validate_physics_caption_binding(
    physics: Mapping[str, Any],
    caption: str | Mapping[str, Any],
) -> CollisionPhysicsBinding:
    """Validate the collision numbering authority without consulting old masks."""

    physics_value: Any = physics.get("physics", physics)
    if not isinstance(physics_value, Mapping):
        raise ValueError("physics must contain an object mapping")
    objects = physics_value.get("objects")
    if not isinstance(objects, Mapping) or not objects:
        raise ValueError("physics.objects must be a non-empty mapping")
    object_ids = tuple(str(value) for value in objects)
    expected = tuple(f"object_{index}" for index in range(1, len(objects) + 1))
    if object_ids != expected:
        raise ValueError(
            f"physics object IDs must be contiguous: observed {object_ids}, expected {expected}"
        )
    caption_text: Any = caption.get("caption") if isinstance(caption, Mapping) else caption
    if not isinstance(caption_text, str) or not caption_text:
        raise ValueError("caption must be non-empty text")
    radii: list[float] = []
    for object_id in object_ids:
        quantities = objects[object_id]
        if not isinstance(quantities, Mapping):
            raise ValueError(f"physics {object_id} quantities must be a mapping")
        radius = quantities.get("radius")
        if not isinstance(radius, Mapping):
            raise ValueError(f"physics {object_id} misses radius")
        radius_value = float(radius.get("value", math.nan))
        if not math.isfinite(radius_value) or radius_value <= 0:
            raise ValueError(f"physics {object_id} radius must be positive and finite")
        radii.append(radius_value)
        for quantity in quantities.values():
            if not isinstance(quantity, Mapping):
                continue
            symbol = quantity.get("symbol")
            if isinstance(symbol, str) and symbol not in caption_text:
                raise ValueError(f"caption misses {symbol}")
    return CollisionPhysicsBinding(object_ids=object_ids, radii_m=tuple(radii))


def row_major_candidates(
    candidates: Iterable[GtCandidate],
) -> tuple[GtCandidate, ...]:
    values = tuple(candidates)
    if not values:
        return ()
    tolerance = max(1.0, 0.5 * float(median(item.height for item in values)))
    rows: list[list[GtCandidate]] = []
    row_centers: list[float] = []
    for candidate in sorted(
        values,
        key=lambda item: (item.centroid_xy[1], item.centroid_xy[0], item.candidate_id),
    ):
        y_center = candidate.centroid_xy[1]
        if not rows or abs(y_center - row_centers[-1]) > tolerance:
            rows.append([candidate])
            row_centers.append(y_center)
            continue
        rows[-1].append(candidate)
        row_centers[-1] = float(
            sum(item.centroid_xy[1] for item in rows[-1]) / len(rows[-1])
        )
    return tuple(
        item
        for row in rows
        for item in sorted(row, key=lambda value: (value.centroid_xy[0], value.candidate_id))
    )


def bind_row_major_identities(
    candidates: Sequence[GtCandidate],
    binding: CollisionPhysicsBinding,
) -> tuple[OrderedGtIdentity, ...]:
    ordered = row_major_candidates(candidates)
    if len(ordered) != len(binding.object_ids):
        raise ValueError(
            "selected candidate count differs from physics object count: "
            f"found {len(ordered)}, expected {len(binding.object_ids)}"
        )
    return tuple(
        OrderedGtIdentity(
            object_id=object_id,
            evaluator_object_id=f"ball_{index}",
            mask_id=f"{index:02d}",
            candidate=candidate,
        )
        for index, (object_id, candidate) in enumerate(
            zip(binding.object_ids, ordered, strict=True), start=1
        )
    )


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("SAM3.1 GT candidate masks must share one frame shape")
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return float(intersection / union) if union else 1.0


def deduplicate_candidates(
    candidates: Iterable[GtCandidate],
    *,
    mask_iou_threshold: float = 0.7,
    centroid_height_fraction: float = 0.35,
) -> tuple[GtCandidate, ...]:
    if not 0 < mask_iou_threshold <= 1 or centroid_height_fraction < 0:
        raise ValueError("SAM3.1 GT deduplication thresholds are invalid")
    ordered = sorted(
        candidates,
        key=lambda item: (-item.confidence, item.candidate_id),
    )
    result: list[GtCandidate] = []
    for candidate in ordered:
        duplicate = False
        for kept in result:
            distance = math.dist(candidate.centroid_xy, kept.centroid_xy)
            height_scale = max(candidate.height, kept.height)
            if (
                _mask_iou(candidate.mask, kept.mask) >= mask_iou_threshold
                or distance <= centroid_height_fraction * height_scale
            ):
                duplicate = True
                break
        if not duplicate:
            result.append(candidate)
    return tuple(sorted(result, key=lambda item: item.candidate_id))


def _combination_score(
    candidates: Sequence[GtCandidate],
    expected_radii: Sequence[float],
) -> float:
    ordered = row_major_candidates(candidates)
    heights = np.asarray([item.height for item in ordered], dtype=np.float64)
    centers_y = np.asarray([item.centroid_xy[1] for item in ordered], dtype=np.float64)
    median_height = max(1.0, float(np.median(heights)))
    row_dispersion = float(np.std(centers_y) / median_height)

    observed_radii = np.sqrt(
        np.asarray([item.area_pixels for item in ordered], dtype=np.float64) / np.pi
    )
    expected = np.asarray(expected_radii, dtype=np.float64)
    log_error = np.log(observed_radii) - np.log(expected)
    usable = np.asarray([not item.touches_border for item in ordered], dtype=bool)
    if np.count_nonzero(usable) >= 2:
        normalized = log_error[usable] - float(np.mean(log_error[usable]))
        radius_mismatch = float(np.sqrt(np.mean(normalized**2)))
    else:
        radius_mismatch = 0.0

    confidence = float(np.mean([item.confidence for item in ordered]))
    compactness = float(np.mean([min(1.0, item.compactness) for item in ordered]))
    return confidence + 0.35 * compactness - 2.5 * row_dispersion - radius_mismatch


def select_collision_candidates(
    candidates: Iterable[GtCandidate],
    *,
    expected_radii: Sequence[float],
    ambiguity_margin: float = 0.03,
) -> tuple[GtCandidate, ...]:
    expected = tuple(float(value) for value in expected_radii)
    if not expected or any(not math.isfinite(value) or value <= 0 for value in expected):
        raise ValueError("expected collision radii must be positive and finite")
    if ambiguity_margin < 0:
        raise ValueError("collision candidate ambiguity margin must be nonnegative")
    unique = deduplicate_candidates(candidates)
    expected_count = len(expected)
    if len(unique) < expected_count:
        raise ValueError(
            "SAM3.1 candidates cannot explain the physics subject count: "
            f"found {len(unique)}, expected {expected_count}"
        )
    scored = sorted(
        (
            (_combination_score(combination, expected), tuple(combination))
            for combination in itertools.combinations(unique, expected_count)
        ),
        key=lambda item: (-item[0], tuple(value.candidate_id for value in item[1])),
    )
    if len(scored) > 1 and scored[0][0] - scored[1][0] < ambiguity_margin:
        raise ValueError(
            "SAM3.1 collision candidate selection is ambiguous: "
            f"best margin {scored[0][0] - scored[1][0]:.6f} is below "
            f"{ambiguity_margin:.6f}"
        )
    return row_major_candidates(scored[0][1])


__all__ = [
    "CollisionPhysicsBinding",
    "GtCandidate",
    "OrderedGtIdentity",
    "bind_row_major_identities",
    "deduplicate_candidates",
    "row_major_candidates",
    "select_collision_candidates",
    "validate_physics_caption_binding",
]
