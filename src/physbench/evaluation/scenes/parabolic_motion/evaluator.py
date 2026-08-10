from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import cv2
import numpy as np

from ...common.artifacts import save_iou_curve, save_series_comparison
from ...common.artifacts.tables import write_rows_csv
from ...common.base import ReferenceCaseEvaluator, SceneAnalysis
from ...common.csti import build_csti_input_from_aligned_masks
from ...common.entities import (
    ReferenceCapability,
    build_common_time_grid,
    materialize_entity_manifest,
)
from ...common.errors import ReferenceAnalysisError
from ...common.media import SampledVideo
from ...common.subject import infer_reference_mode
from ....io import write_json
from .visualization import write_parabolic_visualization


EPSILON = 1e-12


@dataclass(frozen=True)
class BallCandidate:
    xy: np.ndarray
    radius: float
    mask: np.ndarray
    histogram: np.ndarray
    contrast: float
    source: str


@dataclass(frozen=True)
class ParabolicObservation:
    xy: np.ndarray
    radii: np.ndarray
    observed: np.ndarray
    interpolated: np.ndarray
    masks: tuple[np.ndarray, ...]
    histograms: tuple[np.ndarray | None, ...]
    cardinality: np.ndarray
    seed: BallCandidate | None
    diagnostics: dict[str, Any]

    @property
    def observed_count(self) -> int:
        return int(np.count_nonzero(self.observed))


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.size == 0:
        return 0.0
    denominator = float(weights.sum())
    if denominator <= EPSILON:
        return float(np.mean(values))
    return float(np.dot(values, weights) / denominator)


def _similarity(error: float, scale: float) -> float:
    if not math.isfinite(error) or scale <= 0.0:
        return 0.0
    return _bounded(math.exp(-0.5 * (float(error) / float(scale)) ** 2))


def _mask_histogram(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pixels = mask > 0
    if not np.any(pixels):
        return np.zeros(24, dtype=np.float64)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    histogram = cv2.calcHist(
        [lab], [0, 1], mask.astype(np.uint8), [8, 3], [0, 256, 0, 256]
    ).astype(np.float64).reshape(-1)
    total = float(histogram.sum())
    if total > EPSILON:
        histogram /= total
    return histogram


def _histogram_similarity(
    reference: np.ndarray | None,
    prediction: np.ndarray | None,
) -> float:
    if reference is None or prediction is None:
        return 0.0
    if reference.shape != prediction.shape:
        return 0.0
    return _bounded(float(np.minimum(reference, prediction).sum()))


def _candidate_mask(
    frame: np.ndarray,
    *,
    x: float,
    y: float,
    radius: float,
) -> tuple[np.ndarray, float]:
    """Segment one compact ball inside a circle proposal.

    The annular local background makes this polarity agnostic: a dark steel
    ball on the light wall and a bright ball on the black panel use the same
    rule.  A filled proposal circle is the deterministic fallback when glare
    splits the photometric component.
    """

    height, width = frame.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
    core = distance <= max(2.0, 0.9 * radius)
    support = distance <= max(3.0, 1.18 * radius)
    annulus = (distance >= 1.25 * radius) & (distance <= 1.9 * radius)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if not np.any(core) or not np.any(annulus):
        mask = support.astype(np.uint8) * 255
        return mask, 0.0
    background = float(np.median(gray[annulus]))
    foreground = float(np.mean(gray[core]))
    signed = foreground - background
    contrast = abs(signed)
    threshold = max(5.0, 0.22 * contrast)
    selected = (
        (gray >= background + threshold)
        if signed >= 0.0
        else (gray <= background - threshold)
    ) & support
    raw = selected.astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        raw
    )
    best_index: int | None = None
    best_cost = float("inf")
    minimum_area = max(4.0, 0.08 * math.pi * radius * radius)
    for index in range(1, component_count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        center = centroids[index]
        cost = float(np.linalg.norm(center - np.asarray([x, y])))
        if cost < best_cost:
            best_cost = cost
            best_index = index
    if best_index is None or best_cost > 1.1 * radius:
        mask = support.astype(np.uint8) * 255
    else:
        mask = (labels == best_index).astype(np.uint8) * 255
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8))
        mask[support == 0] = 0
    return mask, contrast


def _deduplicate_candidates(
    candidates: Iterable[BallCandidate],
) -> list[BallCandidate]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.contrast, item.radius),
        reverse=True,
    )
    retained: list[BallCandidate] = []
    for candidate in ordered:
        duplicate = any(
            float(np.linalg.norm(candidate.xy - other.xy))
            <= 1.5 * max(candidate.radius, other.radius)
            for other in retained
        )
        if not duplicate:
            retained.append(candidate)
    return retained


def _hough_candidates(
    frame: np.ndarray,
    *,
    config: dict[str, Any],
    initial: bool,
) -> list[BallCandidate]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = int(config.get("blur_kernel", 7))
    if kernel < 3:
        kernel = 3
    if kernel % 2 == 0:
        kernel += 1
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 1.5)
    thresholds = config.get(
        "initial_hough_thresholds" if initial else "hough_thresholds",
        [24, 20, 16, 13, 10] if initial else [20, 16, 13, 10],
    )
    height, width = gray.shape
    candidates: list[BallCandidate] = []
    for accumulator_threshold in thresholds:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=float(config.get("hough_dp", 1.2)),
            minDist=float(config.get("minimum_center_distance_px", 12.0)),
            param1=float(config.get("edge_threshold", 90.0)),
            param2=float(accumulator_threshold),
            minRadius=int(config.get("minimum_radius_px", 3)),
            maxRadius=int(config.get("maximum_radius_px", 24)),
        )
        if circles is None:
            continue
        for x, y, radius in np.asarray(circles[0], dtype=np.float64):
            if initial and (
                x < float(config.get("initial_minimum_x_ratio", 0.65)) * width
                or y
                > float(config.get("initial_maximum_y_ratio", 0.45)) * height
            ):
                continue
            mask, contrast = _candidate_mask(
                frame, x=float(x), y=float(y), radius=float(radius)
            )
            if contrast < float(config.get("minimum_local_contrast", 8.0)):
                continue
            candidates.append(
                BallCandidate(
                    xy=np.asarray([x, y], dtype=np.float64),
                    radius=float(radius),
                    mask=mask,
                    histogram=_mask_histogram(frame, mask),
                    contrast=float(contrast),
                    source=f"hough_{accumulator_threshold:g}",
                )
            )
        if candidates:
            # The first successful threshold is the strongest stable proposal
            # set.  Lower thresholds are a fallback, not a way to manufacture
            # extra entities from texture.
            break
    return _deduplicate_candidates(candidates)


def _foreground_candidates(
    frame: np.ndarray,
    *,
    background: np.ndarray,
    seed: BallCandidate,
    config: dict[str, Any],
) -> list[BallCandidate]:
    difference = cv2.absdiff(frame, background)
    magnitude = np.max(difference, axis=2)
    threshold = float(config.get("foreground_threshold", 12.0))
    mask = (magnitude >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    nominal_area = math.pi * seed.radius * seed.radius
    candidates: list[BallCandidate] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index].tolist()
        area = float(area)
        if not (
            float(config.get("minimum_area_ratio", 0.12)) * nominal_area
            <= area
            <= float(config.get("maximum_area_ratio", 5.0)) * nominal_area
        ):
            continue
        if width < 2 or height < 2:
            continue
        aspect = max(width / height, height / width)
        if aspect > float(config.get("maximum_aspect_ratio", 3.5)):
            continue
        component = (labels == index).astype(np.uint8) * 255
        center = np.asarray(centroids[index], dtype=np.float64)
        radius = float(math.sqrt(area / math.pi))
        candidates.append(
            BallCandidate(
                xy=center,
                radius=radius,
                mask=component,
                histogram=_mask_histogram(frame, component),
                contrast=float(np.mean(magnitude[labels == index])),
                source="temporal_foreground",
            )
        )
    return _deduplicate_candidates(candidates)


def _candidate_is_seed_like(
    candidate: BallCandidate,
    seed: BallCandidate,
    *,
    config: dict[str, Any],
) -> bool:
    ratio = candidate.radius / max(seed.radius, EPSILON)
    if not (
        float(config.get("minimum_radius_ratio", 0.35))
        <= ratio
        <= float(config.get("maximum_radius_ratio", 2.5))
    ):
        return False
    appearance = _histogram_similarity(seed.histogram, candidate.histogram)
    return appearance >= float(config.get("minimum_histogram_similarity", 0.05))


def _empty_observation(
    frame_shape: tuple[int, int],
    frame_count: int,
    *,
    code: str,
    reason: str,
) -> ParabolicObservation:
    height, width = frame_shape
    return ParabolicObservation(
        xy=np.full((frame_count, 2), np.nan, dtype=np.float64),
        radii=np.full(frame_count, np.nan, dtype=np.float64),
        observed=np.zeros(frame_count, dtype=bool),
        interpolated=np.zeros(frame_count, dtype=bool),
        masks=tuple(
            np.zeros((height, width), dtype=np.uint8)
            for _ in range(frame_count)
        ),
        histograms=tuple(None for _ in range(frame_count)),
        cardinality=np.zeros(frame_count, dtype=np.int64),
        seed=None,
        diagnostics={"status": "failed_closed", "code": code, "reason": reason},
    )


def observe_projectile(
    frames: list[np.ndarray],
    *,
    available: np.ndarray | None,
    config: dict[str, Any],
) -> ParabolicObservation:
    """Discover and track the single manifest-declared projectile ball.

    Discovery is independent for reference and prediction.  It does not copy
    the GT mask into the prediction, and it keeps a null state when no
    plausible ball is present.
    """

    if not frames:
        raise ValueError("projectile observation requires at least one frame")
    height, width = frames[0].shape[:2]
    frame_count = len(frames)
    availability = (
        np.ones(frame_count, dtype=bool)
        if available is None
        else np.asarray(available, dtype=bool)
    )
    if availability.shape != (frame_count,):
        raise ValueError("availability must have one value per frame")
    seed_candidates = (
        _hough_candidates(frames[0], config=config, initial=True)
        if availability[0]
        else []
    )
    if not seed_candidates:
        return _empty_observation(
            (height, width),
            frame_count,
            code="projectile_seed_not_found",
            reason="no compact ball candidate was found in prediction frame zero",
        )
    seed = max(
        seed_candidates,
        key=lambda item: (
            item.contrast
            + 12.0 * item.xy[0] / max(width, 1)
            - 2.0 * item.xy[1] / max(height, 1)
        ),
    )
    background = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    xy = np.full((frame_count, 2), np.nan, dtype=np.float64)
    radii = np.full(frame_count, np.nan, dtype=np.float64)
    observed = np.zeros(frame_count, dtype=bool)
    interpolated = np.zeros(frame_count, dtype=bool)
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in frames]
    histograms: list[np.ndarray | None] = [None for _ in frames]
    cardinality = np.zeros(frame_count, dtype=np.int64)
    sources: list[str | None] = [None for _ in frames]
    candidate_counts: list[int] = [0 for _ in frames]
    last_xy: np.ndarray | None = None
    velocity = np.zeros(2, dtype=np.float64)
    maximum_jump = max(
        float(config.get("maximum_jump_diagonal_ratio", 0.2))
        * math.hypot(width, height),
        float(config.get("maximum_jump_radius_ratio", 18.0)) * seed.radius,
    )
    for frame_index, frame in enumerate(frames):
        if not availability[frame_index]:
            continue
        if frame_index == 0:
            hough_candidates = seed_candidates
            candidates = seed_candidates
        else:
            hough_candidates = _hough_candidates(
                frame, config=config, initial=False
            )
            foreground = _foreground_candidates(
                frame, background=background, seed=seed, config=config
            )
            candidates = _deduplicate_candidates(
                [*hough_candidates, *foreground]
            )
        plausible_hough = [
            candidate
            for candidate in hough_candidates
            if _candidate_is_seed_like(candidate, seed, config=config)
        ]
        plausible = [
            candidate
            for candidate in candidates
            if _candidate_is_seed_like(candidate, seed, config=config)
        ]
        candidate_counts[frame_index] = len(plausible)
        # Foreground components are a recall fallback for the bound identity,
        # not sufficient evidence for extra entities on their own.  Residual
        # cardinality requires the stronger compact-circle observation.
        cardinality[frame_index] = len(plausible_hough)
        if frame_index == 0:
            selected = seed
        elif not plausible:
            selected = None
        else:
            predicted_xy = (
                last_xy + velocity if last_xy is not None else seed.xy
            )

            def assignment_cost(candidate: BallCandidate) -> float:
                distance = float(np.linalg.norm(candidate.xy - predicted_xy))
                scale = max(seed.radius * 5.0, 8.0)
                radius_cost = abs(
                    math.log(candidate.radius / max(seed.radius, EPSILON))
                )
                appearance_cost = 1.0 - _histogram_similarity(
                    seed.histogram, candidate.histogram
                )
                return distance / scale + 0.45 * radius_cost + 0.35 * appearance_cost

            selected = min(plausible, key=assignment_cost)
            if last_xy is not None:
                distance = float(
                    np.linalg.norm(selected.xy - (last_xy + velocity))
                )
                if distance > maximum_jump:
                    selected = None
        if selected is None:
            velocity *= 0.75
            continue
        cardinality[frame_index] = max(1, cardinality[frame_index])
        xy[frame_index] = selected.xy
        radii[frame_index] = selected.radius
        observed[frame_index] = True
        masks[frame_index] = selected.mask
        histograms[frame_index] = selected.histogram
        sources[frame_index] = selected.source
        if last_xy is not None:
            instantaneous = selected.xy - last_xy
            velocity = 0.45 * velocity + 0.55 * instantaneous
        last_xy = selected.xy.copy()

    maximum_gap = int(config.get("maximum_interpolation_gap_frames", 2))
    observed_indices = np.flatnonzero(observed)
    for left, right in zip(observed_indices[:-1], observed_indices[1:]):
        gap = int(right - left - 1)
        if gap <= 0 or gap > maximum_gap:
            continue
        for index in range(int(left) + 1, int(right)):
            alpha = (index - left) / (right - left)
            xy[index] = (1.0 - alpha) * xy[left] + alpha * xy[right]
            radii[index] = (1.0 - alpha) * radii[left] + alpha * radii[right]
            center = tuple(np.rint(xy[index]).astype(int).tolist())
            cv2.circle(
                masks[index], center, max(1, round(radii[index])), 255, -1
            )
            histograms[index] = _mask_histogram(frames[index], masks[index])
            observed[index] = True
            interpolated[index] = True
            sources[index] = "short_gap_interpolation"
            cardinality[index] = max(1, cardinality[index])
    return ParabolicObservation(
        xy=xy,
        radii=radii,
        observed=observed,
        interpolated=interpolated,
        masks=tuple(masks),
        histograms=tuple(histograms),
        cardinality=cardinality,
        seed=seed,
        diagnostics={
            "status": "observed",
            "observer": "manifest_guided_compact_ball_v1",
            "seed_source": seed.source,
            "seed_xy": seed.xy.tolist(),
            "seed_radius_px": seed.radius,
            "observed_frames": int(np.count_nonzero(observed)),
            "interpolated_frames": int(np.count_nonzero(interpolated)),
            "candidate_counts": candidate_counts,
            "selected_sources": sources,
        },
    )


def _binding_score(
    reference: ParabolicObservation,
    prediction: ParabolicObservation,
    *,
    frame_shape: tuple[int, int],
    config: dict[str, Any],
) -> dict[str, Any]:
    if reference.seed is None or prediction.seed is None:
        return {
            "score": 0.0,
            "accepted": False,
            "position": 0.0,
            "scale": 0.0,
            "appearance": 0.0,
        }
    diagonal = math.hypot(*frame_shape)
    position = _similarity(
        float(np.linalg.norm(reference.seed.xy - prediction.seed.xy)) / diagonal,
        float(config.get("binding_position_scale", 0.08)),
    )
    scale = _similarity(
        abs(math.log(prediction.seed.radius / reference.seed.radius)),
        float(config.get("binding_log_radius_scale", 0.35)),
    )
    appearance = _histogram_similarity(
        reference.seed.histogram, prediction.seed.histogram
    )
    weights = config.get(
        "binding_weights", {"position": 0.5, "scale": 0.2, "appearance": 0.3}
    )
    denominator = sum(float(value) for value in weights.values())
    score = (
        float(weights["position"]) * position
        + float(weights["scale"]) * scale
        + float(weights["appearance"]) * appearance
    ) / max(denominator, EPSILON)
    threshold = float(config.get("minimum_binding_score", 0.15))
    return {
        "score": _bounded(score),
        "accepted": bool(score >= threshold),
        "minimum_score": threshold,
        "position": position,
        "scale": scale,
        "appearance": appearance,
    }


def _fit_polynomial(
    independent: np.ndarray,
    dependent: np.ndarray,
    *,
    degree: int,
) -> tuple[np.ndarray | None, float | None]:
    independent = np.asarray(independent, dtype=np.float64)
    dependent = np.asarray(dependent, dtype=np.float64)
    valid = np.isfinite(independent) & np.isfinite(dependent)
    if int(np.count_nonzero(valid)) < degree + 1:
        return None, None
    x = independent[valid]
    y = dependent[valid]
    if float(np.ptp(x)) <= EPSILON:
        return None, None
    try:
        coefficients = np.polyfit(x, y, degree)
        fitted = np.polyval(coefficients, x)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, None
    scale = max(float(np.ptp(y)), EPSILON)
    rmse = float(np.sqrt(np.mean((y - fitted) ** 2)) / scale)
    return coefficients, rmse


def _model_curve_similarity(
    reference_coefficients: np.ndarray | None,
    prediction_coefficients: np.ndarray | None,
    *,
    domain: np.ndarray,
    scale: float,
) -> float:
    if reference_coefficients is None or prediction_coefficients is None:
        return 0.0
    reference_curve = np.polyval(reference_coefficients, domain)
    prediction_curve = np.polyval(prediction_coefficients, domain)
    error = float(np.sqrt(np.mean((reference_curve - prediction_curve) ** 2)))
    return _similarity(error, scale)


def _mask_iou(reference: np.ndarray, prediction: np.ndarray) -> float:
    first = reference > 0
    second = prediction > 0
    union = np.logical_or(first, second)
    if not np.any(union):
        return 1.0
    return float(np.count_nonzero(first & second) / np.count_nonzero(union))


def score_parabolic_observations(
    reference: ParabolicObservation,
    prediction: ParabolicObservation,
    *,
    times_s: list[float],
    frame_shape: tuple[int, int],
    available: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    frame_count = len(times_s)
    if reference.xy.shape != (frame_count, 2) or prediction.xy.shape != (
        frame_count,
        2,
    ):
        raise ValueError("parabolic observations differ from the common timeline")
    time_grid = build_common_time_grid(times_s)
    weights = np.asarray(time_grid.cell_weights_s, dtype=np.float64)
    if float(weights.sum()) <= EPSILON:
        weights = np.ones(frame_count, dtype=np.float64)
    binding = _binding_score(
        reference, prediction, frame_shape=frame_shape, config=config
    )
    expected = np.asarray(reference.observed, dtype=bool)
    matched = (
        expected
        & np.asarray(prediction.observed, dtype=bool)
        & np.asarray(available, dtype=bool)
        & bool(binding["accepted"])
    )
    diagonal = math.hypot(*frame_shape)
    per_frame_position = np.zeros(frame_count, dtype=np.float64)
    if np.any(matched):
        distances = np.linalg.norm(
            reference.xy[matched] - prediction.xy[matched], axis=1
        ) / diagonal
        per_frame_position[matched] = np.exp(
            -0.5
            * (
                distances / float(config.get("trajectory_distance_scale", 0.06))
            )
            ** 2
        )
    expected_weights = weights * expected.astype(np.float64)
    timed_trajectory = _weighted_mean(per_frame_position, expected_weights)
    exposure = _weighted_mean(matched.astype(np.float64), expected_weights)
    reference_cardinality = expected.astype(np.int64)
    prediction_cardinality = (
        reference_cardinality.copy()
        if prediction is reference
        else prediction.cardinality.astype(np.int64)
    )
    cardinality_similarity = np.exp(
        -np.abs(
            prediction_cardinality.astype(np.float64)
            - reference_cardinality.astype(np.float64)
        )
    )
    integrity = _weighted_mean(
        cardinality_similarity,
        weights * np.asarray(available, dtype=np.float64),
    )
    lifecycle = _bounded(exposure * integrity)

    t = np.asarray(times_s, dtype=np.float64)
    duration = max(float(t[-1] - t[0]), EPSILON)
    normalized_t = (t - t[0]) / duration
    reference_xy = reference.xy / diagonal
    prediction_xy = prediction.xy / diagonal
    reference_valid = expected & np.isfinite(reference_xy).all(axis=1)
    prediction_valid = matched & np.isfinite(prediction_xy).all(axis=1)
    ref_x_line, ref_x_residual = _fit_polynomial(
        normalized_t[reference_valid], reference_xy[reference_valid, 0], degree=1
    )
    pred_x_line, pred_x_residual = _fit_polynomial(
        normalized_t[prediction_valid],
        prediction_xy[prediction_valid, 0],
        degree=1,
    )
    horizontal_curve = _model_curve_similarity(
        ref_x_line,
        pred_x_line,
        domain=np.linspace(0.0, 1.0, 32),
        scale=float(config.get("horizontal_model_scale", 0.04)),
    )
    horizontal_residual = (
        _similarity(
            abs(float(pred_x_residual) - float(ref_x_residual)),
            float(config.get("horizontal_residual_scale", 0.03)),
        )
        if ref_x_residual is not None and pred_x_residual is not None
        else 0.0
    )
    horizontal_uniform = _bounded(0.75 * horizontal_curve + 0.25 * horizontal_residual)

    ref_y_quad, ref_y_residual = _fit_polynomial(
        normalized_t[reference_valid], reference_xy[reference_valid, 1], degree=2
    )
    pred_y_quad, pred_y_residual = _fit_polynomial(
        normalized_t[prediction_valid],
        prediction_xy[prediction_valid, 1],
        degree=2,
    )
    vertical_curve = _model_curve_similarity(
        ref_y_quad,
        pred_y_quad,
        domain=np.linspace(0.0, 1.0, 32),
        scale=float(config.get("vertical_model_scale", 0.05)),
    )
    vertical_residual = (
        _similarity(
            abs(float(pred_y_residual) - float(ref_y_residual)),
            float(config.get("vertical_residual_scale", 0.03)),
        )
        if ref_y_residual is not None and pred_y_residual is not None
        else 0.0
    )
    vertical_acceleration = _bounded(0.8 * vertical_curve + 0.2 * vertical_residual)

    reference_relative = reference_xy - reference_xy[0]
    prediction_relative = prediction_xy - prediction_xy[0]
    ref_parabola, ref_parabola_residual = _fit_polynomial(
        reference_relative[reference_valid, 0],
        reference_relative[reference_valid, 1],
        degree=2,
    )
    pred_parabola, pred_parabola_residual = _fit_polynomial(
        prediction_relative[prediction_valid, 0],
        prediction_relative[prediction_valid, 1],
        degree=2,
    )
    if np.any(reference_valid):
        x_values = reference_relative[reference_valid, 0]
        domain = np.linspace(float(np.min(x_values)), float(np.max(x_values)), 32)
    else:
        domain = np.linspace(-0.5, 0.0, 32)
    parabolic_curve = _model_curve_similarity(
        ref_parabola,
        pred_parabola,
        domain=domain,
        scale=float(config.get("parabolic_model_scale", 0.05)),
    )
    parabolic_residual = (
        _similarity(
            abs(float(pred_parabola_residual) - float(ref_parabola_residual)),
            float(config.get("parabolic_residual_scale", 0.04)),
        )
        if ref_parabola_residual is not None
        and pred_parabola_residual is not None
        else 0.0
    )
    parabolic_geometry = _bounded(0.8 * parabolic_curve + 0.2 * parabolic_residual)

    state_components = {
        "time_parameterized_trajectory": timed_trajectory,
        "horizontal_uniform_motion": horizontal_uniform,
        "vertical_uniform_acceleration": vertical_acceleration,
        "parabolic_geometry": parabolic_geometry,
        "lifecycle_and_cardinality": lifecycle,
    }
    state_weights = config.get(
        "weights",
        {
            "time_parameterized_trajectory": 0.45,
            "horizontal_uniform_motion": 0.15,
            "vertical_uniform_acceleration": 0.15,
            "parabolic_geometry": 0.15,
            "lifecycle_and_cardinality": 0.10,
        },
    )
    weight_sum = sum(float(value) for value in state_weights.values())
    physics_score = sum(
        float(state_weights[name]) * float(value)
        for name, value in state_components.items()
    ) / max(weight_sum, EPSILON)

    shape_values = np.zeros(frame_count, dtype=np.float64)
    appearance_values = np.zeros(frame_count, dtype=np.float64)
    for index in np.flatnonzero(matched):
        shape_values[index] = _similarity(
            abs(
                math.log(
                    prediction.radii[index] / reference.radii[index]
                )
            ),
            float(config.get("subject_log_radius_scale", 0.35)),
        )
        appearance_values[index] = _histogram_similarity(
            reference.histograms[index], prediction.histograms[index]
        )
    shape_score = _weighted_mean(shape_values, expected_weights)
    appearance_score = _weighted_mean(appearance_values, expected_weights)
    subject_weights = config.get(
        "subject_weights", {"binding": 0.4, "shape": 0.3, "appearance": 0.3}
    )
    subject_weight_sum = sum(float(value) for value in subject_weights.values())
    subject_score = (
        float(subject_weights["binding"]) * float(binding["score"])
        + float(subject_weights["shape"]) * shape_score
        + float(subject_weights["appearance"]) * appearance_score
    ) / max(subject_weight_sum, EPSILON)
    case_weights = config.get(
        "case_weights", {"physics_state": 0.75, "subject": 0.25}
    )
    case_weight_sum = sum(float(value) for value in case_weights.values())
    score = (
        float(case_weights["physics_state"]) * physics_score
        + float(case_weights["subject"]) * subject_score
    ) / max(case_weight_sum, EPSILON)
    score *= 0.5 + 0.5 * integrity
    ious = [
        _mask_iou(reference.masks[index], prediction.masks[index])
        if expected[index]
        else None
        for index in range(frame_count)
    ]
    return {
        "score": _bounded(score),
        "physics_score": _bounded(physics_score),
        "subject_score": _bounded(subject_score),
        "integrity_score": _bounded(integrity),
        "state_components": state_components,
        "state_weights": state_weights,
        "binding": binding,
        "shape_score": shape_score,
        "appearance_score": appearance_score,
        "expected": expected,
        "matched": matched,
        "position_scores": per_frame_position,
        "ious": ious,
        "reference_cardinality": reference_cardinality,
        "prediction_cardinality": prediction_cardinality,
        "diagnostics": {
            "reference_horizontal_linearity": (
                None if ref_x_residual is None else _similarity(ref_x_residual, 0.03)
            ),
            "prediction_horizontal_linearity": (
                None
                if pred_x_residual is None
                else _similarity(pred_x_residual, 0.03)
            ),
            "reference_vertical_quadratic_fit": (
                None if ref_y_residual is None else _similarity(ref_y_residual, 0.03)
            ),
            "prediction_vertical_quadratic_fit": (
                None
                if pred_y_residual is None
                else _similarity(pred_y_residual, 0.03)
            ),
            "reference_parabolic_fit": (
                None
                if ref_parabola_residual is None
                else _similarity(ref_parabola_residual, 0.04)
            ),
            "prediction_parabolic_fit": (
                None
                if pred_parabola_residual is None
                else _similarity(pred_parabola_residual, 0.04)
            ),
            "matched_exposure": exposure,
        },
    }


class ParabolicMotionCaseEvaluator(ReferenceCaseEvaluator):
    evaluator_id = "parabolic_motion_state"
    evaluator_version = "1.0"
    sequential_evaluator_version = "1.0"
    robust_evaluator_version = "1.0"
    scene_id = "parabolic_motion"
    primary_score = "parabolic_motion_state_similarity"
    allow_partial_prediction = True

    def __init__(self, config: dict[str, Any]):
        if config.get("observer_protocol") != "open_world_v2":
            raise ValueError(
                "parabolic evaluator requires observer_protocol=open_world_v2"
            )
        super().__init__(config)

    def describe_observation(self) -> dict[str, Any]:
        return {
            "protocol": "open_world_v2",
            "reference_discovery": "manifest_guided_compact_ball_v1",
            "prediction_discovery": "independent_compact_ball_with_null_assignment",
            "segmentation": "local_annular_contrast_with_circle_fallback",
            "tracking": "appearance_scale_continuity_with_short_gap_interpolation",
            "lifecycle": "reference_frozen_may_exit",
            "distance": "scene_specific_empirical_projectile_trajectory_v1",
        }

    def analyze(
        self,
        request,
        *,
        times_s: list[float],
        reference_video: SampledVideo,
        prediction_video: SampledVideo,
    ) -> SceneAnalysis:
        try:
            manifest = materialize_entity_manifest(request.case)
        except Exception as exc:
            raise ReferenceAnalysisError(
                "invalid_parabolic_entity_manifest",
                f"parabolic entity manifest failed validation: {exc}",
            ) from exc
        if manifest.scene_id != self.scene_id or len(manifest.entities) != 1:
            raise ReferenceAnalysisError(
                "invalid_parabolic_entity_manifest",
                "parabolic_motion requires exactly one manifest entity",
            )
        entity = manifest.entities[0]
        if entity.entity_class != "ball":
            raise ReferenceAnalysisError(
                "invalid_parabolic_entity_class",
                f"projectile entity must be a ball, got {entity.entity_class!r}",
            )
        observation_config = dict(self.config.get("open_world_observation", {}))
        try:
            reference = observe_projectile(
                list(reference_video.frames),
                available=np.ones(len(times_s), dtype=bool),
                config=observation_config,
            )
        except Exception as exc:
            raise ReferenceAnalysisError(
                "reference_projectile_observation_failed",
                f"reference projectile observation failed: {type(exc).__name__}: {exc}",
            ) from exc
        minimum_reference_frames = int(
            self.config.get("quality", {}).get(
                "minimum_reference_observed_frames", 3
            )
        )
        if reference.seed is None or reference.observed_count < minimum_reference_frames:
            raise ReferenceAnalysisError(
                "insufficient_reference_projectile_observation",
                "reference projectile observation has "
                f"{reference.observed_count} frames; requires "
                f"{minimum_reference_frames}",
            )
        prediction_frames = list(prediction_video.frames)
        available = np.asarray(
            prediction_video.available
            if prediction_video.available is not None
            else [True] * len(times_s),
            dtype=bool,
        )
        prediction_failures: list[dict[str, str]] = []
        identical_samples = bool(
            np.all(available)
            and len(reference_video.frames) == len(prediction_frames)
            and all(
                np.array_equal(first, second)
                for first, second in zip(reference_video.frames, prediction_frames)
            )
        )
        if identical_samples:
            prediction = reference
            prediction_observation_policy = "exact_sample_reuse"
        else:
            prediction_observation_policy = "independent_observation"
            try:
                prediction = observe_projectile(
                    prediction_frames,
                    available=available,
                    config=observation_config,
                )
            except Exception as exc:
                failure = {
                    "code": "prediction_projectile_observation_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
                prediction_failures.append(failure)
                prediction = _empty_observation(
                    reference_video.frames[0].shape[:2],
                    len(times_s),
                    code=failure["code"],
                    reason=failure["reason"],
                )
        if not bool(np.all(available)):
            prediction_failures.append(
                {
                    "code": "prediction_partial_timeline",
                    "reason": (
                        f"prediction supplies {int(available.sum())}/"
                        f"{len(available)} common-time samples"
                    ),
                }
            )
        scoring_config = dict(self.config.get("scoring", {}))
        scored = score_parabolic_observations(
            reference,
            prediction,
            times_s=times_s,
            frame_shape=reference_video.frames[0].shape[:2],
            available=available,
            config=scoring_config,
        )
        if prediction.seed is None:
            prediction_failures.append(
                {
                    "code": "prediction_projectile_entity_missing",
                    "reason": "prediction frame zero contains no bindable projectile ball",
                }
            )
        elif not bool(scored["binding"]["accepted"]):
            prediction_failures.append(
                {
                    "code": "prediction_projectile_entity_unmatched",
                    "reason": "prediction projectile failed the frame-zero binding gate",
                }
            )
        artifact_failures: list[dict[str, str]] = []
        artifacts: dict[str, str] = {}
        rows: list[dict[str, Any]] = []
        for index, time_s in enumerate(times_s):
            rows.append(
                {
                    "frame_index": index,
                    "time_s": time_s,
                    "prediction_available": bool(available[index]),
                    "reference_expected": bool(scored["expected"][index]),
                    "reference_observed": bool(reference.observed[index]),
                    "prediction_observed": bool(prediction.observed[index]),
                    "matched": bool(scored["matched"][index]),
                    "reference_x": (
                        None if not reference.observed[index] else float(reference.xy[index, 0])
                    ),
                    "reference_y": (
                        None if not reference.observed[index] else float(reference.xy[index, 1])
                    ),
                    "prediction_x": (
                        None if not prediction.observed[index] else float(prediction.xy[index, 0])
                    ),
                    "prediction_y": (
                        None if not prediction.observed[index] else float(prediction.xy[index, 1])
                    ),
                    "position_similarity": float(scored["position_scores"][index]),
                    "reference_cardinality": int(
                        scored["reference_cardinality"][index]
                    ),
                    "reference_candidate_cardinality": int(
                        reference.cardinality[index]
                    ),
                    "prediction_cardinality": int(
                        scored["prediction_cardinality"][index]
                    ),
                    "prediction_candidate_cardinality": int(
                        prediction.cardinality[index]
                    ),
                    "mask_iou": scored["ious"][index],
                }
            )
        try:
            csv_path = request.artifact_dir / "per_frame.csv"
            audit_path = request.artifact_dir / "open_world_audit.json"
            position_path = request.artifact_dir / "entity_position_curve.png"
            iou_path = request.artifact_dir / "physical_subject_iou_curve.png"
            write_rows_csv(csv_path, rows)
            write_json(
                audit_path,
                {
                    "schema_version": "1.0",
                    "scene_id": self.scene_id,
                    "case_id": request.case["case_id"],
                    "entity_manifest": manifest.to_canonical_dict(),
                    "reference_observation": reference.diagnostics,
                    "prediction_observation": prediction.diagnostics,
                    "prediction_observation_policy": prediction_observation_policy,
                    "binding": scored["binding"],
                    "state_components": scored["state_components"],
                    "diagnostics": scored["diagnostics"],
                    "prediction_failures": prediction_failures,
                },
            )
            save_iou_curve(
                position_path,
                times_s=times_s,
                ious=[float(value) for value in scored["position_scores"]],
                case_id=request.case["case_id"],
                scene_name="parabolic motion entity position",
                series_label="Matched projectile position similarity",
                y_label="Position similarity",
                metric_name="score",
            )
            save_iou_curve(
                iou_path,
                times_s=times_s,
                ious=scored["ious"],
                case_id=request.case["case_id"],
                scene_name="parabolic motion",
            )
            trajectory_path = request.artifact_dir / "projectile_trajectory_xy.png"
            save_series_comparison(
                trajectory_path,
                times_s=times_s,
                reference=reference.xy[:, 1],
                prediction=prediction.xy[:, 1],
                ylabel="Image y coordinate (px)",
                title=f"Parabolic vertical trajectory — {request.case['case_id']}",
            )
            artifacts = {
                "per_frame_csv": str(csv_path),
                "open_world_audit": str(audit_path),
                "entity_position_curve": str(position_path),
                "physical_subject_iou_curve": str(iou_path),
                "projectile_trajectory_curve": str(trajectory_path),
            }
        except Exception as exc:
            artifact_failures.append(
                {
                    "code": "parabolic_artifact_write_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        primary = {
            "score": float(scored["score"]),
            "components": {
                "physics_state": float(scored["physics_score"]),
                "subject": float(scored["subject_score"]),
                "object_centric_integrity": float(scored["integrity_score"]),
            },
            "case_weights": scoring_config.get(
                "case_weights", {"physics_state": 0.75, "subject": 0.25}
            ),
            "integrity_composition": "multiply_by_0.5_plus_0.5_integrity",
        }
        try:
            reference_mode = infer_reference_mode(request.case)
            artifacts.update(
                write_parabolic_visualization(
                    request,
                    config=self.config.get("visualization", {}),
                    times_s=times_s,
                    reference_frames=reference_video.frames,
                    prediction_frames=prediction_frames,
                    entity=entity,
                    capability=manifest.reference_capability,
                    reference=reference,
                    prediction=prediction,
                    scored=scored,
                    prediction_available=available.tolist(),
                    reference_role=(
                        "PHYSICS REFERENCE"
                        if reference_mode == "parent_physics_reference"
                        else "REFERENCE"
                    ),
                    score_summary=primary,
                    has_issues=bool(prediction_failures),
                )
            )
        except Exception as exc:
            artifact_failures.append(
                {
                    "code": "parabolic_visualization_write_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        finite_ious = [float(value) for value in scored["ious"] if value is not None]
        csti_input = (
            build_csti_input_from_aligned_masks(
                reference_capability=manifest.reference_capability,
                times_s=times_s,
                frame_shape=reference_video.frames[0].shape[:2],
                expected_entities=((entity.entity_id, entity.role_id),),
                reference_masks_by_entity={
                    entity.entity_id: tuple(reference.masks)
                },
                prediction_masks_by_entity={
                    entity.entity_id: (
                        tuple(prediction.masks)
                        if bool(scored["binding"]["accepted"])
                        else None
                    )
                },
                matched_track_ids_by_entity={
                    entity.entity_id: (
                        ("bound_projectile",)
                        if bool(scored["binding"]["accepted"])
                        else ()
                    )
                },
            )
            if (
                self.csti_enabled
                and manifest.reference_capability
                is ReferenceCapability.SAME_CASE_GT
            )
            else None
        )
        return SceneAnalysis(
            score=float(scored["score"]),
            metrics={
                "scene_subject_state_similarity": primary,
                "parabolic_motion_state_similarity": {
                    "score": float(scored["physics_score"]),
                    "components": scored["state_components"],
                    "weights": scored["state_weights"],
                    "reference_relative_diagnostics": scored["diagnostics"],
                },
                "physical_subject_similarity": {
                    "score": float(scored["subject_score"]),
                    "components": {
                        "frame_zero_binding": scored["binding"],
                        "shape": float(scored["shape_score"]),
                        "appearance": float(scored["appearance_score"]),
                    },
                },
                "physical_subject_mask_iou": {
                    "mean": float(np.mean(finite_ious)) if finite_ious else None,
                    "minimum": min(finite_ious) if finite_ious else None,
                    "maximum": max(finite_ious) if finite_ious else None,
                    "role": "segmentation_diagnostic_not_primary_score",
                },
                "object_centric_integrity": {
                    "score": float(scored["integrity_score"]),
                    "reference_cardinality": scored[
                        "reference_cardinality"
                    ].tolist(),
                    "reference_candidate_cardinality": (
                        reference.cardinality.tolist()
                    ),
                    "prediction_cardinality": scored[
                        "prediction_cardinality"
                    ].tolist(),
                    "prediction_candidate_cardinality": (
                        prediction.cardinality.tolist()
                    ),
                    "identity_switches": 0,
                    "new_or_duplicate_entity_cells": int(
                        np.count_nonzero(
                            scored["prediction_cardinality"]
                            > scored["reference_cardinality"]
                        )
                    ),
                },
                "entity_manifest": manifest.to_canonical_dict(),
            },
            quality={
                "degraded": bool(prediction_failures),
                "degradation_codes": [
                    item["code"] for item in prediction_failures
                ],
                "degradation_reason": (
                    "; ".join(item["reason"] for item in prediction_failures)
                    if prediction_failures
                    else None
                ),
                "artifact_failures": artifact_failures,
                "reference_observed_frames": reference.observed_count,
                "prediction_observed_frames": prediction.observed_count,
                "reference_valid_mask_ratio": float(np.mean(reference.observed)),
                "prediction_matched_exposure_ratio": float(
                    scored["diagnostics"]["matched_exposure"]
                ),
                "expected_entity_count": 1,
                "prediction_entity_bound": bool(scored["binding"]["accepted"]),
                "available_prediction_frames": int(np.count_nonzero(available)),
                "expected_prediction_frames": len(times_s),
            },
            artifacts=artifacts,
            provenance={
                "entity_manifest": {
                    "materializer_id": manifest.materializer_id,
                    "digest": manifest.digest,
                },
                "observation": {
                    "reference": reference.diagnostics,
                    "prediction": prediction.diagnostics,
                    "prediction_policy": prediction_observation_policy,
                },
                "identity_policy": "frame_zero_binding_then_continuity_with_null_state",
                "time_alignment": "common_reference_bounded_physical_time_grid_no_dtw",
                "reference_lifecycle": "manifest_may_exit_observed_timeline_frozen",
                "ideal_law_metrics_role": "diagnostic_only_empirical_gt_is_scoring_reference",
            },
            csti_input=csti_input,
        )


__all__ = [
    "ParabolicMotionCaseEvaluator",
    "ParabolicObservation",
    "observe_projectile",
    "score_parabolic_observations",
]
