#!/usr/bin/env python3
"""Track pendulum/spring balls with temporally constrained circle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from physbench.reference_observations import (
    EntityObservation,
    load_reference_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import load_curation_cases


Circle = tuple[float, float, float]


def pendulum_anchor_is_support(mask: np.ndarray) -> bool:
    """Identify a vertically elongated fixture mask mistaken for the round bob."""
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        return False
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    return height / max(1, width) >= 1.5


def select_pendulum_bob_candidate(
    candidates: Iterable[Circle],
    *,
    expected_radius: float,
    frame_height: int,
) -> Circle | None:
    """Select the hanging bob below static circular fixture details."""
    minimum_y = 0.40 * frame_height
    accepted = [candidate for candidate in candidates if candidate[1] >= minimum_y]
    if not accepted:
        return None
    return min(
        accepted,
        key=lambda candidate: (
            -float(candidate[1]),
            abs(float(candidate[2]) - expected_radius),
        ),
    )


def bootstrap_detection_state(detected: np.ndarray) -> np.ndarray:
    """Treat undetected bootstrap samples as unresolved, not visible fixtures."""
    values = np.asarray(detected, dtype=bool)
    if values.ndim != 1:
        raise ValueError("bootstrap detections must be one-dimensional")
    return np.where(values, 0, 3).astype(np.uint8)


def select_circle_candidate(
    candidates: Iterable[Circle],
    *,
    predicted_xy: np.ndarray,
    guide_xy: np.ndarray,
    expected_radius: float,
    maximum_distance: float,
) -> Circle | None:
    accepted = []
    for candidate in candidates:
        center = np.asarray(candidate[:2], np.float64)
        prediction_error = float(np.linalg.norm(center - predicted_xy))
        if prediction_error > maximum_distance:
            continue
        guide_error = (
            float(np.linalg.norm(center - guide_xy))
            if np.all(np.isfinite(guide_xy))
            else 0.0
        )
        radius_error = abs(float(candidate[2]) - expected_radius)
        score = prediction_error + 0.20 * guide_error + 1.5 * radius_error
        accepted.append((score, candidate))
    return min(accepted, default=(0.0, None), key=lambda item: item[0])[1]


def select_spring_axis_circle_candidate(
    candidates: Iterable[Circle],
    *,
    guide_xy: np.ndarray,
    expected_radius: float,
) -> Circle | None:
    """Select the round mass while rejecting ruler markings beside the spring.

    A spring mass can be far above or below a contaminated canonical centroid,
    but it remains close to the spring's vertical axis.  Weighting horizontal
    error much more strongly than vertical error captures that geometry.
    """
    accepted = []
    for candidate in candidates:
        center = np.asarray(candidate[:2], np.float64)
        horizontal_error = abs(float(center[0] - guide_xy[0]))
        vertical_error = abs(float(center[1] - guide_xy[1]))
        radius_error = abs(float(candidate[2]) - expected_radius)
        score = 4.0 * horizontal_error + vertical_error + 1.5 * radius_error
        accepted.append((score, candidate))
    return min(accepted, default=(0.0, None), key=lambda item: item[0])[1]


def resolve_spring_boundary_candidate(
    selected_xy: np.ndarray,
    *,
    guide_xy: np.ndarray,
    frame_height: int,
    expected_radius: float,
) -> tuple[np.ndarray, bool]:
    """Reject a spring hook mistaken for a partly out-of-frame ball."""
    selected = np.asarray(selected_xy, np.float64)
    guide = np.asarray(guide_xy, np.float64)
    near_lower_edge = guide[1] > frame_height - 2.0 * expected_radius
    jumped_to_hook = selected[1] < guide[1] - 0.75 * expected_radius
    if near_lower_edge and jumped_to_hook:
        return guide.copy(), False
    return selected.copy(), True


def _decode_video(path: Path, source_indices: Iterable[int]) -> list[np.ndarray]:
    requested = tuple(int(value) for value in source_indices)
    capture = cv2.VideoCapture(str(path))
    decoded: dict[int, np.ndarray] = {}
    wanted = set(requested)
    try:
        index = 0
        while index <= requested[-1]:
            okay, frame = capture.read()
            if not okay:
                break
            if index in wanted:
                decoded[index] = frame
            index += 1
    finally:
        capture.release()
    if len(decoded) != len(wanted):
        raise ValueError(f"video does not cover the requested timeline: {path}")
    return [decoded[index].copy() for index in requested]


def _circle_candidates(
    frame: np.ndarray,
    *,
    center_xy: np.ndarray,
    expected_radius: float,
) -> tuple[Circle, ...]:
    height, width = frame.shape[:2]
    reach = int(round(5.0 * expected_radius))
    center_x = int(round(np.clip(center_xy[0], 0, width - 1)))
    center_y = int(round(np.clip(center_xy[1], 0, height - 1)))
    left, right = max(0, center_x - reach), min(width, center_x + reach + 1)
    top, bottom = max(0, center_y - reach), min(height, center_y + reach + 1)
    crop = frame[top:bottom, left:right]
    gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 5)
    minimum_radius = max(3, int(round(0.55 * expected_radius)))
    maximum_radius = max(minimum_radius + 1, int(round(1.45 * expected_radius)))
    output: list[Circle] = []
    for threshold in (30.0, 24.0, 18.0, 14.0):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8.0, expected_radius),
            param1=100.0,
            param2=threshold,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )
        if circles is None:
            continue
        for raw_x, raw_y, raw_radius in circles[0]:
            candidate = (float(raw_x + left), float(raw_y + top), float(raw_radius))
            if any(
                np.linalg.norm(np.asarray(candidate[:2]) - np.asarray(old[:2]))
                <= 0.55 * expected_radius
                for old in output
            ):
                continue
            output.append(candidate)
        if output:
            break
    return tuple(output)


def _global_circle_candidates(
    frame: np.ndarray,
    *,
    expected_radius: float,
) -> tuple[Circle, ...]:
    gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 5)
    minimum_radius = max(3, int(round(0.55 * expected_radius)))
    maximum_radius = max(minimum_radius + 1, int(round(1.45 * expected_radius)))
    for threshold in (30.0, 24.0, 18.0, 14.0):
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8.0, expected_radius),
            param1=100.0,
            param2=threshold,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )
        if circles is not None:
            return tuple(
                (float(x), float(y), float(radius))
                for x, y, radius in circles[0]
            )
    return ()


def _anchor_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        mask = np.asarray(payload["masks"])
    if mask.ndim == 3:
        mask = mask[0]
    return mask > 0


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.asarray([np.nan, np.nan], np.float64)
    return np.asarray([xs.mean(), ys.mean()], np.float64)


def _track(
    frames: list[np.ndarray],
    *,
    initial_xy: np.ndarray,
    guide_xy: np.ndarray,
    expected_radius: float,
    spring_axis: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.full((len(frames), 2), np.nan, np.float64)
    guides = np.asarray(guide_xy, np.float64).copy()
    valid = np.all(np.isfinite(guides), axis=1)
    if not np.any(valid):
        raise ValueError("circle tracking requires at least one finite trajectory guide")
    sample_indices = np.arange(len(guides), dtype=np.float64)
    for axis in range(2):
        guides[:, axis] = np.interp(
            sample_indices, sample_indices[valid], guides[valid, axis]
        )
    detected = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        guide = guides[index]
        predicted = initial_xy if index == 0 else guide
        height, width = frame.shape[:2]
        predicted = np.asarray(
            [
                np.clip(predicted[0], 0, width - 1),
                np.clip(predicted[1], 0, height - 1),
            ],
            np.float64,
        )
        candidates = _circle_candidates(
            frame, center_xy=predicted, expected_radius=expected_radius
        )
        maximum_distance = max(3.0 * expected_radius, 12.0)
        if spring_axis:
            selected = select_spring_axis_circle_candidate(
                candidates,
                guide_xy=guide,
                expected_radius=expected_radius,
            )
        else:
            selected = select_circle_candidate(
                candidates,
                predicted_xy=predicted,
                guide_xy=guide,
                expected_radius=expected_radius,
                maximum_distance=maximum_distance,
            )
        if selected is not None:
            center = np.asarray(selected[:2], np.float64)
            accepted = True
            if spring_axis:
                center, accepted = resolve_spring_boundary_candidate(
                    center,
                    guide_xy=guide,
                    frame_height=height,
                    expected_radius=expected_radius,
                )
            detected[index] = accepted
        else:
            center = guide
        centers[index] = center
    return centers, detected, guides


def _track_bootstrapped_pendulum(
    frames: list[np.ndarray],
    *,
    expected_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_centers = np.full((len(frames), 2), np.nan, np.float64)
    detected = np.zeros(len(frames), dtype=bool)
    for index, frame in enumerate(frames):
        selected = select_pendulum_bob_candidate(
            _global_circle_candidates(frame, expected_radius=expected_radius),
            expected_radius=expected_radius,
            frame_height=frame.shape[0],
        )
        if selected is not None:
            raw_centers[index] = selected[:2]
            detected[index] = True
    valid = np.all(np.isfinite(raw_centers), axis=1)
    if not np.any(valid):
        raise ValueError("global pendulum bootstrap could not locate the bob")
    sample_indices = np.arange(len(frames), dtype=np.float64)
    centers = raw_centers.copy()
    for axis in range(2):
        centers[:, axis] = np.interp(
            sample_indices,
            sample_indices[valid],
            raw_centers[valid, axis],
        )
    return centers, detected, centers.copy()


def resolve_subject_lifecycle(
    masks: np.ndarray,
    *,
    canonical_state: np.ndarray,
    interpolated_guides: np.ndarray,
    detected: np.ndarray,
    expected_radius: float,
) -> np.ndarray:
    """Preserve known lifecycle and classify unresolved boundary excursions.

    Hough evidence near an image edge is especially vulnerable to rail and
    fixture circles.  When the canonical tube already loses the subject and
    its interpolated center lies within one subject radius of the boundary,
    the video evidence denotes a complete exit rather than a visible disk.
    """
    values = np.asarray(masks)
    states = np.asarray(canonical_state, dtype=np.uint8).copy()
    guides = np.asarray(interpolated_guides, dtype=np.float64)
    detections = np.asarray(detected, dtype=bool)
    if values.ndim != 3 or states.shape != (len(values),):
        raise ValueError("round-subject masks and lifecycle must use THW/T layouts")
    if guides.shape != (len(values), 2) or detections.shape != (len(values),):
        raise ValueError("round-subject guide evidence must align with masks")
    height, width = values.shape[1:]
    for index, state in enumerate(states):
        if state in (1, 2):
            values[index] = 0
            continue
        if state != 3:
            states[index] = 0
            continue
        x, y = guides[index]
        boundary_excursion = (
            x < expected_radius
            or x > (width - 1 - expected_radius)
            or y < expected_radius
            or y > (height - 1 - expected_radius)
        )
        if boundary_excursion:
            values[index] = 0
            states[index] = 2
        elif detections[index]:
            states[index] = 0
        else:
            values[index] = 0
    return states


def resolve_bootstrap_subject_lifecycle(
    masks: np.ndarray,
    *,
    interpolated_guides: np.ndarray,
    detected: np.ndarray,
    expected_radius: float,
) -> np.ndarray:
    """Keep smooth interior interpolation but suppress inferred boundary exits."""
    source_state = bootstrap_detection_state(detected)
    return resolve_subject_lifecycle(
        masks,
        canonical_state=source_state,
        interpolated_guides=interpolated_guides,
        detected=np.ones_like(np.asarray(detected, dtype=bool)),
        expected_radius=expected_radius,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    case_ids = tuple(arguments.case_list.read_text(encoding="utf-8").split())
    cases = load_curation_cases(arguments.dataset, case_ids=case_ids)
    for case in cases:
        if len(case.entities) != 1 or case.scene_id not in {
            "pendulum",
            "vertical_spring_oscillator",
        }:
            raise ValueError(f"unsupported round-subject Case: {case.case_id}")
        frames = _decode_video(
            case.reference_video_path,
            (sample["source_frame_index"] for sample in case.timeline["samples"]),
        )
        canonical = load_reference_observation(
            case.asset_root,
            case.observation_manifest_path,
            bundle_root=case.asset_root,
        ).entities["object_1"]
        anchor = _anchor_mask(case.entities[0].anchor_npz_path)
        initial = _centroid(anchor)
        if case.scene_id == "pendulum":
            radius = 0.040 * min(frames[0].shape[:2])
        else:
            radius = float(np.sqrt(np.count_nonzero(anchor) / np.pi))
        use_pendulum_bootstrap = (
            case.scene_id == "pendulum" and pendulum_anchor_is_support(anchor)
        )
        if use_pendulum_bootstrap:
            centers, detected, interpolated_guides = _track_bootstrapped_pendulum(
                frames,
                expected_radius=radius,
            )
        else:
            centers, detected, interpolated_guides = _track(
                frames,
                initial_xy=initial,
                guide_xy=np.asarray(canonical.centroid_xy, np.float64),
                expected_radius=radius,
                spring_axis=case.scene_id == "vertical_spring_oscillator",
            )
        masks = np.zeros((len(frames), *frames[0].shape[:2]), np.uint8)
        for index, center in enumerate(centers):
            cv2.circle(
                masks[index],
                tuple(int(round(value)) for value in center),
                max(2, int(round(radius))),
                1,
                -1,
            )
        if use_pendulum_bootstrap:
            state = resolve_bootstrap_subject_lifecycle(
                masks,
                interpolated_guides=interpolated_guides,
                detected=detected,
                expected_radius=radius,
            )
        else:
            state = resolve_subject_lifecycle(
                masks,
                canonical_state=np.asarray(canonical.state, np.uint8),
                interpolated_guides=interpolated_guides,
                detected=detected,
                expected_radius=radius,
            )
        area = masks.reshape(len(masks), -1).sum(axis=1, dtype=np.int64)
        exact_centers = np.full((len(masks), 2), np.nan, np.float32)
        bbox = np.full((len(masks), 4), np.nan, np.float32)
        for index, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            exact_centers[index] = (float(xs.mean()), float(ys.mean()))
            bbox[index] = (xs.min(), ys.min(), xs.max(), ys.max())
        output = arguments.output_root / case.scene_id / case.case_id
        write_entity_observation(
            output / "entities" / "object_1",
            EntityObservation(
                object_id="object_1",
                mask_id="01",
                masks=masks,
                centroid_xy=exact_centers,
                bbox_xyxy=bbox,
                area_pixels=area,
                state=state,
            ),
        )
        (output / "candidate_tracking.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "case_id": case.case_id,
                    "model_id": "temporal_hough_round_subject_v1",
                    "expected_radius_px": radius,
                    "pendulum_global_bootstrap": use_pendulum_bootstrap,
                    "hough_detected_samples": int(np.count_nonzero(detected)),
                    "sample_count": len(frames),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
