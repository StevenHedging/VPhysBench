#!/usr/bin/env python3
"""Repair reviewed collision tubes with ranked Hough-circle evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from physbench.reference_observations import (
    EntityObservation,
    load_entity_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import load_curation_cases


Circle = tuple[float, float, float]


def filter_circles_by_x(
    circles: Iterable[Circle], *, x_range: tuple[float, float]
) -> tuple[Circle, ...]:
    minimum, maximum = (float(value) for value in x_range)
    if maximum < minimum:
        raise ValueError("circle x range must be increasing")
    return tuple(
        sorted(
            (circle for circle in circles if minimum <= circle[0] <= maximum),
            key=lambda value: value[0],
        )
    )


def select_ranked_circle(
    circles: Iterable[Circle], *, rank_from_left: int
) -> Circle:
    ordered = sorted(circles, key=lambda value: float(value[0]))
    if rank_from_left < 0 or rank_from_left >= len(ordered):
        raise ValueError(
            f"circle rank {rank_from_left} is unavailable among {len(ordered)} detections"
        )
    return ordered[rank_from_left]


def select_temporal_circle(
    circles: Iterable[Circle],
    *,
    predicted_xy: np.ndarray,
    expected_radius: float,
    maximum_distance: float,
) -> Circle | None:
    prediction = np.asarray(predicted_xy, dtype=np.float64)
    accepted: list[tuple[float, Circle]] = []
    for circle in circles:
        center = np.asarray(circle[:2], dtype=np.float64)
        distance = float(np.linalg.norm(center - prediction))
        if distance > maximum_distance:
            continue
        score = distance + abs(float(circle[2]) - expected_radius)
        accepted.append((score, circle))
    return min(accepted, default=(0.0, None), key=lambda item: item[0])[1]


def clamp_velocity(velocity_xy: np.ndarray, maximum_speed: float) -> np.ndarray:
    velocity = np.asarray(velocity_xy, dtype=np.float64)
    speed = float(np.linalg.norm(velocity))
    if maximum_speed <= 0:
        raise ValueError("maximum temporal Hough speed must be positive")
    if speed <= maximum_speed or speed == 0:
        return velocity.copy()
    return velocity * (maximum_speed / speed)


def enforce_direction(
    selected_xy: np.ndarray,
    *,
    previous_xy: np.ndarray,
    predicted_xy: np.ndarray,
    direction: int,
    maximum_backward_jitter: float,
) -> np.ndarray:
    selected = np.asarray(selected_xy, dtype=np.float64).copy()
    previous = np.asarray(previous_xy, dtype=np.float64)
    predicted = np.asarray(predicted_xy, dtype=np.float64)
    backward = direction * float(selected[0] - previous[0])
    if not direction or backward >= 0:
        return selected
    if abs(backward) <= maximum_backward_jitter:
        selected[0] = previous[0]
        return selected
    return predicted.copy()


def _decode_samples(case) -> list[np.ndarray]:
    requested = tuple(
        int(sample["source_frame_index"]) for sample in case.timeline["samples"]
    )
    wanted = set(requested)
    decoded: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(case.reference_video_path))
    try:
        index = 0
        while index <= requested[-1]:
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f"cannot decode source frame {index} for {case.case_id}")
            if index in wanted:
                decoded[index] = frame
            index += 1
    finally:
        capture.release()
    return [decoded[index].copy() for index in requested]


def detect_rail_circles(
    frame: np.ndarray,
    *,
    y_range: tuple[int, int],
    minimum_radius: int,
    maximum_radius: int,
    threshold: float,
    minimum_distance: float,
) -> tuple[Circle, ...]:
    top, bottom = y_range
    if top < 0 or bottom <= top or bottom > frame.shape[0]:
        raise ValueError("collision Hough y range is outside the frame")
    crop = frame[top:bottom]
    gray = cv2.medianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minimum_distance,
        param1=100.0,
        param2=threshold,
        minRadius=minimum_radius,
        maxRadius=maximum_radius,
    )
    if circles is None:
        return ()
    result = [
        (float(x), float(y + top), float(radius))
        for x, y, radius in circles[0]
    ]
    return tuple(sorted(result, key=lambda value: value[0]))


def _trajectory(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(masks)
    centroid = np.full((count, 2), np.nan, np.float32)
    bbox = np.full((count, 4), np.nan, np.float32)
    area = masks.reshape(count, -1).sum(axis=1, dtype=np.int64)
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask)
        if len(xs):
            centroid[index] = (float(xs.mean()), float(ys.mean()))
            bbox[index] = (xs.min(), ys.min(), xs.max(), ys.max())
    return centroid, bbox, area


def _load_candidate_entity(root: Path, object_id: str, count: int) -> EntityObservation:
    entity = root / "entities" / object_id
    return load_entity_observation(
        entity / "mask_tube.npz",
        entity / "trajectory.npz",
        object_id=object_id,
        mask_id=f"{int(object_id.rsplit('_', 1)[-1]):02d}",
        expected_samples=count,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    arguments = parser.parse_args(argv)

    case_ids = tuple(arguments.case_list.read_text(encoding="utf-8").split())
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    if set(plan) != set(case_ids):
        raise ValueError("special Hough plan and case list must contain the same Cases")
    cases = load_curation_cases(arguments.dataset, case_ids=case_ids)
    for case in cases:
        if case.scene_id != "collision_1d":
            raise ValueError(f"unsupported non-collision Case: {case.case_id}")
        specification = plan[case.case_id]
        frames = _decode_samples(case)
        count = len(frames)
        source = next(
            arguments.candidate_root.rglob(f"{case.case_id}/candidate_tracking.json")
        ).parent
        entities = {
            item.identity.object_id: _load_candidate_entity(
                source, item.identity.object_id, count
            )
            for item in case.entities
        }
        detector = specification["detector"]
        detections = [
            detect_rail_circles(
                frame,
                y_range=tuple(int(value) for value in detector["y_range"]),
                minimum_radius=int(detector["minimum_radius"]),
                maximum_radius=int(detector["maximum_radius"]),
                threshold=float(detector["threshold"]),
                minimum_distance=float(detector["minimum_distance"]),
            )
            for frame in frames
        ]
        output = arguments.output_root / case.scene_id / case.case_id
        for object_id, original in entities.items():
            masks = np.asarray(original.masks, dtype=np.uint8).copy()
            states = np.asarray(original.state, dtype=np.uint8).copy()
            entity_spec = specification.get("entities", {}).get(object_id)
            if entity_spec is not None:
                start = int(entity_spec["start_index"])
                end = int(entity_spec["end_index"])
                radius = float(entity_spec["radius"])
                yy, xx = np.ogrid[: masks.shape[1], : masks.shape[2]]
                mode = str(entity_spec.get("selection", "rank"))
                if mode not in {"rank", "temporal"}:
                    raise ValueError(f"unknown Hough selection mode: {mode}")
                if mode == "rank":
                    rank = int(entity_spec["rank_from_left"])
                else:
                    center = np.asarray(entity_spec["seed_center_xy"], np.float64)
                    velocity = np.asarray(entity_spec["initial_velocity_xy"], np.float64)
                    maximum_distance = float(entity_spec["maximum_distance"])
                    maximum_speed = float(entity_spec.get("maximum_speed", maximum_distance))
                    direction = int(entity_spec.get("direction", 0))
                    maximum_backward_jitter = float(
                        entity_spec.get("maximum_backward_jitter", 0.0)
                    )
                    if center.shape != (2,) or velocity.shape != (2,):
                        raise ValueError("temporal Hough seed and velocity must be xy pairs")
                for index in range(start, end + 1):
                    frame_detections = detections[index]
                    if "x_range" in entity_spec:
                        frame_detections = filter_circles_by_x(
                            frame_detections,
                            x_range=tuple(
                                float(value) for value in entity_spec["x_range"]
                            ),
                        )
                    if mode == "rank":
                        x, y, _ = select_ranked_circle(
                            frame_detections, rank_from_left=rank
                        )
                    else:
                        prediction = center if index == start else center + velocity
                        selected = select_temporal_circle(
                            frame_detections,
                            predicted_xy=prediction,
                            expected_radius=radius,
                            maximum_distance=maximum_distance,
                        )
                        previous = center.copy()
                        center = (
                            prediction
                            if selected is None
                            else np.asarray(selected[:2], dtype=np.float64)
                        )
                        center = enforce_direction(
                            center,
                            previous_xy=previous,
                            predicted_xy=prediction,
                            direction=direction,
                            maximum_backward_jitter=maximum_backward_jitter,
                        )
                        if "fixed_y" in entity_spec:
                            center[1] = float(entity_spec["fixed_y"])
                        velocity = clamp_velocity(center - previous, maximum_speed)
                        x, y = center
                        if x + radius < 0 or x - radius >= masks.shape[2]:
                            masks[index:] = 0
                            states[index:] = 2
                            break
                    masks[index] = (
                        (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
                    ).astype(np.uint8)
                    states[index] = 0
                if "out_start_index" in entity_spec:
                    out_start = int(entity_spec["out_start_index"])
                    masks[out_start:] = 0
                    states[out_start:] = 2
            centroid, bbox, area = _trajectory(masks)
            write_entity_observation(
                output / "entities" / object_id,
                EntityObservation(
                    object_id=object_id,
                    mask_id=original.mask_id,
                    masks=masks,
                    centroid_xy=centroid,
                    bbox_xyxy=bbox,
                    area_pixels=area,
                    state=states,
                ),
            )
        (output / "candidate_tracking.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "case_id": case.case_id,
                    "model_id": "reviewed_ranked_hough_collision_tracker",
                    "plan": specification,
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
