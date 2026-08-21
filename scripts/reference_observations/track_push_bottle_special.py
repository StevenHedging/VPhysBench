#!/usr/bin/env python3
"""Repair transparent push-bottle cases whose human anchor selected a board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physbench.reference_observations import write_entity_observation
from physbench.reference_observations.curation import (
    load_curation_cases,
    render_dense_event_sheet,
)
from physbench.reference_observations.curation.finalize import entity_from_masks
from physbench.reference_observations.curation.push_bottle import (
    RigidMaskTracker,
    apply_pose_keyframes,
    clean_binary_mask,
    fit_rigid_mask_to_evidence,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _template_mask(frame: np.ndarray, specification: dict[str, Any]) -> np.ndarray:
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        frame,
        mask,
        tuple(int(value) for value in specification["rect_xywh"]),
        background,
        foreground,
        int(specification["iterations"]),
        cv2.GC_INIT_WITH_RECT,
    )
    foreground_mask = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(foreground_mask, 8)
    point_x, point_y = (int(value) for value in specification["keep_point_xy"])
    label = int(labels[point_y, point_x])
    if not 0 < label < count:
        raise ValueError("template GrabCut keep point is not foreground")
    result = (labels == label).astype(np.uint8)
    result[: int(specification["clear_above_y"])] = 0
    kernel_size = int(specification["opening_kernel"])
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
    if np.count_nonzero(result) < 50_000:
        raise ValueError("template GrabCut did not recover the bottle silhouette")
    return result.astype(np.uint8)


def _translated_mask(mask: np.ndarray, center_xy: tuple[float, float]) -> np.ndarray:
    y, x = np.nonzero(mask)
    source_center = np.asarray((x.mean(), y.mean()), dtype=np.float64)
    transform = np.asarray(
        (
            (1.0, 0.0, float(center_xy[0]) - source_center[0]),
            (0.0, 1.0, float(center_xy[1]) - source_center[1]),
        ),
        dtype=np.float64,
    )
    return cv2.warpAffine(
        mask,
        transform,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def _principal_axis_delta(mask: np.ndarray, template: np.ndarray) -> float:
    def axis(value: np.ndarray) -> float:
        y, x = np.nonzero(value)
        points = np.stack((x, y), axis=1)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
        direction = eigenvectors[:, int(np.argmax(eigenvalues))]
        return float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)

    return (axis(mask) - axis(template) + 90.0) % 180.0 - 90.0


def _liquid_evidence(
    frame: np.ndarray,
    predicted_mask: np.ndarray,
    hsv_specification: dict[str, Any],
) -> np.ndarray | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_minimum, hue_maximum = (
        int(value) for value in hsv_specification["hue"]
    )
    value_minimum, value_maximum = (
        int(value) for value in hsv_specification["value"]
    )
    support = cv2.dilate(
        predicted_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101)),
    )
    raw = (
        (hsv[:, :, 0] >= hue_minimum)
        & (hsv[:, :, 0] <= hue_maximum)
        & (hsv[:, :, 1] >= int(hsv_specification["saturation_minimum"]))
        & (hsv[:, :, 2] >= value_minimum)
        & (hsv[:, :, 2] <= value_maximum)
        & (support != 0)
    ).astype(np.uint8)
    count, labels, statistics, centroids = cv2.connectedComponentsWithStats(raw, 8)
    predicted_y, predicted_x = np.nonzero(predicted_mask)
    center_y = float(predicted_y.mean())
    eligible = [
        index
        for index in range(1, count)
        if int(statistics[index, cv2.CC_STAT_AREA]) >= 500
        and float(centroids[index, 1]) > center_y
    ]
    if not eligible:
        return None
    selected = max(eligible, key=lambda index: int(statistics[index, cv2.CC_STAT_AREA]))
    return (labels == selected).astype(np.uint8)


def _initial_mask(
    frame: np.ndarray,
    *,
    common_template: np.ndarray,
    center_xy: tuple[float, float],
    hsv_specification: dict[str, Any],
) -> np.ndarray:
    predicted = _translated_mask(common_template, center_xy)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue_minimum, hue_maximum = (
        int(value) for value in hsv_specification["hue"]
    )
    value_minimum, value_maximum = (
        int(value) for value in hsv_specification["value"]
    )
    interior = cv2.erode(
        predicted, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    evidence = (
        (hsv[:, :, 0] >= hue_minimum)
        & (hsv[:, :, 0] <= hue_maximum)
        & (hsv[:, :, 1] >= int(hsv_specification["saturation_minimum"]))
        & (hsv[:, :, 2] >= value_minimum)
        & (hsv[:, :, 2] <= value_maximum)
        & (interior != 0)
    ).astype(np.uint8)
    if np.count_nonzero(evidence) < 500:
        raise ValueError("initial bottle liquid evidence is insufficient")
    fitted = fit_rigid_mask_to_evidence(
        frame=frame,
        template_mask=common_template,
        predicted_mask=predicted,
        interior_evidence=evidence,
        angle_radius=5.0,
        center_radius=30,
    )
    return clean_binary_mask(fitted, opening_kernel=3)


def _decode_and_track(
    case: Any,
    *,
    initial_mask: np.ndarray,
    tracking_scale: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    source_indices = tuple(
        int(sample["source_frame_index"]) for sample in case.timeline["samples"]
    )
    wanted = set(source_indices)
    capture = cv2.VideoCapture(str(case.reference_video_path))
    okay, first = capture.read()
    if not okay:
        capture.release()
        raise ValueError(f"cannot decode source frame zero for {case.case_id}")
    small_first = cv2.resize(
        first,
        None,
        fx=tracking_scale,
        fy=tracking_scale,
        interpolation=cv2.INTER_AREA,
    )
    small_mask = cv2.resize(
        initial_mask,
        (small_first.shape[1], small_first.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    tracker = RigidMaskTracker(
        small_first,
        small_mask,
        maximum_translation=25.0,
    )
    frames_by_source = {0: first.copy()}
    masks_by_source = {0: initial_mask.copy()}
    source_index = 0
    try:
        while source_index < source_indices[-1]:
            okay, frame = capture.read()
            source_index += 1
            if not okay:
                raise ValueError(
                    f"cannot decode source frame {source_index} for {case.case_id}"
                )
            small = cv2.resize(
                frame,
                None,
                fx=tracking_scale,
                fy=tracking_scale,
                interpolation=cv2.INTER_AREA,
            )
            tracked = tracker.update(small)
            if source_index in wanted:
                full = cv2.resize(
                    tracked,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                frames_by_source[source_index] = frame.copy()
                masks_by_source[source_index] = full.astype(np.uint8)
    finally:
        capture.release()
    return (
        [frames_by_source[index] for index in source_indices],
        np.stack([masks_by_source[index] for index in source_indices]),
    )


def _render_anchor(path: Path, frame: np.ndarray, mask: np.ndarray) -> None:
    image = frame.copy()
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, (0, 255, 255), 4)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"cannot write {path}")


def process_case(
    case: Any,
    *,
    common_template: np.ndarray,
    specification: dict[str, Any],
    plan: dict[str, Any],
    output_root: Path,
) -> None:
    capture = cv2.VideoCapture(str(case.reference_video_path))
    okay, first = capture.read()
    capture.release()
    if not okay:
        raise ValueError(f"cannot decode first frame for {case.case_id}")
    initial = _initial_mask(
        first,
        common_template=common_template,
        center_xy=tuple(float(value) for value in specification["rough_initial_center_xy"]),
        hsv_specification=plan["liquid_hsv"],
    )
    frames, masks = _decode_and_track(
        case,
        initial_mask=initial,
        tracking_scale=float(plan["tracking_scale"]),
    )
    refined_indices: list[int] = []
    for index, (frame, predicted) in enumerate(zip(frames, masks, strict=True)):
        delta = abs(_principal_axis_delta(predicted, initial))
        if delta < float(plan["late_pose_axis_delta_degrees"]):
            continue
        evidence = _liquid_evidence(frame, predicted, plan["liquid_hsv"])
        if evidence is None:
            continue
        masks[index] = fit_rigid_mask_to_evidence(
            frame=frame,
            template_mask=initial,
            predicted_mask=predicted,
            interior_evidence=evidence,
        )
        refined_indices.append(index)

    raw_pose_keyframes = specification.get("manual_pose_keyframes", {})
    pose_keyframes = {int(index): pose for index, pose in raw_pose_keyframes.items()}
    if pose_keyframes:
        masks = apply_pose_keyframes(
            masks,
            template_mask=initial,
            keyframes=pose_keyframes,
        )

    output = output_root / case.scene_id / case.case_id
    output.mkdir(parents=True, exist_ok=True)
    states = np.zeros(len(masks), dtype=np.uint8)
    entity = entity_from_masks("object_1", "01", masks, states)
    write_entity_observation(output / "entities" / "object_1", entity)
    _render_anchor(output / "anchor.png", frames[0], masks[0])
    sample_indices = tuple(
        dict.fromkeys(
            int(value)
            for value in np.linspace(0, len(frames) - 1, min(20, len(frames)), dtype=int)
        )
    )
    contact = render_dense_event_sheet(
        frames,
        masks_by_object={"object_1": masks},
        states_by_object={"object_1": states},
        observation_indices=sample_indices,
        source_indices=tuple(
            int(case.timeline["samples"][index]["source_frame_index"])
            for index in sample_indices
        ),
        columns=4,
    )
    if not cv2.imwrite(str(output / "contact_sheet.png"), contact):
        raise RuntimeError(f"cannot write contact sheet for {case.case_id}")
    _write_json(
        output / "candidate_tracking.json",
        {
            "schema_version": "1.0",
            "case_id": case.case_id,
            "model_id": "opencv_dense_lk_rigid_edge_liquid_v1",
            "anchor_source": "independent_case_specific_template",
            "seed_frame_by_observation": [0] * len(masks),
            "propagation": {
                "backend": "dense_source_frame_lk_plus_rigid_edge_liquid_fit",
                "source_frames_processed": int(
                    case.timeline["samples"][-1]["source_frame_index"]
                )
                + 1,
                "observation_frames": len(masks),
                "late_pose_refined_observations": refined_indices,
                "manual_pose_keyframes": {
                    str(index): pose_keyframes[index]
                    for index in sorted(pose_keyframes)
                },
                "tracking_scale": float(plan["tracking_scale"]),
            },
            "evidence": {
                "anchor": "anchor.png",
                "contact_sheet": "contact_sheet.png",
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    plan = json.loads(arguments.plan.read_text(encoding="utf-8"))
    case_specs = plan["cases"]
    cases = load_curation_cases(arguments.dataset, case_ids=set(case_specs))
    cases_by_id = {case.case_id: case for case in cases}
    if set(cases_by_id) != set(case_specs):
        raise ValueError("special push-bottle plan does not match the Dataset")
    if any(case.scene_id != "push_bottle" for case in cases):
        raise ValueError("special push-bottle tracker received another scene")

    template_case = cases_by_id[plan["template_case_id"]]
    capture = cv2.VideoCapture(str(template_case.reference_video_path))
    okay, template_frame = capture.read()
    capture.release()
    if not okay:
        raise ValueError("cannot decode the push-bottle template frame")
    common_template = _template_mask(template_frame, plan["template_grabcut"])
    for case in cases:
        process_case(
            case,
            common_template=common_template,
            specification=case_specs[case.case_id],
            plan=plan,
            output_root=arguments.output_root,
        )
        print(json.dumps({"case_id": case.case_id, "status": "tracked"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
