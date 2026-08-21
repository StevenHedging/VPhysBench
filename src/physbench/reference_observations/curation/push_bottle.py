"""Classical dense-frame tracking helpers for the transparent push bottle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


def clean_binary_mask(mask: np.ndarray, *, opening_kernel: int = 3) -> np.ndarray:
    """Remove thin warp artifacts and retain the principal connected silhouette."""

    value = (np.asarray(mask) != 0).astype(np.uint8)
    if value.ndim != 2 or not np.count_nonzero(value):
        raise ValueError("mask cleaning requires a non-empty 2D mask")
    if opening_kernel < 1 or opening_kernel % 2 == 0:
        raise ValueError("opening_kernel must be a positive odd integer")
    if opening_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (opening_kernel, opening_kernel)
        )
        value = cv2.morphologyEx(value, cv2.MORPH_OPEN, kernel)
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(value, 8)
    if count <= 1:
        raise ValueError("mask cleaning removed the complete silhouette")
    largest = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _compose(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left3 = np.vstack((np.asarray(left, dtype=np.float64), (0.0, 0.0, 1.0)))
    right3 = np.vstack((np.asarray(right, dtype=np.float64), (0.0, 0.0, 1.0)))
    return (left3 @ right3)[:2]


def _inverse(transform: np.ndarray) -> np.ndarray:
    matrix = np.vstack(
        (np.asarray(transform, dtype=np.float64), (0.0, 0.0, 1.0))
    )
    return np.linalg.inv(matrix)[:2]


def _validated_similarity(
    transform: np.ndarray,
    *,
    maximum_rotation_degrees: float,
    maximum_translation: float,
) -> np.ndarray | None:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.all(np.isfinite(matrix)):
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    angle = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    translation = float(np.linalg.norm(matrix[:, 2]))
    if (
        not 0.94 <= scale <= 1.06
        or abs(angle) > maximum_rotation_degrees
        or translation > maximum_translation
    ):
        return None
    return matrix


def _mask_centroid_and_axis(mask: np.ndarray) -> tuple[np.ndarray, float]:
    y, x = np.nonzero(mask)
    if len(x) < 3:
        raise ValueError("pose estimation requires a non-empty mask")
    points = np.stack((x, y), axis=1).astype(np.float64)
    values, vectors = np.linalg.eigh(np.cov(points.T))
    direction = vectors[:, int(np.argmax(values))]
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    return points.mean(axis=0), angle


def _mask_for_pose(
    template: np.ndarray,
    template_centroid: np.ndarray,
    *,
    angle: float,
    scale: float,
    center_xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    transform = cv2.getRotationMatrix2D(
        tuple(float(value) for value in template_centroid), angle, scale
    )
    mapped = transform @ np.asarray((*template_centroid, 1.0))
    transform[:, 2] += np.asarray(center_xy, dtype=np.float64) - mapped
    output = cv2.warpAffine(
        template,
        transform,
        (template.shape[1], template.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.uint8)
    return output, transform


def apply_pose_keyframes(
    masks: np.ndarray,
    *,
    template_mask: np.ndarray,
    keyframes: Mapping[int, Mapping[str, object]],
) -> np.ndarray:
    """Override a mask-tube interval with interpolated directed rigid poses.

    ``axis_degrees`` is deliberately unwrapped: increasing through 180 degrees
    preserves which end of an asymmetric bottle contains the cap.  Treating the
    PCA axis modulo 180 would flip that end during a near-horizontal impact.
    """

    output = (np.asarray(masks) != 0).astype(np.uint8)
    template = (np.asarray(template_mask) != 0).astype(np.uint8)
    if output.ndim != 3 or template.shape != output.shape[1:]:
        raise ValueError("pose-keyframe masks must share one image shape")
    if not keyframes:
        return output.copy()

    parsed: list[tuple[int, float, np.ndarray, float]] = []
    for raw_index, raw_pose in keyframes.items():
        index = int(raw_index)
        if not 0 <= index < len(output):
            raise ValueError("pose keyframe index is outside the mask tube")
        center = np.asarray(raw_pose["center_xy"], dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("pose keyframe center_xy must contain two numbers")
        axis = float(raw_pose["axis_degrees"])
        scale = float(raw_pose.get("scale", 1.0))
        if not np.isfinite(axis) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("pose keyframe axis and scale must be finite")
        parsed.append((index, axis, center, scale))
    parsed.sort(key=lambda item: item[0])
    if len({item[0] for item in parsed}) != len(parsed):
        raise ValueError("pose keyframe indices must be unique")

    template_center, template_axis = _mask_centroid_and_axis(template)

    def render(index: int, axis: float, center: np.ndarray, scale: float) -> None:
        output[index], _ = _mask_for_pose(
            template,
            template_center,
            angle=template_axis - axis,
            scale=scale,
            center_xy=(float(center[0]), float(center[1])),
        )

    if len(parsed) == 1:
        index, axis, center, scale = parsed[0]
        render(index, axis, center, scale)
        return output

    for left, right in zip(parsed, parsed[1:]):
        left_index, left_axis, left_center, left_scale = left
        right_index, right_axis, right_center, right_scale = right
        if right_index <= left_index:
            raise ValueError("pose keyframes must have increasing indices")
        for index in range(left_index, right_index + 1):
            weight = (index - left_index) / (right_index - left_index)
            axis = (1.0 - weight) * left_axis + weight * right_axis
            center = (1.0 - weight) * left_center + weight * right_center
            scale = (1.0 - weight) * left_scale + weight * right_scale
            render(index, axis, center, scale)
    return output


def fit_rigid_mask_to_evidence(
    *,
    frame: np.ndarray,
    template_mask: np.ndarray,
    predicted_mask: np.ndarray,
    interior_evidence: np.ndarray,
    angle_radius: float = 22.0,
    center_radius: int = 60,
) -> np.ndarray:
    """Correct a drifted pose using silhouette edges and known interior pixels.

    ``interior_evidence`` need not cover the object.  It represents pixels that
    are known to lie inside it (for the push-bottle cases, the amber liquid).
    """

    image = np.asarray(frame)
    template = (np.asarray(template_mask) != 0).astype(np.uint8)
    predicted = (np.asarray(predicted_mask) != 0).astype(np.uint8)
    evidence = (np.asarray(interior_evidence) != 0).astype(np.uint8)
    if image.shape[:2] != template.shape or predicted.shape != template.shape:
        raise ValueError("pose-fitting inputs must share one image shape")
    if evidence.shape != template.shape or not np.count_nonzero(evidence):
        raise ValueError("interior evidence must be non-empty and image-sized")

    template_center, template_axis = _mask_centroid_and_axis(template)
    predicted_center, predicted_axis = _mask_centroid_and_axis(predicted)
    axis_delta = (predicted_axis - template_axis + 90.0) % 180.0 - 90.0
    predicted_angle = -axis_delta
    predicted_scale = float(
        np.sqrt(np.count_nonzero(predicted) / np.count_nonzero(template))
    )

    contours, _ = cv2.findContours(
        (template * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    boundary = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    boundary = boundary[:: max(1, len(boundary) // 700)]
    evidence_y, evidence_x = np.nonzero(evidence)
    evidence_points = np.stack((evidence_x, evidence_y), axis=1).astype(np.float64)
    evidence_points = evidence_points[
        :: max(1, len(evidence_points) // 1200)
    ]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8), cv2.DIST_L2, 3
    )
    height, width = template.shape

    def score(
        angle: float, scale: float, center_x: float, center_y: float
    ) -> tuple[float, np.ndarray]:
        _, transform = _mask_for_pose(
            template,
            template_center,
            angle=angle,
            scale=scale,
            center_xy=(center_x, center_y),
        )
        mapped_boundary = cv2.transform(
            boundary.reshape(1, -1, 2), transform
        ).reshape(-1, 2)
        rounded = np.rint(mapped_boundary).astype(int)
        valid = (
            (rounded[:, 0] >= 0)
            & (rounded[:, 0] < width)
            & (rounded[:, 1] >= 0)
            & (rounded[:, 1] < height)
        )
        if np.count_nonzero(valid) != len(rounded):
            return float("inf"), transform
        distances = edge_distance[rounded[:, 1], rounded[:, 0]]
        edge_cost = float(np.mean(np.minimum(distances, 12.0)))

        inverse = cv2.invertAffineTransform(transform)
        source_evidence = cv2.transform(
            evidence_points.reshape(1, -1, 2), inverse
        ).reshape(-1, 2)
        source = np.rint(source_evidence).astype(int)
        inside_bounds = (
            (source[:, 0] >= 0)
            & (source[:, 0] < width)
            & (source[:, 1] >= 0)
            & (source[:, 1] < height)
        )
        inside = np.zeros(len(source), dtype=bool)
        inside[inside_bounds] = (
            template[source[inside_bounds, 1], source[inside_bounds, 0]] != 0
        )
        return edge_cost + 20.0 * (1.0 - float(inside.mean())), transform

    candidates: list[tuple[float, float, float, float, float]] = []
    angle_values = np.arange(
        predicted_angle - angle_radius,
        predicted_angle + angle_radius + 0.1,
        2.0,
    )
    scale_values = np.arange(
        max(0.90, predicted_scale - 0.04),
        min(1.10, predicted_scale + 0.04) + 0.001,
        0.02,
    )
    for angle in angle_values:
        for scale in scale_values:
            for center_x in np.arange(
                predicted_center[0] - center_radius,
                predicted_center[0] + center_radius + 0.1,
                10.0,
            ):
                for center_y in np.arange(
                    predicted_center[1] - center_radius,
                    predicted_center[1] + center_radius + 0.1,
                    10.0,
                ):
                    value, _ = score(angle, scale, center_x, center_y)
                    candidates.append((value, angle, scale, center_x, center_y))
    _, best_angle, best_scale, best_x, best_y = min(candidates)

    fine: list[tuple[float, float, float, float, float]] = []
    for angle in np.arange(best_angle - 2.0, best_angle + 2.01, 0.5):
        for scale in np.arange(
            max(0.90, best_scale - 0.02),
            min(1.10, best_scale + 0.02) + 0.001,
            0.01,
        ):
            for center_x in np.arange(best_x - 8.0, best_x + 8.1, 2.0):
                for center_y in np.arange(best_y - 8.0, best_y + 8.1, 2.0):
                    value, _ = score(angle, scale, center_x, center_y)
                    fine.append((value, angle, scale, center_x, center_y))
    _, best_angle, best_scale, best_x, best_y = min(fine)
    fitted, _ = _mask_for_pose(
        template,
        template_center,
        angle=best_angle,
        scale=best_scale,
        center_xy=(best_x, best_y),
    )
    return fitted


@dataclass
class RigidMaskTracker:
    """Track one rigid silhouette through consecutive (not subsampled) frames."""

    initial_frame: np.ndarray
    initial_mask: np.ndarray
    maximum_rotation_degrees: float = 8.0
    maximum_translation: float = 40.0

    def __post_init__(self) -> None:
        frame = np.asarray(self.initial_frame)
        mask = np.asarray(self.initial_mask)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("initial frame must use HWC BGR layout")
        if mask.shape != frame.shape[:2]:
            raise ValueError("initial mask shape differs from the frame")
        if not np.count_nonzero(mask):
            raise ValueError("initial mask cannot be empty")
        self._initial_mask = (mask != 0).astype(np.uint8)
        self._previous_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._cumulative = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float64
        )
        self._shape = frame.shape[:2]
        self._current_mask = self._initial_mask.copy()
        initial_y, initial_x = np.nonzero(self._initial_mask)
        self._initial_centroid = np.asarray(
            (initial_x.mean(), initial_y.mean()), dtype=np.float64
        )
        self._points = self._detect_points(self._previous_gray, self._current_mask)
        self._origin_points = self._points.copy()

    @staticmethod
    def _detect_points(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ys, xs = np.nonzero(mask)
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        erosion = max(1, min(width, height) // 24)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * erosion + 1, 2 * erosion + 1)
        )
        support = cv2.erode(mask, kernel)
        points = cv2.goodFeaturesToTrack(
            gray,
            mask=(support * 255).astype(np.uint8),
            maxCorners=800,
            qualityLevel=0.003,
            minDistance=3,
            blockSize=5,
        )
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        return points.reshape(-1, 2).astype(np.float32)

    def _supplement_points(self) -> None:
        if len(self._points) >= 80:
            return
        support = self._current_mask.copy()
        for x, y in np.rint(self._points).astype(int):
            cv2.circle(support, (x, y), 6, 0, -1)
        additions = self._detect_points(self._previous_gray, support)
        if not len(additions):
            return
        inverse = _inverse(self._cumulative)
        origin = cv2.transform(additions.reshape(1, -1, 2), inverse).reshape(-1, 2)
        self._points = np.concatenate((self._points, additions)).astype(np.float32)
        self._origin_points = np.concatenate((self._origin_points, origin)).astype(
            np.float32
        )

    @property
    def current_mask(self) -> np.ndarray:
        return self._current_mask.copy()

    def update(self, frame: np.ndarray) -> np.ndarray:
        image = np.asarray(frame)
        if image.shape[:2] != self._shape or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("tracking frame shape differs from the initial frame")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self._supplement_points()
        accepted: np.ndarray | None = None
        if len(self._points) >= 6:
            points = self._points.reshape(-1, 1, 2)
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                self._previous_gray,
                gray,
                points,
                None,
                winSize=(25, 25),
                maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01),
            )
            if forward is not None and forward_status is not None:
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self._previous_gray,
                    forward,
                    None,
                    winSize=(25, 25),
                    maxLevel=4,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        40,
                        0.01,
                    ),
                )
                if backward is not None and backward_status is not None:
                    previous = points.reshape(-1, 2)
                    target = forward.reshape(-1, 2)
                    roundtrip = np.linalg.norm(
                        backward.reshape(-1, 2) - previous, axis=1
                    )
                    valid = (
                        forward_status.reshape(-1).astype(bool)
                        & backward_status.reshape(-1).astype(bool)
                        & (roundtrip <= 1.5)
                    )
                    origin = self._origin_points[valid]
                    previous = previous[valid]
                    target = target[valid]
                    if len(origin) >= 6:
                        candidate, inliers = cv2.estimateAffinePartial2D(
                            origin,
                            target,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.0,
                            maxIters=3000,
                            confidence=0.995,
                            refineIters=20,
                        )
                        if candidate is not None and inliers is not None:
                            if int(inliers.sum()) >= 6:
                                # The tabletop camera is fixed and the bottle
                                # remains at essentially constant depth.  LK
                                # scale error otherwise compounds into visible
                                # silhouette shrinkage over a thousand frames.
                                absolute_scale = float(
                                    np.hypot(candidate[0, 0], candidate[1, 0])
                                )
                                clamped_scale = float(
                                    np.clip(absolute_scale, 0.985, 1.015)
                                )
                                if absolute_scale > 0:
                                    mapped_centroid = (
                                        candidate[:, :2] @ self._initial_centroid
                                        + candidate[:, 2]
                                    )
                                    candidate[:, :2] *= clamped_scale / absolute_scale
                                    candidate[:, 2] = mapped_centroid - (
                                        candidate[:, :2] @ self._initial_centroid
                                    )
                                relative = _compose(candidate, _inverse(self._cumulative))
                                accepted = _validated_similarity(
                                    relative,
                                    maximum_rotation_degrees=self.maximum_rotation_degrees,
                                    maximum_translation=self.maximum_translation,
                                )
                                if accepted is not None:
                                    self._cumulative = candidate
                                    keep = inliers.reshape(-1).astype(bool)
                                    self._origin_points = origin[keep].astype(np.float32)
                                    self._points = target[keep].astype(np.float32)

        if accepted is None:
            # Keep only forward/backward-consistent tracks even when the pose
            # estimate is temporarily unavailable; the next dense frame can
            # recover by adding features inside the last trusted silhouette.
            if "valid" in locals():
                self._origin_points = self._origin_points[valid]
                self._points = forward.reshape(-1, 2)[valid].astype(np.float32)
        self._current_mask = cv2.warpAffine(
            self._initial_mask,
            self._cumulative,
            (self._shape[1], self._shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.uint8)
        self._previous_gray = gray
        return self.current_mask


def track_rigid_mask_sequence(
    frames: Sequence[np.ndarray], initial_mask: np.ndarray
) -> np.ndarray:
    """Return one binary mask per consecutive frame, including frame zero."""

    if not frames:
        raise ValueError("tracking requires at least one frame")
    tracker = RigidMaskTracker(frames[0], initial_mask)
    masks = [tracker.current_mask]
    masks.extend(tracker.update(frame) for frame in frames[1:])
    return np.stack(masks).astype(np.uint8)
