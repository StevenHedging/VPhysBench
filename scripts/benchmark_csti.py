#!/usr/bin/env python3
"""Benchmark the exact CSTI reference backend on deterministic Tube pairs."""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import time

import numpy as np

from physbench.evaluation.common.csti import CSTIConfig, score_postcondition_tube
from physbench.io import write_json


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _frame_count(value: str) -> int:
    parsed = int(value)
    if parsed < 4:
        raise argparse.ArgumentTypeError("frames must be at least 4")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact cumulative spatio-temporal soft Tube IoU",
    )
    parser.add_argument("--frames", type=_frame_count, required=True)
    parser.add_argument("--height", type=_positive_int, required=True)
    parser.add_argument("--width", type=_positive_int, required=True)
    parser.add_argument("--objects", type=_positive_int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _tube_pair(
    *,
    object_index: int,
    object_count: int,
    frames: int,
    height: int,
    width: int,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    yy, xx = np.ogrid[:height, :width]
    radius = max(1, min(height, width) // 12)
    y_center = min(
        height - 1,
        max(0, round((object_index + 1) * (height - 1) / (object_count + 1))),
    )
    reference: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    gap_index = frames // 2

    for frame_index in range(frames):
        progress = frame_index / max(frames - 1, 1)
        x_center = round(radius + progress * max(width - 1 - 2 * radius, 0))
        x_center = (x_center + object_index * max(2 * radius + 1, 1)) % width
        gt_mask = (yy - y_center) ** 2 + (xx - x_center) ** 2 <= radius**2
        prediction_center = min(width - 1, x_center + 1)
        prediction_mask = (
            (yy - y_center) ** 2 + (xx - prediction_center) ** 2 <= radius**2
        )
        if frame_index == gap_index:
            prediction_mask = np.zeros((height, width), dtype=bool)
        reference.append(np.asarray(gt_mask, dtype=bool))
        prediction.append(np.asarray(prediction_mask, dtype=bool))

    return tuple(reference), tuple(prediction)


def main() -> int:
    args = _parser().parse_args()
    config = CSTIConfig(
        enabled=True,
        algorithm="exact_full_tube_edt",
        spatial_tolerance_fraction=0.004204482076268572,
        temporal_tolerance_s=0.025,
        condition_frame_policy="exclude_initial_samples",
        initial_frames_excluded=3,
        score_aggregation="full_tube",
        diagnostic_prefix_fractions=(0.25, 0.5, 0.75, 1.0),
        case_aggregation="mean_gt_entities",
        timeline_policy="physical_overlap",
        mask_resolution="scene_analysis_native",
    )

    scores: list[float] = []
    diagnostic_points: list[int] = []
    wall_time_s = 0.0
    sampling_fps = 24.0
    times_s = tuple(index / sampling_fps for index in range(args.frames))
    for object_index in range(args.objects):
        reference, prediction = _tube_pair(
            object_index=object_index,
            object_count=args.objects,
            frames=args.frames,
            height=args.height,
            width=args.width,
        )
        started = time.perf_counter()
        score, diagnostics = score_postcondition_tube(
            reference,
            prediction,
            times_s=times_s,
            config=config,
        )
        wall_time_s += time.perf_counter() - started
        scores.append(float(score))
        diagnostic_points.append(len(diagnostics))

    write_json(
        args.output,
        {
            "algorithm": config.algorithm,
            "sampling_fps": sampling_fps,
            "spatial_tolerance_fraction": config.spatial_tolerance_fraction,
            "temporal_tolerance_s": config.temporal_tolerance_s,
            "frames": args.frames,
            "initial_frames_excluded": config.initial_frames_excluded,
            "scored_frames": args.frames - config.initial_frames_excluded,
            "height": args.height,
            "width": args.width,
            "objects": args.objects,
            "scores": scores,
            "diagnostic_points": diagnostic_points,
            "wall_time_s": wall_time_s,
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
