#!/usr/bin/env python3
"""Audit collision-v4 reference observability without generating predictions."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from physbench.data_layout import LATEST_DATASET
from physbench.datasets import load_dataset
from physbench.evaluation.common.media import reference_timeline, sample_video
from physbench.evaluation.protocols import load_evaluation_protocol
from physbench.evaluation.registry import SceneEvaluatorRegistry
from physbench.evaluation.scenes.collision.scoring import (
    extract_collision_trace,
)
from physbench.io import write_json


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the collision-v4 observation stack over immutable reference "
            "videos and record coverage, prompts, masks, and contact events."
        )
    )
    parser.add_argument("--dataset", type=Path, default=LATEST_DATASET)
    parser.add_argument("--protocol", default="scene_default_v4")
    parser.add_argument("--view", default="view_b")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _view_case_ids(dataset: Any, view_id: str) -> list[str]:
    view = dataset.views[view_id]
    scene = view["scenes"]["collision_1d"]
    return [
        case_id
        for partition in scene.values()
        for case_id in partition
    ]


def main() -> None:
    args = _arguments()
    dataset = load_dataset(args.dataset)
    catalog = {case["case_id"]: case for case in dataset.cases}
    protocol = copy.deepcopy(load_evaluation_protocol(args.protocol))
    collision_config = protocol["scenes"]["collision_1d"]
    collision_config["sam2"]["device"] = args.device
    evaluator = SceneEvaluatorRegistry(protocol).resolve("collision_1d")
    selected = args.case_id or _view_case_ids(dataset, args.view)
    records: list[dict[str, Any]] = []

    for case_id in selected:
        case = catalog[case_id]
        reference = (
            dataset.asset_root / case["assets"]["reference_video"]
        ).resolve()
        timeline = collision_config["timeline"]
        spatial = collision_config["spatial"]
        record: dict[str, Any] = {
            "case_id": case_id,
            "reference_video": str(reference),
        }
        try:
            times_s = reference_timeline(
                reference,
                fps=float(timeline["fps"]),
                max_duration_s=float(timeline["maximum_duration_s"]),
                minimum_duration_s=float(timeline["minimum_duration_s"]),
            )
            video = sample_video(
                reference,
                sample_times_s=times_s,
                width=int(spatial["width"]),
                height=int(spatial["height"]),
                pad_value=int(spatial.get("pad_value", 0)),
                min_source_fps=float(timeline["minimum_source_fps"]),
                duration_tolerance_s=float(
                    timeline.get("duration_tolerance_s", 0.02)
                ),
                decode_policy=str(timeline["decode_policy"]),
            )
            xy, valid, _, _, observation = evaluator._observe(
                video.frames,
                minimum_valid_ratio=collision_config["quality"].get(
                    "minimum_reference_valid_frame_ratio"
                ),
            )
            physics = case["physics"]
            masses = np.asarray(
                [
                    float(physics[f"ball_{index}_mass"]["value"])
                    for index in range(1, 4)
                ]
            )
            quality = collision_config["quality"]
            trace = extract_collision_trace(
                xy,
                valid,
                times_s,
                masses_kg=masses,
                minimum_span_px=float(quality["minimum_motion_span_px"]),
                velocity_window_fraction=float(
                    quality["velocity_window_fraction"]
                ),
            )
            record.update(
                {
                    "status": "observable",
                    "sampled_frames": len(times_s),
                    "valid_track_ratios": trace.valid_ratio.tolist(),
                    "axis_explained_ratio": trace.axis.explained_ratio,
                    "event_frame": trace.event_frame,
                    "event_time_s": trace.event_time_s,
                    "observation": observation,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "failed",
                    "error": {
                        "type": type(exc).__name__,
                        "code": getattr(exc, "code", None),
                        "message": str(exc),
                    },
                }
            )
        records.append(record)
        print(
            f"[{len(records)}/{len(selected)}] {case_id}: "
            f"{record['status']}",
            flush=True,
        )

    observable = sum(record["status"] == "observable" for record in records)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "digest": dataset.digest,
            "descriptor": str(Path(args.dataset).resolve()),
        },
        "protocol": {
            "id": protocol["protocol_id"],
            "fingerprint": protocol["fingerprint"],
        },
        "scope": {
            "view": args.view,
            "scene_id": "collision_1d",
            "case_ids": selected,
        },
        "summary": {
            "cases": len(records),
            "observable": observable,
            "failed": len(records) - observable,
            "coverage": observable / max(len(records), 1),
        },
        "cases": records,
    }
    write_json(args.output, report)
    print(report["summary"], flush=True)


if __name__ == "__main__":
    main()
