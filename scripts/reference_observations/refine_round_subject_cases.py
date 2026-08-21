#!/usr/bin/env python3
"""Remove thin strings/springs from reviewed round-subject mask tubes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physbench.reference_observations import (
    EntityObservation,
    load_entity_observation,
    load_reference_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import load_curation_cases
from physbench.reference_observations.curation.tracking import (
    isolate_compact_round_subject,
)


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


def _candidate_entity(root: Path, object_id: str, samples: int) -> EntityObservation:
    return load_entity_observation(
        root / "entities" / object_id / "mask_tube.npz",
        root / "entities" / object_id / "trajectory.npz",
        object_id=object_id,
        mask_id=f"{int(object_id.rsplit('_', 1)[-1]):02d}",
        expected_samples=samples,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--source", choices=("canonical", "candidate"), required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--radius-scale", type=float, default=1.45)
    arguments = parser.parse_args(argv)
    if arguments.source == "candidate" and arguments.candidate_root is None:
        parser.error("--candidate-root is required for candidate input")

    case_ids = tuple(arguments.case_list.read_text(encoding="utf-8").split())
    cases = load_curation_cases(arguments.dataset, case_ids=case_ids)
    for case in cases:
        if len(case.entities) != 1:
            raise ValueError(f"round-subject refinement expects one entity: {case.case_id}")
        object_id = case.entities[0].identity.object_id
        if arguments.source == "canonical":
            source = load_reference_observation(
                case.asset_root,
                case.observation_manifest_path,
                bundle_root=case.asset_root,
            ).entities[object_id]
        else:
            assert arguments.candidate_root is not None
            candidate_case = next(
                arguments.candidate_root.rglob(f"{case.case_id}/candidate_tracking.json")
            ).parent
            source = _candidate_entity(
                candidate_case, object_id, len(case.timeline["samples"])
            )
        masks, states = isolate_compact_round_subject(
            source.masks,
            source.state,
            radius_scale=arguments.radius_scale,
        )
        centroid, bbox, area = _trajectory(masks)
        output = arguments.output_root / case.scene_id / case.case_id
        write_entity_observation(
            output / "entities" / object_id,
            EntityObservation(
                object_id=object_id,
                mask_id="01",
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
                    "model_id": "distance_transform_round_subject_refinement",
                    "source": arguments.source,
                    "radius_scale": arguments.radius_scale,
                    "unresolved_samples": int(np.count_nonzero(states == 3)),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"cases": len(cases), "source": arguments.source}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
