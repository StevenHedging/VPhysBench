#!/usr/bin/env python3
"""Build candidates that relabel geometrically proven trailing boundary exits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from physbench.reference_observations import (
    EntityObservation,
    load_reference_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import load_curation_cases
from physbench.reference_observations.curation.tracking import (
    resolve_trailing_boundary_exit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--case-id", action="append", default=[])
    arguments = parser.parse_args(argv)

    changes: list[dict[str, object]] = []
    cases = load_curation_cases(
        arguments.dataset, case_ids=arguments.case_id or None
    )
    for case in cases:
        observation = load_reference_observation(
            case.asset_root,
            case.observation_manifest_path,
            bundle_root=case.asset_root,
        )
        resolved: dict[str, EntityObservation] = {}
        case_changes: list[dict[str, object]] = []
        for object_id, entity in observation.entities.items():
            state = resolve_trailing_boundary_exit(entity.masks, entity.state)
            changed = np.flatnonzero(state != entity.state)
            if len(changed):
                case_changes.append(
                    {
                        "object_id": object_id,
                        "start_index": int(changed[0]),
                        "end_index": int(changed[-1]),
                        "old_state": int(entity.state[changed[0]]),
                        "new_state": int(state[changed[0]]),
                    }
                )
            resolved[object_id] = EntityObservation(
                object_id=entity.object_id,
                mask_id=entity.mask_id,
                masks=np.asarray(entity.masks).copy(),
                centroid_xy=np.asarray(entity.centroid_xy).copy(),
                bbox_xyxy=np.asarray(entity.bbox_xyxy).copy(),
                area_pixels=np.asarray(entity.area_pixels).copy(),
                state=state,
            )
        if not case_changes:
            continue
        output = arguments.output_root / case.scene_id / case.case_id
        for entity in resolved.values():
            write_entity_observation(output / "entities" / entity.object_id, entity)
        (output / "candidate_tracking.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "case_id": case.case_id,
                    "model_id": "reviewed_boundary_lifecycle_relabel",
                    "anchor_source": "unchanged_canonical_observation_zero",
                    "changes": case_changes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        changes.append(
            {"case_id": case.case_id, "scene_id": case.scene_id, "changes": case_changes}
        )

    report = arguments.output_root / "changes.jsonl"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in changes),
        encoding="utf-8",
    )
    print(json.dumps({"cases_scanned": len(cases), "cases_changed": len(changes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
