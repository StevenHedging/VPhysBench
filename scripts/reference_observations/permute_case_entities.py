#!/usr/bin/env python3
"""Build a reviewable candidate by permuting already-accurate entity tubes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from physbench.reference_observations import (
    load_reference_observation,
    write_entity_observation,
)
from physbench.reference_observations.curation import (
    load_curation_cases,
    render_dense_event_sheet,
)
from physbench.reference_observations.curation.finalize import (
    permute_entity_observations,
)


def _decode_samples(case) -> list[np.ndarray]:
    requested = tuple(int(item["source_frame_index"]) for item in case.timeline["samples"])
    wanted = set(requested)
    frames: list[np.ndarray] = []
    capture = cv2.VideoCapture(str(case.reference_video_path))
    try:
        for index in range(requested[-1] + 1):
            okay, frame = capture.read()
            if not okay:
                raise ValueError(f"cannot decode source frame {index} for {case.case_id}")
            if index in wanted:
                frames.append(frame)
    finally:
        capture.release()
    return frames


def _mapping(values: list[str], object_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        new, separator, old = value.partition("=")
        if not separator or new in result:
            raise ValueError("entity mappings must use unique NEW=OLD pairs")
        result[new] = old
    if set(result) != object_ids or set(result.values()) != object_ids:
        raise ValueError("entity mapping must be a complete permutation")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--map", action="append", required=True)
    arguments = parser.parse_args(argv)

    case = load_curation_cases(arguments.dataset, case_ids=[arguments.case_id])[0]
    observation = load_reference_observation(
        case.asset_root,
        case.observation_manifest_path,
        bundle_root=case.asset_root,
    )
    mapping = _mapping(arguments.map, set(observation.entities))
    entities = permute_entity_observations(observation.entities, new_to_old=mapping)
    output = arguments.output_root / case.scene_id / case.case_id
    for entity in entities.values():
        write_entity_observation(output / "entities" / entity.object_id, entity)

    frames = _decode_samples(case)
    sample_indices = tuple(
        dict.fromkeys(
            int(value)
            for value in np.linspace(0, len(frames) - 1, min(16, len(frames)), dtype=int)
        )
    )
    contact = render_dense_event_sheet(
        frames,
        masks_by_object={key: value.masks for key, value in entities.items()},
        states_by_object={key: value.state for key, value in entities.items()},
        observation_indices=sample_indices,
        source_indices=tuple(
            int(case.timeline["samples"][index]["source_frame_index"])
            for index in sample_indices
        ),
        columns=4,
    )
    if not cv2.imwrite(str(output / "contact_sheet.png"), contact):
        raise RuntimeError(f"cannot write permutation evidence for {case.case_id}")
    (output / "candidate_tracking.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "case_id": case.case_id,
                "model_id": "identity_permutation_of_reviewed_canonical_tubes",
                "anchor_source": "permuted_canonical_observation_zero",
                "new_to_old": mapping,
                "evidence": {"contact_sheet": "contact_sheet.png"},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"case_id": case.case_id, "new_to_old": mapping}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
