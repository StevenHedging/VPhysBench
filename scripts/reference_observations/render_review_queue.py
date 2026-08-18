#!/usr/bin/env python3
"""Render one Case's first-frame identity evidence for curation review."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from physbench.reference_observations.curation import (
    load_curation_cases,
    render_anchor_sheet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _load_anchor(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "masks" not in payload.files:
            raise ValueError(f"anchor NPZ lacks masks: {path}")
        masks = np.asarray(payload["masks"])
    if masks.ndim == 3 and masks.shape[0] == 1:
        masks = masks[0]
    if masks.ndim != 2:
        raise ValueError(f"anchor NPZ must contain one HW mask: {path}")
    return np.where(masks > 0, 1, 0).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    case = load_curation_cases(
        arguments.dataset,
        case_ids={arguments.case_id},
    )[0]
    frame = cv2.imread(str(case.first_frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot decode first frame: {case.first_frame_path}")
    existing = {
        entity.identity.object_id: _load_anchor(entity.anchor_npz_path)
        for entity in case.entities
    }
    rendered = render_anchor_sheet(
        frame,
        existing_masks=existing,
        candidate_masks=existing,
        metadata={"case_id": case.case_id, "scene_id": case.scene_id},
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(arguments.output), rendered):
        raise RuntimeError(f"failed to write review image: {arguments.output}")
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
