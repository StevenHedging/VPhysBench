#!/usr/bin/env python3
"""Build compact, read-only pages from tracking or anchor review evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

from physbench.reference_observations.curation import load_curation_cases
from physbench.reference_observations.curation.review_atlas import (
    render_review_atlas_pages,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument(
        "--evidence",
        choices=("contact-sheet", "independent-anchor"),
        default="contact-sheet",
    )
    parser.add_argument(
        "--review-root",
        type=Path,
        help="Root containing <scene>/<case>/anchor.png audit evidence",
    )
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=720)
    parser.add_argument("--tile-height", type=int, default=480)
    args = parser.parse_args()
    if args.evidence == "independent-anchor" and args.review_root is None:
        parser.error("--review-root is required for independent-anchor evidence")

    requested_scenes = set(args.scene_id)
    requested_cases = set(args.case_id)
    cases = load_curation_cases(
        args.dataset,
        case_ids=requested_cases or None,
    )
    grouped = defaultdict(list)
    for case in cases:
        if requested_scenes and case.scene_id not in requested_scenes:
            continue
        if args.evidence == "contact-sheet":
            evidence_path = (
                case.visualization_manifest_path.parent / "contact_sheet.png"
            )
        else:
            evidence_path = (
                args.review_root / case.scene_id / case.case_id / "anchor.png"
            )
        grouped[case.scene_id].append((case.case_id, evidence_path))
    if not grouped:
        raise ValueError("no release cases matched the requested filters")

    page_size = args.columns * args.rows
    for scene_id, records in grouped.items():
        scene_root = args.output_root / scene_id
        scene_root.mkdir(parents=True, exist_ok=True)
        page_count = 0
        for offset in range(0, len(records), page_size):
            entries = []
            for case_id, evidence_path in records[offset : offset + page_size]:
                image = cv2.imread(str(evidence_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"could not read evidence: {evidence_path}")
                entries.append((case_id, image))
            (page,) = render_review_atlas_pages(
                entries,
                columns=args.columns,
                rows=args.rows,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
            )
            page_count += 1
            index = page_count
            path = scene_root / f"page_{index:03d}.png"
            if not cv2.imwrite(str(path), page):
                raise RuntimeError(f"could not write review page: {path}")
        print(f"{scene_id}: {len(records)} cases -> {page_count} pages")


if __name__ == "__main__":
    main()
