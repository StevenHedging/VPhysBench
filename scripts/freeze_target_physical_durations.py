#!/usr/bin/env python3
"""Freeze per-Case output-duration targets from canonical reference videos."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from physbench.baseline_runtime.media_contract import probe_media


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(
            "datasets/physics_video/releases/7.0.0/cases.jsonl"
        ),
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def _reference_case(
    case: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if (
        case.get("has_real_reference_video", False)
        and case.get("assets", {}).get("physics_reference_video")
    ):
        return case
    parent_id = case.get("provenance", {}).get("parent_case_id")
    parent = catalog.get(parent_id) if parent_id else None
    if parent is None or parent.get("physics") != case.get("physics"):
        raise ValueError(
            f"case {case['case_id']} has no physics-identical reference"
        )
    return parent


def _duration(
    case: dict[str, Any],
    *,
    catalog: dict[str, dict[str, Any]],
    asset_root: Path,
) -> float:
    reference = _reference_case(case, catalog)
    relative = reference.get("assets", {}).get(
        "physics_reference_video"
    )
    if not isinstance(relative, str) or not relative:
        raise ValueError(
            f"case {case['case_id']} reference has no physics video"
        )
    path = (asset_root / relative).resolve()
    path.relative_to(asset_root.resolve())
    probe = probe_media(path, count_frames=False)
    if probe.get("frames") is None:
        probe = probe_media(path, count_frames=True)
    frames, fps = probe.get("frames"), probe.get("fps")
    if frames is None or int(frames) < 2 or fps is None:
        raise ValueError(f"cannot resolve last-frame timestamp for {path}")
    scale = float(
        reference.get("temporal", {}).get(
            "encoded_to_physical_speed", 1.0
        )
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            f"case {case['case_id']} has invalid physical time scale"
        )
    return round((int(frames) - 1) / float(fps) / scale, 12)


def main() -> None:
    args = _arguments()
    cases_path = args.cases.resolve()
    asset_root = cases_path.parents[2]
    rows = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = {case["case_id"]: case for case in rows}
    changed = 0
    values: list[float] = []
    for case in rows:
        value = _duration(
            case,
            catalog=catalog,
            asset_root=asset_root,
        )
        values.append(value)
        previous = case.setdefault("temporal", {}).get(
            "target_physical_duration_s"
        )
        if previous is None or abs(float(previous) - value) > 1e-9:
            changed += 1
            case["temporal"]["target_physical_duration_s"] = value
    if args.check:
        if changed:
            raise SystemExit(
                f"{changed} Cases have missing or stale duration targets"
            )
    elif changed:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cases_path.parent,
            prefix=f".{cases_path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for case in rows:
                handle.write(
                    json.dumps(
                        case,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        os.replace(temporary, cases_path)
    print(
        {
            "cases": len(rows),
            "changed": changed,
            "minimum_target_physical_duration_s": min(values),
            "maximum_target_physical_duration_s": max(values),
        }
    )


if __name__ == "__main__":
    main()
