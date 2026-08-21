#!/usr/bin/env python3
"""Rebind collision physics and appearance to reviewed left-to-right identities."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from physbench.reference_observations.curation import load_curation_cases
from physbench.reference_observations.curation.finalize import (
    remap_ordered_appearance,
    remap_physics_objects,
    visual_size_order_matches_physics,
)
from physbench.reference_observations.curation.install import refresh_locked_files


_SWAP = {"object_1": "object_2", "object_2": "object_1"}


class SemanticSwapPlan(NamedTuple):
    physics_document: dict[str, Any]
    appearance: dict[str, Any]
    caption: str


def _velocity(physics: Mapping[str, Any], object_id: str) -> float:
    quantity = physics["objects"][object_id].get("initial_velocity")
    if not isinstance(quantity, Mapping):
        raise ValueError(f"{object_id} lacks initial_velocity")
    return float(quantity["value"])


def _collision_caption(physics: Mapping[str, Any]) -> str:
    left_moving = abs(_velocity(physics, "object_1")) > 1e-12
    right_moving = abs(_velocity(physics, "object_2")) > 1e-12
    prefix = (
        "The left ball has mass m_1, radius r_1, and initial speed v_1, "
        "while the right ball has mass m_2, radius r_2, and initial speed v_2. "
    )
    if left_moving and not right_moving:
        motion = "The left ball moves right, the right ball is initially stationary"
    elif right_moving and not left_moving:
        motion = "The left ball is initially stationary, the right ball moves left"
    elif left_moving and right_moving:
        motion = "The left ball moves right and the right ball moves left"
    else:
        motion = "Both balls are initially stationary"
    return prefix + motion + ", and they undergo a one-dimensional central collision."


def plan_two_object_collision_swap(
    *,
    physics_document: Mapping[str, Any],
    appearance: Mapping[str, Any],
    bbox_areas: Mapping[str, int],
) -> SemanticSwapPlan | None:
    """Plan an idempotent whole-object swap when visible size proves reversal."""
    if physics_document.get("scene_id") != "collision_1d":
        raise ValueError("semantic collision swaps require collision_1d")
    physics = physics_document.get("physics")
    if not isinstance(physics, Mapping) or set(physics.get("objects", {})) != set(_SWAP):
        raise ValueError("semantic collision swaps require exactly two objects")
    if visual_size_order_matches_physics(physics, bbox_areas):
        return None
    remapped_physics = remap_physics_objects(physics, new_to_old=_SWAP)
    if not visual_size_order_matches_physics(remapped_physics, bbox_areas):
        raise ValueError("semantic swap did not align physics with reviewed geometry")
    document = copy.deepcopy(dict(physics_document))
    document["physics"] = remapped_physics
    return SemanticSwapPlan(
        physics_document=document,
        appearance=remap_ordered_appearance(appearance, new_to_old=_SWAP),
        caption=_collision_caption(remapped_physics),
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _replace_files_transactionally(payloads: Mapping[Path, bytes]) -> None:
    staged: list[tuple[Path, Path]] = []
    replaced: list[tuple[Path, Path]] = []
    try:
        for target, payload in payloads.items():
            descriptor, temporary_value = tempfile.mkstemp(
                prefix=f".{target.name}.semantic-new-", dir=target.parent
            )
            temporary = Path(temporary_value)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, target))
        try:
            for temporary, target in staged:
                descriptor, backup_value = tempfile.mkstemp(
                    prefix=f".{target.name}.semantic-old-", dir=target.parent
                )
                os.close(descriptor)
                backup = Path(backup_value)
                backup.unlink()
                os.replace(target, backup)
                replaced.append((target, backup))
                os.replace(temporary, target)
        except BaseException:
            for target, backup in reversed(replaced):
                target.unlink(missing_ok=True)
                os.replace(backup, target)
            raise
        for _, backup in replaced:
            backup.unlink(missing_ok=True)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--case-list", required=True, type=Path)
    parser.add_argument("--install", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    case_ids = tuple(arguments.case_list.read_text(encoding="utf-8").split())
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("semantic collision case list contains duplicates")
    cases = load_curation_cases(arguments.dataset, case_ids=case_ids)
    plans: dict[str, SemanticSwapPlan] = {}
    for case in cases:
        bbox_areas = {
            entity.identity.object_id: int(
                (entity.identity.bbox_xyxy[2] - entity.identity.bbox_xyxy[0])
                * (entity.identity.bbox_xyxy[3] - entity.identity.bbox_xyxy[1])
            )
            for entity in case.entities
        }
        physics_path = case.asset_root / case.assets["physics_annotation"]
        plan = plan_two_object_collision_swap(
            physics_document=json.loads(physics_path.read_text(encoding="utf-8")),
            appearance=case.appearance,
            bbox_areas=bbox_areas,
        )
        if plan is not None:
            plans[case.case_id] = plan

    if arguments.install and plans:
        descriptor = json.loads(arguments.dataset.read_text(encoding="utf-8"))
        cases_path = arguments.dataset.parent / descriptor["cases"]
        rows = [
            json.loads(line)
            for line in cases_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        by_id = {case.case_id: case for case in cases}
        payloads: dict[Path, bytes] = {}
        changed_asset_paths: list[str] = []
        for case_id, plan in plans.items():
            case = by_id[case_id]
            physics_path = case.asset_root / case.assets["physics_annotation"]
            caption_path = case.asset_root / case.assets["caption"]
            payloads[physics_path] = _json_bytes(plan.physics_document)
            payloads[caption_path] = _json_bytes(
                {"case_id": case_id, "scene_id": case.scene_id, "caption": plan.caption}
            )
            changed_asset_paths.extend(
                path.relative_to(case.asset_root).as_posix()
                for path in (physics_path, caption_path)
            )
        seen: set[str] = set()
        for row in rows:
            case_id = str(row["case_id"])
            if case_id in plans:
                row["appearance"] = plans[case_id].appearance
                seen.add(case_id)
        if seen != set(plans):
            raise ValueError(f"release misses semantic collision cases: {sorted(set(plans) - seen)}")
        payloads[cases_path] = b"".join(
            (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
        )
        _replace_files_transactionally(payloads)
        refresh_locked_files(
            arguments.dataset.parent / descriptor["asset_lock"],
            cases[0].asset_root,
            changed_asset_paths,
        )
    print(
        json.dumps(
            {
                "requested": len(case_ids),
                "requires_swap": len(plans),
                "already_aligned": len(case_ids) - len(plans),
                "installed": bool(arguments.install and plans),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
