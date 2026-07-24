from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Any

from .io import canonical_sha256


def build_view_a(cases: list[dict[str, Any]]) -> dict[str, Any]:
    scenes: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"train": [], "test_id": [], "test_ood1": []}
    )
    for case in cases:
        scenes[case["scene_id"]][case["view_a_split"]].append(case["case_id"])
    frozen = {
        scene: {split: sorted(ids) for split, ids in groups.items()}
        for scene, groups in sorted(scenes.items())
    }
    all_ids = [case_id for groups in frozen.values() for ids in groups.values() for case_id in ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("view A leakage: a case appears in multiple splits")
    return {
        "schema_version": "1.0",
        "view": "A",
        "case_set_sha256": canonical_sha256(sorted(all_ids)),
        "scenes": frozen,
    }


def _scene_seed(seed: int, scene_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{scene_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def build_view_b(cases: list[dict[str, Any]], groups: int, seed: int) -> dict[str, Any]:
    if groups < 1:
        raise ValueError("groups must be >= 1")
    by_scene: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        by_scene[case["scene_id"]].append(case["case_id"])
    output: dict[str, dict[str, list[str]]] = {}
    for scene_id, ids in sorted(by_scene.items()):
        shuffled = sorted(ids)
        random.Random(_scene_seed(seed, scene_id)).shuffle(shuffled)
        scene_groups = {f"group_{index + 1}": [] for index in range(groups)}
        for index, case_id in enumerate(shuffled):
            scene_groups[f"group_{index % groups + 1}"].append(case_id)
        output[scene_id] = scene_groups
        sizes = [len(items) for items in scene_groups.values()]
        if sizes and max(sizes) - min(sizes) > 1:
            raise AssertionError("unbalanced view B groups")
    return {
        "schema_version": "1.0",
        "view": "B",
        "seed": seed,
        "group_count": groups,
        "case_set_sha256": canonical_sha256(sorted(case["case_id"] for case in cases)),
        "scenes": output,
    }

