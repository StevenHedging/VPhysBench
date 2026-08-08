from __future__ import annotations

from pathlib import Path

from physbench.baseline_runtime import create_baseline_scaffold
from physbench.io import load_json, write_json


def create_generic_baseline_pair(root: Path) -> tuple[Path, Path]:
    generic = create_baseline_scaffold(
        name="generic_fixture",
        backend="managed-i2v",
        root=root,
    )
    physics = create_baseline_scaffold(
        name="physics_fixture",
        backend="managed-i2v",
        root=root,
    )
    for directory in (generic, physics):
        value = load_json(directory / "baseline.json")
        value["supported_scenes"] = "all"
        value["capabilities"].update({
            "task_families": ["direct_eval", "finetune_eval"],
            "train": True,
            "finetune": True,
        })
        value["trainer"] = {"type": "fixture", "config": {}}
        write_json(directory / "baseline.json", value)

    value = load_json(physics / "baseline.json")
    value["input_policy"]["physics"] = {
        "source": "case.physics",
        "usage": "required",
        "representations": ["structured_text"],
    }
    value["adapter"]["physics_transform"] = {
        "type": "append_structured_text_v1",
        "template_set": "six_scene_physics_clauses_v2",
    }
    write_json(physics / "baseline.json", value)
    return generic / "baseline.json", physics / "baseline.json"
