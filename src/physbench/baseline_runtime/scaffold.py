from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..io import write_json


SCENES = [
    "pendulum",
    "free_fall",
    "collision_1d",
    "inclined_plane_slide",
    "uniform_circular_motion",
]


def _adapter(generation_mode: str) -> dict[str, Any]:
    adapter = {
        "kind": "standard",
        "preset": f"standard_{generation_mode}_v1",
        "profile_set": "five_scene_i2v_v1",
        "spatial": {
            "scene_profiles": {
                scene_id: {"width": 832, "height": 480}
                for scene_id in SCENES
            }
        },
        "temporal": {
            "fps": 24,
            "num_frames": 121,
            "valid_frame_rule": "4n+1",
        },
    }
    if generation_mode == "i2v":
        adapter["first_frame_policy"] = "require_asset"
    elif generation_mode == "v2v":
        adapter["video_asset_key"] = "input_video"
    return adapter


def _common(name: str, generation_mode: str) -> dict[str, Any]:
    return {
        "schema_version": "4.0",
        "baseline_id": name,
        "baseline_version": "0.1.0",
        "description": f"Physics Video Benchmark Baseline: {name}.",
        "supported_scenes": SCENES,
        "capabilities": {
            "task_families": ["direct_eval"],
            "conditioning": ["generic", "physics"],
            "generation_modes": [generation_mode],
            "physics_representations": ["structured_text"],
            "train": False,
            "finetune": False,
            "generate": True,
        },
        "model": {"model_id": name, "checkpoint": None},
        "runtime": {},
        "adapter": _adapter(generation_mode),
    }


def create_baseline_scaffold(
    *,
    name: str,
    backend: str,
    root: str | Path,
) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(
            "baseline name may only contain letters, digits, '.', '_' and '-'"
        )
    if backend not in {"managed-i2v", "managed-v2v", "submission"}:
        raise ValueError(
            "baseline backend must be managed-i2v, managed-v2v or submission"
        )
    target = (Path(root).resolve() / name).resolve()
    if target.exists():
        raise FileExistsError(
            f"baseline scaffold target already exists: {target}"
        )
    target.mkdir(parents=True)
    generation_mode = "v2v" if backend == "managed-v2v" else "i2v"
    value = _common(name, generation_mode)
    if backend in {"managed-i2v", "managed-v2v"}:
        driver_module = (
            "subprocess_v2v"
            if backend == "managed-v2v"
            else "subprocess_i2v"
        )
        driver_type = (
            "StandardV2VCLIDriver"
            if backend == "managed-v2v"
            else "StandardI2VCLIDriver"
        )
        value["implementation"] = {
            "kind": "managed",
            "driver": "driver.py",
            "fingerprint_paths": ["*.py"],
        }
        value["runner"] = {
            "type": f"standard_{generation_mode}_cli_v1",
            "config": {
                "command": ["python", "inference.py"],
                "extra_args": [],
            },
        }
        (target / "driver.py").write_text(
            "from physbench.baseline_runtime.drivers."
            f"{driver_module} import {driver_type} as Driver\n",
            encoding="utf-8",
        )
    else:
        value["implementation"] = {
            "kind": "submission",
            "fingerprint_paths": [],
        }
    write_json(target / "baseline.json", value)
    write_json(target / "baseline.local.example.json", {
        "model": {"checkpoint": "/absolute/path/to/checkpoint"},
        "runtime": (
            {}
            if backend in {"managed-i2v", "managed-v2v"}
            else {
                "submission_manifest": (
                    "/absolute/path/to/submission.jsonl"
                )
            }
        ),
    })
    (target / ".gitignore").write_text(
        "baseline.local.json\n__pycache__/\n",
        encoding="utf-8",
    )
    readme = (
        f"# {name}\n\n"
        f"Backend: `{backend}`.\n\n"
        "Copy `baseline.local.example.json` to `baseline.local.json` and "
        "fill only machine-local paths. See the repository operations guide "
        "for the input/output contract.\n"
    )
    (target / "README.md").write_text(readme, encoding="utf-8")
    return target
