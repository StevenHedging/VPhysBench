#!/usr/bin/env python3
"""Build Dataset 4.0.0 by attaching canonical case text to release 3.0.0."""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.data_layout import V3_DATASET, V4_RELEASE_ROOT  # noqa: E402
from physbench.datasets import load_dataset  # noqa: E402
from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)


DATASET_ID = "physics_video_five_scene_v4"
RELEASE = "4.0.0"
TEXT_ANNOTATION_SOURCE = "five_scene_prompt_v1"
SOURCE_PROMPT_PROFILE = (
    "src/physbench/baseline_plugins/resources/"
    "five_scene_i2v_v1/generic.json"
)

# Immutable snapshot of the generic profile used before prompts became
# Dataset-owned in release 4.0.0.  Keeping the migration input here makes the
# historical release reproducible after the old runtime resource is removed.
SOURCE_GENERIC_PROFILE: dict[str, Any] = {
    "schema_version": "1.0",
    "profile_id": "generic",
    "display_name": "Five-scene generic I2V text adaptation",
    "language": "en",
    "includes_physical_parameters": False,
    "scenes": {
        "pendulum": {
            "base_prompt": (
                "A fixed-camera real-world laboratory video of a simple "
                "pendulum released from rest at the first frame. The support "
                "and camera remain stationary, and the bob swings naturally "
                "under gravity."
            ),
            "parameter_clauses": [],
        },
        "free_fall": {
            "base_prompt": (
                "A fixed-camera real-world laboratory video of a spherical "
                "ball released from rest in free fall. The camera remains "
                "stationary, and the ball accelerates vertically downward "
                "under gravity at the true physical time scale."
            ),
            "parameter_clauses": [],
        },
        "collision_1d": {
            "base_prompt": (
                "A fixed-camera real-world laboratory video of a "
                "one-dimensional central collision among three aligned "
                "balls. At frame 0, all three balls are visible, with ball 1 "
                "on the left. Ball 1 moves toward initially stationary balls "
                "2 and 3. The camera and track remain stationary, and the "
                "collision unfolds at the true physical time scale."
            ),
            "parameter_clauses": [],
        },
        "inclined_plane_slide": {
            "base_prompt": (
                "A fixed-camera real-world laboratory video of a block "
                "released from rest at the top of a wooden inclined plane. "
                "Frame 0 is immediately before or at the onset of sliding. "
                "The block accelerates down the track under gravity while "
                "maintaining contact with the surface."
            ),
            "parameter_clauses": [],
        },
        "uniform_circular_motion": {
            "base_prompt": (
                "A fixed overhead real-world laboratory video of one or more "
                "blocks undergoing uniform circular motion on a green "
                "rotating disk. The rotation center and camera remain fixed, "
                "and each block maintains an approximately constant orbit "
                "radius and angular speed."
            ),
            "parameter_clauses": [],
        },
    },
}


def _validate_prompt_profile(profile: dict[str, Any]) -> dict[str, str]:
    if profile.get("schema_version") != "1.0":
        raise ValueError("prompt profile must use schema_version=1.0")
    if profile.get("profile_id") != "generic":
        raise ValueError("Dataset v4 must be built from the generic prompt profile")
    if profile.get("language") != "en":
        raise ValueError("Dataset v4 currently requires an English prompt profile")
    if profile.get("includes_physical_parameters") is not False:
        raise ValueError("canonical case text must not inject physical parameters")
    scenes = profile.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("prompt profile scenes must be a non-empty object")

    prompts: dict[str, str] = {}
    for scene_id, scene in scenes.items():
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("prompt profile contains an invalid scene ID")
        if not isinstance(scene, dict):
            raise ValueError(f"prompt profile scene {scene_id} must be an object")
        prompt = scene.get("base_prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"prompt profile scene {scene_id} lacks base_prompt")
        clauses = scene.get("parameter_clauses")
        if clauses not in (None, []):
            raise ValueError(
                f"generic prompt scene {scene_id} unexpectedly injects parameters"
            )
        prompts[scene_id] = prompt.strip()
    return prompts


def _migrate_cases(
    source_cases: list[dict[str, Any]],
    prompts: dict[str, str],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_case in source_cases:
        case = copy.deepcopy(source_case)
        case_id = case.get("case_id")
        scene_id = case.get("scene_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("source release contains a case without case_id")
        if case_id in seen:
            raise ValueError(f"source release contains duplicate case ID {case_id}")
        seen.add(case_id)
        if not isinstance(scene_id, str) or scene_id not in prompts:
            raise ValueError(
                f"case {case_id} has no generic prompt for scene {scene_id!r}"
            )
        if "text" in case:
            raise ValueError(f"source case {case_id} already contains text")
        if case.get("schema_version") != "2.0":
            raise ValueError(f"source case {case_id} must use schema_version=2.0")
        case["schema_version"] = "3.0"
        case["text"] = {
            "schema_version": "1.0",
            "prompt": prompts[scene_id],
            "language": "en",
            "annotation_source": TEXT_ANNOTATION_SOURCE,
        }
        cases.append(case)
    return cases


def _copy_release_metadata(source_root: Path, output: Path) -> None:
    for directory_name in ("scenes", "views"):
        source_directory = source_root / directory_name
        if not source_directory.is_dir():
            raise FileNotFoundError(source_directory)
        shutil.copytree(source_directory, output / directory_name)
    source_audit = source_root / "expansion_audit.json"
    if not source_audit.is_file():
        raise FileNotFoundError(source_audit)
    shutil.copy2(source_audit, output / source_audit.name)


def _build_readme(case_count: int, asset_count: int) -> str:
    return f"""# Physics Video Dataset 4.0.0

这是正式五场景 DatasetSnapshot 的 case-owned text release：

```text
dataset_id: physics_video_five_scene_v4
cases:      {case_count}
assets:     {asset_count}
```

相对 3.0.0，本 release 不改 case 集合、场景、视图或资产，只进行数据契约迁移：

- 每个 case 升级到 schema 3.0；
- `text.prompt` 固化自旧 generic profile 的场景通用描述，其完整迁移快照保存在
  `scripts/migrate_dataset_v4.py`；
- 结构化 `physics` 与原始文本并列保存，由 Baseline 自行决定是否以及如何使用；
- `assets.lock.json` 仅更新 dataset identity，`files` 保持不变。

文件：

- `dataset.json`：唯一加载入口；
- `release.json`：loader-compatible Dataset 与资产集合 digest；
- `cases.jsonl`：带原始文本、媒体与结构化物理标注的 case；
- `migration_audit.json`：3.0.0 到 4.0.0 的迁移审计；
- `expansion_audit.json`、`scenes/`、`views/`：逐字节复制自 3.0.0。

验收：

```bash
PYTHONPATH=src /root/miniconda3/envs/phybench/bin/python \\
  scripts/migrate_dataset_v4.py --force
```
"""


def build_release(
    source_descriptor_path: Path,
    output: Path,
) -> str:
    source_descriptor_path = source_descriptor_path.resolve()
    output = output.resolve()
    source_root = source_descriptor_path.parent
    if output == source_root:
        raise ValueError("output must not overwrite the source release")

    source_descriptor = load_json(source_descriptor_path)
    if source_descriptor.get("dataset_id") != "physics_video_five_scene_v3":
        raise ValueError("source dataset must be physics_video_five_scene_v3")
    if source_descriptor.get("release") != "3.0.0":
        raise ValueError("source dataset must be release 3.0.0")
    if source_descriptor.get("schema_version") != "2.0":
        raise ValueError("source descriptor must use schema_version=2.0")

    profile = copy.deepcopy(SOURCE_GENERIC_PROFILE)
    prompts = _validate_prompt_profile(profile)
    source_cases = load_jsonl(source_root / source_descriptor["cases"])
    cases = _migrate_cases(source_cases, prompts)
    case_scene_ids = {case["scene_id"] for case in cases}
    unused_prompt_scenes = sorted(set(prompts) - case_scene_ids)
    if unused_prompt_scenes:
        raise ValueError(
            f"prompt profile has scenes absent from the release: {unused_prompt_scenes}"
        )

    output.mkdir(parents=True)
    _copy_release_metadata(source_root, output)
    write_jsonl(output / "cases.jsonl", cases)

    source_lock = load_json(source_root / source_descriptor["asset_lock"])
    if source_lock.get("dataset_id") != source_descriptor["dataset_id"]:
        raise ValueError("source asset lock dataset_id mismatch")
    if source_lock.get("release") != source_descriptor["release"]:
        raise ValueError("source asset lock release mismatch")
    files = copy.deepcopy(source_lock.get("files"))
    if not isinstance(files, list):
        raise ValueError("source asset lock files must be a list")
    if source_lock.get("files_digest") != canonical_sha256(files):
        raise ValueError("source asset lock files_digest mismatch")
    asset_lock = copy.deepcopy(source_lock)
    asset_lock["dataset_id"] = DATASET_ID
    asset_lock["release"] = RELEASE
    write_json(output / "assets.lock.json", asset_lock)

    descriptor = {
        "schema_version": "3.0",
        "dataset_id": DATASET_ID,
        "release": RELEASE,
        "cases": "cases.jsonl",
        "asset_root": "../..",
        "asset_lock": "assets.lock.json",
        "release_manifest": "release.json",
        "scene_catalog": "scenes",
        "views": {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        },
    }
    write_json(output / "dataset.json", descriptor)

    scene_configs: dict[str, dict[str, Any]] = {}
    for path in sorted((output / "scenes").glob("*.json")):
        value = load_json(path)
        scene_id = value.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError(f"scene config lacks scene_id: {path}")
        if scene_id in scene_configs:
            raise ValueError(f"duplicate scene config: {scene_id}")
        scene_configs[scene_id] = value
    views = {
        view_id: load_json(output / relative_path)
        for view_id, relative_path in descriptor["views"].items()
    }
    dataset_digest = canonical_sha256(
        {
            "descriptor": descriptor,
            "cases": tuple(cases),
            "views": views,
            "scenes": scene_configs,
            "asset_lock": asset_lock,
        }
    )
    release_manifest = {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "release": RELEASE,
        "dataset_digest": dataset_digest,
        "asset_files": len(files),
        "asset_files_digest": asset_lock["files_digest"],
    }
    write_json(output / "release.json", release_manifest)

    write_json(
        output / "migration_audit.json",
        {
            "schema_version": "1.0",
            "source_dataset": "../3.0.0/dataset.json",
            "source_dataset_id": source_descriptor["dataset_id"],
            "source_release": source_descriptor["release"],
            "source_case_schema_version": "2.0",
            "target_case_schema_version": "3.0",
            "source_prompt_profile": (
                "../../../../" + SOURCE_PROMPT_PROFILE
            ),
            "prompt_profile_storage": "migration_script_snapshot",
            "prompt_profile_digest": canonical_sha256(profile),
            "text_annotation_source": TEXT_ANNOTATION_SOURCE,
            "case_count": len(cases),
            "case_ids_sha256": canonical_sha256(
                sorted(case["case_id"] for case in cases)
            ),
            "scene_prompt_count": len(prompts),
            "added_case_fields": ["text"],
            "copied_metadata": [
                "expansion_audit.json",
                "scenes",
                "views",
            ],
            "assets_copied": False,
            "source_assets_mutated": False,
            "asset_lock_files_unchanged": files == source_lock["files"],
            "dataset_digest": dataset_digest,
        },
    )
    (output / "README.md").write_text(
        _build_readme(len(cases), len(files)),
        encoding="utf-8",
    )

    snapshot = load_dataset(output / "dataset.json")
    if snapshot.digest != dataset_digest:
        raise ValueError(
            f"loader digest mismatch: expected={dataset_digest}, "
            f"actual={snapshot.digest}"
        )
    return dataset_digest


def _validate_force_target(
    requested_output: Path,
    source_descriptor_path: Path,
    *,
    releases_root: Path = V4_RELEASE_ROOT.parent,
    expected_dataset_id: str = DATASET_ID,
    expected_release: str = RELEASE,
) -> Path:
    """Return a verified release directory that may be recursively removed.

    ``--force`` is intentionally narrower than ``build_release``: destructive
    replacement is only allowed for the expected, direct child of the
    repository's release root, and only when the existing bundle identifies
    itself consistently in all three identity-bearing manifests.
    """

    requested_output = Path(requested_output)
    if requested_output.is_symlink():
        raise ValueError(
            f"refusing to recursively remove a symlink: {requested_output}"
        )
    output = requested_output.resolve()
    allowed_root = Path(releases_root).resolve()
    if output.parent != allowed_root or output.name != expected_release:
        raise ValueError(
            "--force may only replace the expected release directory "
            f"{allowed_root / expected_release}; got {output}"
        )
    if output == Path(source_descriptor_path).resolve().parent:
        raise ValueError("refusing to remove the source release")
    if not output.is_dir():
        raise ValueError(
            f"--force target must be an existing release directory: {output}"
        )

    identity_files = {
        "dataset.json": "3.0",
        "assets.lock.json": "1.0",
        "release.json": "1.0",
    }
    for filename, schema_version in identity_files.items():
        path = output / filename
        if not path.is_file():
            raise ValueError(
                f"refusing to remove an unverified release; missing {path}"
            )
        value = load_json(path)
        if value.get("schema_version") != schema_version:
            raise ValueError(
                f"refusing to remove release with unexpected {filename} "
                f"schema_version={value.get('schema_version')!r}"
            )
        if value.get("dataset_id") != expected_dataset_id:
            raise ValueError(
                f"refusing to remove release with unexpected {filename} "
                f"dataset_id={value.get('dataset_id')!r}"
            )
        if value.get("release") != expected_release:
            raise ValueError(
                f"refusing to remove release with unexpected {filename} "
                f"release={value.get('release')!r}"
            )
    return output


def _remove_verified_release(
    requested_output: Path,
    source_descriptor_path: Path,
    *,
    releases_root: Path = V4_RELEASE_ROOT.parent,
    expected_dataset_id: str = DATASET_ID,
    expected_release: str = RELEASE,
) -> Path:
    output = _validate_force_target(
        requested_output,
        source_descriptor_path,
        releases_root=releases_root,
        expected_dataset_id=expected_dataset_id,
        expected_release=expected_release,
    )
    shutil.rmtree(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=V3_DATASET,
        help="Dataset 3.0.0 descriptor",
    )
    parser.add_argument("--output", type=Path, default=V4_RELEASE_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    requested_output = args.output
    output = requested_output.resolve()
    digest: str
    if output.exists():
        if not args.force:
            raise FileExistsError(
                f"refusing to overwrite existing Dataset bundle: {output}"
            )
        output = _validate_force_target(
            requested_output,
            args.source_dataset,
        )
        # Build and validate the replacement first. A malformed source must
        # never cause a known-good target release to be deleted.
        with tempfile.TemporaryDirectory(
            prefix=f".{RELEASE}.migration-",
            dir=output.parent,
        ) as temporary:
            staged_output = Path(temporary) / RELEASE
            digest = build_release(
                args.source_dataset,
                staged_output,
            )
            _remove_verified_release(
                output,
                args.source_dataset,
            )
            staged_output.rename(output)
    else:
        digest = build_release(
            args.source_dataset,
            output,
        )
    print(
        f"{output / 'dataset.json'} cases="
        f"{len(load_jsonl(output / 'cases.jsonl'))} dataset_digest={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
