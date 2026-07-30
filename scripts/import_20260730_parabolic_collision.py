#!/usr/bin/env python3
"""Publish the reviewed parabolic-motion and collision supplement.

The importer consumes only visually approved staging records produced by
``prepare_20260730_parabolic_collision.py``.  It preserves both uploaded ZIP
files byte-for-byte, materializes immutable source/canonical assets, records
annotation and cleaning provenance, and derives Dataset release 5.0.0 without
mutating release 4.0.0.
"""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.datasets import load_dataset  # noqa: E402
from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)
from physbench.splitters import build_view_b  # noqa: E402
from build_dataset_asset_lock import rebuild_asset_lock  # noqa: E402


DATA_ROOT = ROOT / "datasets" / "physics_video"
ASSET_ROOT = DATA_ROOT / "assets"
RELEASES_ROOT = DATA_ROOT / "releases"
BASE_ROOT = RELEASES_ROOT / "4.0.0"
OUTPUT_ROOT = RELEASES_ROOT / "5.0.0"
DATASET_ID = "physics_video_six_scene_v5"
RELEASE = "5.0.0"

WORK_ROOT = Path(
    "/mnt/nvme1/physics_video_benchmark/import_work_20260730/prepared_v1"
)
ALIGNMENT_INPUT = WORK_ROOT / "alignment"
STAGED_ASSETS = WORK_ROOT / "assets"
STAGED_REVIEW = WORK_ROOT / "review"

PARABOLIC_ARCHIVE_INPUT = Path(
    "/root/Brady/data/source_archives/parabolic_motion/parabolic_motion.zip"
)
COLLISION_ARCHIVE_INPUT = Path(
    "/root/Brady/collision/补充_碰撞试验7_29.zip"
)
PARABOLIC_ARCHIVE_SHA256 = (
    "d53e2d36cc2f8172825eae5bf8e9c3483cb1bed1e4066a085fe099e654862331"
)
COLLISION_ARCHIVE_SHA256 = (
    "15dc8d638ebd0674784f9d518cfee605aed11952b72465600cb8d038344517e1"
)
SOURCE_ARCHIVE_ROOT = (
    ASSET_ROOT / "source_archives" / "20260730_parabolic_collision"
)
PARABOLIC_ARCHIVE = SOURCE_ARCHIVE_ROOT / "parabolic_motion.zip"
COLLISION_ARCHIVE = SOURCE_ARCHIVE_ROOT / "collision_supplement_20260729.zip"

PARABOLIC_SOURCE_ROOT = (
    Path("/root/Brady/data/assets/parabolic_motion") / "平抛运动"
)
PARABOLIC_WORKBOOK_ROOT = Path(
    "/root/Brady/data/assets/parabolic_motion"
)
PARABOLIC_ANNOTATION_INPUTS = {
    "normalized_annotations.jsonl": Path(
        "/root/Brady/data/manifests/parabolic_parameter_import_v2.jsonl"
    ),
    "normalized_annotations.csv": Path(
        "/root/Brady/data/source_docs/"
        "parabolic_motion_parameters_imported_v2.csv"
    ),
    "annotation_import_report.md": Path(
        "/root/Brady/data/PARABOLIC_PARAMETER_IMPORT_REPORT_20260729.md"
    ),
}

COLLISION_BATCH_ROOT = Path(
    "/root/Brady/data/assets/collision_1d_supplement_20260729"
)
COLLISION_SOURCE_ROOT = COLLISION_BATCH_ROOT / "raw"
COLLISION_ANNOTATION_ROOT = COLLISION_BATCH_ROOT / "annotations"
COLLISION_CURATION_AUDIT = (
    COLLISION_BATCH_ROOT / "processed_v1/metadata/curation_audit.jsonl"
)
COLLISION_CURATION_SUMMARY = (
    COLLISION_BATCH_ROOT / "processed_v1/metadata/curation_summary.json"
)
COLLISION_INGEST_AUDIT = (
    COLLISION_BATCH_ROOT / "metadata/ingest_audit.json"
)

DOC_ROOT = (
    DATA_ROOT / "provenance" / "source_docs"
    / "20260730_parabolic_collision"
)
PROVENANCE_ALIGNMENT_ROOT = (
    DATA_ROOT / "provenance" / "alignment"
    / "photogate_cleaning_20260730_v1"
)
IMPORT_AUDIT = (
    DATA_ROOT / "provenance" / "imports"
    / "parabolic_collision_20260730_import_audit.jsonl"
)
EXCLUSION_AUDIT = (
    DATA_ROOT / "provenance" / "imports"
    / "parabolic_collision_20260730_exclusions.json"
)

PARABOLIC_PROMPT = (
    "A fixed-camera real-world laboratory video of a shiny steel ball in "
    "horizontal projectile motion. At frame 0, the ball has just cleared the "
    "launcher at the right side of the image and moves leftward with an "
    "initially horizontal velocity. The launcher is outside the crop, the "
    "camera remains stationary, and the ball follows a continuous downward "
    "parabolic trajectory under gravity at the true physical time scale."
)
COLLISION_PROMPTS = {
    "two_ball_single_incident": (
        "A fixed-camera real-world laboratory video of a one-dimensional "
        "central collision between two aligned shiny steel balls. At frame 0, "
        "both balls are fully visible: the striker is on the right and moves "
        "left toward the initially stationary target on the left. The camera "
        "and track remain stationary, the balls remain distinct, and the "
        "collision unfolds at the true physical time scale."
    ),
    "two_ball_opposed_incident": (
        "A fixed-camera real-world laboratory video of a one-dimensional "
        "central collision between two aligned shiny steel balls moving "
        "toward each other. At frame 0, both balls are fully visible: the left "
        "ball moves right and the right ball moves left. The camera and track "
        "remain stationary, the balls remain distinct, and the collision "
        "unfolds at the true physical time scale."
    ),
    "three_ball_single_incident": (
        "A fixed-camera real-world laboratory video of a one-dimensional "
        "central collision among three aligned shiny steel balls. At frame 0, "
        "all three balls are fully visible: the left ball moves right toward "
        "two initially stationary touching target balls. The camera and track "
        "remain stationary, the balls remain distinct, and the collision "
        "unfolds at the true physical time scale."
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_verified(
    source: Path,
    target: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Copy one immutable file and refuse any conflicting existing target."""

    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest = sha256(source)
    if expected_sha256 is not None and source_digest != expected_sha256:
        raise ValueError(
            f"source digest mismatch for {source}: "
            f"expected={expected_sha256}, actual={source_digest}"
        )
    if target.exists():
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"immutable target is not a regular file: {target}")
        target_digest = sha256(target)
        if target_digest != source_digest:
            raise ValueError(
                f"refusing to overwrite conflicting asset {target}: "
                f"existing={target_digest}, incoming={source_digest}"
            )
        return source_digest
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.import-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        copied_digest = sha256(temporary)
        if copied_digest != source_digest:
            raise ValueError(f"copy verification failed for {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return source_digest


def asset_path(path: Path) -> str:
    return str(path.relative_to(DATA_ROOT))


def quantity(
    value: float,
    unit: str,
    *,
    annotated: bool = True,
) -> dict[str, Any]:
    return {
        "value": float(value),
        "unit": unit,
        "annotated": annotated,
    }


def copy_quantity(value: dict[str, Any]) -> dict[str, Any]:
    return quantity(
        float(value["value"]),
        str(value["unit"]),
        annotated=bool(value["annotated"]),
    )


def load_reviewed_records(stem: str, expected: int) -> list[dict[str, Any]]:
    path = ALIGNMENT_INPUT / f"{stem}_encoded.jsonl"
    records = load_jsonl(path)
    if len(records) != expected:
        raise ValueError(f"{stem} expected {expected} records, found {len(records)}")
    identifiers = [item["case_id"] for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{stem} staging contains duplicate case IDs")
    pending = [
        item["case_id"]
        for item in records
        if item.get("review_status") != "visually_verified"
    ]
    if pending:
        raise ValueError(f"{stem} has unapproved records: {pending[:5]}")
    for item in records:
        assets = item["staged_assets"]
        video = Path(assets["reference_video"])
        first = Path(assets["first_frame"])
        if sha256(video) != assets["reference_video_sha256"]:
            raise ValueError(f"staged video changed after review: {video}")
        if sha256(first) != assets["first_frame_sha256"]:
            raise ValueError(f"staged first frame changed after review: {first}")
    return records


def verify_archive_inventory(
    records: list[dict[str, Any]],
    archive: Path,
    *,
    member_prefix: str,
) -> dict[int, str]:
    with zipfile.ZipFile(archive) as bundle:
        infos = {item.filename: item for item in bundle.infolist()}
    members: dict[int, str] = {}
    for item in records:
        number = int(item["image_number"])
        member = f"{member_prefix}/IMG_{number:04d}.MOV"
        info = infos.get(member)
        if info is None:
            raise ValueError(f"archive member missing for {item['case_id']}: {member}")
        source = Path(item["source_video"])
        if info.file_size != source.stat().st_size:
            raise ValueError(
                f"archive/source size mismatch for {item['case_id']}: "
                f"archive={info.file_size}, source={source.stat().st_size}"
            )
        members[number] = member
    return members


def materialize_case_assets(
    record: dict[str, Any],
) -> dict[str, str]:
    scene_id = record["scene_id"]
    case_id = record["case_id"]
    root = ASSET_ROOT / scene_id / case_id
    source_target = root / "source" / "reference.mov"
    canonical_video = root / "canonical" / "reference.mp4"
    canonical_first = root / "canonical" / "first_frame.png"
    source_digest = copy_verified(
        Path(record["source_video"]),
        source_target,
        expected_sha256=record["source_sha256"],
    )
    staged = record["staged_assets"]
    video_digest = copy_verified(
        Path(staged["reference_video"]),
        canonical_video,
        expected_sha256=staged["reference_video_sha256"],
    )
    first_digest = copy_verified(
        Path(staged["first_frame"]),
        canonical_first,
        expected_sha256=staged["first_frame_sha256"],
    )
    return {
        "source_video": asset_path(source_target),
        "reference_video": asset_path(canonical_video),
        "first_frame": asset_path(canonical_first),
        "source_sha256": source_digest,
        "reference_video_sha256": video_digest,
        "first_frame_sha256": first_digest,
    }


def parabolic_case(
    record: dict[str, Any],
    assets: dict[str, str],
    archive_member: str,
) -> dict[str, Any]:
    annotation = record["annotation"]
    physics = {
        "ball_radius": quantity(annotation["ball_diameter_m"] / 2.0, "m"),
        "ball_mass": quantity(annotation["ball_mass_kg"], "kg"),
        "launch_height": quantity(annotation["launch_height_m"], "m"),
        "initial_horizontal_velocity": quantity(
            annotation["initial_horizontal_velocity_m_per_s"],
            "m/s",
        ),
        "photogate_block_time": quantity(
            annotation["interpreted_block_time_s"],
            "s",
        ),
        "photogate_distance_before_launch": quantity(
            annotation["photogate_distance_before_launch_m"],
            "m",
        ),
    }
    if annotation.get("ramp_angle_deg") is not None:
        physics["ramp_angle"] = quantity(annotation["ramp_angle_deg"], "deg")
    if annotation.get("release_distance_m") is not None:
        physics["release_distance"] = quantity(
            annotation["release_distance_m"],
            "m",
        )
    quality_flags = list(record.get("quality_flags", []))
    appearance = {
        "ball_material": "steel",
        "ball_size_class": (
            "medium"
            if annotation["object_kind_zh"] == "中型钢球"
            else "large"
        ),
        "ball_source_label_zh": annotation["object_kind_zh"],
        "background": (
            annotation["background_description"]
            if annotation["background_changed"]
            else "default_lab_background"
        ),
        "camera": "fixed",
        "launch_side": "right",
        "motion_direction": "left",
        "capture_session": "20260729_parabolic_motion",
    }
    source_workbook = Path(annotation["source_workbook"]).name
    return {
        "schema_version": "3.0",
        "case_id": record["case_id"],
        "scene_id": "parabolic_motion",
        "text": {
            "schema_version": "1.0",
            "prompt": PARABOLIC_PROMPT,
            "language": "en",
            "annotation_source": "parabolic_motion_prompt_v1",
        },
        "assets": {
            "first_frame": assets["first_frame"],
            "reference_video": assets["reference_video"],
            "physics_reference_video": assets["reference_video"],
            "source_video": assets["source_video"],
            "source_archive": asset_path(PARABOLIC_ARCHIVE),
            "subject_mask": None,
        },
        "physics": physics,
        "appearance": appearance,
        "temporal": {
            "encoded_to_physical_speed": 1.0,
            "time_scale": "real_time",
            "annotation_source": (
                "native high-frame-rate capture normalized to 24 fps while "
                "preserving physical time"
            ),
        },
        "alignment": {
            "version": "photogate_cleaning_20260730_v1",
            "method": record["alignment"]["method"],
            "review_status": record["review_status"],
            "source_start_frame": record["alignment"]["source_start_frame"],
            "source_end_frame_exclusive": (
                record["alignment"]["source_end_frame_exclusive"]
            ),
            "source_fps": record["alignment"]["source_fps"],
            "source_crop": record["alignment"]["crop"],
            "output_size": {
                "width": record["alignment"]["output_width"],
                "height": record["alignment"]["output_height"],
            },
            "audit_record": (
                "provenance/alignment/photogate_cleaning_20260730_v1/"
                "parabolic_encoded.jsonl"
            ),
        },
        "provenance": {
            "source_kind": "real_capture",
            "parent_case_id": None,
            "generator": None,
            "source_locator": {
                "archive": asset_path(PARABOLIC_ARCHIVE),
                "member": archive_member,
                "annotation_manifest": (
                    "provenance/source_docs/20260730_parabolic_collision/"
                    "parabolic/normalized_annotations.jsonl"
                ),
                "annotation_workbook": (
                    "provenance/source_docs/20260730_parabolic_collision/"
                    f"parabolic/{source_workbook}"
                ),
                "sheet": annotation["source_sheet"],
                "row": annotation["source_row"],
                "trial_id": annotation["trial_id"],
                "mapping_correction": annotation["mapping_correction"],
                "object_kind_correction": annotation["object_kind_correction"],
                "quality_flags": quality_flags,
            },
        },
        "ood": {"level": "id", "factors": []},
        "has_real_reference_video": True,
    }


def collision_balls(
    physical: dict[str, Any],
) -> tuple[list[dict[str, dict[str, Any]]], list[int]]:
    structure = physical["collision_structure"]
    if structure == "two_ball_single_incident":
        return [
            {
                "mass": physical["target_mass"],
                "diameter": physical["target_diameter"],
                "velocity": physical["target_initial_velocity"],
            },
            {
                "mass": physical["striker_mass"],
                "diameter": physical["striker_diameter"],
                "velocity": physical["striker_initial_velocity"],
            },
        ], [2]
    if structure == "two_ball_opposed_incident":
        return [
            {
                "mass": physical["left_ball_mass"],
                "diameter": physical["left_ball_diameter"],
                "velocity": physical["left_ball_initial_velocity"],
            },
            {
                "mass": physical["right_ball_mass"],
                "diameter": physical["right_ball_diameter"],
                "velocity": physical["right_ball_initial_velocity"],
            },
        ], [1, 2]
    if structure == "three_ball_single_incident":
        return [
            {
                "mass": physical["striker_mass"],
                "diameter": physical["striker_diameter"],
                "velocity": physical["striker_initial_velocity"],
            },
            {
                "mass": physical["target_1_mass"],
                "diameter": physical["target_1_diameter"],
                "velocity": physical["target_1_initial_velocity"],
            },
            {
                "mass": physical["target_2_mass"],
                "diameter": physical["target_2_diameter"],
                "velocity": physical["target_2_initial_velocity"],
            },
        ], [1]
    raise ValueError(f"unknown collision structure: {structure}")


def ball_label(ball: dict[str, dict[str, Any]]) -> str:
    diameter_mm = float(ball["diameter"]["value"]) * 1000.0
    mass_g = float(ball["mass"]["value"]) * 1000.0
    diameter = f"{diameter_mm:g}".replace(".", "p")
    mass = f"{mass_g:g}".replace(".", "p")
    return f"steel_ball_d{diameter}mm_m{mass}g"


def collision_case(
    record: dict[str, Any],
    assets: dict[str, str],
    archive_member: str,
) -> dict[str, Any]:
    physical = record["physical_parameters"]
    structure = physical["collision_structure"]
    balls, striker_indices = collision_balls(physical)
    physics: dict[str, Any] = {}
    for index, ball in enumerate(balls, 1):
        physics[f"ball_{index}_mass"] = copy_quantity(ball["mass"])
        physics[f"ball_{index}_radius"] = quantity(
            float(ball["diameter"]["value"]) / 2.0,
            "m",
            annotated=bool(ball["diameter"]["annotated"]),
        )
        physics[f"ball_{index}_initial_velocity"] = copy_quantity(
            ball["velocity"]
        )
    physics["striker_initial_velocity"] = copy.deepcopy(
        physics[f"ball_{striker_indices[0]}_initial_velocity"]
    )
    signatures = {
        (
            physics[f"ball_{index}_mass"]["value"],
            physics[f"ball_{index}_radius"]["value"],
        )
        for index in range(1, len(balls) + 1)
    }
    homogeneous = len(signatures) == 1
    source_annotation = record["source_annotation"]
    workbook = Path(source_annotation["workbook"]).name
    appearance: dict[str, Any] = {
        "ball_materials": ["steel"] * len(balls),
        "ball_sequence": [ball_label(ball) for ball in balls],
        "camera": "fixed",
        "background": "laboratory_track",
        "capture_session": "20260729_collision_supplement",
        "collision_structure": structure,
        "striker_ball_index": striker_indices[0],
    }
    if len(striker_indices) > 1:
        appearance["opposing_striker_ball_index"] = striker_indices[1]
    return {
        "schema_version": "3.0",
        "case_id": record["case_id"],
        "scene_id": "collision_1d",
        "text": {
            "schema_version": "1.0",
            "prompt": COLLISION_PROMPTS[structure],
            "language": "en",
            "annotation_source": "collision_supplement_prompt_v1",
        },
        "assets": {
            "first_frame": assets["first_frame"],
            "reference_video": assets["reference_video"],
            "physics_reference_video": assets["reference_video"],
            "source_video": assets["source_video"],
            "source_archive": asset_path(COLLISION_ARCHIVE),
            "subject_mask": None,
        },
        "physics": physics,
        "appearance": appearance,
        "temporal": {
            "encoded_to_physical_speed": 1.0,
            "time_scale": "real_time",
            "annotation_source": (
                "native approximately 240 fps capture normalized to 24 fps "
                "while preserving physical time"
            ),
        },
        "alignment": {
            "version": "photogate_cleaning_20260730_v1",
            "method": record["alignment"]["method"],
            "review_status": record["review_status"],
            "source_start_frame": record["alignment"]["source_start_frame"],
            "source_end_frame_exclusive": (
                record["alignment"]["source_end_frame_exclusive"]
            ),
            "source_fps": record["alignment"]["source_fps"],
            "source_crop": record["alignment"]["crop"],
            "output_size": {
                "width": record["alignment"]["output_width"],
                "height": record["alignment"]["output_height"],
            },
            "audit_record": (
                "provenance/alignment/photogate_cleaning_20260730_v1/"
                "collision_encoded.jsonl"
            ),
        },
        "provenance": {
            "source_kind": "real_capture",
            "parent_case_id": None,
            "generator": None,
            "source_locator": {
                "archive": asset_path(COLLISION_ARCHIVE),
                "member": archive_member,
                "annotation_workbook": (
                    "provenance/source_docs/20260730_parabolic_collision/"
                    f"collision/{workbook}"
                ),
                "sheet": source_annotation["sheet"],
                "row": source_annotation["row"],
                "source_cells": source_annotation["source_cells"],
                "normalization_correction": record["normalization_correction"],
                "formula_checks": physical["formula_checks"],
                "quality_flags": record.get("quality_flags", []),
            },
        },
        "ood": {
            "level": "id" if homogeneous else "ood1",
            "factors": [] if homogeneous else ["collision_pair"],
        },
        "has_real_reference_video": True,
    }


def materialize_provenance() -> None:
    copy_verified(
        PARABOLIC_ARCHIVE_INPUT,
        PARABOLIC_ARCHIVE,
        expected_sha256=PARABOLIC_ARCHIVE_SHA256,
    )
    copy_verified(
        COLLISION_ARCHIVE_INPUT,
        COLLISION_ARCHIVE,
        expected_sha256=COLLISION_ARCHIVE_SHA256,
    )
    parabolic_doc_root = DOC_ROOT / "parabolic"
    collision_doc_root = DOC_ROOT / "collision"
    for name, source in PARABOLIC_ANNOTATION_INPUTS.items():
        copy_verified(source, parabolic_doc_root / name)
    for number in (2, 3, 4):
        name = f"{number}.平抛运动实验记录表.xlsx"
        copy_verified(
            PARABOLIC_WORKBOOK_ROOT / name,
            parabolic_doc_root / name,
        )
    for source in sorted(COLLISION_ANNOTATION_ROOT.glob("*.xlsx")):
        copy_verified(source, collision_doc_root / source.name)
    for source in (
        COLLISION_INGEST_AUDIT,
        COLLISION_CURATION_AUDIT,
        COLLISION_CURATION_SUMMARY,
    ):
        copy_verified(source, collision_doc_root / source.name)

    PROVENANCE_ALIGNMENT_ROOT.mkdir(parents=True, exist_ok=True)
    for source in sorted(ALIGNMENT_INPUT.glob("*")):
        if source.is_file():
            copy_verified(source, PROVENANCE_ALIGNMENT_ROOT / source.name)
    review_target = PROVENANCE_ALIGNMENT_ROOT / "review"
    for scene_id in ("parabolic_motion", "collision_1d"):
        for source in sorted((STAGED_REVIEW / scene_id).rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(STAGED_REVIEW)
            copy_verified(source, review_target / relative)


def build_view_a(
    base: dict[str, Any],
    parabolic_records: list[dict[str, Any]],
    collision_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    scenes = copy.deepcopy(base["scenes"])
    exclusions: dict[str, str] = {}
    scenes["parabolic_motion"] = {
        "train": sorted(
            item["case_id"]
            for item in parabolic_records
            if item["annotation"]["split"] == "train"
        ),
        "test_id": sorted(
            item["case_id"]
            for item in parabolic_records
            if item["annotation"]["split"] == "test"
        ),
        "test_ood1": [],
    }

    curation = {
        item["case_id"]: item["curation"]
        for item in load_jsonl(COLLISION_CURATION_AUDIT)
    }
    collision_groups = scenes["collision_1d"]
    for record in collision_records:
        case_id = record["case_id"]
        balls, _ = collision_balls(record["physical_parameters"])
        signatures = {
            (
                float(ball["mass"]["value"]),
                float(ball["diameter"]["value"]),
            )
            for ball in balls
        }
        homogeneous = len(signatures) == 1
        decision = curation[case_id]
        if decision["status"] != "included":
            exclusions[case_id] = "excluded from View A by source near-replicate curation"
        elif homogeneous and decision["split"] == "train":
            collision_groups["train"].append(case_id)
        elif homogeneous and decision["split"] == "test":
            collision_groups["test_id"].append(case_id)
        elif not homogeneous and decision["split"] == "test":
            collision_groups["test_ood1"].append(case_id)
        else:
            exclusions[case_id] = (
                "heterogeneous collision reserved outside View A training "
                "to preserve collision-pair OOD purity"
            )
    for groups in scenes.values():
        for values in groups.values():
            values.sort()
    selected = sorted(
        case_id
        for groups in scenes.values()
        for values in groups.values()
        for case_id in values
    )
    if len(selected) != len(set(selected)):
        raise ValueError("View A contains duplicate case IDs")
    view = {
        "schema_version": "2.0",
        "view_id": "view_a",
        "coverage": "subset",
        "selection_policy": (
            "pure_numeric_id_environment_ood1_with_20260730_source_splits_v1"
        ),
        "case_set_sha256": canonical_sha256(selected),
        "scenes": dict(sorted(scenes.items())),
    }
    return view, dict(sorted(exclusions.items()))


def build_release(
    new_cases: list[dict[str, Any]],
    parabolic_records: list[dict[str, Any]],
    collision_records: list[dict[str, Any]],
    exclusions: dict[str, Any],
) -> tuple[Path, str]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable release: {OUTPUT_ROOT}"
        )
    base_descriptor = load_json(BASE_ROOT / "dataset.json")
    if (
        base_descriptor.get("dataset_id") != "physics_video_five_scene_v4"
        or base_descriptor.get("release") != "4.0.0"
    ):
        raise ValueError("release 5.0.0 requires the frozen 4.0.0 base")
    old_cases = load_jsonl(BASE_ROOT / base_descriptor["cases"])
    cases = sorted(old_cases + new_cases, key=lambda item: item["case_id"])
    identifiers = [item["case_id"] for item in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate case ID while building release 5.0.0")

    stage = Path(
        tempfile.mkdtemp(prefix=".5.0.0.import-", dir=RELEASES_ROOT)
    )
    try:
        (stage / "views").mkdir()
        (stage / "scenes").mkdir()
        write_jsonl(stage / "cases.jsonl", cases)
        used_scene_ids = {item["scene_id"] for item in cases}
        for source in sorted((BASE_ROOT / "scenes").glob("*.json")):
            if source.stem in used_scene_ids:
                shutil.copy2(source, stage / "scenes" / source.name)
        shutil.copy2(
            ROOT / "configs/scenes/parabolic_motion.json",
            stage / "scenes/parabolic_motion.json",
        )

        view_a, view_a_exclusions = build_view_a(
            load_json(BASE_ROOT / "views/view_a.json"),
            parabolic_records,
            collision_records,
        )
        write_json(stage / "views/view_a.json", view_a)
        view_b = build_view_b(cases, groups=5, seed=42)
        view_b["schema_version"] = "2.0"
        view_b["view_id"] = "view_b"
        view_b["coverage"] = "complete"
        view_b.pop("view", None)
        write_json(stage / "views/view_b.json", view_b)

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
        write_json(stage / "dataset.json", descriptor)

        scene_counts = Counter(item["scene_id"] for item in cases)
        added_counts = Counter(item["scene_id"] for item in new_cases)
        selected_counts = {
            scene_id: {
                partition: len(values)
                for partition, values in view_a["scenes"][scene_id].items()
            }
            for scene_id in ("parabolic_motion", "collision_1d")
        }
        write_json(
            stage / "expansion_audit.json",
            {
                "schema_version": "1.0",
                "base_release": "../4.0.0/dataset.json",
                "base_dataset_id": base_descriptor["dataset_id"],
                "base_case_count": len(old_cases),
                "added_case_count": len(new_cases),
                "case_count": len(cases),
                "added_scenes": ["parabolic_motion"],
                "expanded_scenes": ["collision_1d"],
                "accepted_source_counts": dict(sorted(added_counts.items())),
                "scene_case_counts": dict(sorted(scene_counts.items())),
                "source_archive_sha256": {
                    "parabolic_motion.zip": PARABOLIC_ARCHIVE_SHA256,
                    "collision_supplement_20260729.zip": (
                        COLLISION_ARCHIVE_SHA256
                    ),
                },
                "excluded_source_counts": exclusions,
                "collision_existing_benchmark_duplicate_count": 0,
                "collision_existing_benchmark_duplicate_audit": (
                    "provenance/alignment/"
                    "photogate_cleaning_20260730_v1/"
                    "collision_duplicate_audit.json"
                ),
                "visual_review_approval": (
                    "provenance/alignment/"
                    "photogate_cleaning_20260730_v1/"
                    "visual_review_approval.json"
                ),
                "view_a_selected_counts": selected_counts,
                "view_a_excluded_added_cases": view_a_exclusions,
                "case_ids_sha256": canonical_sha256(identifiers),
            },
        )
        (stage / "README.md").write_text(
            f"""# Physics Video Dataset 5.0.0

这是从不可变 `4.0.0` 派生的六场景数据 release。

- 基础 case：{len(old_cases)}
- 新增平抛：{len(parabolic_records)}
- 新增碰撞：{len(collision_records)}
- 总 case：{len(cases)}

新增视频均有独立物理标注，并已完成光电门出门时刻对齐、无门体裁剪、
首帧全体球完整可见检查和末帧事件完整性检查。平抛中 31 个无独立物理标注
的视频未导入；碰撞补充数据与旧 32 个碰撞 source 的重复数为 0。

`View A` 使用来源侧冻结 split，并保持 collision-pair OOD 纯度；
`View B` 完整覆盖全部 {len(cases)} 个 case。
""",
            encoding="utf-8",
        )
        _, release_manifest = rebuild_asset_lock(stage / "dataset.json")
        snapshot = load_dataset(
            stage / "dataset.json",
            check_assets=True,
            check_asset_hashes=True,
        )
        if snapshot.digest != release_manifest["dataset_digest"]:
            raise ValueError("staged release digest changed after lock build")
        stage.rename(OUTPUT_ROOT)
        return OUTPUT_ROOT, snapshot.digest
    except BaseException:
        if stage.exists() and stage.parent == RELEASES_ROOT:
            shutil.rmtree(stage)
        raise


def main() -> int:
    parabolic_records = load_reviewed_records("parabolic", 97)
    collision_records = load_reviewed_records("collision", 298)
    approval = load_json(ALIGNMENT_INPUT / "visual_review_approval.json")
    if any(
        value.get("review_status") != "visually_verified"
        for value in approval["scenes"].values()
    ):
        raise ValueError("visual review approval is incomplete")
    duplicate_audit = load_json(
        ALIGNMENT_INPUT / "collision_duplicate_audit.json"
    )
    if duplicate_audit.get("exact_byte_duplicate_count") != 0:
        raise ValueError("collision supplement contains an old Dataset duplicate")
    if (
        duplicate_audit["decoded_motion_fingerprint"][
            "candidate_count_at_or_above_threshold"
        ]
        != 0
    ):
        raise ValueError("collision duplicate audit still has review candidates")

    materialize_provenance()
    parabolic_members = verify_archive_inventory(
        parabolic_records,
        PARABOLIC_ARCHIVE,
        member_prefix="平抛运动",
    )
    collision_members = verify_archive_inventory(
        collision_records,
        COLLISION_ARCHIVE,
        member_prefix="补充_碰撞试验7_29",
    )

    new_cases: list[dict[str, Any]] = []
    import_audit: list[dict[str, Any]] = []
    for record in parabolic_records + collision_records:
        assets = materialize_case_assets(record)
        if record["scene_id"] == "parabolic_motion":
            member = parabolic_members[int(record["image_number"])]
            case = parabolic_case(record, assets, member)
        else:
            member = collision_members[int(record["image_number"])]
            case = collision_case(record, assets, member)
        new_cases.append(case)
        import_audit.append(
            {
                "schema_version": "parabolic-collision-import-audit-v1",
                "case_id": record["case_id"],
                "scene_id": record["scene_id"],
                "image_number": record["image_number"],
                "archive_member": member,
                "annotation_status": "matched_unique",
                "duplicate_status": (
                    "no_match_in_existing_benchmark"
                    if record["scene_id"] == "collision_1d"
                    else "not_applicable"
                ),
                "review_status": record["review_status"],
                "source_sha256": assets["source_sha256"],
                "reference_video_sha256": assets[
                    "reference_video_sha256"
                ],
                "first_frame_sha256": assets["first_frame_sha256"],
                "normalization_correction": record.get(
                    "normalization_correction"
                ),
                "quality_flags": record.get("quality_flags", []),
            }
        )
    write_jsonl(
        IMPORT_AUDIT,
        sorted(import_audit, key=lambda item: item["case_id"]),
    )

    accepted_parabolic_numbers = {
        int(item["image_number"]) for item in parabolic_records
    }
    with zipfile.ZipFile(PARABOLIC_ARCHIVE) as archive:
        all_parabolic_numbers = {
            int(Path(item.filename).stem.split("_")[-1])
            for item in archive.infolist()
            if (
                not item.is_dir()
                and Path(item.filename).suffix.lower() == ".mov"
            )
        }
    unannotated = sorted(all_parabolic_numbers - accepted_parabolic_numbers)
    if len(unannotated) != 31:
        raise ValueError(
            f"expected 31 unannotated parabolic videos, found {len(unannotated)}"
        )
    collision_ingest = load_json(COLLISION_INGEST_AUDIT)
    exclusions = {
        "parabolic_video_without_independent_annotation": len(unannotated),
        "parabolic_video_without_independent_annotation_ids": [
            f"IMG_{number:04d}" for number in unannotated
        ],
        "collision_annotation_row_without_video": (
            collision_ingest["content"]["workbook_ids_without_video_count"]
        ),
        "collision_annotation_row_without_video_ids": (
            collision_ingest["content"]["workbook_ids_without_video"]
        ),
        "collision_video_without_annotation": 0,
        "collision_duplicate_of_existing_benchmark": 0,
    }
    write_json(EXCLUSION_AUDIT, exclusions)
    output, digest = build_release(
        new_cases,
        parabolic_records,
        collision_records,
        exclusions,
    )
    print(
        f"{output / 'dataset.json'} added_cases={len(new_cases)} "
        f"total_cases={len(load_jsonl(output / 'cases.jsonl'))} "
        f"dataset_digest={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
