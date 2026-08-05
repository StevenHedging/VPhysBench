#!/usr/bin/env python3
"""Import the annotation-backed inclined-plane and circular-motion captures.

The two immutable source archives are stored once under the Dataset asset root.
Cases reference those archives through provenance and use per-case canonical
videos for training/evaluation.  Inclined-plane canonical videos must already
have been materialized by ``align_inclined_plane.py`` after visual review.

This script deliberately excludes any source member without an unambiguous XLSX
row and any annotation row without a complete video member.  It builds the native
v2-contract Dataset release 3.0.0 without mutating releases 1.0.0 or 2.0.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)
from physbench.splitters import build_view_b  # noqa: E402


DATA_ROOT = ROOT / "datasets"
ASSET_ROOT = DATA_ROOT / "assets"
SOURCE_ROOT = ASSET_ROOT / "source_archives" / "20260723_new_scenes"
DOC_ROOT = (
    DATA_ROOT / "provenance" / "source_docs" / "20260723_new_scenes"
)
IMPORT_ROOT = DATA_ROOT / "provenance" / "imports"
ALIGNMENT_ROOT = (
    DATA_ROOT / "provenance" / "alignment" / "inclined_plane_start_v1"
)
INCLINE_ARCHIVE = SOURCE_ROOT / "inclined_plane_raw.zip"
CIRCULAR_ARCHIVE = SOURCE_ROOT / "uniform_circular_motion_processed.zip"
INCLINE_XLSX = DOC_ROOT / "inclined_plane_annotations.xlsx"
CIRCULAR_XLSX = DOC_ROOT / "uniform_circular_motion_annotations.xlsx"
NORMALIZED_ANNOTATIONS = DOC_ROOT / "normalized_annotations.json"
IMPORT_AUDIT = IMPORT_ROOT / "new_scenes_20260723_import_audit.jsonl"
REVIEWED_FRAMES = ALIGNMENT_ROOT / "reviewed_frames.json"
ALIGNMENT_AUDIT = ALIGNMENT_ROOT / "alignment_audit.jsonl"
V2_ROOT = DATA_ROOT / "releases" / "2.0.0"
V3_ROOT = DATA_ROOT / "releases" / "3.0.0"

XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quantity(value: float, unit: str, *, annotated: bool = True) -> dict[str, Any]:
    if not math.isfinite(value):
        raise ValueError(f"non-finite physical quantity: {value}")
    return {"value": value, "unit": unit, "annotated": annotated}


def normalized_stem(value: str) -> str:
    match = re.search(r"(\d+)", value)
    if match is None:
        raise ValueError(f"filename has no numeric identifier: {value}")
    return f"IMG_{int(match.group(1)):04d}"


def _xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, str | None]]:
    with zipfile.ZipFile(path) as workbook:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.iterfind(".//a:t", XML_NS))
                for item in root.findall("a:si", XML_NS)
            ]
        book = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships = ET.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        targets = {
            item.attrib["Id"]: item.attrib["Target"] for item in relationships
        }
        sheet = next(
            item
            for item in book.find("a:sheets", XML_NS) or []
            if item.attrib["name"] == sheet_name
        )
        relation_key = f"{{{XML_NS['r']}}}id"
        target = targets[sheet.attrib[relation_key]]
        member = (
            target.lstrip("/")
            if target.startswith("/")
            else str(PurePosixPath("xl") / target)
        )
        root = ET.fromstring(workbook.read(member))
        rows: list[dict[str, str | None]] = []
        for row in root.findall(".//a:sheetData/a:row", XML_NS):
            values: dict[str, str | None] = {}
            for cell in row.findall("a:c", XML_NS):
                column = re.match(r"[A-Z]+", cell.attrib["r"])
                if column is None:
                    continue
                value_node = cell.find("a:v", XML_NS)
                kind = cell.attrib.get("t")
                value: str | None = None
                if kind == "s" and value_node is not None:
                    value = strings[int(value_node.text or "0")]
                elif kind == "inlineStr":
                    inline = cell.find("a:is", XML_NS)
                    if inline is not None:
                        value = "".join(
                            node.text or ""
                            for node in inline.iterfind(".//a:t", XML_NS)
                        )
                elif value_node is not None:
                    value = value_node.text
                values[column.group()] = value
            rows.append(values)
        return rows


def incline_settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for row in _xlsx_rows(INCLINE_XLSX, "实验设置"):
        key = row.get("A")
        value = row.get("B")
        if key and value:
            values[key] = value
    required = {
        "实验批次名称",
        "采集日期",
        "手机型号",
        "录像模式",
        "物体说明",
        "物体质量_kg",
        "轨道材质",
        "动摩擦系数_mu_k",
        "标定参照长度_cm",
        "固定释放点",
        "重力加速度_m_s2",
        "整体备注",
    }
    missing = required - set(values)
    if missing:
        raise ValueError(f"incline settings missing {sorted(missing)}")
    return values


def incline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in _xlsx_rows(INCLINE_XLSX, "Trial记录"):
        filename = value.get("B")
        angle = value.get("C")
        if not filename or not angle or not re.search(r"\d", filename):
            continue
        source_background = (value.get("G") or "").strip()
        background = {
            "": "default_white",
            "油画": "oil_painting",
            "黑色泡沫板": "black_foam_board",
            "绿色卡纸": "green_cardstock",
        }.get(source_background)
        if background is None:
            raise ValueError(f"unknown incline background: {source_background}")
        rows.append(
            {
                "trial_id": value.get("A"),
                "source_stem": normalized_stem(filename),
                "angle_deg": int(float(angle)),
                "friction_force_n": float(value["D"] or "nan"),
                "theoretical_acceleration_m_s2": float(value["E"] or "nan"),
                "background": background,
                "background_source_value": source_background or "default",
            }
        )
    if len(rows) != 100:
        raise ValueError(f"expected 100 incline annotations, found {len(rows)}")
    return rows


def _distance_cm(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*cm\s*", value)
    if match is None:
        raise ValueError(f"ambiguous distance annotation: {value}")
    number = float(match.group(1))
    if not number.is_integer():
        raise ValueError(f"non-integer centimeter annotation: {value}")
    return int(number)


def _angular_velocity(value: str | None) -> float:
    if not value:
        raise ValueError("missing angular velocity")
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*度/秒\s*", value)
    if match is None:
        raise ValueError(f"ambiguous angular velocity annotation: {value}")
    return float(match.group(1))


def circular_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in _xlsx_rows(CIRCULAR_XLSX, "实验数据"):
        filename = value.get("A")
        if not filename or not re.search(r"\d", filename):
            continue
        silver = _distance_cm(value.get("C"))
        wood = _distance_cm(value.get("D"))
        if silver is None and wood is None:
            raise ValueError(f"no radius annotation for {filename}")
        rows.append(
            {
                "source_stem": normalized_stem(filename),
                "angular_velocity_deg_s": _angular_velocity(value.get("B")),
                "silver_radius_cm": silver,
                "wood_radius_cm": wood,
                "moving_objects": [
                    name
                    for name, present in (
                        ("silver_metal_block", silver is not None),
                        ("rectangular_wood_block", wood is not None),
                    )
                    if present
                ],
            }
        )
    if len(rows) != 36:
        raise ValueError(f"expected 36 circular annotations, found {len(rows)}")
    return rows


def archive_members(path: Path) -> tuple[dict[str, zipfile.ZipInfo], dict[str, zipfile.ZipInfo]]:
    complete: dict[str, zipfile.ZipInfo] = {}
    partial: dict[str, zipfile.ZipInfo] = {}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            stem = normalized_stem(Path(item.filename).stem)
            target = partial if item.filename.endswith(".drivedownload") else complete
            if stem in target:
                raise ValueError(f"duplicate normalized archive member: {stem}")
            target[stem] = item
    return complete, partial


def incline_case_id(row: dict[str, Any]) -> str:
    backgrounds = {
        "default_white": "bgwhite",
        "oil_painting": "bgoil",
        "black_foam_board": "bgblack",
        "green_cardstock": "bggreen",
    }
    return (
        f"incline_r1_a{row['angle_deg']:02d}deg_"
        f"{backgrounds[row['background']]}_{row['source_stem'].lower()}"
    )


def circular_case_id(row: dict[str, Any]) -> str:
    silver = row["silver_radius_cm"]
    wood = row["wood_radius_cm"]
    if silver is not None and wood is not None:
        condition = f"silver{silver:02d}cm_wood{wood:02d}cm"
    elif silver is not None:
        condition = f"silver{silver:02d}cm"
    else:
        condition = f"wood{wood:02d}cm"
    return f"circular_r1_{condition}_{row['source_stem'].lower()}"


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "frames": (
            int(stream["nb_frames"])
            if stream.get("nb_frames") not in {None, "N/A"}
            else None
        ),
        "duration_s": (
            float(stream["duration"])
            if stream.get("duration") not in {None, "N/A"}
            else None
        ),
    }


def first_frame(video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.png")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, output)


def extract_stable(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        if destination.stat().st_size != member.file_size:
            raise FileExistsError(f"destination size mismatch: {destination}")
        return
    temporary = destination.with_suffix(".tmp" + destination.suffix)
    with archive.open(member) as source, temporary.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if temporary.stat().st_size != member.file_size:
        raise OSError(f"incomplete archive extraction: {member.filename}")
    os.replace(temporary, destination)


def materialize_circular(
    rows: list[dict[str, Any]],
    members: dict[str, zipfile.ZipInfo],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    with zipfile.ZipFile(CIRCULAR_ARCHIVE) as archive:
        for index, row in enumerate(rows, 1):
            member = members[row["source_stem"]]
            identifier = circular_case_id(row)
            case_root = ASSET_ROOT / "uniform_circular_motion" / identifier
            video = case_root / "canonical" / "reference.mp4"
            image = case_root / "canonical" / "first_frame.png"
            extract_stable(archive, member, video, overwrite=overwrite)
            first_frame(video, image)
            audits.append(
                {
                    "case_id": identifier,
                    "scene_id": "uniform_circular_motion",
                    "source_archive": str(CIRCULAR_ARCHIVE.relative_to(ROOT)),
                    "archive_member": member.filename,
                    "source_member_crc32": f"{member.CRC:08x}",
                    "source_member_bytes": member.file_size,
                    "copy_policy": "byte_preserving_member_extraction",
                    "destination": str(video.relative_to(ROOT)),
                    "destination_sha256": sha256(video),
                    "first_frame": str(image.relative_to(ROOT)),
                    "first_frame_sha256": sha256(image),
                    "probe": probe(video),
                }
            )
            print(f"[{index:02d}/{len(rows):02d}] {identifier}", flush=True)
    return audits


def build_normalized_annotations(
    incline: list[dict[str, Any]],
    circular: list[dict[str, Any]],
    incline_complete: dict[str, zipfile.ZipInfo],
    incline_partial: dict[str, zipfile.ZipInfo],
    circular_complete: dict[str, zipfile.ZipInfo],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    incline_by_stem = {row["source_stem"]: row for row in incline}
    circular_by_stem = {row["source_stem"]: row for row in circular}
    accepted_incline = [
        {**row, "archive_member": incline_complete[row["source_stem"]].filename}
        for row in incline
        if row["source_stem"] in incline_complete
    ]
    accepted_circular = [
        {**row, "archive_member": circular_complete[row["source_stem"]].filename}
        for row in circular
        if row["source_stem"] in circular_complete
    ]
    missing_incline = sorted(set(incline_by_stem) - set(incline_complete))
    partial_incline = sorted(set(incline_by_stem) & set(incline_partial))
    absent_incline = sorted(set(missing_incline) - set(partial_incline))
    unannotated_incline = sorted(set(incline_complete) - set(incline_by_stem))
    unannotated_circular = sorted(set(circular_complete) - set(circular_by_stem))
    if len(accepted_incline) != 95:
        raise ValueError(f"expected 95 accepted incline cases, got {len(accepted_incline)}")
    if len(accepted_circular) != 36:
        raise ValueError(f"expected 36 accepted circular cases, got {len(accepted_circular)}")
    if unannotated_circular:
        raise ValueError(f"unexpected unannotated circular clips: {unannotated_circular}")
    report = {
        "schema_version": "1.0",
        "policy": (
            "accept only exact normalized filename joins with complete media and "
            "unambiguous physical annotations"
        ),
        "sources": {
            "inclined_plane": {
                "archive": str(INCLINE_ARCHIVE.relative_to(ROOT)),
                "annotation_workbook": str(INCLINE_XLSX.relative_to(ROOT)),
                "annotation_rows": len(incline),
                "accepted_rows": len(accepted_incline),
                "partial_annotated_members_excluded": partial_incline,
                "absent_annotated_members_excluded": absent_incline,
                "unannotated_complete_members_excluded": unannotated_incline,
            },
            "uniform_circular_motion": {
                "archive": str(CIRCULAR_ARCHIVE.relative_to(ROOT)),
                "annotation_workbook": str(CIRCULAR_XLSX.relative_to(ROOT)),
                "annotation_rows": len(circular),
                "accepted_rows": len(accepted_circular),
                "unannotated_complete_members_excluded": unannotated_circular,
            },
        },
        "inclined_plane_fixed_settings": incline_settings(),
        "inclined_plane_rows": accepted_incline,
        "uniform_circular_motion_rows": accepted_circular,
    }
    write_json(NORMALIZED_ANNOTATIONS, report)
    return accepted_incline, accepted_circular, report


def asset_path(scene: str, identifier: str, filename: str) -> str:
    return f"assets/{scene}/{identifier}/canonical/{filename}"


def source_archive_path(path: Path) -> str:
    return str(path.relative_to(DATA_ROOT))


def incline_cases(
    rows: list[dict[str, Any]], reviewed: dict[str, Any]
) -> list[dict[str, Any]]:
    settings = incline_settings()
    frames = reviewed["frames"]
    mass = float(settings["物体质量_kg"])
    friction_coefficient = float(settings["动摩擦系数_mu_k"])
    gravity = float(settings["重力加速度_m_s2"])
    calibration_m = float(settings["标定参照长度_cm"]) / 100.0
    block_length_match = re.search(r"(\d+(?:\.\d+)?)\s*cm", settings["整体备注"])
    if block_length_match is None:
        raise ValueError("cannot parse block length from incline settings")
    block_length_m = float(block_length_match.group(1)) / 100.0
    cases: list[dict[str, Any]] = []
    for row in rows:
        identifier = incline_case_id(row)
        if identifier not in frames:
            raise ValueError(f"missing reviewed start frame for {identifier}")
        frame = frames[identifier]
        video = asset_path("inclined_plane_slide", identifier, "reference.mp4")
        image = asset_path("inclined_plane_slide", identifier, "first_frame.png")
        cases.append(
            {
                "schema_version": "2.0",
                "case_id": identifier,
                "scene_id": "inclined_plane_slide",
                "assets": {
                    "first_frame": image,
                    "reference_video": video,
                    "physics_reference_video": video,
                    "source_archive": source_archive_path(INCLINE_ARCHIVE),
                    "subject_mask": None,
                },
                "physics": {
                    "incline_angle": quantity(float(row["angle_deg"]), "deg"),
                    "block_mass": quantity(mass, "kg"),
                    "kinetic_friction_coefficient": quantity(
                        friction_coefficient, "1"
                    ),
                    "gravity_acceleration": quantity(gravity, "m/s^2"),
                    "friction_force": quantity(row["friction_force_n"], "N"),
                    "theoretical_acceleration": quantity(
                        row["theoretical_acceleration_m_s2"], "m/s^2"
                    ),
                    "calibration_length": quantity(calibration_m, "m"),
                    "block_length": quantity(block_length_m, "m"),
                    "initial_velocity": quantity(0.0, "m/s", annotated=False),
                },
                "appearance": {
                    "background": row["background"],
                    "block": "black_base_round_hole_wood_slider",
                    "track_material": "unfinished_wood_board",
                    "camera": "fixed",
                    "capture_device": settings["手机型号"],
                    "release_point": "top",
                    "capture_session": settings["实验批次名称"],
                },
                "temporal": {
                    "encoded_to_physical_speed": 1.0,
                    "time_scale": "real_time",
                    "annotation_source": (
                        "native approximately 240 fps capture; container timestamps "
                        "represent physical time"
                    ),
                },
                "alignment": {
                    "version": "inclined_plane_start_v1",
                    "method": "decoded_source_frame_trim",
                    "review_status": reviewed["review_status"],
                    "source_start_frame": int(frame["source_start_frame"]),
                    "source_start_time_s_approx": (
                        int(frame["source_start_frame"]) / float(frame["fps"])
                    ),
                    "source_frame_zero_exception": (
                        int(frame["source_start_frame"]) == 0
                    ),
                    "audit_record": (
                        "provenance/alignment/inclined_plane_start_v1/"
                        "alignment_audit.jsonl"
                    ),
                },
                "provenance": {
                    "source_kind": "real_capture",
                    "parent_case_id": None,
                    "generator": None,
                    "source_locator": {
                        "archive": source_archive_path(INCLINE_ARCHIVE),
                        "member": row["archive_member"],
                        "annotation_workbook": str(INCLINE_XLSX.relative_to(DATA_ROOT)),
                        "trial_id": row["trial_id"],
                    },
                },
                "ood": {
                    "level": (
                        "ood1"
                        if row["background"] == "black_foam_board"
                        else "id"
                    ),
                    "factors": (
                        ["background"]
                        if row["background"] == "black_foam_board"
                        else []
                    ),
                },
                "has_real_reference_video": True,
            }
        )
    return cases


def circular_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in rows:
        identifier = circular_case_id(row)
        video = asset_path("uniform_circular_motion", identifier, "reference.mp4")
        image = asset_path("uniform_circular_motion", identifier, "first_frame.png")
        radii = [
            value
            for value in (row["silver_radius_cm"], row["wood_radius_cm"])
            if value is not None
        ]
        physics = {
            "angular_velocity": quantity(row["angular_velocity_deg_s"], "deg/s"),
            "angular_velocity_rad_s": quantity(
                math.radians(row["angular_velocity_deg_s"]),
                "rad/s",
                annotated=False,
            ),
            "object_1_orbit_radius": quantity(radii[0] / 100.0, "m"),
        }
        if len(radii) == 2:
            physics["object_2_orbit_radius"] = quantity(radii[1] / 100.0, "m")
        both = len(row["moving_objects"]) == 2
        cases.append(
            {
                "schema_version": "2.0",
                "case_id": identifier,
                "scene_id": "uniform_circular_motion",
                "assets": {
                    "first_frame": image,
                    "reference_video": video,
                    "physics_reference_video": video,
                    "source_archive": source_archive_path(CIRCULAR_ARCHIVE),
                    "subject_mask": None,
                },
                "physics": physics,
                "appearance": {
                    "moving_objects": row["moving_objects"],
                    "object_count": len(row["moving_objects"]),
                    "disk_color": "green",
                    "background": "wood_tabletop",
                    "camera": "fixed_overhead",
                    "capture_session": "20260723_uniform_circular_motion",
                },
                "temporal": {
                    "encoded_to_physical_speed": 1.0,
                    "time_scale": "real_time",
                    "annotation_source": (
                        "processed 60 fps video paired with measured angular velocity"
                    ),
                },
                "provenance": {
                    "source_kind": "real_capture",
                    "parent_case_id": None,
                    "generator": None,
                    "source_locator": {
                        "archive": source_archive_path(CIRCULAR_ARCHIVE),
                        "member": row["archive_member"],
                        "annotation_workbook": str(CIRCULAR_XLSX.relative_to(DATA_ROOT)),
                    },
                },
                "ood": {
                    "level": "ood1" if both else "id",
                    "factors": ["moving_object_composition"] if both else [],
                },
                "has_real_reference_video": True,
            }
        )
    return cases


def new_view_a(
    base: dict[str, Any],
    incline: list[dict[str, Any]],
    circular: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    scenes = json.loads(json.dumps(base["scenes"]))
    exclusions: dict[str, str] = {}

    incline_groups = {"train": [], "test_id": [], "test_ood1": []}
    by_environment_angle: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in incline:
        by_environment_angle[(row["background"], row["angle_deg"])].append(row)
    for values in by_environment_angle.values():
        values.sort(key=lambda item: item["source_stem"])
    seen_backgrounds = {"default_white", "oil_painting", "green_cardstock"}
    train_angles = {32, 35, 41, 44}
    for row in incline:
        identifier = incline_case_id(row)
        if row["background"] in seen_backgrounds and row["angle_deg"] in train_angles:
            incline_groups["train"].append(identifier)
        else:
            exclusions[identifier] = "not selected by pure-factor View A sampling"
    for background in sorted(seen_backgrounds):
        values = by_environment_angle[(background, 38)]
        for row in values[:2]:
            identifier = incline_case_id(row)
            incline_groups["test_id"].append(identifier)
            exclusions.pop(identifier, None)
    for angle in sorted(train_angles):
        values = by_environment_angle[("black_foam_board", angle)]
        for row in values[:4]:
            identifier = incline_case_id(row)
            incline_groups["test_ood1"].append(identifier)
            exclusions.pop(identifier, None)
    for group in incline_groups.values():
        group.sort()
    scenes["inclined_plane_slide"] = incline_groups

    circular_groups = {"train": [], "test_id": [], "test_ood1": []}
    test_id_stems = {"IMG_0369", "IMG_0381"}
    ood_stems = {"IMG_0392", "IMG_0393", "IMG_0394", "IMG_0395"}
    for row in circular:
        identifier = circular_case_id(row)
        single = len(row["moving_objects"]) == 1
        radius = row["silver_radius_cm"] or row["wood_radius_cm"]
        if single and radius in {2, 6, 8}:
            circular_groups["train"].append(identifier)
        elif single and row["source_stem"] in test_id_stems:
            circular_groups["test_id"].append(identifier)
        elif not single and row["source_stem"] in ood_stems:
            circular_groups["test_ood1"].append(identifier)
        else:
            exclusions[identifier] = "duplicate held-out condition not selected for View A"
    for group in circular_groups.values():
        group.sort()
    scenes["uniform_circular_motion"] = circular_groups

    selected = sorted(
        identifier
        for groups in scenes.values()
        for identifiers in groups.values()
        for identifier in identifiers
    )
    view = {
        "schema_version": "2.0",
        "view_id": "view_a",
        "coverage": "subset",
        "selection_policy": "pure_numeric_id_and_environment_ood1_v1",
        "case_set_sha256": canonical_sha256(selected),
        "scenes": scenes,
    }
    return view, exclusions


def build_release(
    new_cases: list[dict[str, Any]],
    incline: list[dict[str, Any]],
    circular: list[dict[str, Any]],
    normalized: dict[str, Any],
) -> None:
    old_cases = load_jsonl(V2_ROOT / "cases.jsonl")
    cases = sorted(old_cases + new_cases, key=lambda item: item["case_id"])
    identifiers = [case["case_id"] for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate case ID while building release 3.0.0")

    (V3_ROOT / "views").mkdir(parents=True, exist_ok=True)
    (V3_ROOT / "scenes").mkdir(parents=True, exist_ok=True)
    write_jsonl(V3_ROOT / "cases.jsonl", cases)
    base_view_a = load_json(V2_ROOT / "views" / "view_a.json")
    view_a, exclusions = new_view_a(base_view_a, incline, circular)
    write_json(V3_ROOT / "views" / "view_a.json", view_a)
    view_b = build_view_b(cases, groups=5, seed=42)
    view_b["schema_version"] = "2.0"
    view_b["view_id"] = "view_b"
    view_b["coverage"] = "complete"
    view_b.pop("view", None)
    write_json(V3_ROOT / "views" / "view_b.json", view_b)

    for source in sorted((ROOT / "configs" / "scenes").glob("*.json")):
        shutil.copy2(source, V3_ROOT / "scenes" / source.name)
    descriptor = {
        "schema_version": "2.0",
        "dataset_id": "physics_video_five_scene_v3",
        "release": "3.0.0",
        "cases": "cases.jsonl",
        "asset_root": "../..",
        "asset_lock": "assets.lock.json",
        "scene_catalog": "scenes",
        "views": {
            "view_a": "views/view_a.json",
            "view_b": "views/view_b.json",
        },
    }
    write_json(V3_ROOT / "dataset.json", descriptor)
    write_json(
        V3_ROOT / "expansion_audit.json",
        {
            "schema_version": "1.0",
            "base_release": "../2.0.0/dataset.json",
            "base_case_count": len(old_cases),
            "added_case_count": len(new_cases),
            "case_count": len(cases),
            "added_scenes": ["inclined_plane_slide", "uniform_circular_motion"],
            "accepted_source_counts": {
                "inclined_plane_slide": len(incline),
                "uniform_circular_motion": len(circular),
            },
            "view_a_selected_counts": {
                scene: {
                    partition: len(values)
                    for partition, values in view_a["scenes"][scene].items()
                }
                for scene in ("inclined_plane_slide", "uniform_circular_motion")
            },
            "view_a_excluded_new_cases": dict(sorted(exclusions.items())),
            "view_a_exclusion_reason": (
                "retain all valid cases in Dataset/View B while keeping View A "
                "factor-pure and training-dominant; mixed unseen-number plus "
                "unseen-environment cases are reserved for future OOD2"
            ),
            "normalized_annotation_digest": canonical_sha256(normalized),
            "case_ids_sha256": canonical_sha256(identifiers),
        },
    )
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-circular-materialization",
        action="store_true",
        help="only rebuild metadata; canonical circular assets must already exist",
    )
    parser.add_argument("--overwrite-circular", action="store_true")
    args = parser.parse_args()

    required = [
        INCLINE_ARCHIVE,
        CIRCULAR_ARCHIVE,
        INCLINE_XLSX,
        CIRCULAR_XLSX,
        REVIEWED_FRAMES,
        ALIGNMENT_AUDIT,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required import inputs missing: {missing}")

    incline = incline_rows()
    circular = circular_rows()
    incline_complete, incline_partial = archive_members(INCLINE_ARCHIVE)
    circular_complete, circular_partial = archive_members(CIRCULAR_ARCHIVE)
    if circular_partial:
        raise ValueError("circular archive unexpectedly contains partial members")
    accepted_incline, accepted_circular, normalized = build_normalized_annotations(
        incline,
        circular,
        incline_complete,
        incline_partial,
        circular_complete,
    )
    reviewed = load_json(REVIEWED_FRAMES)
    if reviewed.get("review_status") != "visually_verified":
        raise ValueError("incline reviewed frame map is not visually verified")

    circular_audits: list[dict[str, Any]] = []
    if not args.skip_circular_materialization:
        circular_audits = materialize_circular(
            accepted_circular,
            circular_complete,
            overwrite=args.overwrite_circular,
        )
    elif IMPORT_AUDIT.is_file():
        circular_audits = [
            item
            for item in load_jsonl(IMPORT_AUDIT)
            if item.get("scene_id") == "uniform_circular_motion"
        ]
    if len(circular_audits) != 36:
        raise ValueError("circular import audit must cover all 36 cases")
    write_jsonl(IMPORT_AUDIT, sorted(circular_audits, key=lambda item: item["case_id"]))

    new_cases = incline_cases(accepted_incline, reviewed) + circular_cases(
        accepted_circular
    )
    for case in new_cases:
        for role, relative in case["assets"].items():
            if relative is None:
                continue
            path = DATA_ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"{case['case_id']} missing {role}: {relative}"
                )
    build_release(new_cases, accepted_incline, accepted_circular, normalized)
    counts = Counter(case["scene_id"] for case in new_cases)
    print(
        f"{V3_ROOT / 'dataset.json'} added_cases={len(new_cases)} "
        f"scenes={dict(sorted(counts.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
