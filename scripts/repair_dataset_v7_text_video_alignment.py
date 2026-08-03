#!/usr/bin/env python3
"""Repair the audited Dataset 7.0.0 text/video alignment defects in place.

The repair is deliberately scoped to release 7.0.0:

* neutralize ``central``/``head-on`` wording only for the 185 collision
  Cases whose first colliding pair has unequal radii;
* create release-7-only real-time media for ``parabolic_img_0539`` so older
  releases that share the original slow-motion assets remain byte-valid;
* refresh the release asset lock and digest transactionally;
* record a machine-readable repair audit.

No new Dataset release is created.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from physbench.io import (  # noqa: E402
    canonical_sha256,
    load_json,
    load_jsonl,
    write_json,
    write_jsonl,
)


DATA_ROOT = ROOT / "datasets" / "physics_video"
RELEASE_ROOT = DATA_ROOT / "releases" / "7.0.0"
DESCRIPTOR_PATH = RELEASE_ROOT / "dataset.json"
CASES_PATH = RELEASE_ROOT / "cases.jsonl"
MAPPING_PATH = RELEASE_ROOT / "asset_directory_mapping.json"
MIGRATION_AUDIT_PATH = RELEASE_ROOT / "migration_audit.json"
LOCK_PATH = RELEASE_ROOT / "assets.lock.json"
RELEASE_PATH = RELEASE_ROOT / "release.json"
REPAIR_AUDIT_PATH = RELEASE_ROOT / "text_video_alignment_repair.json"

PARABOLIC_CASE_ID = "parabolic_img_0539"
OLD_REFERENCE = (
    "assets/parabolic_motion/"
    "projectile_steelM-d20mm-m33p13g_h0p77m_v1p8051mps_img0539/"
    "canonical/reference.mp4"
)
NEW_REFERENCE = OLD_REFERENCE.replace("reference.mp4", "reference_realtime.mp4")
OLD_FIRST_FRAME = OLD_REFERENCE.replace("reference.mp4", "first_frame.png")
NEW_FIRST_FRAME = OLD_REFERENCE.replace(
    "reference.mp4", "first_frame_realtime.png"
)

PROMPT_C3_OLD = (
    "A fixed-camera real-world laboratory video of aligned balls undergoing a "
    "one-dimensional collision. The left ball moves right toward two initially "
    "stationary balls and collides centrally with the middle ball. The camera and "
    "track remain stationary, and the motion unfolds at the true physical time scale."
)
PROMPT_C3_NEW = (
    "A fixed-camera real-world laboratory video of three balls moving along a straight "
    "track. The left ball moves right toward two initially stationary balls and collides "
    "with the middle ball. The camera and track remain stationary, and the motion "
    "unfolds at the true physical time scale."
)
PROMPT_C2_SINGLE_OLD = (
    "A fixed-camera real-world laboratory video of aligned balls undergoing a "
    "one-dimensional collision. The left ball is initially stationary while the right "
    "ball moves left toward it, and the balls collide centrally. The camera and track "
    "remain stationary, and the motion unfolds at the true physical time scale."
)
PROMPT_C2_SINGLE_NEW = (
    "A fixed-camera real-world laboratory video of two balls moving along a straight "
    "track. The left ball is initially stationary while the right ball moves left toward "
    "it, and the balls collide. The camera and track remain stationary, and the motion "
    "unfolds at the true physical time scale."
)
PROMPT_C2_OPPOSED_OLD = (
    "A fixed-camera real-world laboratory video of aligned balls undergoing a "
    "one-dimensional collision. The left ball moves right while the right ball moves "
    "left, and the balls collide head-on. The camera and track remain stationary, and "
    "the motion unfolds at the true physical time scale."
)
PROMPT_C2_OPPOSED_NEW = (
    "A fixed-camera real-world laboratory video of two balls moving along a straight "
    "track. The left ball moves right while the right ball moves left, and the balls "
    "collide. The camera and track remain stationary, and the motion unfolds at the true "
    "physical time scale."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate(value: str) -> float:
    numerator, denominator = value.split("/")
    return float(numerator) / float(denominator)


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration,time_base",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "avg_frame_rate": stream["avg_frame_rate"],
        "fps": _rate(stream["avg_frame_rate"]),
        "nominal_frame_rate": stream["r_frame_rate"],
        "nominal_fps": _rate(stream["r_frame_rate"]),
        "frames": int(stream["nb_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_s": float(stream["duration"]),
        "time_base": stream["time_base"],
    }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _collision_prompt_pair(case: dict[str, Any]) -> tuple[str, str, str]:
    physics = case["physics"]
    if "ball_3_radius" in physics:
        return "three_ball_single_incident", PROMPT_C3_OLD, PROMPT_C3_NEW
    velocity_1 = float(physics["ball_1_initial_velocity"]["value"])
    velocity_2 = float(physics["ball_2_initial_velocity"]["value"])
    if abs(velocity_1) <= 1e-12 and velocity_2 < 0.0:
        return "two_ball_single_incident", PROMPT_C2_SINGLE_OLD, PROMPT_C2_SINGLE_NEW
    if velocity_1 > 0.0 and velocity_2 < 0.0:
        return "two_ball_opposed_incident", PROMPT_C2_OPPOSED_OLD, PROMPT_C2_OPPOSED_NEW
    raise ValueError(f"unexpected collision initial state: {case['case_id']}")


def repair_collision_prompts(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for case in cases:
        if case["scene_id"] != "collision_1d":
            continue
        physics = case["physics"]
        radius_1 = float(physics["ball_1_radius"]["value"])
        radius_2 = float(physics["ball_2_radius"]["value"])
        if abs(radius_1 - radius_2) <= 1e-12:
            continue
        family, old_prompt, new_prompt = _collision_prompt_pair(case)
        prompt = case["text"]["prompt"]
        if prompt not in {old_prompt, new_prompt}:
            raise ValueError(
                f"unexpected source prompt for audited collision {case['case_id']}"
            )
        case["text"]["prompt"] = new_prompt
        case["text"]["annotation_source"] = (
            "collision_process_prompt_v4_noncentral_safe"
        )
        changes.append({
            "case_id": case["case_id"],
            "prompt_family": family,
            "ball_1_radius_m": radius_1,
            "ball_2_radius_m": radius_2,
            "before": old_prompt,
            "after": new_prompt,
        })
    if len(changes) != 185:
        raise ValueError(f"expected 185 collision prompt repairs, found {len(changes)}")
    return changes


def encode_realtime_parabolic(case: dict[str, Any]) -> dict[str, Any]:
    alignment = case["alignment"]
    source = DATA_ROOT / case["assets"]["source_video"]
    old_reference = DATA_ROOT / OLD_REFERENCE
    old_first_frame = DATA_ROOT / OLD_FIRST_FRAME
    new_reference = DATA_ROOT / NEW_REFERENCE
    new_first_frame = DATA_ROOT / NEW_FIRST_FRAME
    crop = alignment["source_crop"]
    output = alignment["output_size"]
    start = int(alignment["source_start_frame"])
    end = int(alignment["source_end_frame_exclusive"])
    expected_frames = end - start
    physical_speedup = float(alignment["physical_playback_speedup"])
    already_retimed = (
        case["assets"]["reference_video"] == NEW_REFERENCE
        and case["assets"]["physics_reference_video"] == NEW_REFERENCE
        and case["assets"]["first_frame"] == NEW_FIRST_FRAME
        and case["temporal"]["encoded_to_physical_speed"] == 1.0
        and physical_speedup == 1.0
    )
    source_physical_speedup = 8.0
    output_fps = float(alignment["source_fps"]) * source_physical_speedup
    if expected_frames != 79 or output_fps != 240.0:
        raise ValueError("unexpected IMG_0539 source timing contract")

    if already_retimed:
        if not new_reference.is_file() or not new_first_frame.is_file():
            raise ValueError("partially repaired IMG_0539 media is missing")
        before_temporal = {
            "annotation_source": (
                "source frame rate and all frames in the retained source window "
                "are preserved; baseline adapters handle timing"
            ),
            "encoded_to_physical_speed": 8.0,
            "time_scale": "source_timing",
        }
        before_alignment = {
            "output_fps": 30.0,
            "output_frames": expected_frames,
            "physical_playback_speedup": 8.0,
            "timing_policy": "preserve_source_fps_and_all_trimmed_frames",
            "timing_audit_record": (
                "provenance/alignment/native_timing_20260731_v1/audit.jsonl"
            ),
        }
    else:
        if physical_speedup != 8.0:
            raise ValueError("unexpected IMG_0539 pre-repair speed factor")
        with tempfile.TemporaryDirectory(
            prefix=".v7-realtime-repair-", dir=new_reference.parent
        ) as temporary:
            temporary_root = Path(temporary)
            candidate_video = temporary_root / new_reference.name
            candidate_frame = temporary_root / new_first_frame.name
            filters = (
                f"trim=start_frame={start}:end_frame={end},"
                f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']},"
                f"scale={output['width']}:{output['height']}:flags=lanczos,"
                f"setpts=N/({output_fps:g}*TB)"
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-vf",
                    filters,
                    "-an",
                    "-r",
                    f"{output_fps:g}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-vsync",
                    "cfr",
                    "-video_track_timescale",
                    "24000",
                    "-movflags",
                    "+faststart",
                    str(candidate_video),
                ],
                check=True,
            )
            candidate_probe = probe(candidate_video)
            if (
                candidate_probe["frames"] != expected_frames
                or candidate_probe["nominal_fps"] != output_fps
                or candidate_probe["fps"] != output_fps
                or candidate_probe["width"] != int(output["width"])
                or candidate_probe["height"] != int(output["height"])
            ):
                raise ValueError(f"invalid real-time candidate: {candidate_probe}")
            expected_duration = expected_frames / output_fps
            if abs(candidate_probe["duration_s"] - expected_duration) > 1.0 / output_fps:
                raise ValueError(f"unexpected real-time duration: {candidate_probe}")
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(candidate_video),
                    "-frames:v",
                    "1",
                    str(candidate_frame),
                ],
                check=True,
            )
            os.replace(candidate_video, new_reference)
            os.replace(candidate_frame, new_first_frame)

        before_temporal = dict(case["temporal"])
        before_alignment = {
            key: alignment.get(key)
            for key in (
                "output_fps",
                "output_frames",
                "physical_playback_speedup",
                "timing_policy",
                "timing_audit_record",
            )
        }
        case["assets"]["reference_video"] = NEW_REFERENCE
        case["assets"]["physics_reference_video"] = NEW_REFERENCE
        case["assets"]["first_frame"] = NEW_FIRST_FRAME
        case["temporal"] = {
            "annotation_source": (
                "30 fps slow-motion source retimed to the measured physical 240 "
                "fps; all retained frames are preserved"
            ),
            "encoded_to_physical_speed": 1.0,
            "time_scale": "real_time",
        }
        alignment["output_fps"] = output_fps
        alignment["output_frames"] = expected_frames
        alignment["physical_playback_speedup"] = 1.0
        alignment["timing_policy"] = (
            "retime_slow_motion_source_to_physical_fps_preserve_all_frames"
        )
        alignment["timing_audit_record"] = (
            "releases/7.0.0/text_video_alignment_repair.json"
        )

    return {
        "case_id": case["case_id"],
        "source_video": case["assets"]["source_video"],
        "source_trim": {
            "start_frame": start,
            "end_frame_exclusive": end,
            "frames": expected_frames,
        },
        "retiming": {
            "source_encoded_fps": float(alignment["source_fps"]),
            "physical_playback_speedup_before": source_physical_speedup,
            "output_fps": output_fps,
            "ffmpeg_setpts": f"N/({output_fps:g}*TB)",
        },
        "before": {
            "reference_video": OLD_REFERENCE,
            "first_frame": OLD_FIRST_FRAME,
            "reference_sha256": sha256(old_reference),
            "first_frame_sha256": sha256(old_first_frame),
            "reference_probe": probe(old_reference),
            "temporal": before_temporal,
            "alignment_timing": before_alignment,
        },
        "after": {
            "reference_video": NEW_REFERENCE,
            "first_frame": NEW_FIRST_FRAME,
            "reference_sha256": sha256(new_reference),
            "first_frame_sha256": sha256(new_first_frame),
            "reference_size_bytes": new_reference.stat().st_size,
            "first_frame_size_bytes": new_first_frame.stat().st_size,
            "reference_probe": probe(new_reference),
            "temporal": dict(case["temporal"]),
            "alignment_timing": {
                key: alignment.get(key)
                for key in (
                    "output_fps",
                    "output_frames",
                    "physical_playback_speedup",
                    "timing_policy",
                    "timing_audit_record",
                )
            },
        },
    }


def refresh_release_lock() -> dict[str, Any]:
    """Replace only the two changed locked assets and recompute canonical digests."""

    descriptor = load_json(DESCRIPTOR_PATH)
    cases = load_jsonl(CASES_PATH)
    lock = load_json(LOCK_PATH)
    replacements = {
        OLD_REFERENCE: NEW_REFERENCE,
        OLD_FIRST_FRAME: NEW_FIRST_FRAME,
    }
    files_by_path = {item["path"]: item for item in lock["files"]}
    if len(files_by_path) != len(lock["files"]):
        raise ValueError("asset lock contains duplicate paths")
    for old_path, new_path in replacements.items():
        old = files_by_path.pop(old_path, None)
        if old is None:
            existing = files_by_path.get(new_path)
            if existing is None:
                raise ValueError(f"asset lock lacks repair source path: {old_path}")
            old = existing
        asset = DATA_ROOT / new_path
        files_by_path[new_path] = {
            **old,
            "path": new_path,
            "size_bytes": asset.stat().st_size,
            "sha256": sha256(asset),
        }
    files = [files_by_path[path] for path in sorted(files_by_path)]
    if len(files) != 1617:
        raise ValueError(f"unexpected repaired asset count: {len(files)}")
    lock["files"] = files
    lock["files_digest"] = canonical_sha256(files)

    views = {
        view_id: load_json(RELEASE_ROOT / relative)
        for view_id, relative in descriptor["views"].items()
    }
    scenes = {
        path.stem: load_json(path)
        for path in sorted((RELEASE_ROOT / descriptor["scene_catalog"]).glob("*.json"))
    }
    dataset_digest = canonical_sha256({
        "descriptor": descriptor,
        "cases": tuple(cases),
        "views": views,
        "scenes": scenes,
        "asset_lock": lock,
    })
    release_manifest = {
        "schema_version": "1.0",
        "dataset_id": descriptor["dataset_id"],
        "release": descriptor["release"],
        "dataset_digest": dataset_digest,
        "asset_files": len(files),
        "asset_files_digest": lock["files_digest"],
    }
    write_json(LOCK_PATH, lock)
    write_json(RELEASE_PATH, release_manifest)
    return release_manifest


def update_mapping(media_repair: dict[str, Any]) -> None:
    mapping = load_json(MAPPING_PATH)
    records = [
        record
        for record in mapping["records"]
        if record["case_id"] == PARABOLIC_CASE_ID
    ]
    if len(records) != 1:
        raise ValueError("IMG_0539 asset mapping record is missing or duplicated")
    paths = records[0]["paths"]
    for role in ("reference_video", "physics_reference_video"):
        paths[role]["corrected"] = NEW_REFERENCE
        paths[role]["same_inode"] = False
        paths[role]["sha256"] = media_repair["after"]["reference_sha256"]
    paths["first_frame"]["corrected"] = NEW_FIRST_FRAME
    paths["first_frame"]["same_inode"] = False
    paths["first_frame"]["sha256"] = media_repair["after"][
        "first_frame_sha256"
    ]
    write_json(MAPPING_PATH, mapping)


def main() -> int:
    descriptor = load_json(DESCRIPTOR_PATH)
    if (
        descriptor.get("dataset_id") != "physics_video_five_scene_v7"
        or descriptor.get("release") != "7.0.0"
        or descriptor.get("schema_version") != "4.0"
    ):
        raise ValueError("repair target is not Dataset 7.0.0 schema 4.0")

    if REPAIR_AUDIT_PATH.is_file():
        existing_audit = load_json(REPAIR_AUDIT_PATH)
        existing_release = load_json(RELEASE_PATH)
        cases = load_jsonl(CASES_PATH)
        collision_changes = repair_collision_prompts(cases)
        parabolic = [
            case for case in cases if case["case_id"] == PARABOLIC_CASE_ID
        ]
        state_is_repaired = (
            existing_audit.get("repair_id")
            == "dataset_v7_text_video_alignment_20260803_v1"
            and existing_audit.get("post_repair_dataset_digest")
            == existing_release.get("dataset_digest")
            and len(collision_changes) == 185
            and all(
                change["after"]
                == next(
                    case["text"]["prompt"]
                    for case in cases
                    if case["case_id"] == change["case_id"]
                )
                for change in collision_changes
            )
            and len(parabolic) == 1
            and parabolic[0]["assets"]["reference_video"] == NEW_REFERENCE
            and parabolic[0]["assets"]["first_frame"] == NEW_FIRST_FRAME
            and parabolic[0]["temporal"]["encoded_to_physical_speed"] == 1.0
            and (DATA_ROOT / NEW_REFERENCE).is_file()
            and (DATA_ROOT / NEW_FIRST_FRAME).is_file()
        )
        if not state_is_repaired:
            raise ValueError("repair audit exists but Dataset 7.0.0 is inconsistent")
        reference_probe = probe(DATA_ROOT / NEW_REFERENCE)
        if (
            reference_probe["frames"] != 79
            or reference_probe["fps"] != 240.0
            or reference_probe["nominal_fps"] != 240.0
        ):
            raise ValueError("repair audit exists but real-time media is invalid")
        print(json.dumps({
            "release": "7.0.0",
            "status": "already_repaired",
            "collision_prompts_repaired": 185,
            "parabolic_case_retimed": PARABOLIC_CASE_ID,
            "dataset_digest": existing_release["dataset_digest"],
        }, indent=2, sort_keys=True))
        return 0

    protected_paths = [
        CASES_PATH,
        MAPPING_PATH,
        MIGRATION_AUDIT_PATH,
        LOCK_PATH,
        RELEASE_PATH,
        REPAIR_AUDIT_PATH,
        DATA_ROOT / NEW_REFERENCE,
        DATA_ROOT / NEW_FIRST_FRAME,
    ]
    backups = {
        path: path.read_bytes() if path.is_file() else None
        for path in protected_paths
    }
    pre_release = load_json(RELEASE_PATH)

    try:
        cases = load_jsonl(CASES_PATH)
        if len(cases) != 593:
            raise ValueError(f"unexpected 7.0.0 Case count: {len(cases)}")
        collision_changes = repair_collision_prompts(cases)
        parabolic = [case for case in cases if case["case_id"] == PARABOLIC_CASE_ID]
        if len(parabolic) != 1:
            raise ValueError("IMG_0539 Case is missing or duplicated")
        media_repair = encode_realtime_parabolic(parabolic[0])
        write_jsonl(CASES_PATH, cases)
        update_mapping(media_repair)

        release_manifest = refresh_release_lock()
        post_cases = load_jsonl(CASES_PATH)
        repaired = {
            case["case_id"]: case for case in post_cases
        }[PARABOLIC_CASE_ID]
        if repaired["temporal"]["encoded_to_physical_speed"] != 1.0:
            raise ValueError("IMG_0539 did not become a real-time Case")

        family_counts: dict[str, int] = {}
        for change in collision_changes:
            family = change["prompt_family"]
            family_counts[family] = family_counts.get(family, 0) + 1
        repair_audit = {
            "schema_version": "1.0",
            "repair_id": "dataset_v7_text_video_alignment_20260803_v1",
            "dataset_id": descriptor["dataset_id"],
            "release": descriptor["release"],
            "release_strategy": "in_place_no_new_release",
            "pre_repair_dataset_digest": pre_release["dataset_digest"],
            "post_repair_dataset_digest": release_manifest["dataset_digest"],
            "collision_prompt_repair": {
                "selection": (
                    "first colliding ball pair has unequal annotated radii"
                ),
                "count": len(collision_changes),
                "counts_by_prompt_family": family_counts,
                "case_ids": sorted(change["case_id"] for change in collision_changes),
                "changes": collision_changes,
            },
            "parabolic_realtime_media_repair": media_repair,
            "historical_release_safety": {
                "old_shared_assets_overwritten": False,
                "old_shared_assets_retained_for_releases": ["5.1.0", "6.0.0"],
                "release_7_uses_dedicated_realtime_assets": True,
            },
            "asset_lock": {
                "asset_files": release_manifest["asset_files"],
                "asset_files_digest": release_manifest["asset_files_digest"],
            },
        }
        write_json(REPAIR_AUDIT_PATH, repair_audit)

        migration = load_json(MIGRATION_AUDIT_PATH)
        migration["output_dataset_digest"] = release_manifest["dataset_digest"]
        migration["media_files_modified"] = 1
        migration["text_records_modified"] = len(collision_changes)
        migration["in_place_repair_audit"] = REPAIR_AUDIT_PATH.name
        write_json(MIGRATION_AUDIT_PATH, migration)
    except BaseException:
        for path, payload in backups.items():
            if payload is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write_bytes(path, payload)
        raise

    print(json.dumps({
        "release": "7.0.0",
        "collision_prompts_repaired": len(collision_changes),
        "parabolic_case_retimed": PARABOLIC_CASE_ID,
        "new_reference": NEW_REFERENCE,
        "dataset_digest": release_manifest["dataset_digest"],
        "asset_files_digest": release_manifest["asset_files_digest"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
