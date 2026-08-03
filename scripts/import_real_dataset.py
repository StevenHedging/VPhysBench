#!/usr/bin/env python3
"""Import the confirmed real captures without rewriting their media streams.

The importer copies source bytes into stable per-case directories, derives only a
PNG first frame, records hashes/probes, writes the case manifest, and freezes both
benchmark views.  Re-running selected scenes replaces their manifest/audit rows
but refuses to silently overwrite a different destination video.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from physbench.io import load_jsonl, write_json, write_jsonl  # noqa: E402
from physbench.data_layout import (  # noqa: E402
    PHYSICS_VIDEO_ASSETS as ASSET_ROOT,
    PHYSICS_VIDEO_PROVENANCE,
    V1_CASES as MANIFEST,
    V1_VIEW_A as VIEW_A,
    V1_VIEW_B as VIEW_B,
)
from physbench.splitters import build_view_a, build_view_b  # noqa: E402


AUDIT = PHYSICS_VIDEO_PROVENANCE / "imports" / "import_audit.jsonl"
SOURCE_DOCS = PHYSICS_VIDEO_PROVENANCE / "source_docs"

COLLISION_FALL_ARCHIVE = WORKSPACE / "7_19碰撞_自由落体数据.zip"
PENDULUM_R2_ARCHIVE = WORKSPACE / "去后缀_处理后_R2_单摆实验.zip"
PENDULUM_R1_DIR = WORKSPACE / "wan22_pendulum_pipeline" / "data" / "raw" / "videos"


def quantity(value: float, unit: str, *, annotated: bool = True) -> dict[str, Any]:
    return {"value": value, "unit": unit, "annotated": annotated}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    duration = stream.get("duration") or payload.get("format", {}).get("duration")
    return {
        "codec": stream.get("codec_name"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames") not in {None, "N/A"} else None,
        "duration_s": float(duration) if duration not in {None, "N/A"} else None,
    }


def first_frame(video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-frames:v", "1", str(output)],
        check=True,
    )


def stable_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if destination.exists():
        if sha256(destination) != source_hash:
            raise FileExistsError(f"refusing to replace different media: {destination}")
        return
    shutil.copy2(source, destination)
    if sha256(destination) != source_hash:
        raise OSError(f"copy verification failed: {destination}")


def asset_path(
    scene: str, case_id: str, name: str, *, group: str = "canonical"
) -> str:
    return os.path.relpath(
        ASSET_ROOT / scene / case_id / group / name,
        MANIFEST.parent,
    )


def base_case(
    *,
    case_id: str,
    scene_id: str,
    split: str,
    physical_parameters: dict[str, Any],
    appearance: dict[str, Any],
    prompt: str,
    description: str,
    extension: str,
    temporal: dict[str, Any],
    ood_factors: list[str] | None = None,
) -> dict[str, Any]:
    first = asset_path(scene_id, case_id, "first_frame.png")
    video = asset_path(scene_id, case_id, f"reference{extension.lower()}")
    factors = ood_factors or []
    return {
        "schema_version": "1.0",
        "case_id": case_id,
        "scene_id": scene_id,
        "view_a_split": split,
        "physical_parameters": physical_parameters,
        "appearance": appearance,
        "temporal": temporal,
        "text": {"description": description, "prompt": prompt},
        "assets": {
            "first_frame": first,
            "reference_video": video,
            "physics_reference_video": video,
            "subject_mask": None,
        },
        "has_real_reference_video": True,
        "ood": {"level": "ood1" if factors else "id", "factors": factors},
        "provenance": {"source_kind": "real_capture", "parent_case_id": None, "generator": None},
        "input_views": {
            "t2v": {"prompt": prompt},
            "i2v": {"prompt": prompt, "first_frame": first},
            "ti2v": {"prompt": prompt, "first_frame": first},
        },
    }


def materialize_case(
    case: dict[str, Any],
    source: Path,
    *,
    source_locator: dict[str, Any],
) -> dict[str, Any]:
    relative = Path(case["assets"]["reference_video"])
    destination = (MANIFEST.parent / relative).resolve()
    stable_copy(source, destination)
    frame = (MANIFEST.parent / case["assets"]["first_frame"]).resolve()
    first_frame(destination, frame)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise OSError(f"source/destination hashes differ for {case['case_id']}")
    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "source": source_locator,
        "source_bytes": source.stat().st_size,
        "source_sha256": source_hash,
        "destination": str(destination.relative_to(ROOT)),
        "destination_sha256": destination_hash,
        "first_frame": str(frame.relative_to(ROOT)),
        "probe": probe(destination),
        "temporal": case["temporal"],
        "copy_policy": "byte_preserving_copy; first_frame_is_derived",
    }


def materialize_source_timing_case(
    case: dict[str, Any],
    source: Path,
    *,
    source_locator: dict[str, Any],
    speed_factor: float,
) -> dict[str, Any]:
    """Keep source bytes/timing and defer physical-time adaptation to baselines."""
    if speed_factor <= 0:
        raise ValueError("speed_factor must be positive")
    destination = (MANIFEST.parent / case["assets"]["reference_video"]).resolve()
    source_asset = (MANIFEST.parent / case["assets"]["source_video"]).resolve()
    stable_copy(source, source_asset)
    stable_copy(source, destination)
    source_probe = probe(source_asset)
    destination_probe = probe(destination)
    if sha256(source_asset) != sha256(destination):
        raise RuntimeError("source-timing reference is not byte-identical")
    if destination_probe != source_probe:
        raise RuntimeError("source-timing reference probe changed")

    frame = (MANIFEST.parent / case["assets"]["first_frame"]).resolve()
    first_frame(destination, frame)
    return {
        "case_id": case["case_id"],
        "scene_id": case["scene_id"],
        "source": source_locator,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "source_asset": str(source_asset.relative_to(ROOT)),
        "source_asset_sha256": sha256(source_asset),
        "source_probe": source_probe,
        "destination": str(destination.relative_to(ROOT)),
        "destination_sha256": sha256(destination),
        "probe": destination_probe,
        "first_frame": str(frame.relative_to(ROOT)),
        "temporal": case["temporal"],
        "transformation": {
            "kind": "preserve_source_timing",
            "encoded_to_physical_speed": speed_factor,
            "codec_policy": "byte_identical_copy_without_reencoding",
            "frame_policy": "preserve_all_encoded_frames",
        },
        "copy_policy": (
            "source and canonical reference are byte-identical; baseline "
            "adapter handles physical-time restoration"
        ),
    }


def zip_videos(archive: Path, predicate: Any) -> Iterable[tuple[str, Path]]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(prefix="physbench_import_") as tmp:
        temporary = Path(tmp)
        for member in sorted(bundle.namelist()):
            if member.endswith("/") or not predicate(member):
                continue
            extracted = temporary / Path(member).name
            with bundle.open(member) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
            yield member, extracted


COLLISION_BALLS = {
    "大": {"name": "large_steel", "material": "steel", "mass_kg": 0.06477, "radius_m": 0.0125},
    "中": {"name": "medium_steel", "material": "steel", "mass_kg": 0.03310, "radius_m": 0.0100},
    "小": {"name": "small_steel", "material": "steel", "mass_kg": 0.01400, "radius_m": 0.0075},
    "波": {"name": "glass_marble", "material": "glass", "mass_kg": 0.00346, "radius_m": 0.00675},
}


def import_collision() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[tuple[str, str, float]] = []
    pattern = re.compile(r"^([大中小波]{3}) ([0-9]+\.[0-9]+)\.MOV$", re.IGNORECASE)
    with zipfile.ZipFile(COLLISION_FALL_ARCHIVE) as bundle:
        for member in sorted(bundle.namelist()):
            if "/R2_碰撞实验/" not in member or not member.lower().endswith(".mov"):
                continue
            match = pattern.match(Path(member).name)
            if not match:
                raise ValueError(f"unrecognized collision filename: {member}")
            records.append((member, match.group(1), float(match.group(2))))
    by_sequence: dict[str, list[float]] = {}
    for _, sequence, speed in records:
        by_sequence.setdefault(sequence, []).append(speed)
    held_out = {
        sequence: sorted(speeds)[len(speeds) // 2]
        for sequence, speeds in by_sequence.items()
        if sequence in {"大大大", "中中中", "小小小"}
    }

    cases: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for member, source in zip_videos(
        COLLISION_FALL_ARCHIVE,
        lambda name: "/R2_碰撞实验/" in name and name.lower().endswith(".mov"),
    ):
        match = pattern.match(Path(member).name)
        assert match is not None
        sequence, speed = match.group(1), float(match.group(2))
        balls = [COLLISION_BALLS[token] for token in sequence]
        sequence_id = "_".join(ball["name"] for ball in balls)
        case_id = f"collision_r2_{sequence_id}_v{round(speed * 10000):05d}"
        homogeneous_steel = sequence in {"大大大", "中中中", "小小小"}
        if homogeneous_steel:
            split = "test_id" if speed == held_out[sequence] else "train"
            factors: list[str] = []
        else:
            split = "test_ood1"
            factors = ["collision_pair"]
            if "波" in sequence:
                factors.append("ball_material")
        parameters = {"striker_initial_velocity": quantity(speed, "m/s")}
        for index, ball in enumerate(balls, 1):
            parameters[f"ball_{index}_mass"] = quantity(ball["mass_kg"], "kg")
            parameters[f"ball_{index}_radius"] = quantity(ball["radius_m"], "m")
            parameters[f"ball_{index}_initial_velocity"] = quantity(speed if index == 1 else 0.0, "m/s")
        ball_text = ", ".join(
            f"ball {index} is {ball['material']} with mass {ball['mass_kg']:.5f} kg and radius {ball['radius_m']:.5f} m"
            for index, ball in enumerate(balls, 1)
        )
        prompt = (
            "A fixed-camera real-world laboratory video of a one-dimensional central collision among three aligned balls. "
            f"Ball 1 moves toward initially stationary balls 2 and 3 at {speed:.4f} m/s; {ball_text}. "
            "The camera and track remain stationary, and the collision unfolds at the true physical time scale."
        )
        case = base_case(
            case_id=case_id,
            scene_id="collision_1d",
            split=split,
            physical_parameters=parameters,
            appearance={
                "capture_session": "R2_collision",
                "camera": "fixed",
                "ball_sequence": [ball["name"] for ball in balls],
                "ball_materials": [ball["material"] for ball in balls],
            },
            prompt=prompt,
            description=f"一维对心碰撞：{sequence}，入射球初速度 {speed:.4f} m/s。",
            extension=".mov",
            temporal={
                "encoded_to_physical_speed": 1.0,
                "time_scale": "real_time",
                "annotation_source": "user-confirmed true physical time scale",
            },
            ood_factors=factors,
        )
        source_asset = asset_path(
            "collision_1d", case_id, "reference.mov", group="source"
        )
        case["assets"]["source_video"] = source_asset
        case["assets"]["reference_video"] = source_asset
        case["assets"]["physics_reference_video"] = source_asset
        audit.append(materialize_case(case, source, source_locator={
            "archive": str(COLLISION_FALL_ARCHIVE), "member": member,
        }))
        cases.append(case)
    return cases, audit


FREE_FALL_BALLS = {
    "XL": {"mass_kg": 0.07267, "radius_m": 0.0250},
    "L": {"mass_kg": 0.02410, "radius_m": 0.0170},
    "M": {"mass_kg": 0.01500, "radius_m": 0.0144},
    "S": {"mass_kg": 0.00450, "radius_m": 0.0100},
}


def import_free_fall() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pattern = re.compile(r"^(XL|L|M|S)_(60|80|100)\.mp4$", re.IGNORECASE)
    cases: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for member, source in zip_videos(
        COLLISION_FALL_ARCHIVE,
        lambda name: "/处理后_R2_自由落体实验/" in name and name.lower().endswith(".mp4"),
    ):
        match = pattern.match(Path(member).name)
        if not match:
            raise ValueError(f"unrecognized free-fall filename: {member}")
        size, height_cm = match.group(1).upper(), int(match.group(2))
        ball = FREE_FALL_BALLS[size]
        case_id = f"freefall_r2_{size.lower()}_h{height_cm:03d}cm"
        split = "test_id" if height_cm == 80 else "train"
        prompt = (
            "A fixed-camera real-world laboratory video of a ball released from rest in free fall. "
            f"The initial vertical drop height is {height_cm / 100:.2f} m, the ball radius is "
            f"{ball['radius_m']:.4f} m, and its mass is {ball['mass_kg']:.5f} kg. "
            "The ball starts with zero velocity and accelerates downward under gravity at the true physical time scale."
        )
        case = base_case(
            case_id=case_id,
            scene_id="free_fall",
            split=split,
            physical_parameters={
                "initial_height": quantity(height_cm / 100, "m"),
                "ball_radius": quantity(ball["radius_m"], "m"),
                "ball_mass": quantity(ball["mass_kg"], "kg"),
                "initial_velocity": quantity(0.0, "m/s"),
            },
            appearance={
                "capture_session": "R2_free_fall",
                "camera": "fixed",
                "ball_size_class": size,
                "ball_material": "not_documented",
            },
            prompt=prompt,
            description=f"自由落体：{size} 球从 {height_cm} cm 高度静止释放。",
            extension=".mp4",
            temporal={
                "encoded_to_physical_speed": 8.0,
                "time_scale": "source_timing",
                "annotation_source": (
                    "original 8x slow-motion timing is preserved; baseline "
                    "adapters handle physical-time restoration"
                ),
            },
        )
        case["assets"]["source_video"] = asset_path(
            "free_fall", case_id, "source_slowmo.mp4", group="source"
        )
        audit.append(materialize_source_timing_case(
            case,
            source,
            source_locator={"archive": str(COLLISION_FALL_ARCHIVE), "member": member},
            speed_factor=8.0,
        ))
        cases.append(case)
    return cases, audit


PENDULUM_PATTERN = re.compile(
    r"^pendulum_ltot(?P<total>[0-9]{4})mm_lrope(?P<string>[0-9]{4})mm_"
    r"r010mm_a(?P<angle>[0-9]{3})deg_r1\.mp4$"
)


def pendulum_case(
    *, source: Path, total_m: float, string_m: float, angle: int, session: str,
    source_locator: dict[str, Any], extension: str = ".mp4",
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = (
        f"pendulum_{session}_ltot{round(total_m * 1000):04d}mm_"
        f"lrope{round(string_m * 1000):04d}mm_r010mm_a{angle:03d}deg"
    )
    prompt = (
        "A fixed-camera real-world laboratory video of a simple pendulum released from rest at the first frame. "
        "The support and camera remain stationary, and the bob swings naturally under gravity. "
        f"Physical parameters: the string length is {string_m:.3f} meters; the spherical bob radius is 0.010 meters; "
        f"the pivot-to-bob-center pendulum length is {total_m:.3f} meters; the initial release angle is {angle:.1f} degrees. "
        f"Parameters: string_length_m={string_m:.3f}; bob_radius_m=0.010; "
        f"pendulum_length_m={total_m:.3f}; initial_angle_deg={angle:.1f}."
    )
    case = base_case(
        case_id=case_id,
        scene_id="pendulum",
        split="test_id" if angle == 20 else "train",
        physical_parameters={
            "initial_angle": quantity(float(angle), "deg"),
            "string_length": quantity(string_m, "m"),
            "bob_radius": quantity(0.010, "m"),
            "pendulum_length": quantity(total_m, "m"),
        },
        appearance={
            "capture_session": session.upper(),
            "camera": "fixed",
            "bob_material": "not_documented",
            "support": "original_real_capture",
            "background": "original_real_capture",
        },
        prompt=prompt,
        description=(
            f"单摆：绳长 {string_m:.3f} m，球半径 0.010 m，"
            f"支点到球心总长 {total_m:.3f} m，初始摆角 {angle}°。"
        ),
        extension=extension,
        temporal={
            "encoded_to_physical_speed": 1.0,
            "time_scale": "real_time",
            "annotation_source": (
                "existing trusted R1 manifest" if session == "r1"
                else "user-confirmed corrected replacement R2"
            ),
        },
    )
    return case, materialize_case(case, source, source_locator=source_locator)


def import_pendulum() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for source in sorted(PENDULUM_R1_DIR.glob("*r1.mp4")):
        match = PENDULUM_PATTERN.match(source.name)
        if not match:
            raise ValueError(f"unrecognized trusted R1 filename: {source.name}")
        case, record = pendulum_case(
            source=source,
            total_m=int(match.group("total")) / 1000,
            string_m=int(match.group("string")) / 1000,
            angle=int(match.group("angle")),
            session="r1",
            source_locator={"path": str(source)},
        )
        cases.append(case)
        audit.append(record)

    pattern = re.compile(r"^(11|13|15\.5|18)_(5|10|20|25|30)\.mp4$")
    for member, source in zip_videos(PENDULUM_R2_ARCHIVE, lambda name: name.lower().endswith(".mp4")):
        match = pattern.match(Path(member).name)
        if not match:
            raise ValueError(f"unrecognized replacement R2 filename: {member}")
        total_cm = float(match.group(1))
        angle = int(match.group(2))
        case, record = pendulum_case(
            source=source,
            total_m=total_cm / 100,
            string_m=(total_cm - 1.0) / 100,
            angle=angle,
            session="r2",
            source_locator={"archive": str(PENDULUM_R2_ARCHIVE), "member": member},
        )
        cases.append(case)
        audit.append(record)
    return cases, audit


def copy_source_docs() -> None:
    SOURCE_DOCS.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(COLLISION_FALL_ARCHIVE) as bundle:
        for member, output in (
            (next(name for name in bundle.namelist() if "/R2_碰撞实验/小球规格.txt" in name),
             SOURCE_DOCS / "collision_1d_ball_spec.txt"),
            (next(name for name in bundle.namelist() if "/处理后_R2_自由落体实验/小球规格.txt" in name),
             SOURCE_DOCS / "free_fall_ball_spec.txt"),
        ):
            output.write_bytes(bundle.read(member))
    write_json(SOURCE_DOCS / "normalized_ball_specs.json", {
        "collision_1d": COLLISION_BALLS,
        "free_fall": FREE_FALL_BALLS,
        "note": "The source documents report diameters; radius_m is diameter/2, user-confirmed 2026-07-19.",
    })


def write_pendulum_annotations() -> None:
    SOURCE_DOCS.mkdir(parents=True, exist_ok=True)
    write_json(SOURCE_DOCS / "pendulum_annotations.json", {
        "annotation_confirmation_date": "2026-07-19",
        "filename_contract": {
            "first_number": "pivot-to-bob-center total pendulum length in centimeters",
            "second_number": "initial release angle in degrees",
        },
        "bob_radius_m": 0.010,
        "string_length_formula": "total_pendulum_length_m - bob_radius_m",
        "R1": {
            "source": "existing trusted R1 captures",
            "time_scale": "real_time",
            "case_count": 16,
        },
        "R2": {
            "source_archive": str(PENDULUM_R2_ARCHIVE),
            "time_scale": "real_time (corrected replacement for polluted old R2)",
            "case_count": 19,
            "known_capture_exception": "11_25.mp4 is 30 fps; the other replacement R2 files are 120 fps",
            "known_missing_capture": "18_30.mp4 was not collected",
        },
        "view_a_policy": "For every captured total length, angle 20 deg is test_id and all other angles are train.",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenes", nargs="+", choices=["collision_1d", "free_fall", "pendulum"],
        default=["collision_1d", "free_fall"],
    )
    parser.add_argument(
        "--confirm-pendulum-r2-real-time", action="store_true",
        help="Required before importing the corrected replacement R2 archive.",
    )
    args = parser.parse_args()
    selected = set(args.scenes)
    if "pendulum" in selected and not args.confirm_pendulum_r2_real_time:
        raise ValueError("pendulum import requires --confirm-pendulum-r2-real-time")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise FileNotFoundError("ffmpeg and ffprobe are required for derived first frames and audit probes")

    imported_cases: list[dict[str, Any]] = []
    imported_audit: list[dict[str, Any]] = []
    importers = {
        "collision_1d": import_collision,
        "free_fall": import_free_fall,
        "pendulum": import_pendulum,
    }
    for scene in ("collision_1d", "free_fall", "pendulum"):
        if scene in selected:
            cases, audit = importers[scene]()
            imported_cases.extend(cases)
            imported_audit.extend(audit)

    existing_cases = load_jsonl(MANIFEST) if MANIFEST.exists() else []
    existing_audit = load_jsonl(AUDIT) if AUDIT.exists() else []
    merged_cases = [case for case in existing_cases if case.get("scene_id") not in selected]
    merged_cases.extend(imported_cases)
    merged_cases.sort(key=lambda case: (case["scene_id"], case["case_id"]))
    merged_audit = [row for row in existing_audit if row.get("scene_id") not in selected]
    merged_audit.extend(imported_audit)
    merged_audit.sort(key=lambda row: (row["scene_id"], row["case_id"]))

    write_jsonl(MANIFEST, merged_cases)
    write_jsonl(AUDIT, merged_audit)
    write_json(VIEW_A, build_view_a(merged_cases))
    write_json(VIEW_B, build_view_b(merged_cases, groups=5, seed=42))
    if {"collision_1d", "free_fall"} & selected:
        copy_source_docs()
    if "pendulum" in selected:
        write_pendulum_annotations()

    counts: dict[str, dict[str, int]] = {}
    for case in merged_cases:
        counts.setdefault(case["scene_id"], {}).setdefault(case["view_a_split"], 0)
        counts[case["scene_id"]][case["view_a_split"]] += 1
    print(json.dumps({
        "manifest": str(MANIFEST),
        "total_cases": len(merged_cases),
        "selected_scenes": sorted(selected),
        "counts": counts,
        "view_a": str(VIEW_A),
        "view_b": str(VIEW_B),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
